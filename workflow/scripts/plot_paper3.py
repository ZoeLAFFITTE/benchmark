from pathlib import Path

import numpy as np
import pandas as pd
import slam.io as sio
import slam.remeshing as srem
import yaml
import slam.texture as stex
import slam.plot as splt
import plotly.graph_objects as go
import plotly.express as px
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Taille minimale (en sommets) d'un îlot de bassin avant nettoyage
MIN_BASIN_SIZE = 20


CONFIG_FILE = Path("../config/config.yaml")

with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

DATA = Path(config["data_dir"])
DATA_TEST = Path(config["data_test"])

ATLAS = Path(config["atlas_dir"])
DIR = config["dir"]
SOURCES = config["sources"]
MESH = config["mesh_types"]

RAW = DIR["rawdata"]
NATIVE = SOURCES["native"]
REG = SOURCES["reg"]

WHITE = MESH["white"]
BASIN = MESH["basin"]
SPHEREREG = MESH["sphere.reg"]
SPHERETEMP = MESH["sphere.template"]
WHITETEMP = MESH["white.template"]

OUT = Path("benchmark_final")

SESSION_TEST = "HCP_test"
SESSION_RETEST = "HCP_retest"

TSV_FILE = OUT / "aggregated" / "basin_metrics_test_retest.tsv"

OUTPUT_DIR = OUT / "population_textures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PATHS
# ============================================================

def path_basins_native(test, sub, side):
    return DATA_TEST / f"{test}_{sub}_{side}_{BASIN}.gii"


def path_sphere_reg(test, sub, side):
    return (
        DATA / RAW / test / str(sub) / REG
        / f"{sub}.{side}.{SPHEREREG}.surf.gii"
    )

def path_zoe(test, sub, side):
    return (
        OUT / test / sub / "zoe"
        / f"{sub}.{side}.{WHITE}.sulci.gii"
    )

def to_template(sphere_mesh, sphere_mesh_template, texture):
    return np.asarray(
        srem.texture_spherical_interpolation_nearest_neighbor(
            sphere_mesh, sphere_mesh_template, texture
        ),
        dtype=float,
    ).squeeze()

def path_white_mesh_native(test, sub, side):
    # return (
    #     DATA / RAW / test / sub / NATIVE
    #     / f"{sub}.{side}.inflated.native.surf.gii"
    # )
    return (
        DATA / RAW / test / sub / NATIVE
        / f"{sub}.{side}.{WHITE}.surf.gii"
    )

def clean_labels(faces, labels, min_size=20, max_iter=20):
    """Supprime les petits îlots de labels (bruit du rééchantillonnage).

    Les composantes connexes d'un même label de moins de min_size sommets
    sont réaffectées au label de leurs voisins.
    """
    faces = np.asarray(faces)
    labels = np.asarray(labels).copy()
    n = len(labels)

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a, b = edges[:, 0], edges[:, 1]

    same = labels[a] == labels[b]
    graph = coo_matrix(
        (np.ones(same.sum()), (a[same], b[same])), shape=(n, n)
    )
    _, comp = connected_components(graph, directed=False)
    small = np.bincount(comp)[comp] < min_size

    for _ in range(max_iter):
        if not small.any():
            break
        for u, v in ((a, b), (b, a)):
            sel = small[u] & ~small[v]
            labels[u[sel]] = labels[v[sel]]
            small[u[sel]] = False
    return labels


def border_mask(faces, labels):
    """Frontière d'un sommet d'épaisseur (vectorisé).

    Pour chaque arête reliant deux bassins différents, seul le sommet du
    bassin de plus petit label est marqué : la ligne ne fait qu'un sommet
    d'épaisseur au lieu de deux.
    """
    faces = np.asarray(faces)
    labels = np.asarray(labels)
    mask = np.zeros(len(labels), dtype=bool)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        va, vb = faces[:, a], faces[:, b]
        diff = labels[va] != labels[vb]
        lower = np.where(labels[va] < labels[vb], va, vb)
        mask[lower[diff]] = True
    return mask


sub = "130518"
side = "L"
test = "HCP_test"

sphere_mesh_template = sio.load_mesh(
        f"{ATLAS}.{side}.{SPHERETEMP}.surf.gii"
    )

white_mesh_template = sio.load_mesh(
        f"{ATLAS}.{side}.{WHITETEMP}.surf.gii"
    )

sphere_mesh = sio.load_mesh(str(path_sphere_reg(test, sub, side)))


# Chargement des données du sujet
dpf_star = sio.load_texture(f"{OUT}/{test}/{sub}/preproc/{sub}.{side}.white.native.DPF_star.gii").darray[0]
basin = sio.load_texture(path_basins_native(test, sub, side)).darray[0]

tex_dpf = to_template(sphere_mesh, sphere_mesh_template, dpf_star)

# Labels de la session reportés sur le template, puis nettoyés
basins_reg = np.rint(
    to_template(sphere_mesh, sphere_mesh_template, basin.astype(float))
).astype(int)

basins_reg = clean_labels(
    sphere_mesh_template.faces, basins_reg, min_size=MIN_BASIN_SIZE
)

# Extraction de la bordure sous forme de texture binaire (1 = bordure, 0 = sinon)
border_texture = border_mask(sphere_mesh_template.faces, basins_reg).astype(int)

white_mesh = sio.load_mesh(path_white_mesh_native(test, sub, side))
border_ind = border_mask(white_mesh.faces, basin).astype(int)

fig1 = splt.plot_mesh(
    {"vertices": white_mesh.vertices, "faces": white_mesh.faces},
    {
        "values": dpf_star,
        "mode": "vertex",
        "cmin": -0.025,
        "cmax": np.max(tex_dpf),
    },
    {"template": None, "colorscale": "Agsunset"},
    caption=False,
)
fig1.write_html(str(OUTPUT_DIR / f"template_dpf.html"))
#fig1.show()


fig2 = splt.plot_mesh(
    {"vertices": white_mesh.vertices, "faces": white_mesh.faces},
    {
        "values": border_ind,
        "mode": "vertex",
    },
    {"template": None, "colorscale":[[0, "white"], [0.5, "red"], [0.5, "red"], [1, "red"]]},
    caption=False,
)
edge_vertices = white_mesh.vertices[white_mesh.edges_unique]
coords = np.full((len(edge_vertices), 3, 3), np.nan)
coords[:, :2, :] = edge_vertices

fig2.add_trace(
    go.Scatter3d(
        x=coords[:, :, 0].ravel(),
        y=coords[:, :, 1].ravel(),
        z=coords[:, :, 2].ravel(),
        mode="lines",
        line={"color": "black", "width": 1},
        showlegend=False,
    )
)
fig2.write_html(str(OUTPUT_DIR / f"border.html"))
#fig2.show()


#test pour guillaume
dpf_star_modif = dpf_star.copy()
dpf_star_modif[border_ind==1]=max(dpf_star_modif)

fig3 = splt.plot_mesh(
    {"vertices": white_mesh.vertices, "faces": white_mesh.faces},
    {
        "values": dpf_star_modif,
        "mode": "vertex",
        "cmin": -0.025,
        "cmax": np.max(tex_dpf),
    },
    {"template": None, "colorscale": "Agsunset"},
    caption=False,
)
#fig3.show()
fig3.write_html(str(OUTPUT_DIR / f"merge.html"))
