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


# ============================================================
# OUTILS
# ============================================================

def make_colorscale(border=False):
    colors = px.colors.sample_colorscale("Turbo", np.linspace(0, 1, 255))
    colorscale = [[0.0, "white"]] + [
        [i / 255 + 1 / 255, color] for i, color in enumerate(colors)
    ]
    if border:
        colorscale[-1] = [1.0, "black"]  # la valeur max (= frontière) est noire
    return colorscale


def save_figure(texture, side, out_html, border=False):
    """Carte sur le template blanc, avec le maillage en surimpression.

    Avec border=True, la texture contient les sommets de frontière à la
    valeur max, affichée en noir.
    """
    mesh = sio.load_mesh(f"{ATLAS}.{side}.{WHITETEMP}.surf.gii")

    fig = splt.plot_mesh(
        {"vertices": mesh.vertices, "faces": mesh.faces},
        {"values": texture, "mode": "vertex"},
        {"template": None, "colorscale": make_colorscale(border)},
        caption=False,
    )

    # edge_vertices = mesh.vertices[mesh.edges_unique]
    # coords = np.full((len(edge_vertices), 3, 3), np.nan)
    # coords[:, :2, :] = edge_vertices

    # fig.add_trace(
    #     go.Scatter3d(
    #         x=coords[:, :, 0].ravel(),
    #         y=coords[:, :, 1].ravel(),
    #         z=coords[:, :, 2].ravel(),
    #         mode="lines",
    #         line={"color": "white", "width": 1},
    #         showlegend=False,
    #     )
    # )

    fig.write_html(str(out_html))


def to_template(sphere_mesh, sphere_mesh_template, texture):
    return np.asarray(
        srem.texture_spherical_interpolation_nearest_neighbor(
            sphere_mesh, sphere_mesh_template, texture
        ),
        dtype=float,
    ).squeeze()


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


def add_border(texture, mask):
    """Copie de la texture avec les frontières à max + 0.1."""
    texture_bis = texture.copy()
    texture_bis[mask] = np.max(texture) + 0.1
    return texture_bis


# ============================================================
# LOAD TSV
# ============================================================

df = pd.read_csv(TSV_FILE, sep="\t")

SUBJECTS = sorted(df["subject"].unique())
SIDES = sorted(df["side"].unique())
METHODS = sorted(df["method"].unique())

print(
    f"{len(df)} lignes, {len(SUBJECTS)} sujets, "
    f"{len(SIDES)} hémisphères, méthodes : {METHODS}"
)


# ============================================================
# TEXTURES INDIVIDUELLES (une par sujet, hémisphère et session)
# ============================================================

MEASURES = {"leaves": "nb_feuilles", "length": "length"}

# clé : (method, side, measure) -> textures de tous les sujets et sessions
population_textures = {}

def basin_values(df_sub, session, prefix):
    """Valeur / sqrt(aire) par bassin, pour une session donnée."""
    values = {}
    for _, row in df_sub.iterrows():
        basin = row[f"basin_{session}"]
        area = row[f"area_basin_{session}"]
        value = row[f"{prefix}_{session}"]
        if (
            pd.isna(basin) or pd.isna(value) or pd.isna(area)
            or not np.isfinite(area) or area <= 0
        ):
            continue
        values[int(basin)] = float(value / np.sqrt(area))
    return values


for (method, sub, side), df_sub in df.groupby(["method", "subject", "side"]):
    print(f"\n===== Méthode {method}, sujet {sub}, hémisphère {side} =====")

    sphere_mesh_template = sio.load_mesh(
        f"{ATLAS}.{side}.{SPHERETEMP}.surf.gii"
    )

    # un jeu de cartes par session : labels, valeurs et frontières
    # proviennent tous de la même session
    for test in [SESSION_TEST, SESSION_RETEST]:
        session = "test" if test == SESSION_TEST else "retest"

        sphere_mesh = sio.load_mesh(str(path_sphere_reg(test, sub, side)))
        basins = np.asarray(
            sio.load_texture(str(path_basins_native(test, sub, side))).darray
        ).squeeze()

        # labels de la session reportés sur le template, puis nettoyés
        basins_reg = np.rint(
            to_template(sphere_mesh, sphere_mesh_template, basins.astype(float))
        ).astype(int)
        basins_reg = clean_labels(
            sphere_mesh_template.faces, basins_reg, min_size=MIN_BASIN_SIZE
        )

        # frontières issues des mêmes labels que ceux utilisés pour colorer
        border = border_mask(sphere_mesh_template.faces, basins_reg)

        for measure, prefix in MEASURES.items():
            values = basin_values(df_sub, session, prefix)
            texture = np.zeros(len(basins_reg), dtype=float)
            for basin, value in values.items():
                texture[basins_reg == basin] = value

            population_textures.setdefault(
                (method, side, measure), []
            ).append(texture)

            stem = f"{method}_{sub}_{side}_{session}_{measure}"

            # carte classique
            sio.write_texture(
                stex.TextureND(darray=texture),
                str(OUTPUT_DIR / f"{stem}.gii"),
            )
            save_figure(texture, side, OUTPUT_DIR / f"{stem}.html")

            # carte bis : mêmes valeurs + frontières des bassins de la session
            texture_bis = add_border(texture, border)
            sio.write_texture(
                stex.TextureND(darray=texture_bis),
                str(OUTPUT_DIR / f"{stem}_bis.gii"),
            )
            save_figure(
                texture_bis, side, OUTPUT_DIR / f"{stem}_bis.html", border=True
            )


# ============================================================
# TEXTURES MOYENNES DE POPULATION (méthode x hémisphère, test + retest)
# ============================================================

for (method, side, measure), textures in population_textures.items():
    textures = np.vstack(textures)

    texture_mean = np.mean(textures, axis=0)
    texture_std = np.std(textures, axis=0)

    stem = f"{method}_{side}_{measure}"

    sio.write_texture(
        stex.TextureND(darray=texture_mean),
        str(OUTPUT_DIR / f"{stem}_mean.gii"),
    )
    sio.write_texture(
        stex.TextureND(darray=texture_std),
        str(OUTPUT_DIR / f"{stem}_std.gii"),
    )

    mesh = sio.load_mesh(f"{ATLAS}.{side}.{WHITETEMP}.surf.gii")

    turbo = px.colors.get_colorscale("Turbo")
    colorscale = [[0.0, "white"], [0.1, "white"]] + [
        [0.1 + (1 - 0.1) * p, c] for p, c in turbo
    ]

    fig = splt.plot_mesh(
        {"vertices": mesh.vertices, "faces": mesh.faces},
        {
            "values": texture_mean,
            "mode": "vertex",
            "cmin": 0.15,
            "cmax": np.max(texture_mean),
        },
        {"template": None},
        caption=False,
    )

    # edge_vertices = mesh.vertices[mesh.edges_unique]
    # coords = np.full((len(edge_vertices), 3, 3), np.nan)
    # coords[:, :2, :] = edge_vertices

    # fig.add_trace(
    #     go.Scatter3d(
    #         x=coords[:, :, 0].ravel(),
    #         y=coords[:, :, 1].ravel(),
    #         z=coords[:, :, 2].ravel(),
    #         mode="lines",
    #         line={"color": "white", "width": 1},
    #         showlegend=False,
    #     )
    # )

    fig.write_html(str(OUTPUT_DIR / f"{stem}_mean.html"))

    print(
        f"Texture moyenne : {method}, {side}, {measure}, "
        f"{len(texture_mean)} sommets, {len(textures)} sujets"
    )

print("\nTraitement terminé.")