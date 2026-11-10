#!/usr/bin/env python3
"""
Carte moyenne de complexité (nb de feuilles par bassin) sur la sphère template.

Pour chaque hémisphère, session (test/retest) et sujet :
  1. nb de feuilles par bassin (TSV de stats, colonne n_endpoints)
  2. carte native : chaque sommet reçoit le nb de feuilles de son bassin (0 sinon)
  3. interpolation plus proche voisin vers la sphère template via la sphere.reg
Puis moyenne de toutes les cartes (par session et globale).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import slam.io as sio
import slam.remeshing as srem
import slam.texture as stex
import slam.plot as splt
import yaml
import plotly.express as px

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_FILE = Path("../config/config.yaml")
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

DATA = Path(config["data_dir"])
DATA_TEST = Path(config["data_test"])      # textures de bassins natifs
ATLAS = config["atlas_dir"]
DIR = config["dir"]
SOURCES = config["sources"]
MESH = config["mesh_types"]

SIDES = config["side"]
sides = [SIDES] if isinstance(SIDES, str) else list(SIDES)   # ["L"] ici
# pour les deux hémisphères : sides = ["L", "R"]

THR_BASINS = config["thr_basins"]          # 0.8 (déjà appliqué dans les .npy)

RAW = DIR["rawdata"]
NATIVE = SOURCES["native"]
REG = SOURCES["reg"]

WHITE = MESH["white"]
BASIN = MESH["basin"]
SPHEREREG = MESH["sphere.reg"]
SPHERETEMP = MESH["sphere.template"]

OUT = Path("benchmark_final")
TSV = OUT / "aggregated" / "zoe_stats_complex.tsv"
OUT_MAPS = Path("benchmark_final/test_stat")
OUT_MAPS.mkdir(parents=True, exist_ok=True)

SESSIONS = ["HCP_test", "HCP_retest"]

ONLY_REPRODUCIBLE = True   # True : bassins >= thr (labels du .npy) ; False : tous les bassins
WHITETEMP = MESH["white.template"]


# ---------------------------------------------------------------------------
# Carte Σ longueur arêtes / √aire du bassin, une carte par méthode
# à partir du TSV unique (celui lu dans R)
# ---------------------------------------------------------------------------
TSV_ALL = OUT / "aggregated" / "methods" / "trace_mindboggle_zoe_all_reproducibility.tsv"
SESSION_SUFFIX = {"HCP_test": "test", "HCP_retest": "retest"}


# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
def path_basins_native(test, sub, side):
    """Texture de bassins sur le mesh natif."""
    return DATA_TEST / f"{test}_{sub}_{side}_{BASIN}.gii"


def path_reproducible_npy(test, sub, side):
    """Labels des bassins reproductibles (>= thr) pour cette session."""
    return OUT / test / sub / "postproc" / f"{sub}.{side}.{SPHERETEMP}.{BASIN}.npy"


def path_sphere_reg(test, sub, side):
    return DATA / RAW / test / sub / REG / f"{sub}.{side}.{SPHEREREG}.surf.gii"


def path_sphere_template(side):
    return Path(f"{ATLAS}.{side}.{SPHERETEMP}.surf.gii")

def path_white_mesh_native(test, sub, side):
    return (
        DATA / RAW / test / sub / NATIVE
        / f"{sub}.{side}.{WHITE}.surf.gii"
    )


# ---------------------------------------------------------------------------
# Traitement d'un sujet
# ---------------------------------------------------------------------------
def process(test, sub, side, stats, sphere_template):
    """Retourne la carte de complexité interpolée sur le template (ou None)."""
    p_basins = path_basins_native(test, sub, side)
    p_reg = path_sphere_reg(test, sub, side)
    needed = [p_basins, p_reg]
    p_npy = path_reproducible_npy(test, sub, side)
    if ONLY_REPRODUCIBLE:
        needed.append(p_npy)

    for p in needed:
        if not p.exists():
            print(f"[skip] {test} {sub} {side} : manquant ({p})")
            return None

    # 1. nb de feuilles par bassin pour cette session
    st = stats[(stats["test"] == test) & (stats["sub"] == sub)
               & (stats["hemisphere"] == side)]
    leaves_per_basin = dict(zip(st["basin"].astype(int), st["n_endpoints"]))

    basins = sio.load_texture(str(p_basins)).darray[0].astype(int)
    labels = np.load(p_npy) if ONLY_REPRODUCIBLE else np.unique(basins)

    # 2. carte native : 1 sommet = nb de feuilles de son bassin, 0 sinon
    cmap = np.zeros(len(basins), dtype=float)
    for lab in labels:
        cmap[basins == lab] = leaves_per_basin.get(int(lab), 0)

    #normalisé par la taille du bassin, racien carré de l'aire du bassin ?
    if cmap.max() == 0:
        print(f"[warn] {test} {sub} {side} : carte nulle "
              f"(labels TSV {sorted(leaves_per_basin)[:5]} vs texture {np.unique(basins)[:5]})")

    sio.write_texture(
        stex.TextureND(darray=cmap),
        str(OUT_MAPS / f"{sub}.{side}.{test}.{WHITE}.complexity.gii"),
    )

    # 3. interpolation plus proche voisin vers le template
    sphere_reg = sio.load_mesh(str(p_reg))
    cmap_reg = np.asarray(
        srem.texture_spherical_interpolation_nearest_neighbor(
            sphere_reg, sphere_template, cmap
        ),
        dtype=float,
    )
    sio.write_texture(
        stex.TextureND(darray=cmap_reg),
        str(OUT_MAPS / f"{sub}.{side}.{test}.{SPHERETEMP}.complexity.gii"),
    )
    return cmap_reg


def process(test, sub, side, stats, sphere_template):
    """Retourne la carte de complexité normalisée sur le template."""

    p_basins = path_basins_native(test, sub, side)
    p_reg = path_sphere_reg(test, sub, side)
    p_white = path_white_mesh_native(test, sub, side)

    needed = [p_basins, p_reg, p_white]
    p_npy = path_reproducible_npy(test, sub, side)

    if ONLY_REPRODUCIBLE:
        needed.append(p_npy)

    for p in needed:
        if not p.exists():
            print(f"[skip] {test} {sub} {side} : manquant ({p})")
            return None

    st = stats[
        (stats["test"] == test)
        & (stats["sub"] == sub)
        & (stats["hemisphere"] == side)
    ]

    leaves_per_basin = dict(
        zip(st["basin"].astype(int), st["n_endpoints"])
    )

    basins = sio.load_texture(str(p_basins)).darray[0].astype(int)
    labels = (
        np.load(p_npy)
        if ONLY_REPRODUCIBLE
        else np.unique(basins)
    )

    # Maillage blanc natif du sujet et de la session
    white_mesh = sio.load_mesh(str(p_white))

    if len(white_mesh.vertices) != len(basins):
        raise ValueError(
            f"Nombre de sommets incompatible : {test} {sub} {side}"
        )

    # Aire de chaque bassin sur le white mesh natif
    area_per_basin = compute_basin_areas(white_mesh, basins)

    # Complexité normalisée : nombre de feuilles / racine carrée de l'aire
    cmap = np.zeros(len(basins), dtype=float)

    for lab in labels:
        lab = int(lab)
        area = area_per_basin.get(lab, 0.0)
        n_leaves = leaves_per_basin.get(lab, np.nan)

        if area > 0 and np.isfinite(n_leaves):
            cmap[basins == lab] = n_leaves / np.sqrt(area)

    if cmap.max() == 0:
        print(f"[warn] {test} {sub} {side} : carte nulle")

    # Sauvegarde native
    sio.write_texture(
        stex.TextureND(darray=cmap),
        str(OUT_MAPS / f"{sub}.{side}.{test}.{WHITE}.complexity.gii"),
    )

    # Interpolation vers la sphère template
    sphere_reg = sio.load_mesh(str(p_reg))

    cmap_reg = np.asarray(
        srem.texture_spherical_interpolation_nearest_neighbor(
            sphere_reg, sphere_template, cmap
        ),
        dtype=float,
    )

    sio.write_texture(
        stex.TextureND(darray=cmap_reg),
        str(OUT_MAPS / f"{sub}.{side}.{test}.{SPHERETEMP}.complexity.gii"),
    )
    diagnose_complexity(
        labels, basins, leaves_per_basin, area_per_basin, cmap_reg
    )

    return cmap_reg


def compute_basin_areas(mesh, basins):
    """Calcule l'aire corticale de chaque bassin (mm²)."""
    faces = np.asarray(mesh.faces)
    face_areas = np.asarray(mesh.area_faces)
    basins = np.asarray(basins).astype(int)

    # Un triangle appartient à un bassin si ses 3 sommets
    # portent le même label non nul.
    face_labels = basins[faces]

    same_label = np.all(face_labels == face_labels[:, :1], axis=1)
    valid = same_label & (face_labels[:, 0] > 0)

    labels = face_labels[valid, 0]
    areas = face_areas[valid]

    return pd.Series(areas).groupby(labels).sum().to_dict()


def diagnose_complexity(labels, basins, leaves_per_basin,
                        area_per_basin, cmap):
    labels = set(np.asarray(labels, dtype=int).ravel().tolist())
    texture_labels = set(np.unique(basins).tolist())
    stats_labels = set(leaves_per_basin.keys())
    area_labels = set(area_per_basin.keys())

    missing_stats = labels - stats_labels
    missing_texture = labels - texture_labels
    missing_area = labels - area_labels

    zero_leaves = []
    positive_leaves = []
    invalid_area = []
    invalid_leaves = []

    for lab in sorted(labels):
        n = leaves_per_basin.get(lab, np.nan)
        area = area_per_basin.get(lab, 0.0)

        if not np.isfinite(n):
            invalid_leaves.append(lab)
        elif not np.isfinite(area) or area <= 0:
            invalid_area.append(lab)
        elif n == 0:
            zero_leaves.append(lab)
        elif n > 0:
            positive_leaves.append(lab)

    print("\n========== DIAGNOSTIC COMPLEXITÉ ==========")
    print("Labels sélectionnés :", len(labels))
    print("Labels absents du TSV :", len(missing_stats), sorted(missing_stats)[:10])
    print("Labels absents de la texture :", len(missing_texture), sorted(missing_texture)[:10])
    print("Labels sans aire calculée :", len(missing_area), sorted(missing_area)[:10])
    print("Bassins avec 0 endpoint et aire valide :", len(zero_leaves), zero_leaves[:10])
    print("Bassins avec endpoints positifs et aire valide :", len(positive_leaves))
    print("Bassins avec endpoints invalides/manquants :", len(invalid_leaves), invalid_leaves[:10])
    print("Bassins avec aire invalide :", len(invalid_area), invalid_area[:10])

    print("\n--- Carte cmap ---")
    print("Sommets à zéro :", int(np.sum(cmap == 0)))
    print("Sommets positifs :", int(np.sum(cmap > 0)))
    print("Sommets négatifs :", int(np.sum(cmap < 0)))
    print("Valeurs non finies :", int(np.sum(~np.isfinite(cmap))))
    print("Min / max :", np.nanmin(cmap), np.nanmax(cmap))
    print("===========================================\n")


def load_subject_basin_data(sub, side, test):
    """Charge les bassins, les endpoints et les aires pour un sujet."""

    stats = pd.read_csv(
        TSV,
        sep="\t",
        dtype={"test": str, "sub": str, "hemisphere": str},
    )

    st = stats[
        (stats["test"] == test)
        & (stats["sub"] == sub)
        & (stats["hemisphere"] == side)
    ]

    if st.empty:
        raise ValueError(
            f"Aucune statistique pour {test}, {sub}, {side}"
        )

    # Nombre d'endpoints par bassin
    leaves_per_basin = dict(
        zip(st["basin"].astype(int), st["n_endpoints"])
    )

    # Maillage blanc natif et texture des bassins
    p_white = path_white_mesh_native(test, sub, side)
    p_basins = path_basins_native(test, sub, side)
    p_npy = path_reproducible_npy(test, sub, side)

    white_mesh = sio.load_mesh(str(p_white))
    basins = sio.load_texture(str(p_basins)).darray[0].astype(int)

    # Uniquement les bassins reproductibles
    labels = np.asarray(np.load(p_npy), dtype=int).ravel()

    if len(white_mesh.vertices) != len(basins):
        raise ValueError("Le maillage blanc et la texture sont incompatibles.")

    # Aires sur le maillage blanc natif
    area_per_basin = compute_basin_areas(white_mesh, basins)

    return (
        white_mesh,
        basins,
        labels,
        leaves_per_basin,
        area_per_basin,
    )


def plot_basin_area_vs_endpoints(sub, side, test):
    """
    Trace le nombre d'endpoints en fonction de l'aire
    pour chaque bassin reproductible.
    """

    import plotly.express as px

    (
        _, _, labels, leaves_per_basin, area_per_basin
    ) = load_subject_basin_data(sub, side, test)

    rows = []

    for lab in labels:
        lab = int(lab)
        area = area_per_basin.get(lab, np.nan)
        n_endpoints = leaves_per_basin.get(lab, np.nan)

        if (
            np.isfinite(area)
            and area > 0
            and np.isfinite(n_endpoints)
        ):
            rows.append({
                "Bassin": lab,
                "Aire (mm²)": area,
                "Nombre d'endpoints": n_endpoints,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Aucun bassin valide à afficher.")

    fig = px.scatter(
        df,
        x="Aire (mm²)",
        y="Nombre d'endpoints",
        hover_data=["Bassin"],
        title=f"Complexité des bassins — {sub}, {side}, {test}",
        labels={
            "Aire (mm²)": "Aire du bassin (mm²)",
            "Nombre d'endpoints": "Nombre de feuilles du squelette",
        },
    )

    fig.update_layout(template=None)

    out = OUT_MAPS / f"{sub}.{side}.{test}.area_vs_endpoints.html"
    fig.write_html(str(out), include_plotlyjs="cdn")

    return df, fig


def plot_endpoint_map_template(sub, side, test):
    """
    Projette la complexité normalisée (nombre d'endpoints / √aire) 
    de chaque bassin sur la white template.
    """

    import plotly.express as px

    # On récupère bien area_per_basin en plus
    (
        _, basins, labels, leaves_per_basin, area_per_basin
    ) = load_subject_basin_data(sub, side, test)

    cmap = np.zeros(len(basins), dtype=float)

    for lab in labels:
        lab = int(lab)

        if lab == -1:
            continue

        n_endpoints = leaves_per_basin.get(lab, np.nan)
        area = area_per_basin.get(lab, 0.0)

        # Normalisation par la racine carrée de l'aire du bassin
        if area > 0 and np.isfinite(n_endpoints):
            cmap[basins == lab] = n_endpoints / np.sqrt(area)

    # Projection sur la sphère template
    sphere_reg = sio.load_mesh(
        str(path_sphere_reg(test, sub, side))
    )
    sphere_template = sio.load_mesh(
        str(path_sphere_template(side))
    )

    cmap_template = np.asarray(
        srem.texture_spherical_interpolation_nearest_neighbor(
            sphere_reg,
            sphere_template,
            cmap,
        ),
        dtype=float,
    )

    # Sauvegarde de la texture projetée
    out_texture = (
        OUT_MAPS / f"{sub}.{side}.{test}.{SPHERETEMP}.complexity.gii"
    )
    sio.write_texture(
        stex.TextureND(darray=cmap_template),
        str(out_texture),
    )

    # Affichage sur le maillage blanc template
    white_template = sio.load_mesh(
        f"{ATLAS}.{side}.{WHITETEMP}.surf.gii"
    )

    vmax = float(np.nanmax(cmap_template))
    turbo = px.colors.get_colorscale("Turbo")

    colorscale = [[0.0, "white"], [1e-6, "white"]] + [
        [1e-6 + (1 - 1e-6) * p, c]
        for p, c in turbo
    ]

    fig = splt.plot_mesh(
        {
            "vertices": white_template.vertices,
            "faces": white_template.faces,
            "title": f"Complexité normalisée — {sub}, {side}, {test}",
        },
        {"values": cmap_template, "mode": "vertex"},
        {
            "template": None,
            "colorbar_label": "Endpoints / √Aire (mm⁻¹)",
            "colorscale": colorscale,
        },
        caption=False,
    )

    fig.update_traces(
        intensity=cmap_template,
        intensitymode="vertex",
        colorscale=colorscale,
        cmin=0,
        cmax=max(vmax, 1),
        selector=dict(type="mesh3d"),
    )

    out_html = (
        OUT_MAPS / f"{sub}.{side}.{test}.complexity_template.html"
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")

    return cmap_template, fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
stats = pd.read_csv(TSV, sep="\t", dtype={"test": str, "sub": str, "hemisphere": str})
print("sides :", sides)
print("TSV test :", stats["test"].unique(), "| hemisphere :", stats["hemisphere"].unique())

for side in sides:
    sphere_template = sio.load_mesh(str(path_sphere_template(side)))
    all_maps = []

    for test in SESSIONS:
        subjects = sorted(stats.loc[stats["test"] == test, "sub"].unique())
        maps = []
        for sub in subjects:
            #m = process(test, sub, side, stats, sphere_template, white_mesh)
            m = process(test, sub, side, stats, sphere_template)
            if m is not None:
                maps.append(m)
        print(f"[{side}] {test} : {len(maps)}/{len(subjects)} sujets")

        if maps:
            stack = np.stack(maps, axis=0)
            all_maps.append(stack)
            sio.write_texture(
                stex.TextureND(darray=stack.mean(axis=0)),
                str(OUT_MAPS / f"{side}.{test}.complexity.mean.gii"),
            )

    # moyenne globale (test + retest empilés)
    if all_maps:
        stack = np.concatenate(all_maps, axis=0)
        sio.write_texture(
            stex.TextureND(darray=stack.mean(axis=0)),
            str(OUT_MAPS / f"{side}.complexity.mean_all.gii"),
        )
        print(f"[{side}] moyenne globale sur {stack.shape[0]} cartes")

    # moyenne globale (test + retest empilés)
    if all_maps:
        stack = np.concatenate(all_maps, axis=0)
        mean_all = stack.mean(axis=0)
        sio.write_texture(
            stex.TextureND(darray=mean_all),
            str(OUT_MAPS / f"{side}.complexity.mean_all.gii"),
        )
        print(f"[{side}] moyenne globale sur {stack.shape[0]} cartes")
        print("Valeurs nulles :", np.sum(mean_all == 0))
        print("Valeurs positives :", np.sum(mean_all > 0))
        print("Valeurs non finies :", np.sum(~np.isfinite(mean_all)))
        print("Minimum :", np.nanmin(mean_all))
        print("Maximum :", np.nanmax(mean_all))
        zero_mask = mean_all == 0
        print(f"Proportion de zéros : {100 * zero_mask.mean():.2f}%")
        print(f"Nombre de sommets nuls : {zero_mask.sum()}")

        # affichage sur le white template
        white_mesh_template = sio.load_mesh(f"{ATLAS}.{side}.{WHITETEMP}.surf.gii")

        vmax = float(np.nanmax(mean_all))

        turbo = px.colors.get_colorscale("Turbo")

        colorscale = [[0.0, "white"], [1e-6, "white"]] + [
            [1e-6 + (1 - 1e-6) * p, c]
            for p, c in turbo
        ]

        fig = splt.plot_mesh(
            {
                "vertices": white_mesh_template.vertices,
                "faces": white_mesh_template.faces,
                "title": f"Complexité moyenne ({side})",
            },
            {"values": mean_all, "mode": "vertex"},
            {
                "template": None,
                "colorbar_label": "Endpoints / √Aire (mm⁻¹)",
                "colorscale": colorscale,
            },
            caption=False,
        )

        fig.update_traces(
            intensity=mean_all,
            intensitymode="vertex",
            colorscale=colorscale,
            cmin=0,
            cmax=vmax,
            selector=dict(type="mesh3d"),
        )

        fig.write_html(
            str(OUT_MAPS / f"{side}.complexity.mean_all.html"),
            include_plotlyjs="cdn",
        )

sub = "103818"
side = "L"
test = "HCP_test"

# 1. Aire du bassin vs nombre d'endpoints
df, fig1 = plot_basin_area_vs_endpoints(sub, side, test)

# 2. Carte sur la white template
cmap, fig2 = plot_endpoint_map_template(sub, side, test)



def native_length_map(df_m, sub, side, session, basins):
    """Tout le bassin prend la valeur  length / sqrt(area)  (0 si NaN ou aire nulle)."""
    d = df_m[(df_m["subject"] == sub) & (df_m["side"] == side)]
    if d.empty:
        raise ValueError(f"Aucune ligne pour {sub}, {side}")

    cmap = np.zeros(len(basins), dtype=float)
    for lab, L, A in zip(d[f"label_{session}"].astype(int),
                         d[f"length_{session}"],
                         d[f"area_{session}"]):
        if lab > 0 and np.isfinite(L) and np.isfinite(A) and A > 0:
            cmap[basins == lab] = L / np.sqrt(A)
    return cmap


def plot_length_maps_all_methods(sub, side, test, tsv=TSV_ALL, param=None):
    session = SESSION_SUFFIX[test]

    df = pd.read_csv(tsv, sep="\t", dtype={"subject": str, "side": str})
    if param is not None:
        df = df[df["param"] == param]

    basins = sio.load_texture(str(path_basins_native(test, sub, side))).darray[0].astype(int)
    sphere_reg = sio.load_mesh(str(path_sphere_reg(test, sub, side)))
    sphere_template = sio.load_mesh(str(path_sphere_template(side)))
    white_template = sio.load_mesh(f"{ATLAS}.{side}.{WHITETEMP}.surf.gii")

    # 1) une carte par méthode, projetée sur le template
    maps = {}
    for method, df_m in df.groupby("method"):
        cmap = native_length_map(df_m, sub, side, session, basins)
        cmap_tpl = np.asarray(
            srem.texture_spherical_interpolation_nearest_neighbor(
                sphere_reg, sphere_template, cmap),
            dtype=float,
        )
        sio.write_texture(
            stex.TextureND(darray=cmap_tpl),
            str(OUT_MAPS / f"{sub}.{side}.{test}.{method}.{SPHERETEMP}.length_norm.gii"),
        )
        maps[method] = cmap_tpl

    # 2) échelle commune
    vmax = max(float(np.nanmax(m)) for m in maps.values())
    turbo = px.colors.get_colorscale("Turbo")
    colorscale = [[0.0, "white"], [1e-6, "white"]] + [
        [1e-6 + (1 - 1e-6) * p, c] for p, c in turbo
    ]

    # 3) un plot par méthode
    figs = {}
    for method, cmap_tpl in maps.items():
        fig = splt.plot_mesh(
            {"vertices": white_template.vertices,
             "faces": white_template.faces,
             "title": f"Longueur squelette / √Aire — {method}, {sub}, {side}, {test}"},
            {"values": cmap_tpl, "mode": "vertex"},
            {"template": None,
             "colorbar_label": "Σ longueur arêtes / √Aire (sans dimension)",
             "colorscale": colorscale},
            caption=False,
        )
        fig.update_traces(
            intensity=cmap_tpl, intensitymode="vertex",
            colorscale=colorscale, cmin=0, cmax=max(vmax, 1e-6),
            selector=dict(type="mesh3d"),
        )
        fig.write_html(
            str(OUT_MAPS / f"{sub}.{side}.{test}.{method}.length_norm_template.html"),
            include_plotlyjs="cdn",
        )
        figs[method] = fig
        
    return maps, figs


maps, figs = plot_length_maps_all_methods("103818", "L", "HCP_test")