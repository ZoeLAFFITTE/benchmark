from pathlib import Path

import numpy as np
import pandas as pd
import slam.io as sio
import yaml
import glob
import os
import re
import networkx as nx
from scipy.spatial import cKDTree
from itertools import product


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

OUT = Path("benchmark_final")


# ============================================================
# Détection des sujets, sessions et hémisphères
# ============================================================

pattern = os.path.join(
    DATA_TEST,
    "*_*_*_basins_100.0_0.005_0.0.gii"
)

regex = re.compile(
    r"([^_]+)_([^_]+)_([^_]+)_basins_100\.0_0\.005_0\.0\.gii$"
)

matches = [
    regex.search(os.path.basename(path))
    for path in glob.glob(pattern)
]

matches = [m for m in matches if m is not None]

TESTS = sorted({m.group(1) for m in matches})
SUBJECTS = sorted({m.group(2) for m in matches})
SIDES = sorted({m.group(3) for m in matches})

# Seuil d'overlap : n'est PAS utilisé pour filtrer les bassins.
# Il sert uniquement à marquer les paires reproductibles (colonne
# "reproducible") et à compter leur nombre.
OVERLAP_THRESHOLD = 0.7


# ============================================================
# PATHS
# ============================================================

def path_basins_native(test, sub, side):
    return DATA_TEST / f"{test}_{sub}_{side}_{BASIN}.gii"


def path_white_mesh_native(test, sub, side):
    return (
        DATA / RAW / test / sub / NATIVE
        / f"{sub}.{side}.{WHITE}.surf.gii"
    )


def path_sphere_reg(test, sub, side):
    return (
        DATA / RAW / test / sub / REG
        / f"{sub}.{side}.{SPHEREREG}.surf.gii"
    )


def path_zoe(test, sub, side):
    return (
        OUT / test / sub / "zoe"
        / f"{sub}.{side}.{WHITE}.sulci.gii"
    )


def path_trace(test, sub, side):
    return (
        OUT / test / sub / "trace"
        / f"{sub}.{side}.{WHITE}.scurve_bary_c.gii"
    )


def path_mb(test, sub, side):
    return (
        OUT / test / sub / "mindboggle"
        / f"{sub}.{side}.{WHITE}.fundi.gii"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_texture(texture):
    """
    Convertit la texture en masque binaire :
    -1 et 0 deviennent 0 ; les autres valeurs deviennent 1.
    """
    texture = np.asarray(texture).squeeze()
    return np.where((texture == -1) | (texture == 0), 0, 1)


def compute_gyrification_index(mesh):
    """
    Indice de gyrification : aire du maillage / aire de son enveloppe convexe.
    """
    return mesh.area / mesh.convex_hull.area


def compute_vertex_areas(mesh):
    """
    Aire associée à chaque sommet : chaque face distribue 1/3 de son aire
    à chacun de ses trois sommets.

    Sommer ces aires sur les sommets d'un bassin revient à sommer l'aire des
    faces entièrement dans le bassin, plus 1/3 de l'aire par sommet du bassin
    pour les faces de bord. La somme sur tous les sommets vaut mesh.area.
    Calcul vectorisé, fait une seule fois par maillage.
    """
    faces = np.asarray(mesh.faces)
    face_areas = np.asarray(mesh.area_faces)

    return np.bincount(
        faces.ravel(),
        weights=np.repeat(face_areas / 3.0, 3),
        minlength=len(mesh.vertices),
    )


def hausdorff_distance(points_test, points_retest):
    """
    Calcule la distance de Hausdorff symétrique, la moyenne des distances
    au plus proche voisin (poolée), le 95e percentile et la moyenne des
    deux moyennes directionnelles.

    Les points doivent être exprimés dans le même espace de coordonnées.
    Ici, ils proviennent des sphères enregistrées.
    Renvoie NaN si l'un des deux ensembles de points est vide.
    """
    points_test = np.asarray(points_test)
    points_retest = np.asarray(points_retest)

    if len(points_test) == 0 or len(points_retest) == 0:
        return np.nan, np.nan, np.nan, np.nan

    tree_test = cKDTree(points_test)
    tree_retest = cKDTree(points_retest)

    dist_test_to_retest, _ = tree_retest.query(points_test, k=1)
    dist_retest_to_test, _ = tree_test.query(points_retest, k=1)

    distances = np.concatenate(
        [dist_test_to_retest, dist_retest_to_test]
    )

    hausdorff = max(
        np.max(dist_test_to_retest),
        np.max(dist_retest_to_test),
    )

    mean_H = np.mean(distances)
    p95_H = np.percentile(distances, 95)
    mean_dist = (
        np.mean(dist_test_to_retest) + np.mean(dist_retest_to_test)
    ) / 2

    return hausdorff, mean_H, p95_H, mean_dist


def compute_basin_metrics(mesh, basin_labels, skeleton_labels, basin, vert_area):
    """
    Calcule les métriques d'un bassin sur le maillage natif.

    Parameters
    ----------
    mesh : maillage natif
    basin_labels : texture donnant le label de bassin de chaque sommet
    skeleton_labels : texture contenant le label du bassin sur les sommets
                      du squelette et 0 ailleurs
    basin : label du bassin à analyser
    vert_area : aire associée à chaque sommet (1/3 des faces adjacentes)

    Returns
    -------
    area_basin : aire du bassin
    nb_feuilles : nombre de sommets terminaux du squelette
    length : longueur totale du squelette
    n_cc_basin : nombre de composantes connexes du bassin
    """

    vertices = np.asarray(mesh.vertices)
    edges = np.asarray(mesh.edges_unique)
    edges_length = np.asarray(mesh.edges_unique_length)

    basin_labels = np.asarray(basin_labels).squeeze()
    skeleton_labels = np.asarray(skeleton_labels).squeeze()
    vert_area = np.asarray(vert_area).squeeze()

    basin_mask = basin_labels == basin
    skeleton_mask = (skeleton_labels == basin) & basin_mask

    # Bassin absent : aucune métrique calculable.
    if not np.any(basin_mask):
        return np.nan, np.nan, np.nan, np.nan

    # Aire du bassin : faces internes + 1/3 par sommet pour les faces de bord.
    area_basin = np.sum(vert_area[basin_mask])

    # Graphe du bassin
    basin_graph = nx.Graph()
    basin_vertices = np.flatnonzero(basin_mask)
    basin_graph.add_nodes_from(basin_vertices.tolist())

    edge_mask_basin = basin_mask[edges[:, 0]] & basin_mask[edges[:, 1]]
    basin_graph.add_edges_from(edges[edge_mask_basin].tolist())

    # Graphe du squelette
    skeleton_vertices = np.flatnonzero(skeleton_mask)
    skeleton_graph = nx.Graph()
    skeleton_graph.add_nodes_from(skeleton_vertices.tolist())

    # Squelette absent dans ce bassin
    if not np.any(skeleton_mask):
        return area_basin, np.nan, np.nan, np.nan

    edge_mask_skeleton = (
        skeleton_mask[edges[:, 0]] & skeleton_mask[edges[:, 1]]
    )
    skeleton_edges = edges[edge_mask_skeleton]
    skeleton_graph.add_edges_from(skeleton_edges.tolist())

    # Composantes connexes du squelette (APRÈS ajout des arêtes)
    n_cc_skeleton = nx.number_connected_components(skeleton_graph)

    nb_feuilles = sum(
        degree == 1 for _, degree in skeleton_graph.degree()
    )

    # Longueur totale : NaN si le squelette n'a aucune arête
    # (sommets isolés uniquement).
    if len(skeleton_edges) > 0:
        length = np.sum(edges_length[edge_mask_skeleton])
    else:
        length = np.nan

    return area_basin, nb_feuilles, length, n_cc_skeleton


# ============================================================
# LOAD MATCHING TABLE
# ============================================================

def load_matching(sub, side):
    match_tsv = (
        OUT / "aggregated" / sub
        / f"{sub}.{side}.matching.tsv"
    )

    if not match_tsv.exists():
        raise FileNotFoundError(
            f"Fichier de matching introuvable : {match_tsv}"
        )

    df_match = pd.read_csv(match_tsv, sep="\t")

    required_columns = {
        "test_basin",
        "retest_basin",
        "overlap",
    }

    missing_columns = required_columns - set(df_match.columns)
    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans {match_tsv} : {missing_columns}"
        )

    # Exclure les bassins de fond.
    df_match = df_match[
        (df_match["test_basin"] != -1)
        & (df_match["retest_basin"] != -1)
    ].copy()

    # Pas de filtrage sur l'overlap : toutes les paires sont conservées.

    # Une seule occurrence par paire de bassins.
    df_match = df_match.drop_duplicates(
        subset=["test_basin", "retest_basin"]
    )

    return df_match


# ============================================================
# MAIN PROCESSING
# ============================================================

results = []

for sub, side in product(SUBJECTS, SIDES):

    print(f"\n===== Sujet {sub}, hémisphère {side} =====")

    # --------------------------------------------------------
    # 1. Chargement des maillages natifs
    # --------------------------------------------------------

    mesh_test = sio.load_mesh(
        str(path_white_mesh_native("HCP_test", sub, side))
    )

    mesh_retest = sio.load_mesh(
        str(path_white_mesh_native("HCP_retest", sub, side))
    )

    vertices_test = np.asarray(mesh_test.vertices)
    vertices_retest = np.asarray(mesh_retest.vertices)

    # --------------------------------------------------------
    # 2. Chargement des sphères enregistrées
    # --------------------------------------------------------

    sphere_test = sio.load_mesh(
        str(path_sphere_reg("HCP_test", sub, side))
    )

    sphere_retest = sio.load_mesh(
        str(path_sphere_reg("HCP_retest", sub, side))
    )

    sphere_vertices_test = np.asarray(sphere_test.vertices)
    sphere_vertices_retest = np.asarray(sphere_retest.vertices)

    # --------------------------------------------------------
    # 3. Chargement des labels de bassins natifs
    # --------------------------------------------------------

    basins_test = np.asarray(
        sio.load_texture(
            str(path_basins_native("HCP_test", sub, side))
        ).darray
    ).squeeze()

    basins_retest = np.asarray(
        sio.load_texture(
            str(path_basins_native("HCP_retest", sub, side))
        ).darray
    ).squeeze()

    # --------------------------------------------------------
    # 4. Chargement des textures de squelettes
    # --------------------------------------------------------

    skeleton_paths = {
        "zoe": (path_zoe("HCP_test", sub, side),
                path_zoe("HCP_retest", sub, side)),
        "trace": (path_trace("HCP_test", sub, side),
                  path_trace("HCP_retest", sub, side)),
        "mb": (path_mb("HCP_test", sub, side),
               path_mb("HCP_retest", sub, side)),
    }

    sulci_test = {}
    sulci_retest = {}

    for method, (file_test, file_retest) in skeleton_paths.items():

        tex_test = sio.load_texture(str(file_test)).darray
        tex_retest = sio.load_texture(str(file_retest)).darray

        sulci_test[method] = normalize_texture(tex_test)
        sulci_retest[method] = normalize_texture(tex_retest)

    # Chaque texture binaire est convertie en labels de bassin
    # uniquement sur les sommets appartenant au squelette.
    basin_labels_test = {
        method: np.where(sulci_test[method] > 0, basins_test, 0)
        for method in sulci_test
    }

    basin_labels_retest = {
        method: np.where(sulci_retest[method] > 0, basins_retest, 0)
        for method in sulci_retest
    }

    # --------------------------------------------------------
    # 6. Métriques calculées sur les maillages natifs
    # --------------------------------------------------------

    vert_area_test = compute_vertex_areas(mesh_test)
    vert_area_retest = compute_vertex_areas(mesh_retest)

    IG_test = compute_gyrification_index(mesh_test)
    IG_retest = compute_gyrification_index(mesh_retest)

    area_mesh_test = mesh_test.area
    area_mesh_retest = mesh_retest.area

    volume_mesh_test = mesh_test.volume
    volume_mesh_retest = mesh_retest.volume

    # --------------------------------------------------------
    # 7. Correspondances test-retest des bassins
    # --------------------------------------------------------

    df_match = load_matching(sub, side)

    # Comptage des paires reproductibles (information uniquement).
    n_pairs = len(df_match)
    n_reproducible = int((df_match["overlap"] >= OVERLAP_THRESHOLD).sum())

    print(f"Nombre de paires de bassins (toutes) : {n_pairs}")
    print(
        f"Nombre de paires avec overlap >= {OVERLAP_THRESHOLD} : "
        f"{n_reproducible}"
    )

    # --------------------------------------------------------
    # 8. Calcul des métriques par méthode et paire de bassins
    # --------------------------------------------------------

    for _, match in df_match.iterrows():

        basin_test = int(match["test_basin"])
        basin_retest = int(match["retest_basin"])
        overlap = float(match["overlap"])
        reproducible = overlap >= OVERLAP_THRESHOLD

        for method in sulci_test:

            # ----- Métriques natives du bassin test -----

            (
                area_basin_test,
                nb_feuilles_test,
                length_test,
                n_cc_basin_test,
            ) = compute_basin_metrics(
                mesh=mesh_test,
                basin_labels=basins_test,
                skeleton_labels=basin_labels_test[method],
                basin=basin_test,
                vert_area=vert_area_test,
            )

            # ----- Métriques natives du bassin retest -----

            (
                area_basin_retest,
                nb_feuilles_retest,
                length_retest,
                n_cc_basin_retest,
            ) = compute_basin_metrics(
                mesh=mesh_retest,
                basin_labels=basins_retest,
                skeleton_labels=basin_labels_retest[method],
                basin=basin_retest,
                vert_area=vert_area_retest,
            )

            # ------------------------------------------------
            # 9. Distances test-retest dans l'espace commun
            # ------------------------------------------------
            #
            # Les masques sont définis à partir des labels natifs.
            # Ils sélectionnent les mêmes indices sur sphere.reg.
            # Les coordonnées utilisées pour les distances sont donc
            # celles des sphères enregistrées, et non celles des
            # maillages blancs natifs.
            # ------------------------------------------------

            mask_test = (
                (basins_test == basin_test)
                & (basin_labels_test[method] == basin_test)
            )

            mask_retest = (
                (basins_retest == basin_retest)
                & (basin_labels_retest[method] == basin_retest)
            )

            points_test = sphere_vertices_test[mask_test]
            points_retest = sphere_vertices_retest[mask_retest]

            hausdorff, mean_H, p95_H, mean_dist = hausdorff_distance(
                points_test,
                points_retest,
            )

            # ------------------------------------------------
            # 10. Enregistrement des résultats
            # ------------------------------------------------

            results.append({
                "method": method,
                "subject": sub,
                "side": side,

                "basin_test": basin_test,
                "basin_retest": basin_retest,
                "overlap": overlap,
                "reproducible": reproducible,

                "IG_test": IG_test,
                "IG_retest": IG_retest,

                "area_mesh_test": area_mesh_test,
                "area_mesh_retest": area_mesh_retest,

                "volume_mesh_test": volume_mesh_test,
                "volume_mesh_retest": volume_mesh_retest,

                # Distances calculées sur sphere.reg
                "mean_H": mean_H,
                "hausdorff": hausdorff,
                "p95_H": p95_H,
                "mean_dist": mean_dist,

                # Métriques du squelette sur le maillage natif (valeurs brutes)
                "nb_feuilles_test": nb_feuilles_test,
                "nb_feuilles_retest": nb_feuilles_retest,

                "length_test": length_test,
                "length_retest": length_retest,

                # Aire des bassins sur le maillage natif
                "area_basin_test": area_basin_test,
                "area_basin_retest": area_basin_retest,

                "n_cc_basin_test": n_cc_basin_test,
                "n_cc_basin_retest": n_cc_basin_retest,

                # Nombre de points utilisés pour les distances
                "n_points_test": int(np.sum(mask_test)),
                "n_points_retest": int(np.sum(mask_retest)),
            })


# ============================================================
# EXPORT
# ============================================================

df_results = pd.DataFrame(results)

output_file = (
    OUT / "aggregated" / "basin_metrics_test_retest.tsv"
)
output_file.parent.mkdir(parents=True, exist_ok=True)

df_results.to_csv(
    output_file,
    sep="\t",
    index=False,
)

print(f"\nRésultats sauvegardés dans : {output_file}")
print(f"Nombre total de lignes : {len(df_results)}")


# ============================================================
# NOMBRE DE BASSINS REPRODUCTIBLES (overlap >= seuil)
# ============================================================
# Une ligne par paire de bassins (les lignes sont répétées par méthode).

df_pairs = df_results.drop_duplicates(
    subset=["subject", "side", "basin_test", "basin_retest"]
)

df_count = (
    df_pairs
    .groupby(["subject", "side"])
    .agg(
        n_pairs=("overlap", "size"),
        n_reproducible=("reproducible", "sum"),
    )
    .reset_index()
)
df_count["fraction_reproducible"] = (
    df_count["n_reproducible"] / df_count["n_pairs"]
)

print(f"\nPaires de bassins avec overlap >= {OVERLAP_THRESHOLD} :")
print(df_count.to_string(index=False))
print(
    f"\nTotal : {int(df_count['n_reproducible'].sum())} / "
    f"{int(df_count['n_pairs'].sum())} paires "
    f"({100 * df_count['n_reproducible'].sum() / df_count['n_pairs'].sum():.1f} %)"
)
