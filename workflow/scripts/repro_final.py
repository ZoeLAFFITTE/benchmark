#!/usr/bin/env python
"""
Reproductibilité test/retest par bassin, pour n'importe quelle méthode
d'extraction de sillons (texture labellisée par les bassins natifs).

Une ligne par paire de bassins reproductibles :
aire, longueur, endpoints, bifurcations, pit, n_cc (nb de composantes connexes du squelette via NetworkX)
+ distances test/retest (mean_nn, hausdorff, p95) sur sphere.reg
+ métriques globales du mesh par session : volume, indice de gyrification, aire totale.
"""

import sys
import numpy as np
import pandas as pd
import slam.io as sio
from scipy.spatial import cKDTree
import networkx as nx

sys.stderr = sys.stdout = open(snakemake.log[0], "w")


# ---------------------------------------------------------------------------
# Volume du maillage (mm³)
# ---------------------------------------------------------------------------
def compute_mesh_volume(mesh):
    """Calcule le volume d'un maillage fermé par la formule de divergence."""
    if hasattr(mesh, "volume"):
        return float(mesh.volume)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return float(np.sum(np.einsum('ij,ij->i', v0, np.cross(v1, v2))) / 6.0)


# ---------------------------------------------------------------------------
# Indice de gyrification : aire corticale / aire de l'enveloppe convexe
# ---------------------------------------------------------------------------
def compute_gyrification_index(mesh):
    """Indice de gyrification = aire du mesh / aire de son enveloppe convexe."""
    cortex_area = float(mesh.area)
    convex_hull_area = float(mesh.convex_hull.area)
    return cortex_area / convex_hull_area


# ---------------------------------------------------------------------------
# Aire des bassins (mm²)
# ---------------------------------------------------------------------------
def compute_basin_areas(mesh, basins):
    """Calcule l'aire corticale de chaque bassin (mm²)."""
    faces = np.asarray(mesh.faces)
    face_areas = np.asarray(mesh.area_faces)
    basins = np.asarray(basins).astype(int)

    face_labels = basins[faces]
    same_label = np.all(face_labels == face_labels[:, :1], axis=1)
    valid = same_label & (face_labels[:, 0] > 0)

    labels = face_labels[valid, 0]
    areas = face_areas[valid]
    return pd.Series(areas).groupby(labels).sum().to_dict()


# ---------------------------------------------------------------------------
# Stats du squelette (graphe induit par le maillage white) via NetworkX
# ---------------------------------------------------------------------------
def skeleton_stats(mesh, sulci):
    """Longueur, endpoints, bifurcations et nombre de composantes connexes (n_cc) par label via NetworkX."""
    sulci = np.asarray(sulci).astype(int)
    edges = np.asarray(mesh.edges_unique)
    lengths = np.asarray(mesh.edges_unique_length)

    lab0, lab1 = sulci[edges[:, 0]], sulci[edges[:, 1]]
    keep = (lab0 == lab1) & (lab0 > 0)
    e, e_len, e_lab = edges[keep], lengths[keep], lab0[keep]

    deg = np.bincount(e.ravel(), minlength=len(sulci))
    on_skel = sulci > 0

    def group(vertices):
        return {
            int(k): [int(v) for v in g.index]
            for k, g in pd.Series(vertices).groupby(sulci[vertices])
        } if len(vertices) else {}

    endpoints = group(np.flatnonzero(on_skel & (deg == 1)))
    bifurcations = group(np.flatnonzero(on_skel & (deg >= 3)))
    length = pd.Series(e_len).groupby(e_lab).sum().to_dict()
    n_vertices = pd.Series(sulci[on_skel]).value_counts().to_dict()

    # Nombre de composantes connexes (n_cc) par label via NetworkX.
    # On ajoute d'abord TOUS les sommets du squelette comme nœuds, afin que
    # les sommets isolés (sans voisin de même label) comptent comme composantes.
    G = nx.Graph()
    G.add_nodes_from(np.flatnonzero(on_skel).tolist())
    G.add_edges_from(e.tolist())

    # Les arêtes relient toujours deux sommets de même label : chaque composante
    # appartient donc à un seul label. On les compte en un seul passage.
    n_cc = {}
    n_loops = {}  # composantes sans aucun nœud (cycles purs, tous les sommets de degré 2)
    for comp in nx.connected_components(G):
        comp = list(comp)
        lab = int(sulci[comp[0]])
        n_cc[lab] = n_cc.get(lab, 0) + 1
        if len(comp) > 1 and np.all(deg[comp] == 2):
            n_loops[lab] = n_loops.get(lab, 0) + 1

    # Nombre de segments par label : un segment est un chemin entre deux "nœuds"
    # (feuille deg == 1 ou bifurcation deg >= 3), en passant par des sommets de degré 2.
    # Chaque segment a deux extrémités sur des nœuds, donc
    # n_segments = (somme des degrés des nœuds) / 2.
    node_mask = on_skel & (deg != 2) & (deg > 0)
    node_deg = pd.Series(deg[node_mask]).groupby(sulci[node_mask]).sum()
    n_segments = (node_deg // 2).astype(int).to_dict()
    # Un cycle pur (sans nœud) compte pour 1 segment
    for lab, k in n_loops.items():
        n_segments[lab] = n_segments.get(lab, 0) + k

    return length, endpoints, bifurcations, n_vertices, n_cc, n_segments


def pit_in_skeleton(basins, dpf, sulci, label):
    """Le sommet de DPF* maximal du bassin appartient-il au squelette ?"""
    verts = np.flatnonzero(basins == label)
    if len(verts) == 0:
        return np.nan, np.nan
    pit = int(verts[np.argmax(dpf[verts])])
    return pit, int(sulci[pit] == label)


# ---------------------------------------------------------------------------
# Distances test/retest
# ---------------------------------------------------------------------------
def nn_distances(a, b):
    d_ab = cKDTree(b).query(a, k=1)[0]
    d_ba = cKDTree(a).query(b, k=1)[0]
    d = np.concatenate([d_ab, d_ba])
    return d.max(), d.mean(), np.percentile(d, 95)


def session_table(white, basins, dpf, sulci):
    """Stats par label natif d'une session."""
    areas = compute_basin_areas(white, basins)
    length, endpoints, bifs, n_vert, n_cc, n_seg = skeleton_stats(white, sulci)

    rows = {}
    for label in np.unique(basins[basins > 0]).astype(int):
        pit, pit_in = pit_in_skeleton(basins, dpf, sulci, label)
        ends = endpoints.get(label, [])

        n_v = n_vert.get(label, 0)
        # sommets du squelette (n'importe quel label) situés dans ce bassin
        n_skel_in_basin = int(np.count_nonzero((basins == label) & (sulci > 0)))

        # Diagnostic : pourquoi length / n_segments sont-ils absents ?
        if n_v > 0 and label in length:
            status = "ok"
        elif n_v > 0:
            status = "no_edge"          # sommets du squelette mais aucune arête
        elif n_skel_in_basin > 0:
            status = "label_mismatch"   # squelette présent dans le bassin sous un autre label
        else:
            status = "empty"            # aucun squelette dans le bassin

        rows[label] = {
            "area": areas.get(label, np.nan),
            "length": length.get(label, np.nan),          # NaN au lieu de 0.0
            "n_vertices": n_v,
            "n_endpoints": len(ends),
            "endpoints": ",".join(map(str, ends)),
            "n_bifurcations": len(bifs.get(label, [])),
            "n_cc": n_cc.get(label, 0),
            "n_segments": n_seg.get(label, np.nan),       # NaN au lieu de 0
            "pit": pit,
            "pit_in_skeleton": pit_in,
            "n_skel_in_basin": n_skel_in_basin,
            "skel_status": status,
        }
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
inp, wc = snakemake.input, snakemake.wildcards

df = pd.read_csv(inp.match_tsv, sep="\t")
df["overlap"] = df["overlap"].astype(float)
df = df[df["overlap"] >= snakemake.params.thr_basins]

sessions = {}
for s in ("test", "retest"):
    white = sio.load_mesh(getattr(inp, f"white_{s}"))
    basins = sio.load_texture(getattr(inp, f"basins_{s}")).darray[0].astype(int)
    dpf = sio.load_texture(getattr(inp, f"dpf_{s}")).darray[0]
    sulci = sio.load_texture(getattr(inp, f"sulci_{s}")).darray[0].astype(int)

    sessions[s] = {
        "table": session_table(white, basins, dpf, sulci),
        "sulci": sulci,
        "sphere": sio.load_mesh(getattr(inp, f"sphere_reg_{s}")),
        "volume": compute_mesh_volume(white),
        "IG": compute_gyrification_index(white),  # indice de gyrification
        "mesh_area": float(white.area),           # aire du mesh total
    }

rows = []
for lt, lr, ov in df[["test_basin", "retest_basin", "overlap"]].itertuples(
    index=False, name=None
):
    lt, lr = int(lt), int(lr)
    row = {
        "subject": wc.sub,
        "side": wc.side,
        "method": getattr(snakemake.params, "method", "zoe"),
        "param": getattr(snakemake.params, "param", "default"),
        "label_test": lt,
        "label_retest": lr,
        "overlap": float(ov),
    }
    for s, lab in (("test", lt), ("retest", lr)):
        stats = sessions[s]["table"].get(lab, {})
        row.update({f"{k}_{s}": v for k, v in stats.items()})

        # Métriques globales du mesh pour cette session
        row[f"mesh_volume_{s}"] = sessions[s]["volume"]
        row[f"gyrification_index_{s}"] = sessions[s]["IG"]
        row[f"mesh_area_{s}"] = sessions[s]["mesh_area"]

    pts_t = sessions["test"]["sphere"].vertices[sessions["test"]["sulci"] == lt]
    pts_r = sessions["retest"]["sphere"].vertices[sessions["retest"]["sulci"] == lr]
    h = m = p = np.nan
    if len(pts_t) and len(pts_r):
        h, m, p = nn_distances(pts_t, pts_r)
    row.update({"hausdorff": h, "mean_nn": m, "p95_nn": p})
    rows.append(row)

# 1. Sauvegarde de la DataFrame principale
pd.DataFrame(rows).to_csv(snakemake.output.tsv, sep="\t", index=False)


# ---------------------------------------------------------------------------
# Seconde DataFrame : alpha × racine cubique du volume du mesh
# ---------------------------------------------------------------------------
alphas = [float(f"{i * 0.05:.2f}") for i in range(7)]
rows_alpha = []

for r in rows:
    for s, basin_key in (("test", "label_test"), ("retest", "label_retest")):
        vol = sessions[s]["volume"]
        cbrt_vol = np.cbrt(np.abs(vol))
        for alpha in alphas:
            rows_alpha.append({
                "subject": r["subject"],
                "side": r["side"],
                "method": r["method"],
                "param": r["param"],
                "session": s,
                "basin": r[basin_key],
                "alpha": alpha,
                "mesh_volume": vol,
                "cbrt_mesh_volume": cbrt_vol,
                "alpha_cbrt_volume": alpha * cbrt_vol,
                "gyrification_index": sessions[s]["IG"],
                "mesh_area": sessions[s]["mesh_area"],
            })

df_alpha = pd.DataFrame(rows_alpha)

# 2. Sauvegarde de la seconde DataFrame
output_alpha_path = getattr(
    snakemake.output, "alpha_tsv",
    str(snakemake.output.tsv).replace(".tsv", "_alpha.tsv"),
)
df_alpha.to_csv(output_alpha_path, sep="\t", index=False)