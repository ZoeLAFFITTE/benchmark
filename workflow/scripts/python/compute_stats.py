#!/usr/bin/env python
"""
Calcule des stats descriptives des squelettes produits par chaque methode.
"""

import sys
import numpy as np
import pandas as pd
import slam.io as sio
import networkx as nx
from utils import utils

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


def compute_graph_metrics(G):
    """
    Calcule les métriques topologiques et géométriques
    d'un graphe final.
    """

    # Classification des sommets
    leaves = [
        node for node, degree in G.degree()
        if degree == 1
    ]

    bifurcations = [
        node for node, degree in G.degree()
        if degree >= 3
    ]

    # Longueur totale
    length = sum(
        data["weight"]
        for _, _, data in G.edges(data=True)
    )

    # Extraction des segments
    characteristic_nodes = set(leaves + bifurcations)

    visited_edges = set()
    segments = []

    for start in characteristic_nodes:

        for neighbor in G.neighbors(start):

            edge = frozenset((start, neighbor))

            if edge in visited_edges:
                continue

            path = [start, neighbor]
            visited_edges.add(edge)

            current = neighbor
            previous = start

            while current not in characteristic_nodes:

                next_nodes = [
                    node for node in G.neighbors(current)
                    if node != previous
                ]

                if not next_nodes:
                    break

                next_node = next_nodes[0]

                visited_edges.add(frozenset((current, next_node)))

                path.append(next_node)

                previous, current = current, next_node

            segments.append(path)

    return {
        "n_vertices": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_cc": nx.number_connected_components(G),
        "n_leaves": len(leaves),
        "n_bifurcations": len(bifurcations),
        "n_segments": len(segments),
        "length": length,
    }


if __name__ == "__main__":

    with open(snakemake.log[0], "w", buffering=1, encoding="utf-8") as log_file:

        sys.stderr = sys.stdout = log_file

        # 1. Wildcards and parameters
        test = snakemake.wildcards.test
        sub = snakemake.wildcards.sub
        side = snakemake.wildcards.side
        method = snakemake.wildcards.method_path

        alpha = np.nan
        beta = np.nan
        if "SFG" in method:
            alpha = float(method.split("/")[1].split("d")[1].split("_")[0])
            beta = float(method.split("l")[1])

        method = snakemake.wildcards.method_path.split("/")[0]

        # 2. Load white mesh and textures
        white = sio.load_mesh(snakemake.input.white_gii)
        basins = sio.load_texture(snakemake.input.basins_gii).darray[0]
        skeletons = sio.load_texture(snakemake.input.skeletons_labeled).darray[0]
        dpf = sio.load_texture(snakemake.input.dpf_white_gii).darray[0]

        # 4. Calcul des stats
        results = []

        IG = compute_gyrification_index(white)
        area_mesh = white.area
        volume_mesh = white.volume
        vert_area = compute_vertex_areas(white)

        results = []

        IG = compute_gyrification_index(white)
        area_mesh = white.area
        volume_mesh = white.volume
        cbrt_vol = np.cbrt(np.abs(volume_mesh)) #racine cubique du volume
        #test
        faces = np.asarray(white.faces)
        face_areas = np.asarray(white.area_faces)
        face_labels = basins[faces]
        same_label = np.all(face_labels == face_labels[:, :1], axis=1)
        valid = same_label & (face_labels[:, 0] > 0)

        labels = face_labels[valid, 0]
        areas = face_areas[valid]

        area_by_basin = (
            pd.Series(areas)
            .groupby(labels)
            .sum()
            .to_dict()
        )

        #métriques par bassin
        for basin in np.unique(basins):

            # 1. Construction du graphe final
            skeleton_graph = utils.build_basin_graph(
                basin=basin,
                skeletons=skeletons,
                edges=white.edges_unique,
                vertices=white.vertices,
                dpf=dpf,
            )

            # 2. Calcul des métriques du graphe final
            metrics = {
                "n_vertices": np.nan,
                "n_edges": np.nan,
                "n_cc": np.nan,
                "n_leaves": np.nan,
                "n_bifurcations": np.nan,
                "n_segments": np.nan,
                "length": np.nan,
            }
            if skeleton_graph is not None:
                metrics = compute_graph_metrics(skeleton_graph)

            # 3. Aire du bassin
            #basin_mask = skeletons == basin
            #area_basin = np.sum(vert_area[basin_mask])

            basin_mask = basins == basin
            area_basin = np.sum(vert_area[basin_mask])
            #area_basin = area_by_basin.get(basin, np.nan)

            # 4. Stockage des résultats
            results.append({
                "method": method,
                "subject": sub,
                "side": side,
                "test": test,
                "basin": basin,
                "alpha": alpha,
                "beta": beta,

                "IG": IG,
                "area_mesh": area_mesh,
                "volume_mesh": volume_mesh,
                "cbrt_volume_mesh": cbrt_vol,
                "alpha_cbrt_volume": alpha * cbrt_vol,
                "beta_cbrt_volume": beta * cbrt_vol,

                "n_leaves_basin": metrics["n_leaves"],
                "length_basin": metrics["length"],
                "n_cc_basin": metrics["n_cc"],
                "n_bifurcations_basin": metrics["n_bifurcations"],
                "n_segments_basin": metrics["n_segments"],

                "area_basin": area_basin,
            })

        df_results = pd.DataFrame(results)
        df_results.to_csv(
            snakemake.output.results_tsv,
            sep="\t",
            index=False,
        )
