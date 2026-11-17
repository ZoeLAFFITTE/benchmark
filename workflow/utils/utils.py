#!/usr/bin/env python
"""
Quelques fonctions utilitaires.
"""

import numpy as np
import networkx as nx

def build_basin_graph(
    basin,
    skeletons,
    edges,
    vertices,
    dpf,
):
    """
    Construit le graphe final d'un bassin à partir des plus courts
    chemins entre chaque extrémité et la cible de DPF minimal.

    Returns
    -------
    G_final : nx.Graph
        Graphe final reconstruit.
    """

    basin_vertices = np.flatnonzero(skeletons == basin)

    if len(basin_vertices) == 0:
        return None

    # Graphe initial
    basin_edges = edges[
        (skeletons[edges[:, 0]] == basin)
        & (skeletons[edges[:, 1]] == basin)
    ]

    G = nx.Graph()
    G.add_nodes_from(basin_vertices)

    for i, j in basin_edges:
        distance = np.linalg.norm(vertices[i] - vertices[j])
        G.add_edge(i, j, weight=distance)

    # Reconstruction du graphe final
    G_final = nx.Graph()

    for component in nx.connected_components(G):

        subgraph = G.subgraph(component)

        # Point isolé
        if len(component) == 1:
            G_final.add_node(next(iter(component)))
            continue

        # Extrémités
        endpoints = [
            node for node, degree in subgraph.degree()
            if degree == 1
        ]

        # Cible DPF minimale
        target = min(component, key=lambda node: dpf[node])

        # Chemins des extrémités vers la cible
        for endpoint in endpoints:

            path = nx.shortest_path(
                subgraph,
                source=endpoint,
                target=target,
                weight="weight",
            )

            G_final.add_nodes_from(path)

            for u, v in zip(path[:-1], path[1:]):
                G_final.add_edge(
                    u,
                    v,
                    weight=subgraph[u][v]["weight"],
                )

    return G_final