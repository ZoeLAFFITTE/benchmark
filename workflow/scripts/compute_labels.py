
#!/usr/bin/env python
"""
Relabel sulcal fundi vertices according to the basins they cross.
"""

import numpy as np
import networkx as nx
import slam.io as sio
import slam.texture as stex
import sys


def build_mesh_graph(mesh):
    """
    Build a weighted graph from a surface mesh.
    """
    graph = nx.Graph()
    edges = mesh.edges_unique
    lengths = mesh.edges_unique_length

    for (u, v), weight in zip(edges, lengths):
        graph.add_edge(
            int(u),
            int(v),
            weight=float(weight),
        )

    return graph


def normalize_texture(texture):
    """
    Convert a sulcal texture to a binary texture.
    """
    return np.where((texture == -1) | (texture == 0), 0, 1)


def labeling_sulci(texture, mesh):
    """
    Label each connected sulcal component with a unique integer.
    """
    graph = build_mesh_graph(mesh)

    sulci_nodes = np.flatnonzero(texture == 1)

    sulci_graph = graph.subgraph(sulci_nodes)
    components = nx.connected_components(sulci_graph)

    labeled_texture = np.zeros_like(texture, dtype=np.int32)

    for label, component in enumerate(components, start=1):
        labeled_texture[list(component)] = label

    return labeled_texture


if __name__ == "__main__":

    with open(snakemake.log[0], "w", buffering=1) as log_file:
        sys.stderr = sys.stdout = log_file

        sphere_reg_mesh = sio.load_mesh(snakemake.input.sphere_reg_gii)

        basins = sio.load_texture(
            snakemake.input.basins_gii
        ).darray[0]

        n_vertices = len(sphere_reg_mesh.vertices)

        # Identification de la méthode
        method = snakemake.wildcards.method_path

        # Chargement du squelette correspondant à cette méthode
        input_file = snakemake.input.skeleton
        output_file = snakemake.output.skeletons_labeled

        print(f"Method: {method}")
        print(f"Input skeleton: {input_file}")
        print(f"Output texture: {output_file}")

        sulci = sio.load_texture(input_file).darray[0]

        # Identification des composantes sulcales
        sulci_normalised = normalize_texture(sulci)
        sulci_comp = labeling_sulci(sulci_normalised, sphere_reg_mesh)

        # Conservation de tous les bassins traversés par le squelette
        sulci_labeled = np.zeros(n_vertices, dtype=np.int32)

        mask = sulci_comp > 0

        sulci_labeled[mask] = basins[mask]

        print(f"Number of labeled vertices: {np.count_nonzero(sulci_labeled)}")
        print(f"Unique labels: {np.unique(sulci_labeled)}")

        sio.write_texture(
            stex.TextureND(darray=sulci_labeled[np.newaxis, :]),
            output_file,
        )
