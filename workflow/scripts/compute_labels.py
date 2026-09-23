#!/usr/bin/env python
"""
Relabel sulcal fundi vertices according to the reproducible basins they cross.
"""

import numpy as np
import networkx as nx
import slam.io as sio
import slam.texture as stex


def build_mesh_graph(mesh):
    """
    Build a weighted graph from a surface mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Surface mesh used to construct the graph.

    Returns
    -------
    graph : networkx.Graph
        Graph representing the mesh, with mesh edges as graph edges and
        edge lengths as weights.
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

    Parameters
    ----------
    texture : numpy.ndarray
        Sulcal texture in which -1 and 0 represent non-sulcal vertices.

    Returns
    -------
    numpy.ndarray
        Binary texture with 0 for non-sulcal vertices and 1 for sulcal
        vertices.
    """
    return np.where((texture == -1) | (texture == 0), 0, 1)


def labeling_sulci(texture, mesh):
    """
    Label each connected sulcal component with a unique integer.

    Parameters
    ----------
    texture : numpy.ndarray
        Binary sulcal texture, with 1 indicating sulcal vertices.
    mesh : trimesh.Trimesh
        Surface mesh corresponding to the texture.

    Returns
    -------
    labeled_texture : numpy.ndarray
        Texture in which each connected sulcal component is assigned a
        unique integer label starting from 1.
    """
    graph = build_mesh_graph(mesh)

    sulci_nodes = np.flatnonzero(texture == 1)

    sulci_graph = graph.subgraph(sulci_nodes)
    components = nx.connected_components(sulci_graph)

    labeled_texture = np.zeros_like(texture, dtype=np.int32)

    for label, component in enumerate(components, start=1):
        labeled_texture[list(component)] = label

    return labeled_texture


sphere_reg_mesh = sio.load_mesh(snakemake.input.sphere_reg_gii)
filtered_basins = np.load(snakemake.input.filtered_basins_npy).tolist()
basins = sio.load_texture(snakemake.input.basins_gii).darray[0]

print(np.unique(basins))
print(min(np.unique(basins)))

n_vertices = len(sphere_reg_mesh.vertices)

input_files = [
    path
    for path in snakemake.input
    if path
    not in {
        snakemake.input.filtered_basins_npy,
        snakemake.input.basins_gii,
        snakemake.input.sphere_reg_gii,
    }
]

for input_file, output_file in zip(input_files, snakemake.output):

    print(input_file)
    print(output_file)

    sulci = sio.load_texture(input_file).darray[0]
    sulci_normalised = normalize_texture(sulci)
    sulci_comp = labeling_sulci(sulci_normalised, sphere_reg_mesh)

    sulci_labeled = np.zeros(n_vertices, dtype=np.int32)

    sulci_labeled[sulci_comp > 0] = basins[sulci_comp > 0]

    print(sulci_labeled)
    sio.write_texture(
        stex.TextureND(darray=sulci_labeled[np.newaxis, :]),
        output_file,
    )
