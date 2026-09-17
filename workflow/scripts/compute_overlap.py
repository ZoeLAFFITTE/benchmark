import numpy as np
import networkx as nx
import slam.io as sio
import slam.texture as stex

def build_mesh_graph(mesh):
    G = nx.Graph()
    edges = mesh.edges_unique
    lengths = mesh.edges_unique_length

    for (u, v), w in zip(edges, lengths):

        G.add_edge(
            int(u),
            int(v),
            weight=float(w)
        )

    return G

def normalize_texture(texture):
    """Set all values except -1 and 0 to 1."""
    return np.where((texture == -1) | (texture == 0), 0, 1)


def labeling_sulci(texture, mesh):
    """Label each connected sulcus component with a unique integer."""
    G = build_mesh_graph(mesh)

    sulci_nodes = np.flatnonzero(texture == 1)

    H = G.subgraph(sulci_nodes)
    components = nx.connected_components(H)

    labeled_texture = np.zeros_like(texture, dtype=np.int32)

    for label, component in enumerate(components, start=1):
        labeled_texture[list(component)] = label

    return labeled_texture


mesh = sio.load_mesh(snakemake.input.sphere_reg_gii)
filtered_basins = np.load(snakemake.input.filtered_basins_npy).tolist()
basins = sio.load_texture(snakemake.input.basins_gii).darray[0]

n_vertices = len(mesh.vertices)

print(n_vertices)
print(len(basins))

input_files = [
    path
    for path in snakemake.input
    if path not in {
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
    sulci_comp = labeling_sulci(sulci_normalised, mesh)

    sulci_labeled = np.zeros(n_vertices, dtype=np.int32)

    sulci_labeled[sulci_comp > 0] = basins[sulci_comp > 0]

    print(sulci_labeled)
    sio.write_texture(
        stex.TextureND(darray=sulci_labeled[np.newaxis, :]),
        output_file,
    )

    # for sulcus_label in np.unique(sulci_comp):

    #     # Ignore background
    #     if sulcus_label == 0:
    #         continue

        # Vertices belonging to this sulcus
        # sulcus_mask = sulci_comp == sulcus_label

        # Basin labels under this sulcus
        # basin_labels = basins[sulcus_mask]

        # # Basin covering the largest number of sulcus vertices
        # labels, counts = np.unique(basin_labels, return_counts=True)
        # covering_basin = labels[np.argmax(counts)]

        # # Assign the basin label to all vertices of this sulcus
        # sulci_labeled[sulcus_mask] = covering_basin


# Garder le sillon si son bassin principal est sélectionné
# if covering_basin in filtered_basins:
#     filtered_sulci_texture[sulcus_mask] = sulcus_label