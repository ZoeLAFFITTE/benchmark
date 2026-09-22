import trimesh
import numpy as np
import pandas as pd
import slam.io as sio
import slam.plot as splt


METHODS = ["cachia", "trace", "mindboggle"]
METRICS = ["hausdorff", "mean", "p95"]



# -------------------------------------------------------------------------
# Charger les 45 textures de bassins
# -------------------------------------------------------------------------
mesh_test = sio.load_mesh(snakemake.input.sphere_reg_gii_test)
mesh_retest = sio.load_mesh(snakemake.input.sphere_reg_gii_retest)

basins_test = np.load(snakemake.input.filtered_basins_npy_test).tolist()
basins_retest = np.load(snakemake.input.filtered_basins_npy_retest).tolist()

subject = snakemake.wildcards.sub
print(subject)

# -------------------------------------------------------------------------
# Trace
# -------------------------------------------------------------------------
trace_test = sio.load_texture(snakemake.input.sulci_trace_labeled_test).darray[0]
trace_retest = sio.load_texture(snakemake.input.sulci_trace_labeled_retest).darray[0]

# -------------------------------------------------------------------------
# Cachia
# -------------------------------------------------------------------------
cachia_test = sio.load_texture(snakemake.input.sulci_cachia_labeled_test).darray[0]
cachia_retest = sio.load_texture(snakemake.input.sulci_cachia_labeled_retest).darray[0]

# -------------------------------------------------------------------------
# Mindboggle
# -------------------------------------------------------------------------
mindboggle_test = sio.load_texture(snakemake.input.sulci_mindboggle_labeled_test).darray[0]
mindboggle_retest = sio.load_texture(snakemake.input.sulci_mindboggle_labeled_retest).darray[0]

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

def extract_submesh(mesh, indices):
    """
    Retourne un sous-mesh propre à partir d'une sélection de vertices.
    """

    vertex_mapping = {old: new for new, old in enumerate(indices)}

    vertices = mesh.vertices[indices]

    faces = [
        [vertex_mapping[v] for v in face]
        for face in mesh.faces
        if all(v in vertex_mapping for v in face)
    ]

    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(faces),
        process=False
    )

def compute_metrics(labels_test, labels_retest, test_texture, retest_texture, test_points, retest_points):
    for label_test in labels_test:

            # Find corresponding label in retest
            label_retest = labels_retest[labels_test.index(label_test)]

            test_mask = test_texture == label_test
            retest_mask = retest_texture == label_retest

            test_sulcus = mesh_test.vertices[test_mask]
            retest_sulcus = mesh_retest.vertices[retest_mask]

            #sous-mesh du bassin
            submesh_test = extract_submesh(mesh_test, test_sulcus)
            submesh_retest = extract_submesh(mesh_retest, retest_sulcus)

            G = build_mesh_graph(submesh_test)
            
            sulci_nodes = np.flatnonzero(texture == 1)
            bifurcation_nodes = np.flatnonzero(texture == 2)
            endpoints = np.flatnonzero(texture == 3)
        
            H = G.subgraph(sulci_nodes)
            components = nx.connected_components(H)
        
            for icomp, component in enumerate(components):
                info = {
                    "sujet": sub,
                    "bassin": label,
                    n_component: 0,
                    "component_id": icomp,
                    "n_bifurcation": 0,
                    "n_segment": 0,
                    "angles": []
                }

