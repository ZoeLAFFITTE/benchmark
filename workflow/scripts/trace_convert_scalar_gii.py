import numpy as np
import slam.io as sio
import slam.texture as stex


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

scurve_file_path = snakemake.input.sulci_curv
scurve_bary_file_path = snakemake.input.sulci_curv_bary
white_mesh = sio.load_mesh(snakemake.input.white_gii)

n_vertices = len(white_mesh.vertices)


# ---------------------------------------------------------------------------
# Sulcal curves
# ---------------------------------------------------------------------------

sulci_texture = np.zeros(n_vertices, dtype=np.int8)
sulci_label_texture = np.zeros(n_vertices, dtype=np.int32)

with open(scurve_file_path, "r") as file:
    for sulcus_id, line in enumerate(file, start=1):
        indices = np.fromstring(line, dtype=int, sep=" ")

        sulci_texture[indices] = 1
        sulci_label_texture[indices] = sulcus_id


# Write binary sulci texture
sio.write_texture(
    stex.TextureND(darray=sulci_texture),
    snakemake.output.sulci_gii,
)

# Write sulci label texture
sio.write_texture(
    stex.TextureND(darray=sulci_label_texture),
    snakemake.output.sulci_c_gii,
)


# ---------------------------------------------------------------------------
# Geodesic barycentric sulcal curves
# ---------------------------------------------------------------------------

geodesic_bary_texture = np.zeros(n_vertices, dtype=np.int8)
geodesic_bary_label_texture = np.zeros(n_vertices, dtype=np.int32)

branches = []      # liste de listes de sommets (ordre conservé)
current = None

with open(scurve_bary_file_path, "r") as file:
    next(file)  # nombre total de branches

    for line in file:
        values = line.split()
        if not values:
            continue

        if len(values) == 1:          # début d'une nouvelle branche
            current = []
            branches.append(current)
            continue

        v1, v2, coef = int(values[0]), int(values[1]), float(values[2])
        current.append(v1 if coef > 0.5 else v2)   # sommet le plus proche

for sulcus_id, verts in enumerate(branches, start=1):
    verts = np.array(verts, dtype=int)

    # np.unique trie : on garde plutôt l'ordre d'apparition
    _, idx = np.unique(verts, return_index=True)
    verts = verts[np.sort(idx)]

    geodesic_bary_texture[verts] = 1
    geodesic_bary_label_texture[verts] = sulcus_id


# Write binary barycentric sulci texture
sio.write_texture(
    stex.TextureND(darray=geodesic_bary_texture),
    snakemake.output.sulci_bary_gii,
)

# Write barycentric sulci label texture
sio.write_texture(
    stex.TextureND(darray=geodesic_bary_label_texture),
    snakemake.output.sulci_bary_c_gii,
)