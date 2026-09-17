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

sulcus_id = 1

with open(scurve_bary_file_path, "r") as file:
    next(file)  # Skip header

    for line in file:
        values = line.split()

        # A single value indicates the beginning of a new sulcus
        if len(values) == 1:
            sulcus_id += 1
            continue

        v1, v2 = map(int, values[:2])

        # Binary texture
        geodesic_bary_texture[v1] = 1
        geodesic_bary_texture[v2] = 1

        # Sulcus label texture
        geodesic_bary_label_texture[v1] = sulcus_id
        geodesic_bary_label_texture[v2] = sulcus_id


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