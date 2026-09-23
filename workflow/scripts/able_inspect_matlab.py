import slam.io as sio
import slam.texture as stex
from scipy.io import loadmat
import numpy as np


def load_morphology(mat_path, side):
    """Load the ABLE morphology structure for one hemisphere."""
    data = loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
    )

    side_name = {
        "L": "left",
        "R": "right",
    }.get(side)

    return data["morph"][side_name]

def inspect_object(obj, name, file_handle, indent=0):
    """Recursively write the structure of a MATLAB object to a text file."""
    prefix = " " * indent

    if hasattr(obj, "_fieldnames"):
        file_handle.write(f"{prefix}{name}: struct\n")

        for field in obj._fieldnames:
            value = getattr(obj, field)
            inspect_object(
                value,
                field,
                file_handle,
                indent=indent + 2,
            )
        return

    shape = getattr(obj, "shape", None)

    file_handle.write(
        f"{prefix}{name}: "
        f"type={type(obj)}, "
        f"shape={shape}\n"
    )

def vertices_to_texture(vertices, n_vertices):
    """Convert vertex indices into a binary texture."""
    texture = np.zeros(n_vertices, dtype=np.float32)

    vertices = np.asarray(vertices, dtype=np.int64)
    vertices = np.unique(vertices)

    if np.any(vertices < 0) or np.any(vertices >= n_vertices):
        raise ValueError(
            "Some vertex indices are outside the mesh range."
        )

    texture[vertices] = 1.0

    return texture


# ------------------------------------------------------------------
# Inputs / outputs
# ------------------------------------------------------------------
mat_path = snakemake.input.mat
white_mesh_path = snakemake.input.white_gii
side = snakemake.input.side

inspect_path = snakemake.output.desc_txt
gyri_path = snakemake.output.gyri_gii
sulci_path = snakemake.output.sulci_gii
depth_path = snakemake.output.depth_gii
basins_path = snakemake.output.basins_gii
endpoints_path = snakemake.output.endpoints_npy

# ------------------------------------------------------------------
# Load mesh and ABLE morphology
# ------------------------------------------------------------------
white_mesh = sio.load_mesh(white_mesh_path)
morph = load_morphology(mat_path, side)

print(morph)
print(morph.sulci)

n_vertices = len(white_mesh.vertices)
depthmap = np.asarray(morph.depthmap)

print(n_vertices)
print(depthmap.shape[0])

# ------------------------------------------------------------------
# Inspect MATLAB structure
# ------------------------------------------------------------------
with inspect_path.open("w") as file_handle:
    for key, value in morph.__dict__.items():
        if not key.startswith("__"):
            inspect_object(
                value,
                key,
                file_handle,
            )

# ------------------------------------------------------------------
# Depthmap texture
# ------------------------------------------------------------------
depth_texture = stex.TextureND(
    darray=[
        depthmap.astype(np.float32)
    ]
)

# ------------------------------------------------------------------
# Sulcal basins texture
# ------------------------------------------------------------------
basins = np.asarray(morph.sulcal_basins)

basins_texture = stex.TextureND(
    darray=[
        basins.astype(np.float32)
    ]
)

# ------------------------------------------------------------------
# Sulci texture
# ------------------------------------------------------------------
sulci_texture = vertices_to_texture(
    morph.sulci,
    n_vertices,
)

sulci_texture = stex.TextureND(
    darray=[sulci_texture]
)

# ------------------------------------------------------------------
# Gyri texture
# ------------------------------------------------------------------
gyri_texture = vertices_to_texture(
    morph.gyri,
    n_vertices,
)

gyri_texture = stex.TextureND(
    darray=[gyri_texture]
)

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
endpoints = np.asarray(
    morph.endpoints,
    dtype=np.int64,
)

np.save(
    endpoints_path,
    endpoints,
)

# ------------------------------------------------------------------
# Write textures
# ------------------------------------------------------------------
sio.write_texture(
    gyri_texture,
    gyri_path,
)

sio.write_texture(
    sulci_texture,
    sulci_path,
)

sio.write_texture(
    depth_texture,
    depth_path,
)

sio.write_texture(
    basins_texture,
    basins_path,
)