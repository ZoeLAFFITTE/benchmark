import numpy as np
import slam.io as sio
import nibabel as nib
import pyvista as pv
from nibabel.gifti import GiftiImage, GiftiDataArray

def write_vtk_fs_ascii(filename, coords, faces):
    """
    Écrit un fichier mesh VTK 1.0 ASCII compatible FreeSurfer.
    coords : (N,3) numpy array
    faces  : (M,3) numpy array
    """
    n_points = coords.shape[0]
    n_faces = faces.shape[0]

    with open(filename, 'w') as f:
        # Header VTK 1.0 ASCII
        f.write("# vtk DataFile Version 1.0\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_points} float\n")

        # Points : un par ligne
        for p in coords:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        # Polygons : triangles
        total_ints = n_faces * 4  # 3 indices + 1 nombre par triangle
        f.write(f"POLYGONS {n_faces} {total_ints}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

    print(f"VTK FreeSurfer ASCII écrit : {filename}")

def vtk_scalar_to_gifti(vtk_file, gii_file, dtype):

    # Read scalar from VTK
    mesh = pv.read(vtk_file)
    texture = np.asarray(mesh[mesh.array_names[0]], dtype=dtype)

    data_array = nib.gifti.GiftiDataArray(
        data=texture, intent="NIFTI_INTENT_SHAPE", datatype="NIFTI_TYPE_FLOAT32"
    )

    gifti_image = nib.gifti.GiftiImage(darrays=[data_array])

    nib.save(gifti_image, gii_file)


input_files = [
    path for path in snakemake.input
    if path != snakemake.input.mesh
]

for input_file, output_file in zip(input_files, snakemake.output):
    vtk_scalar_to_gifti(input_file, output_file, dtype= np.float32)