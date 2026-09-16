import numpy as np
import slam.io as sio
import slam.texture as stex

all_indices = []

scurve_file_path = snakemake.input.sulci_curv
white_mesh_path = snakemake.input.white_gii

white_mesh = sio.load_mesh(white_mesh_path)

texture_c = np.zeros(len(white_mesh.vertices), dtype=int)

with open(scurve_file_path, "r") as f:
    for color, line in enumerate(f):
        line = np.fromstring(line, dtype=int, sep=" ")
        all_indices.extend(line)
        texture_c[line] = int(color)


all_indices = np.unique(all_indices)


# --- Créer une "texture" binaire : 1 si point appartient à une courbe sulcale, 0 sinon ---
texture = np.zeros(len(white_mesh.vertices), dtype=int)
texture[all_indices] = 1

tex = stex.TextureND(darray=texture)
sio.write_texture(tex, snakemake.output.sulci_gii)

# --- Créer une "texture" continue : 1 couleur par segment ---
tex_c = stex.TextureND(darray=texture_c)
sio.write_texture(tex_c, snakemake.output.sulci_c_gii)
