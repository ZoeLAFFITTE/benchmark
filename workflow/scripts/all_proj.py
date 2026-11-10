#!/usr/bin/env python

"""Project distance metrics onto the template white surface (one side)."""

from pathlib import Path

import numpy as np
import pandas as pd
import slam.io as sio
import slam.plot as splt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------
L_white_mesh_path = snakemake.input.L_white_mesh
R_white_mesh_path = snakemake.input.R_white_mesh

L_trace_sulci_path = snakemake.input.L_trace_sulci
L_mindboggle_sulci_path = snakemake.input.L_mindboggle_sulci
L_zoe_sulci_path = snakemake.input.L_zoe_sulci
L_dpf_star_path = snakemake.input.L_dpf_star

R_trace_sulci_path = snakemake.input.R_trace_sulci
R_mindboggle_sulci_path = snakemake.input.R_mindboggle_sulci
R_zoe_sulci_path = snakemake.input.R_zoe_sulci
R_dpf_star_path = snakemake.input.R_dpf_star

L_filtered_basins_path = snakemake.input.L_filtered_basins
R_filtered_basins_path = snakemake.input.R_filtered_basins

# ------------------------------------------------------------------
# 1. Chargement des maillages
# ------------------------------------------------------------------
L_white_mesh = sio.load_mesh(L_white_mesh_path)
R_white_mesh = sio.load_mesh(R_white_mesh_path)

meshes = {
    "L": L_white_mesh,
    "R": R_white_mesh,
}

# ------------------------------------------------------------------
# 2. Chargement des textures
# Remplacer les chemins R_* par tes variables réelles
# ------------------------------------------------------------------
textures = {
    "L": {
        "trace": sio.load_texture(L_trace_sulci_path).darray[0],
        "mindboggle": sio.load_texture(L_mindboggle_sulci_path).darray[0],
        "zoe": sio.load_texture(L_zoe_sulci_path).darray[0],
        "dpf": sio.load_texture(L_dpf_star_path).darray[0],
    },
    "R": {
        "trace": sio.load_texture(R_trace_sulci_path).darray[0],
        "mindboggle": sio.load_texture(R_mindboggle_sulci_path).darray[0],
        "zoe": sio.load_texture(R_zoe_sulci_path).darray[0],
        "dpf": sio.load_texture(R_dpf_star_path).darray[0],
    },
}

filtered_basins = {
    "L": np.load(L_filtered_basins_path),
    "R": np.load(R_filtered_basins_path),
}

# ------------------------------------------------------------------
# 3. Créer les deux scènes Plotly
# ------------------------------------------------------------------
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=("Left hemisphere", "Right hemisphere"),
    horizontal_spacing=0.02,
)

# ------------------------------------------------------------------
# 4. Ajouter les maillages avec SLAM
# ------------------------------------------------------------------
for hemi, col in [("L", 1), ("R", 2)]:

    mesh = meshes[hemi]

    mesh_data = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "title": f"{hemi} white mesh",
    }
    intensity_data = {
        "values": textures[hemi]["dpf"],
        "mode": "vertex",
    }

    # Utiliser plot_mesh pour générer la trace du maillage
    mesh_fig = splt.plot_mesh(
        mesh_data,
        intensity_data,
        display_settings={"template": "plotly_white"},
    )

    mesh_trace = mesh_fig.data[0]
    mesh_trace.update(showscale=False)

    fig.add_trace(
        mesh_trace,
        row=1,
        col=col,
    )


# ------------------------------------------------------------------
# 5. Ajouter les trois méthodes avec splt.plot_points
# ------------------------------------------------------------------
colors = {
    "zoe": "red",
    "trace": "green",
    "mindboggle": "blue",
}

for hemi, col in [("L", 1), ("R", 2)]:

    mesh = meshes[hemi]

    for method, color in colors.items():

        texture = textures[hemi][method]

        # Sommets du squelette : adapter si nécessaire au codage
        # Ne garder que les sulci dans des bassins reproductibles (>= THR)
        indices = np.flatnonzero(
            (texture > 0) & np.isin(texture, filtered_basins[hemi])
        )

        # Décaler légèrement les points selon les normales
        points = (
            mesh.vertices[indices]
            + 0.3 * mesh.vertex_normals[indices]
        )

        trace = splt.plot_points(
            points,
            marker={
                "size": 2,
                "color": color,
            },
        )

        trace.name = method
        trace.legendgroup = method
        trace.showlegend = (col == 1)

        fig.add_trace(
            trace,
            row=1,
            col=col,
        )

        # # Tracer les arêtes du maillage en noir
        # edge_vertices = mesh.vertices[mesh.edges_unique]
        # coords = np.full((len(edge_vertices), 3, 3), np.nan)
        # coords[:, :2, :] = edge_vertices
    
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=coords[:, :, 0].ravel(),
        #         y=coords[:, :, 1].ravel(),
        #         z=coords[:, :, 2].ravel(),
        #         mode="lines",
        #         line={"color": "black", "width": 1},
        #         showlegend=False,
        #     ),
        #     row=1,
        #     col=col,
        # )

# ------------------------------------------------------------------
# 6. Configuration des scènes
# ------------------------------------------------------------------
scene_settings = {
    "aspectmode": "data",
    "xaxis": {"visible": False},
    "yaxis": {"visible": False},
    "zaxis": {"visible": False},
    "camera": {
        "eye": {"x": 2.5, "y": 0, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
    },
}

fig.update_layout(
    template="plotly_white",
    autosize=True,
    scene=scene_settings,
    scene2={
        **scene_settings,
        "camera": {
            **scene_settings["camera"],
            "eye": {"x": -2.5, "y": 0, "z": 0},
        },
    },
    legend={
        "title": "Method",
        "groupclick": "togglegroup",
    },
)

fig.write_html(snakemake.output.fig, include_plotlyjs="cdn")