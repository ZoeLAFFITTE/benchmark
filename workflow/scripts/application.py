#!/usr/bin/env python

"""
Application morphométrique vs complexité des SFG, ISBI 2026.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import slam.io as sio
import slam.plot as splt
import networkx as nx
import slam.remeshing as srem
from utils import utils


if __name__ == "__main__":

    with open(snakemake.log[0], "w", buffering=1) as log_file:

        sys.stderr = sys.stdout = log_file

        # 1. Wildcards and parameters
        method = snakemake.wildcards.method_path
        side = snakemake.wildcards.side
        subjects = list(snakemake.params.subjects)
        tests = list(snakemake.params.tests)

        skeleton_paths = list(snakemake.input.skeletons)
        basin_paths = list(snakemake.input.basins)
        white_paths = list(snakemake.input.white_gii)
        stats_paths = list(snakemake.input.stats)
        sphere_reg_paths = list(snakemake.input.sphere_reg_gii)

        sphere_template_path = snakemake.input.sphere_template_gii
        white_template_path = snakemake.input.white_template_gii

        # 2. Load white mesh and textures
        sphere_template = sio.load_mesh(sphere_template_path)
        white_template = sio.load_mesh(white_template_path)

        # 3. On compute les texture pour le template pour la représentation aprox
        idx = 0

        for test in tests:
            for subject in subjects:
                skeletons = sio.load_texture(skeleton_paths[idx]).darray[0]
                basins = sio.load_texture(basin_paths[idx]).darray[0]
                white = sio.load_mesh(white_paths[idx])
                sphere_reg = sio.load_mesh(sphere_reg_paths[idx])
                stats = pd.read_csv(stats_paths[idx], sep="\t")

                # Plot1
                boundary = get_border_mask(white.faces, basins).astype(int)
                boundary_tex = np.zeros(basins, dtype=int)
                boundary_tex[boundary == 1] = 1

                boundary_template = np.asarray(
                    srem.texture_spherical_interpolation_nearest_neighbor(
                        sphere_reg,
                        sphere_template,
                        boundary_tex,
                    ),
                    dtype=float,
                )

                basins_template = np.asarray(
                    srem.texture_spherical_interpolation_nearest_neighbor(
                        sphere_reg,
                        sphere_template,
                        basins,
                    ),
                    dtype=float,
                )

                #pour basins, remplacer la val de chaque label pas sa stat associée
                for label in np.unique(basins_template):
                    stats_basin = stats[
                        stats["basin"] == label
                    ]
                    label = int(label)
                    area = stats_basin["area_basin"]
                    n_leaves = stats_basin["n_leaves_basin"]

                    if area > 0 and np.isfinite(n_leaves):
                        basins_template[basins == label] = n_leaves / np.sqrt(area)
                
                label_max = float(np.max(basins_template))
                basins_template[boundary_template == 1] = label_max

                fig_1 = splt.plot_mesh(
                    {
                        "vertices": white_template.vertices,
                        "faces": white_template.faces,
                    },
                    {
                        "values": basins_template,
                        "mode": "vertex",
                    },
                    {
                        "template": None,
                        "colorscale": "Turbo"
                    },
                    caption=False,
                )
                fig_1.write_html(snakemake.output.individual_plots)
                
