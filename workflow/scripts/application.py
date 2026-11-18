#!/usr/bin/env python

"""
Application morphométrique vs complexité des SFG, ISBI 2026.
"""

import sys
import numpy as np
import pandas as pd
import slam.io as sio
import slam.plot as splt
import slam.remeshing as srem
from utils import utils


if __name__ == "__main__":

    with open(snakemake.log[0], "w", buffering=1, encoding="utf-8") as log_file:

        sys.stderr = sys.stdout = log_file

        # 1. Wildcards and parameters
        method = snakemake.wildcards.method_path
        side = snakemake.wildcards.side
        subjects = list(snakemake.params.subjects)
        tests = list(snakemake.params.tests)

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
        pop = []
        for test in tests:
            for subject in subjects:
                basins = sio.load_texture(basin_paths[idx]).darray[0]
                white = sio.load_mesh(white_paths[idx])
                sphere_reg = sio.load_mesh(sphere_reg_paths[idx])
                stats = pd.read_csv(stats_paths[idx], sep="\t")

                # Project basin labels onto template
                basins_template = np.asarray(
                    srem.texture_spherical_interpolation_nearest_neighbor(
                        sphere_reg,
                        sphere_template,
                        basins,
                    ),
                    dtype=float,
                )

                # Replace basin labels with morphometric values
                for label in np.unique(basins_template):

                    if label == 0:
                        continue

                    stats_basin = stats[
                        stats["basin"] == label
                    ]
                    label = int(label)
                    
                    area = stats_basin["area_basin"].iloc[0]
                    print(f"area basin {label}: {area}")
                    n_leaves = stats_basin["n_leaves_basin"].iloc[0]
                    print(f"n_leaves basin {label}: {n_leaves}")
                    print(f"res basin{label}: {n_leaves / np.sqrt(area)}")

                    if area > 0 and np.isfinite(n_leaves):
                        basins_template[basins_template == label] = n_leaves / np.sqrt(area)
                
                # Store individual map
                pop.append(basins_template.copy())

                # Compute basin boundaries
                label_max = float(np.max(basins_template))
                boundary = utils.get_border_mask(white_template.faces, basins_template).astype(int)
                basins_template[boundary == 1] = label_max

                vmax = float(np.nanmax(basins_template))

                # Plot individual plot
                fig_1 = splt.plot_mesh(
                    {
                        "vertices": white_template.vertices,
                        "faces": white_template.faces,
                        "title": f"Leaves / √Basin area — {method}, {subject}, {side}, {test}",
                    },
                    {
                        "values": basins_template,
                        "mode": "vertex",
                        "cmin": 0,
                        "cmax": vmax,
                    },
                    {
                        "template": None,
                        "colorscale": "Turbo",
                        "colorbar_label": "Leaves / √Basin area (mm⁻¹)"
                    },
                    caption=False,
                )
                fig_1.write_html(snakemake.output.individual_plots[idx])

                idx += 1
        
        # Plot population mean
        stack = np.stack(pop, axis=0)
        mean_all = np.mean(stack, axis=0)

        # Common colorbar limits
        vmax = float(np.nanmax(mean_all))

        fig_2 = splt.plot_mesh(
            {
                "vertices": white_template.vertices,
                "faces": white_template.faces,
                "title": f"Leaves / √Basin area — {method}, Mean pop, {side}",
            },
            {
                "values": mean_all,
                "mode": "vertex",
                "cmin": 0,
                "cmax": vmax,
            },
            {
                "template": None,
                "colorscale": "Turbo",
                "colorbar_label": "Leaves / √Basin area (mm⁻¹)"
            },
            caption=False,
        )
        fig_2.write_html(snakemake.output.pop_plot)
                
