#!/usr/bin/env python
"""
Generation of HTML visualizations (DPF, basin boundaries, skeletons)
on the white mesh for the ISBI 2026 paper.
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
from utils import utils


def get_border_mask(faces, labels):
    """
    Compute a one-vertex-wide boundary mask between distinct basins.
    """
    mask = np.zeros(len(labels), dtype=bool)

    for a, b in ((0, 1), (1, 2), (2, 0)):
        va, vb = faces[:, a], faces[:, b]
        diff = labels[va] != labels[vb]
        lower = np.where(labels[va] < labels[vb], va, vb)
        mask[lower[diff]] = True

    return mask


def create_base_figure(mesh, values, cmin=None, cmax=None):
    """
    Create a base figure of the white mesh colored by DPF.
    """
    return splt.plot_mesh(
        {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
        },
        {
            "values": values,
            "mode": "vertex",
            "cmin": cmin,
            "cmax": cmax,
        },
        {
            "template": None,
            "colorscale": "Agsunset",
        },
        caption=False,
    )


def create_skeleton_trace(
    skeletons,
    white,
    reproducible_basins,
    dpf,
    method_name,
    color,
    normal_offset=0.3,
):

    x, y, z = [], [], []

    vertices = white.vertices + normal_offset * white.vertex_normals
    faces = np.asarray(white.faces)
    edges = white.edges_unique

    for basin in reproducible_basins:

        G = utils.build_basin_graph(
            basin=basin,
            skeletons=skeletons,
            edges=edges,
            vertices=vertices,
            dpf=dpf,
        )

        if G is None:
            continue

        # Arêtes
        for u, v in G.edges():

            points = vertices[[u, v]]

            x.extend(points[:, 0])
            y.extend(points[:, 1])
            z.extend(points[:, 2])

            x.append(None)
            y.append(None)
            z.append(None)

        # Sommets isolés
        isolated_nodes = [
            node for node, degree in G.degree()
            if degree == 0
        ]

        for node in isolated_nodes:
            x.append(vertices[node, 0])
            y.append(vertices[node, 1])
            z.append(vertices[node, 2])

            x.append(None)
            y.append(None)
            z.append(None)

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines+markers",
        line=dict(width=10, color=color),
        marker=dict(size=2, color=color),
        name=method_name,
        showlegend=True,
    )


def get_reproducible_basins(match_tsv, test, threshold):
    """
    Retrieve reproducible basin labels for the selected acquisition
    (HCP_test or HCP_retest).
    """

    df_matches = pd.read_csv(match_tsv, sep="\t")

    df_matches["overlap"] = pd.to_numeric(
        df_matches["overlap"],
        errors="raise",
    )

    df_reproducible = df_matches[
        df_matches["overlap"] >= threshold
    ]

    basin_column = f"{test}_basin"

    reproducible_basins = (
        df_reproducible[basin_column]
        .dropna()
        .astype(int)
        .to_numpy()
    )

    print(f"Acquisition: {test}")
    print(f"Overlap threshold: {threshold}")
    print(f"Number of reproducible basins: {len(reproducible_basins)}")
    print(f"Reproducible basins: {reproducible_basins}")

    return reproducible_basins


if __name__ == "__main__":

    with open(snakemake.log[0], "w", buffering=1) as log_file:

        sys.stderr = sys.stdout = log_file

        # 1. Wildcards and parameters
        test = snakemake.wildcards.test
        sub = snakemake.wildcards.sub
        side = snakemake.wildcards.side
        threshold = snakemake.params.thr_basins

        print("=" * 70)
        print(f"Subject: {sub}")
        print(f"Hemisphere: {side}")
        print(f"Test: {test}")
        print(f"Reproducibility threshold: {threshold}")
        print("=" * 70)

        # 2. Load reproducible basins
        reproducible_basins = get_reproducible_basins(
            match_tsv=snakemake.input.match_tsv,
            test=test,
            threshold=threshold,
        )

        # 3. Load white mesh and textures
        white = sio.load_mesh(snakemake.input.white_gii)
        dpf = sio.load_texture(snakemake.input.dpf_white_gii).darray[0]
        basins = sio.load_texture(snakemake.input.basins_white_gii).darray[0]

        # 4. Prepare DPF and basin boundaries
        boundary = get_border_mask(white.faces, basins).astype(int)
        dpf_min = float(np.min(dpf))
        dpf_max = float(np.max(dpf))
        dpf_boundary = dpf.copy()
        dpf_boundary[boundary == 1] = dpf_min

        # 5. DPF visualization
        print("Generating DPF visualization...")
        fig_dpf = create_base_figure(white, dpf_boundary, cmin=dpf_min, cmax=dpf_max)
        fig_dpf.write_html(snakemake.output.dpf_plot)

        # 6. Basin boundaries visualization
        print("Generating basin boundaries visualization...")

        fig_boundary = splt.plot_mesh(
            {
                "vertices": white.vertices,
                "faces": white.faces,
            },
            {
                "values": boundary,
                "mode": "vertex",
            },
            {
                "template": None,
                "colorscale": [
                    [0, "white"],
                    [0.5, "red"],
                    [1, "red"],
                ],
            },
            caption=False,
        )

        # Add mesh edges
        edge_vertices = white.vertices[white.edges_unique]
        coords = np.full((len(edge_vertices), 3, 3), np.nan)
        coords[:, :2, :] = edge_vertices

        fig_boundary.add_trace(
            go.Scatter3d(
                x=coords[:, :, 0].ravel(),
                y=coords[:, :, 1].ravel(),
                z=coords[:, :, 2].ravel(),
                mode="lines",
                line={
                    "color": "black",
                    "width": 1,
                },
                showlegend=False,
            )
        )

        fig_boundary.write_html(snakemake.output.boundary_plot)

        # 7. Skeleton visualizations
        print("Generating skeleton visualizations...")

        #colors = px.colors.qualitative.Dark24
        colors = ["red", "yellow", "blue", "green"]

        fig_all = create_base_figure(
            white,
            dpf_boundary,
            cmin=dpf_min,
            cmax=dpf_max,
        )

        # Itération propre et directe via les paramètres et entrées gérés par Snakemake
        method_names = snakemake.params.method_names
        skeleton_paths = snakemake.input.skeletons_labeled
        individual_out_files = snakemake.output.individual_plots

        for i, (method, skeleton_path, out_file) in enumerate(
            zip(method_names, skeleton_paths, individual_out_files)
        ):
            color = colors[i % len(colors)]

            print(f"Processing method: {method}")
            print(f"  -> Path: {skeleton_path}")

            # Load labeled skeleton
            skeletons = sio.load_texture(str(skeleton_path)).darray[0]

            # Create skeleton trace
            trace = create_skeleton_trace(
                skeletons=skeletons,
                white=white,
                reproducible_basins=reproducible_basins,
                dpf=dpf,
                method_name=method,
                color=color,
                normal_offset=0.3,
            )

            # ---------------------------------------------------------
            # Individual method visualization
            # ---------------------------------------------------------
            fig_method = create_base_figure(
                white,
                dpf_boundary,
                cmin=dpf_min,
                cmax=dpf_max,
            )

            fig_method.add_trace(trace)
            fig_method.write_html(out_file)
            print(f"Individual plot saved: {out_file}")

            # ---------------------------------------------------------
            # Collective visualization
            # ---------------------------------------------------------
            fig_all.add_trace(trace)

        # 8. Export collective visualization
        fig_all.write_html(snakemake.output.skeletons_plot)

        print(
            f"Collective skeleton visualization saved: "
            f"{snakemake.output.skeletons_plot}"
        )

        print("=" * 70)
        print("All ISBI visualizations successfully generated.")
        print("=" * 70)