#!/usr/bin/env python
"""
Compute the overlap between the basins of two spherical surfaces test & retest
after interpolation onto a reference mesh.
"""

import numpy as np
import pandas as pd
import slam.io as sio
import slam.texture as stex
import slam.remeshing as srem
import slam.plot as splt
import plotly.express as px


def match_basins(test_basins, retest_basins):
    """
    Compute the overlap between the basins of two surfaces after interpolation
    onto a reference mesh.

    Parameters
    ----------
    test_basins : numpy.ndarray
        Basin labels of the test surface.
    retest_basins : numpy.ndarray
        Basin labels of the retest surface.

    Returns
    -------
    correspondence : dict
        Dictionary mapping each test basin label to its best matching retest
        basin label, the corresponding overlap, and the second-best overlap.
    """

    t_labels = np.unique(test_basins)
    rt_labels = np.unique(retest_basins)

    correspondence = {}

    for t_lab in t_labels:
        # get the indices of vertices corresponding to the label t_lab
        inds = set(np.where(test_basins == t_lab)[0])
        max_overlap = -1
        second_max_overlap = -1
        for rt_lab in rt_labels:
            # get the indices of vertices corresponding to the label rt_lab
            corr_inds = set(np.where(retest_basins == rt_lab)[0])
            # compute the overlap as intersection/union of vertices
            lab_overlap = len(inds.intersection(corr_inds)) / len(inds.union(corr_inds))
            # keep only the two highest results
            if lab_overlap > max_overlap:
                max_overlap = lab_overlap
                best_match = rt_lab
            if max_overlap > lab_overlap > second_max_overlap:
                second_max_overlap = lab_overlap
        correspondence[int(t_lab)] = [int(best_match), max_overlap, second_max_overlap]

    return correspondence


def compute_feature(filtered_matches):
    """
    Create a histogram of overlap values between test and retest basins.

    Parameters
    ----------
    filtered_matches : pandas.DataFrame
        DataFrame containing the overlap value for each basin between test
        and retest.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure containing the histogram of overlap values.
    """

    fig = px.histogram(filtered_matches, x="overlap")

    fig.update_xaxes(
        range=[0, 1],
        tick0=0,
        dtick=0.1,
    )

    return fig


def compute_proj(basins_overlap, white_mesh, title):
    """
    Project basin overlap values onto a white mesh.

    Parameters
    ----------
    basins_overlap : slam.texture.TextureND
        Texture containing the overlap value associated with each basin
        of the input mesh.
    white_mesh : trimesh.Trimesh
        Input white surface mesh.
    title : str
        Title of the figure.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure displaying the overlap values on the mesh.
    """

    mesh_data = {
        "vertices": white_mesh.vertices,
        "faces": white_mesh.faces,
        "title": title,
    }

    intensity_data = {
        "values": basins_overlap,
        "mode": "vertex",
    }

    display_settings = {
        "colorbar_label": "Overlap test-retest",
        "template": "plotly_white",
        "colorscale": "Turbo",
    }

    fig = splt.plot_mesh(
        mesh_data,
        intensity_data,
        display_settings,
        caption=True,
    )

    return fig


# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
subject = snakemake.wildcards.sub
basins_test_path = snakemake.input.basins_test_gii
basins_retest_path = snakemake.input.basins_retest_gii

sphere_reg_test_path = snakemake.input.sphere_reg_test_gii
sphere_reg_retest_path = snakemake.input.sphere_reg_retest_gii
sphere_template_path = snakemake.input.sphere_template_gii

white_test_path = snakemake.input.white_test_gii
white_retest_path = snakemake.input.white_retest_gii

# Load meshes
sphere_reg_test = sio.load_mesh(sphere_reg_test_path)
sphere_reg_retest = sio.load_mesh(sphere_reg_retest_path)
sphere_template = sio.load_mesh(sphere_template_path)

white_test = sio.load_mesh(white_test_path)
white_retest = sio.load_mesh(white_retest_path)

# Load texture
basins_test = sio.load_texture(basins_test_path).darray[0]
basins_retest = sio.load_texture(basins_retest_path).darray[0]


# ------------------------------------------------------------------
# Compute
# ------------------------------------------------------------------

# NN interpolation of basins to the template
basins_test_reg = srem.texture_spherical_interpolation_nearest_neighbor(
    sphere_reg_test,
    sphere_template,
    basins_test,
)

basins_retest_reg = srem.texture_spherical_interpolation_nearest_neighbor(
    sphere_reg_retest,
    sphere_template,
    basins_retest,
)

matches = match_basins(
    basins_test_reg,
    basins_retest_reg,
)

print(np.unique(basins_test_reg))
print(np.unique(basins_retest_reg))

print(len(np.unique(basins_test_reg)))
print(len(np.unique(basins_retest_reg)))

df_matches = pd.DataFrame.from_dict(
    matches,
    orient="index",
    columns=["retest_basin", "overlap", "second_overlap"],
)

df_matches.index.name = "test_basin"

df_matches.to_csv(
    snakemake.output.match_tsv,
    sep="\t",
    index=True,
)

print(df_matches)

df_matches["overlap"] = df_matches["overlap"].astype(float)

df_reproducible = df_matches[df_matches["overlap"] >= snakemake.params.thr_basins]

filtered_test_basins = df_reproducible.index.to_numpy(dtype=int)
filtered_retest_basins = df_reproducible["retest_basin"].to_numpy(dtype=int)

print("Test basins:", filtered_test_basins)
print("Retest basins:", filtered_retest_basins)

print("Test basins:", len(filtered_test_basins))
print("Retest basins:", len(filtered_retest_basins))

# Textures overlap
basins_test_overlap = np.zeros(len(basins_test), dtype=float)
basins_retest_overlap = np.zeros(len(basins_retest), dtype=float)

for label_test, row in df_matches.iterrows():
    label_test = int(label_test)
    label_retest = int(row["retest_basin"])
    overlap = float(row["overlap"])

    # Attribuer l'overlap à tous les sommets du bassin correspondant
    basins_test_overlap[basins_test == label_test] = overlap
    basins_retest_overlap[basins_retest == label_retest] = overlap


# ------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------
np.save(snakemake.output.filtered_basins_test_npy, filtered_test_basins)
np.save(snakemake.output.filtered_basins_retest_npy, filtered_retest_basins)

basins_test_overlap_tex = stex.TextureND(darray=basins_test_overlap)
basins_retest_overlap_tex = stex.TextureND(darray=basins_retest_overlap)
sio.write_texture(basins_test_overlap_tex, snakemake.output.basins_test_overlap_gii)
sio.write_texture(basins_retest_overlap_tex, snakemake.output.basins_retest_overlap_gii)

basins_test_reg_tex = stex.TextureND(darray=basins_test_reg)
basins_retest_reg_tex = stex.TextureND(darray=basins_retest_reg)
sio.write_texture(basins_test_reg_tex, snakemake.output.basins_test_reg_gii)
sio.write_texture(basins_retest_reg_tex, snakemake.output.basins_retest_reg_gii)

hist = compute_feature(df_matches)
hist.write_html(snakemake.output.hist, include_plotlyjs="cdn")

proj_test = compute_proj(
    basins_test_overlap, white_test, f"{subject}_basin_overlap_test"
)
proj_retest = compute_proj(
    basins_retest_overlap, white_retest, f"{subject}_basin_overlap_retest"
)
proj_test.write_html(snakemake.output.proj_overlap_test, include_plotlyjs="cdn")
proj_retest.write_html(snakemake.output.proj_overlap_retest, include_plotlyjs="cdn")
