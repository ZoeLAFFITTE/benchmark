
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
import sys


# ============================================================
# 1. Functions
# ============================================================

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

        inds = set(np.where(test_basins == t_lab)[0])

        max_overlap = -1
        second_max_overlap = -1
        best_match = None

        for rt_lab in rt_labels:

            corr_inds = set(np.where(retest_basins == rt_lab)[0])

            # Intersection over Union
            union_size = len(inds.union(corr_inds))

            if union_size == 0:
                lab_overlap = 0
            else:
                lab_overlap = (
                    len(inds.intersection(corr_inds)) / union_size
                )

            # Keep the two highest overlaps
            if lab_overlap > max_overlap:
                second_max_overlap = max_overlap
                max_overlap = lab_overlap
                best_match = rt_lab

            elif lab_overlap > second_max_overlap:
                second_max_overlap = lab_overlap

        correspondence[int(t_lab)] = [
            int(best_match),
            float(max_overlap),
            float(second_max_overlap),
        ]

    return correspondence


def compute_feature(filtered_matches):
    """
    Create a histogram of overlap values between test and retest basins.
    """

    fig = px.histogram(
        filtered_matches,
        x="overlap",
    )

    fig.update_xaxes(
        range=[0, 1],
        tick0=0,
        dtick=0.1,
    )

    return fig


def compute_proj(basins_overlap, white_mesh, title):
    """
    Project basin overlap values onto a white mesh.
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


# ============================================================
# 2. Inputs
# ============================================================

sys.stderr = sys.stdout = open(snakemake.log[0], "w")

# Retrieve metadata from Snakemake wildcards
subject = snakemake.wildcards.sub
side = snakemake.wildcards.side

# Input paths
basins_test_path = snakemake.input.basins_test_gii
basins_retest_path = snakemake.input.basins_retest_gii

sphere_reg_test_path = snakemake.input.sphere_reg_test_gii
sphere_reg_retest_path = snakemake.input.sphere_reg_retest_gii
sphere_template_path = snakemake.input.sphere_template_gii

white_test_path = snakemake.input.white_test_gii
white_retest_path = snakemake.input.white_retest_gii


print("============================================")
print("Matching analysis")
print("============================================")

print(f"Subject: {subject}")
print(f"Side: {side}")


# ============================================================
# 3. Load meshes
# ============================================================

sphere_reg_test = sio.load_mesh(sphere_reg_test_path)
sphere_reg_retest = sio.load_mesh(sphere_reg_retest_path)
sphere_template = sio.load_mesh(sphere_template_path)

white_test = sio.load_mesh(white_test_path)
white_retest = sio.load_mesh(white_retest_path)


# ============================================================
# 4. Load textures
# ============================================================

basins_test = sio.load_texture(
    basins_test_path
).darray[0]

basins_retest = sio.load_texture(
    basins_retest_path
).darray[0]


# ============================================================
# 5. Compute
# ============================================================

# ------------------------------------------------------------
# 5.1. Nearest-neighbor interpolation onto template
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 5.2. Compute basin matching
# ------------------------------------------------------------

matches = match_basins(
    basins_test_reg,
    basins_retest_reg,
)

print("Test labels:", np.unique(basins_test_reg))
print("Retest labels:", np.unique(basins_retest_reg))

print("Number of test basins:", len(np.unique(basins_test_reg)))
print("Number of retest basins:", len(np.unique(basins_retest_reg)))


# ============================================================
# 6. Create matching DataFrame
# ============================================================

df_matches = pd.DataFrame.from_dict(
    matches,
    orient="index",
    columns=[
        "label_HCP_retest",
        "overlap",
        "second_overlap",
    ],
)

df_matches.index.name = "label_HCP_test"

# Add subject and side metadata
df_matches = (
    df_matches
    .reset_index()
    .assign(
        subject=subject,
        side=side,
    )
)

# Organize columns
df_matches = df_matches[
    [
        "subject",
        "side",
        "label_HCP_test",
        "label_HCP_retest",
        "overlap",
        "second_overlap",
    ]
]

# Ensure correct data types
df_matches["label_HCP_test"] = df_matches[
    "label_HCP_test"
].astype(int)

df_matches["label_HCP_retest"] = df_matches[
    "label_HCP_retest"
].astype(int)

df_matches["overlap"] = df_matches[
    "overlap"
].astype(float)

df_matches["second_overlap"] = df_matches[
    "second_overlap"
].astype(float)


# ------------------------------------------------------------
# 6.1. Export matching TSV
# ------------------------------------------------------------

df_matches.to_csv(
    snakemake.output.match_tsv,
    sep="\t",
    index=False,
)

print("\nMatching results:")
print(df_matches)


# ============================================================
# 7. Filter reproducible basins
# ============================================================

threshold = snakemake.params.thr_basins

df_reproducible = df_matches[
    df_matches["overlap"] >= threshold
].copy()

filtered_test_basins = df_reproducible[
    "label_HCP_test"
].to_numpy(dtype=int)

filtered_retest_basins = df_reproducible[
    "label_HCP_retest"
].to_numpy(dtype=int)

print("\nReproducibility threshold:", threshold)

print("Test basins:", filtered_test_basins)
print("Retest basins:", filtered_retest_basins)

print("Number of reproducible test basins:", len(filtered_test_basins))
print("Number of reproducible retest basins:", len(filtered_retest_basins))


# ============================================================
# 8. Compute overlap textures
# ============================================================

basins_test_overlap = np.zeros(
    len(basins_test),
    dtype=float,
)

basins_retest_overlap = np.zeros(
    len(basins_retest),
    dtype=float,
)

for _, row in df_matches.iterrows():

    label_test = int(row["label_HCP_test"])
    label_retest = int(row["label_HCP_retest"])
    overlap = float(row["overlap"])

    # Assign overlap to all vertices of the corresponding basin
    basins_test_overlap[
        basins_test == label_test
    ] = overlap

    basins_retest_overlap[
        basins_retest == label_retest
    ] = overlap


# ============================================================
# 9. Outputs
# ============================================================

# ------------------------------------------------------------
# 9.1. Export registered basin textures
# ------------------------------------------------------------

basins_test_reg_tex = stex.TextureND(
    darray=basins_test_reg
)

basins_retest_reg_tex = stex.TextureND(
    darray=basins_retest_reg
)

sio.write_texture(
    basins_test_reg_tex,
    snakemake.output.basins_test_reg_gii,
)

sio.write_texture(
    basins_retest_reg_tex,
    snakemake.output.basins_retest_reg_gii,
)


# ------------------------------------------------------------
# 9.2. Optional outputs
# ------------------------------------------------------------

# np.save(
#     snakemake.output.filtered_basins_test_npy,
#     filtered_test_basins,
# )

# np.save(
#     snakemake.output.filtered_basins_retest_npy,
#     filtered_retest_basins,
# )

# basins_test_overlap_tex = stex.TextureND(
#     darray=basins_test_overlap
# )

# basins_retest_overlap_tex = stex.TextureND(
#     darray=basins_retest_overlap
# )

# sio.write_texture(
#     basins_test_overlap_tex,
#     snakemake.output.basins_test_overlap_gii,
# )

# sio.write_texture(
#     basins_retest_overlap_tex,
#     snakemake.output.basins_retest_overlap_gii,
# )

# hist = compute_feature(df_matches)
# hist.write_html(
#     snakemake.output.hist,
#     include_plotlyjs="cdn",
# )

# proj_test = compute_proj(
#     basins_test_overlap,
#     white_test,
#     f"{subject}_{side}_basin_overlap_test",
# )

# proj_retest = compute_proj(
#     basins_retest_overlap,
#     white_retest,
#     f"{subject}_{side}_basin_overlap_retest",
# )

# proj_test.write_html(
#     snakemake.output.proj_overlap_test,
#     include_plotlyjs="cdn",
# )

# proj_retest.write_html(
#     snakemake.output.proj_overlap_retest,
#     include_plotlyjs="cdn",
# )


# ============================================================
# 10. End
# ============================================================

print("\n============================================")
print("Matching analysis completed successfully")
print(f"Subject: {subject}")
print(f"Side: {side}")
print(f"Total basins: {len(df_matches)}")
print(f"Reproducible basins: {len(df_reproducible)}")
print("============================================")
