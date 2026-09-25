#!/usr/bin/env python

"""Project distance metrics onto the template white surface."""

from pathlib import Path

import numpy as np
import pandas as pd
import slam.io as sio
import slam.plot as splt


METRICS = {
    "log_hausdorff": "log_hausdorff",
    "hausdorff": "hausdorff",
    "mean": "mean_nn",
    "p95": "p95_nn",
}

# -------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------
white_template = sio.load_mesh(snakemake.input.white_template_gii)
n_vertices = len(white_template.vertices)


# Load aggregated distance results
df = pd.read_csv(snakemake.input.all_methods_tsv, sep="\t")
df["log_hausdorff"] = np.log10(df["hausdorff"])
df["subject"] = df["subject"].astype(str)
df["label_test"] = df["label_test"].astype(int)
df["label_retest"] = df["label_retest"].astype(int)


# -------------------------------------------------------------------------
# Outputs
# -------------------------------------------------------------------------
outputs = {
    ("Cachia", "log_hausdorff"): snakemake.output.cachia_log_hausdorff,
    ("Cachia", "hausdorff"): snakemake.output.cachia_hausdorff,
    ("Cachia", "mean"): snakemake.output.cachia_mean,
    ("Cachia", "p95"): snakemake.output.cachia_p95,

    ("Trace", "log_hausdorff"): snakemake.output.trace_log_hausdorff,
    ("Trace", "hausdorff"): snakemake.output.trace_hausdorff,
    ("Trace", "mean"): snakemake.output.trace_mean,
    ("Trace", "p95"): snakemake.output.trace_p95,

    ("Mindboggle", "log_hausdorff"): snakemake.output.mindboggle_log_hausdorff,
    ("Mindboggle", "hausdorff"): snakemake.output.mindboggle_hausdorff,
    ("Mindboggle", "mean"): snakemake.output.mindboggle_mean,
    ("Mindboggle", "p95"): snakemake.output.mindboggle_p95,
}


# -------------------------------------------------------------------------
# Projection
# -------------------------------------------------------------------------
colorscale = [
    [0.0, "white"],
    [0.1, "blue"],
    [1.0, "red"],
]


subjects = set(df["subject"].astype(str).unique())

basins_per_sub = {
    subject: path
    for path in snakemake.input.basins_test_reg_gii
    for subject in subjects
    if subject in Path(path).name
}

for (method, metric), output_file in outputs.items():


    df_method = df.loc[df["method"] == method]

    subject_textures = []

    for subject, df_subject in df_method.groupby("subject", sort=False):

        basins_path = basins_per_sub[subject]
        basins = sio.load_texture(basins_path).darray[0]

        subject_texture = np.zeros(n_vertices, dtype=np.float32)

        for row in df_subject.itertuples(index=False):

            label = row.label_test
            distance = getattr(row, METRICS[metric])

            if pd.isna(distance):
                continue

            # Ignore la coupe médiale
            # if label < 0:
            #     continue

            subject_texture[basins == label] = distance

        subject_textures.append(subject_texture)

    # Mean across subjects
    mean_texture = np.mean(np.stack(subject_textures), axis=0)

    cmax_color = max(mean_texture)
    if metric == "log_hausdorff":
        cmax_color = np.percentile(mean_texture, 75)

    mesh_data = {
        "vertices": white_template.vertices,
        "faces": white_template.faces,
        "title": f"{method} - {metric} - Mean",
    }

    intensity_data = {
        "values": mean_texture,
        "mode": "vertex",
        "cmin": 0,
        "cmax": cmax_color,
        "colorbar": {
            "title": metric,
        },
    }

    display_settings = {
        "template": "plotly_white",
        "colorscale": colorscale,
    }

    fig = splt.plot_mesh(
        mesh_data,
        intensity_data,
        display_settings,
        caption=False,
    )

    fig.write_html(output_file, include_plotlyjs="cdn")