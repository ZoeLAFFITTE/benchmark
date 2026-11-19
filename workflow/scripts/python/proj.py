#!/usr/bin/env python

"""Project distance metrics onto the template white surface (one side)."""

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

side = snakemake.wildcards.side

# -------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------
white_template = sio.load_mesh(snakemake.input.white_template_gii)
n_vertices = len(white_template.vertices)

df = pd.read_csv(snakemake.input.all_methods_tsv, sep="\t")

# Colonne method absente : un seul algorithme (Zoe) pour l'instant
if "method" not in df.columns:
    df["method"] = "zoe"

df["subject"] = df["subject"].astype(str)
df["label_test"] = df["label_test"].astype(int)
df["label_retest"] = df["label_retest"].astype(int)

# Un seul côté par job
df = df.loc[df["side"].astype(str) == str(side)]

# log10 : on évite -inf si hausdorff == 0
df["log_hausdorff"] = np.log10(df["hausdorff"].where(df["hausdorff"] > 0))

# -------------------------------------------------------------------------
# Outputs : {(méthode, métrique): fichier}, construit depuis la règle
# -------------------------------------------------------------------------
outputs = {}
for method in ("zoe",):  # ajouter "cachia", "trace", "mindboggle" plus tard
    for metric in METRICS:
        attr = f"{method.lower()}_{metric}"
        if hasattr(snakemake.output, attr):
            outputs[(method, metric)] = getattr(snakemake.output, attr)

# -------------------------------------------------------------------------
# Textures de bassins par sujet (identifiant exact : "<sub>.<side>....gii")
# -------------------------------------------------------------------------
basins_per_sub = {
    Path(path).name.split(".")[0]: path
    for path in snakemake.input.basins_test_reg_gii
}

colorscale = [
    [0.0, "white"],
    [0.1, "blue"],
    [1.0, "red"],
]

# -------------------------------------------------------------------------
# Projection
# -------------------------------------------------------------------------
for (method, metric), output_file in outputs.items():

    df_method = df.loc[df["method"].str.lower() == method.lower()]

    subject_textures = []

    for subject, df_subject in df_method.groupby("subject", sort=False):

        if subject not in basins_per_sub:
            continue

        basins = sio.load_texture(basins_per_sub[subject]).darray[0]
        subject_texture = np.zeros(n_vertices, dtype=np.float32)

        for row in df_subject.itertuples(index=False):
            distance = getattr(row, METRICS[metric])
            if pd.isna(distance):
                continue
            subject_texture[basins == row.label_test] = distance

        subject_textures.append(subject_texture)

    if not subject_textures:
        raise ValueError(
            f"Aucune donnée pour method={method}, side={side}, "
            f"metric={metric}"
        )

    mean_texture = np.mean(np.stack(subject_textures), axis=0)

    cmax_color = mean_texture.max()
    if metric == "log_hausdorff":
        cmax_color = np.percentile(mean_texture, 75)

    mesh_data = {
        "vertices": white_template.vertices,
        "faces": white_template.faces,
        "title": f"{method} - {side} - {metric} - Mean",
    }
    intensity_data = {
        "values": mean_texture,
        "mode": "vertex",
        "cmin": 0,
        "cmax": cmax_color,
        "colorbar": {"title": metric},
    }
    display_settings = {
        "template": "plotly_white",
        "colorscale": colorscale,
    }

    fig = splt.plot_mesh(
        mesh_data, intensity_data, display_settings, caption=False
    )
    fig.write_html(output_file, include_plotlyjs="cdn")