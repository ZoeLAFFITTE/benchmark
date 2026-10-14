#!/usr/bin/env python

"""
Generate interactive Plotly figures with two facets (one for each hemisphere)
for overlap and distance metrics (filtered for Zoe method).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRICS = {
    "hausdorff": "Hausdorff",
    "mean_nn": "Mean nearest-neighbor",
}


def load_overlaps(match_paths):
    """Load overlap results for all subjects and all sides."""
    all_matches = []

    for match_path in match_paths:
        match = pd.read_csv(match_path, sep="\t")
        match["overlap"] = match["overlap"].astype(float)

        subject = Path(match_path).parent.name
        # Extraction du side depuis le nom du fichier (ex: sub.side.matching.tsv)
        filename = Path(match_path).name
        parts = filename.split(".")
        side = parts[1] if len(parts) > 1 else "unknown"

        match["subject"] = subject
        match["side"] = side

        all_matches.append(match)

    return pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()


def plot_overlap(df_overlaps):
    """Plot overlap values across subjects and sides using facets."""
    if df_overlaps.empty:
        fig = go.Figure()
        fig.update_layout(title="No overlap data available")
        return fig

    sides = sorted(df_overlaps["side"].dropna().unique())

    fig = make_subplots(
        rows=1,
        cols=len(sides),
        subplot_titles=[f"Side: {side}" for side in sides],
        shared_yaxes=True
    )

    for col_idx, side in enumerate(sides, start=1):
        side_subset = df_overlaps.loc[df_overlaps["side"] == side]
        subjects = sorted(side_subset["subject"].unique())

        fig.add_trace(
            go.Box(
                x=side_subset["subject"],
                y=side_subset["overlap"],
                name=side,
                showlegend=False,
                boxpoints="all",
                boxmean="sd",
                jitter=0.25,
                pointpos=0,
                customdata=side_subset[["subject", "side", "test_basin", "retest_basin", "overlap"]],
                hovertemplate=(
                    "<b>Subject:</b> %{customdata[0]}<br>"
                    "<b>Side:</b> %{customdata[1]}<br>"
                    "<b>Test Basin:</b> %{customdata[2]}<br>"
                    "<b>Retest Basin:</b> %{customdata[3]}<br>"
                    "<b>Overlap:</b> %{customdata[4]:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx
        )

        fig.update_xaxes(
            title="Subject",
            type="category",
            categoryorder="array",
            categoryarray=subjects,
            range=[-0.5, len(subjects) - 0.5],
            row=1,
            col=col_idx
        )

    fig.update_layout(
        width=1800,
        height=600,
        margin={"l": 50, "r": 50, "t": 80, "b": 50},
        title="All overlap values across hemispheres",
    )

    fig.update_yaxes(
        range=[0, 1],
        dtick=0.1,
        title="Overlap",
        row=1,
        col=1
    )

    return fig


def plot_metric(data, metric, title):
    """Generate an interactive boxplot with two facets (one for each side)."""
    subset = data.loc[data["metric"] == metric].copy()
    subset["subject"] = subset["subject"].astype(str)
    subset["label_test"] = subset["label_test"].astype(str)
    subset["label_retest"] = subset["label_retest"].astype(str)

    sides = sorted(subset["side"].dropna().unique())
    methods = sorted(subset["method"].dropna().unique())

    fig = make_subplots(
        rows=1,
        cols=len(sides),
        subplot_titles=[f"Side: {side}" for side in sides],
        shared_yaxes=True
    )

    for col_idx, side in enumerate(sides, start=1):
        side_subset = subset.loc[subset["side"] == side]
        subjects = sorted(side_subset["subject"].unique())

        for i, method in enumerate(methods):
            method_data = side_subset.loc[side_subset["method"] == method]
            if method_data.empty:
                continue

            fig.add_trace(
                go.Box(
                    x=method_data["subject"],
                    y=method_data["distance"],
                    name=method,
                    legendgroup=method,
                    showlegend=(col_idx == 1),
                    offsetgroup=i,
                    alignmentgroup="all",
                    boxpoints="all",
                    boxmean="sd",
                    jitter=0.25,
                    pointpos=0,
                    customdata=method_data[
                        ["subject", "side", "label_test", "label_retest", "distance"]
                    ],
                    hovertemplate=(
                        "<b>Subject:</b> %{customdata[0]}<br>"
                        "<b>Side:</b> %{customdata[1]}<br>"
                        "<b>Label test:</b> %{customdata[2]}<br>"
                        "<b>Label retest:</b> %{customdata[3]}<br>"
                        "<b>Distance:</b> %{customdata[4]:.3f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=1,
                col=col_idx
            )

        fig.update_xaxes(
            title="Subject",
            type="category",
            categoryorder="array",
            categoryarray=subjects,
            range=[-0.5, len(subjects) - 0.5],
            row=1,
            col=col_idx
        )

    n_subjects = subset["subject"].nunique()
    n_labels = len(subset)

    fig.update_layout(
        width=1800,
        height=600,
        margin={"l": 50, "r": 50, "t": 80, "b": 50},
        boxmode="group",
        boxgap=0.3,
        boxgroupgap=0.1,
        title=(
            f"{title} distance "
            f"(n = {n_subjects} subjects, "
            f"{n_labels} label measurements)"
        ),
        legend={"title": "Method"},
    )

    fig.update_yaxes(
        title="Distance",
        row=1,
        col=1
    )

    return fig


def intermethod_metrics(data, metric, title):
    """Generate an interactive boxplot for one inter-method distance metric."""

    subset = data.loc[data["metric"] == metric].copy()

    subset["subject"] = subset["subject"].astype(str)
    subset["label"] = subset["label"].astype(str)
    subset["method_1"] = subset["method_1"].astype(str)
    subset["method_2"] = subset["method_2"].astype(str)
    subset["session"] = subset["session"].astype(str)

    subjects = sorted(subset["subject"].unique())

    method_pairs = (
        subset[["method_1", "method_2"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    fig = go.Figure()

    for i, (m1, m2) in enumerate(method_pairs):

        comparison = f"{m1}-{m2}"

        data_pair = subset.loc[
            (subset["method_1"] == m1)
            & (subset["method_2"] == m2)
        ]

        fig.add_trace(
            go.Box(
                x=data_pair["subject"],
                y=data_pair["distance"],
                name=comparison,
                offsetgroup=i,
                alignmentgroup="all",
                boxpoints="all",
                boxmean="sd",
                jitter=0.25,
                pointpos=0,
                customdata=data_pair[
                    [
                        "subject",
                        "session",
                        "side",
                        "method_1",
                        "method_2",
                        "label",
                        "distance",
                    ]
                ],
                hovertemplate=(
                    "<b>Subject:</b> %{customdata[0]}<br>"
                    "<b>Session:</b> %{customdata[1]}<br>"
                    "<b>Side:</b> %{customdata[2]}<br>"
                    "<b>Method1:</b> %{customdata[3]}<br>"
                    "<b>Method2:</b> %{customdata[4]}<br>"
                    "<b>Label:</b> %{customdata[5]}<br>"
                    "<b>Distance:</b> %{customdata[6]:.3f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    n_subjects = subset["subject"].nunique()
    n_labels = len(subset)

    fig.update_layout(
        width=1800,
        margin={
            "l": 0,
            "r": 0,
            "t": 50,
            "b": 50,
        },
        boxmode="group",
        boxgap=0.3,
        boxgroupgap=0.1,
        title=(
            f"{title} distance "
            f"(n = {n_subjects} subjects, "
            f"{n_labels} label measurements)"
        ),
        legend={
            "title": "Comparison method",
        },
    )

    fig.update_yaxes(
        range=[0, max(data_pair["distance"])],
        #dtick=5,
        title="Distance",
    )

    fig.update_xaxes(
        title="Subject",
        type="category",
        categoryorder="array",
        categoryarray=subjects,
        range=[-0.5, n_subjects - 0.5],
    )

    return fig
    

#------------------------------------------------------------------
# Inputs & Parameters from Snakemake
# ------------------------------------------------------------------
#target_method = snakemake.params.get("method", "zoe").capitalize()

df_overlaps = load_overlaps(snakemake.input.match_tsv)
df_distances_raw = pd.read_csv(snakemake.input.all_methods_tsv, sep="\t")

# Filtrer uniquement pour la méthode souhaitée (ex: Zoe)
# df_distances_raw = df_distances_raw.loc[
#     df_distances_raw["method"].str.lower() == target_method.lower()
# ]

df_distances = df_distances_raw.melt(
    id_vars=[
        "subject",
        "side",
        "label_test",
        "label_retest",
        "method",
    ],
    value_vars=list(METRICS.keys()),
    var_name="metric",
    value_name="distance",
)

# ------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------

# Plot overlaps
overlap_fig = plot_overlap(df_overlaps)
overlap_fig.write_html(
    snakemake.output.hist_overlaps,
    include_plotlyjs="cdn",
)

# Plot distances
repro_output_names = {
    "hausdorff": "hist_hausdorff_plot",
    "mean_nn": "hist_mean_plot",
}

for metric, title in METRICS.items():
    output_file = getattr(
        snakemake.output,
        repro_output_names[metric],
    )

    plot_dist = plot_metric(
        data=df_distances,
        metric=metric,
        title=title,
    )

    plot_dist.write_html(
        output_file,
        include_plotlyjs="cdn",
    )

# # ------------------------------------------------------------------
# # Plot inter-method distances
# # ------------------------------------------------------------------
# df_intermethod = pd.read_csv(
#     snakemake.input.intermethod_results_tsv,
#     sep="\t",
# )

# df_intermethod = df_intermethod.rename(
#     columns={"basin_label": "label"},
# )

# df_intermethod["log(hausdorff)"] = np.log10(
#     df_intermethod["hausdorff"]
# )

# df_intermethod = df_intermethod.melt(
#     id_vars=[
#         "subject",
#         "session",
#         "side",
#         "method_1",
#         "method_2",
#         "label",
#     ],
#     value_vars=list(METRICS.keys()),
#     var_name="metric",
#     value_name="distance",
# )

# for metric, title in METRICS.items():

#     output_file = getattr(
#         snakemake.output,
#         intermethod_output_names[metric],
#     )

#     plot_concordance = intermethod_metrics(
#         data=df_intermethod,
#         metric=metric,
#         title=title,
#     )

#     plot_concordance.write_html(
#         output_file,
#         include_plotlyjs="cdn",
#     )