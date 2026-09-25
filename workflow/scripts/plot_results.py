#!/usr/bin/env python

"""
Generate interactive Plotly figures for overlap and distance metrics.

For each distance metric:
    x = subject
    y = distance
    one point = one label for one subject
    color = method (Cachia / Trace / Mindboggle)

For the overlap plot:
    x = subject
    y = overlap
    one point = one basin for one subject
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


METRICS = {
    "log(hausdorff)": "Hausdorff",
    "hausdorff": "Hausdorff",
    "mean_nn": "Mean nearest-neighbor",
    "p95_nn": "P95 nearest-neighbor",
}


def load_overlaps(match_paths):
    """Load overlap results for all subjects."""

    all_matches = []

    for match_path in match_paths:
        match = pd.read_csv(match_path, sep="\t")
        match["overlap"] = match["overlap"].astype(float)

        subject = Path(match_path).parent.name
        match["subject"] = subject

        all_matches.append(match)

    return pd.concat(all_matches, ignore_index=True)


def load_distances(path): 
    """ 
    Load aggregated test-retest distance results. Expected columns:
        subject
        side
        label_test
        label_retest
        method
        hausdorff
        mean_nn
        p95_nn
    
    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame.
    """

    df = pd.read_csv(path, sep="\t")
    long = df.melt(
        id_vars=[ "subject", "side", "label_test", "label_retest", "method"],
        value_vars=list(METRICS.keys()),
        var_name="metric",
        value_name="distance",
    )
    return long


def plot_overlap(df_overlaps):
    """Plot overlap values across subjects."""

    fig = px.box(
        df_overlaps,
        title="All overlap values",
        x="subject",
        y="overlap",
        color="subject",
        points="all",
        hover_data=[
            "subject",
            "test_basin",
            "retest_basin",
            "overlap",
        ],
        category_orders={
            "subject": sorted(df_overlaps["subject"].unique()),
        },
    )

    fig.update_traces(boxmean="sd")

    fig.update_yaxes(
        range=[0, 1],
        dtick=0.1,
        title="Overlap",
    )

    fig.update_xaxes(title="Subject")

    return fig


def plot_metric(data, metric, title):
    """Generate an interactive boxplot for one distance metric."""

    subset = data.loc[data["metric"] == metric].copy()
    subset["subject"] = subset["subject"].astype(str)
    subset["label_test"] = subset["label_test"].astype(str)
    subset["label_retest"] = subset["label_retest"].astype(str)

    subjects = sorted(subset["subject"].unique())
    methods = sorted(subset["method"].dropna().unique())

    fig = go.Figure()

    for i, method in enumerate(methods):

        method_data = subset.loc[subset["method"] == method]

        fig.add_trace(
            go.Box(
                x=method_data["subject"],
                y=method_data["distance"],
                name=method,
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
            "title": "Method",
        },
    )

    fig.update_traces(boxmean="sd")

    fig.update_yaxes(
        range=[0, max(method_data["distance"])],
        #dtick=5,
        title="Distance",
    )

    fig.update_xaxes(
        title="Subject",
        type="category",
        categoryorder="array",
        categoryarray=subjects,
        range=[
            -0.5,
            n_subjects - 0.5,
        ],
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
    

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
df_overlaps = load_overlaps(snakemake.input.match_tsv)

df_distances_raw = pd.read_csv(
    snakemake.input.all_methods_tsv,
    sep="\t",
)

df_distances_raw["log(hausdorff)"] = np.log10(
    df_distances_raw["hausdorff"]
)

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
    "log(hausdorff)": "repro_log_hausdorff_plot",
    "hausdorff": "repro_hausdorff_plot",
    "mean_nn": "repro_mean_plot",
    "p95_nn": "repro_p95_plot",
}

intermethod_output_names = {
    "log(hausdorff)": "concordance_log_hausdorff_plot",
    "hausdorff": "concordance_hausdorff_plot",
    "mean_nn": "concordance_mean_plot",
    "p95_nn": "concordance_p95_plot",
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

# ------------------------------------------------------------------
# Plot inter-method distances
# ------------------------------------------------------------------
df_intermethod = pd.read_csv(
    snakemake.input.intermethod_results_tsv,
    sep="\t",
)

df_intermethod = df_intermethod.rename(
    columns={"basin_label": "label"},
)

df_intermethod["log(hausdorff)"] = np.log10(
    df_intermethod["hausdorff"]
)

df_intermethod = df_intermethod.melt(
    id_vars=[
        "subject",
        "session",
        "side",
        "method_1",
        "method_2",
        "label",
    ],
    value_vars=list(METRICS.keys()),
    var_name="metric",
    value_name="distance",
)

for metric, title in METRICS.items():

    output_file = getattr(
        snakemake.output,
        intermethod_output_names[metric],
    )

    plot_concordance = intermethod_metrics(
        data=df_intermethod,
        metric=metric,
        title=title,
    )

    plot_concordance.write_html(
        output_file,
        include_plotlyjs="cdn",
    )