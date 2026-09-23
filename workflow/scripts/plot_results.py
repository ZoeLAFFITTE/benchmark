#!/usr/bin/env python
"""
Generate interactive Plotly figures for overlap and distance metrics.

For each distance metric:
    x = subject
    y = distance
    one point = one basin for one subject
    color = method (Cachia / Trace / Mindboggle)

For the overlap plot:
    x = subject
    y = overlap
    one point = one reproducible basin for one subject
    color = method
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


METHODS = {
    "cachia": "Cachia",
    "trace": "Trace",
    "mindboggle": "Mindboggle",
}

METRICS = {
    "hausdorff": "Hausdorff",
    "mean": "Mean nearest-neighbor",
    "p95": "P95 nearest-neighbor",
}


def load_overlaps(match_paths):
    """
    Load and filter basin overlap results for all subjects and methods.

    Parameters
    ----------
    match_paths : list of str
        Paths to the matching TSV files.
    threshold : float
        Minimum overlap required for a basin to be considered reproducible.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the reproducible basin matches for all subjects
        and methods.
    """
    all_matches = []

    for match_path in match_paths:
        match = pd.read_csv(
            match_path,
            sep="\t",
        )

        match["overlap"] = match["overlap"].astype(float)

        subject = Path(match_path).parent.name

        match["subject"] = subject

        all_matches.append(match)

    return pd.concat(
        all_matches,
        ignore_index=True,
    )


def load_distances(snakemake):
    """
    Load and combine distance results for all methods and metrics.

    Parameters
    ----------
    snakemake : snakemake object
        Snakemake object containing the input TSV files.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame containing subject, basin, method, metric,
        and distance columns.
    """
    all_distances = []

    for method, method_name in METHODS.items():
        for metric in METRICS:
            path = getattr(
                snakemake.input,
                f"{method}_{metric}_tsv",
            )

            wide = pd.read_csv(
                path,
                sep="\t",
            )

            long = wide.melt(
                id_vars="subject",
                var_name="label",
                value_name="distance",
            )

            long["method"] = method_name
            long["metric"] = metric

            all_distances.append(long)

    return pd.concat(
        all_distances,
        ignore_index=True,
    )


def plot_overlap(df_overlaps):
    """
    Plot overlap values of reproducible basins across subjects and methods.

    Parameters
    ----------
    df_overlaps : pandas.DataFrame
        DataFrame containing the subject, test basin, retest basin, overlap,
        and method columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure showing overlap values for each basin.
        Each point represents one test/retest basin pair. Subjects are shown
        on the x-axis, methods are distinguished by color, and boxplots are
        displayed for each subject and method.
    """
    fig = px.box(
        df_overlaps,
        title= "all overlap values",
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

    fig.update_traces(
        boxmean="sd",
    )

    fig.update_yaxes(
        range=[0, 1],
        dtick=0.1,
        title="Overlap",
    )

    fig.update_xaxes(
        title="Subject",
    )

    return fig


def plot_metric(data, metric, title):
    """
    Generate an interactive boxplot for one distance metric.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format DataFrame containing subject, basin label, method,
        metric, and distance columns.
    metric : str
        Metric to plot.
    title : str
        Title of the metric.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure containing the boxplots.
    """
    subset = data.loc[
        data["metric"] == metric
    ].copy()

    subset["subject"] = subset["subject"].astype(str)
    subset["label"] = subset["label"].astype(str)

    subjects = sorted(subset["subject"].unique())
    methods = list(METHODS.values())

    fig = go.Figure()

    for i, method in enumerate(methods):
        method_data = subset.loc[
            subset["method"] == method
        ]

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
                    ["subject", "label", "distance"]
                ],
                hovertemplate=(
                    "<b>Subject:</b> %{customdata[0]}<br>"
                    "<b>Basin:</b> %{customdata[1]}<br>"
                    "<b>Distance:</b> %{customdata[2]:.3f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    n_subjects = subset["subject"].nunique()
    n_basins = len(subset)

    fig.update_layout(
        width=1800,
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=50,
        ),
        boxmode="group",
        boxgap=0.3,
        boxgroupgap=0.1,
        title=(
            f"{title} distance "
            f"(n = {n_subjects} subjects, "
            f"{n_basins} basin measurements)"
        ),
        legend=dict(
            title="Method",
        ),
    )
    fig.update_traces(
        boxmean="sd",
    )

    fig.update_yaxes(
        range=[0, 200],
        dtick=5,
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
df_overlaps = load_overlaps(
    snakemake.input.match_tsv
)

df_distances = load_distances(snakemake)


# ------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------

# Plot overlaps
overlap_fig = plot_overlap(df_overlaps)
overlap_fig.write_html(snakemake.output.hist_overlaps, include_plotlyjs="cdn")

# Plot distances
for metric, title in METRICS.items():
    output_file = getattr(
        snakemake.output,
        f"{metric}_plot",
    )

    plot_dist = plot_metric(
        data=df_distances,
        metric=metric,
        title=title,
    )
    plot_dist.write_html(output_file, include_plotlyjs="cdn")