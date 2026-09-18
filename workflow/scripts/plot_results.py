"""
Génère un plot interactif Plotly par métrique (Hausdorff, Mean, P95).

Pour chaque métrique :
    x = sujet
    y = distance
    1 point = 1 bassin pour 1 sujet
    couleur = méthode (Cachia / Trace / Mindboggle)

Tous les bassins présents dans les données sont affichés.
"""

import pandas as pd
import plotly.graph_objects as go


METHODS = ["cachia", "trace", "mindboggle"]

METRICS = {
    "hausdorff": "Hausdorff",
    "mean": "Mean nearest-neighbor",
    "p95": "P95",
}


def load_all_data(inputs):
    """Load and concatenate the 9 TSV files in long format."""

    frames = []

    for method in METHODS:
        for metric in METRICS:
            path = getattr(inputs, f"{method}_{metric}_tsv")
            wide = pd.read_csv(path, sep="\t")
            long = wide.melt(
                id_vars="subject",
                var_name="label",
                value_name="distance",
            )
            long["method"] = method.capitalize()
            long["metric"] = metric
            frames.append(long)

    return pd.concat(frames, ignore_index=True)


def plot_metric(data, metric, title, output_file):
    """Generate an interactive Plotly plot for one metric."""

    subset = data.loc[data["metric"] == metric].copy()

    # Make sure identifiers are strings
    subset["subject"] = subset["subject"].astype(str)
    subset["label"] = subset["label"].astype(str)

    fig = go.Figure()

    for method in [m.capitalize() for m in METHODS]:

        method_data = subset.loc[
            subset["method"] == method
        ]

        fig.add_trace(
            go.Scatter(
                x=method_data["subject"],
                y=method_data["distance"],
                mode="markers",
                name=method,
                customdata=method_data[
                    ["subject", "label", "distance"]
                ],
                hovertemplate=(
                    "<b>Subject:</b> %{customdata[0]}<br>"
                    "<b>Basin:</b> %{customdata[1]}<br>"
                    "<b>Distance:</b> %{customdata[2]:.3f}"
                    "<extra>%{fullData.name}</extra>"
                ),
                marker=dict(
                    size=6,
                    opacity=0.7,
                ),
            )
        )

    n_subjects = subset["subject"].nunique()
    n_basins = subset["label"].nunique()

    fig.update_layout(
        title=(
            f"{title} distance "
            f"(n = {n_subjects} subjects, {n_basins} basins)"
        ),
        xaxis=dict(
            title="Subject",
            type="category",
            categoryorder="category ascending",
        ),
        yaxis=dict(
            title="Distance",
        ),
        legend=dict(
            title="Method",
        ),
        hovermode="closest",
        template="plotly_white",
    )

    fig.write_html(
        output_file,
        include_plotlyjs="cdn",
    )

data = load_all_data(snakemake.input)

for metric, title in METRICS.items():

    output_file = getattr(
        snakemake.output,
        f"{metric}_plot",
    )

    plot_metric(
        data=data,
        metric=metric,
        title=title,
        output_file=output_file,
    )


