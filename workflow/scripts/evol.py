#!/usr/bin/env python
"""Évolution de mean_nn selon le seuil : 2 facettes (hémisphère gauche / droit)."""

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathlib import Path
import plotly.express as px

METRIC = "mean_nn"
HOVER_COLS = [
    "subject", "side", "label_test", "label_retest",
    "seuil", "hausdorff", "mean_nn", "p95_nn",
]
STATS = {
    # nom: (fonction, couleur, dash)
    "min":    (lambda s: s.min(),           "#9ecae1", "dot"),
    "Q1":     (lambda s: s.quantile(0.25),  "#3182bd", "dash"),
    "median": (lambda s: s.median(),        "#d62728", "solid"),
    "mean":   (lambda s: s.mean(),          "#2ca02c", "solid"),
    "Q3":     (lambda s: s.quantile(0.75),  "#3182bd", "dash"),
    "max":    (lambda s: s.max(),           "#9ecae1", "dot"),
}


def load_data(path):
    """Charge le TSV agrégé et vérifie la présence du seuil."""
    df = pd.read_csv(path, sep="\t")
    if "seuil" not in df.columns:
        raise ValueError(
            "Colonne 'seuil' absente : ajoute params.seuils dans la règle "
            "aggregate_results."
        )
    df = df.dropna(subset=[METRIC, "seuil", "side"]).copy()
    df["seuil"] = df["seuil"].astype(int)

    # Léger décalage horizontal pour limiter le recouvrement des points
    rng = np.random.default_rng(0)
    df["seuil_jitter"] = df["seuil"] + rng.uniform(-0.25, 0.25, len(df))
    return df


def compute_stats(df):
    """Statistiques de METRIC par (side, seuil)."""
    rows = []
    for (side, seuil), group in df.groupby(["side", "seuil"]):
        row = {"side": side, "seuil": seuil}
        for name, (func, _, _) in STATS.items():
            row[name] = func(group[METRIC])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_evol(df, stats):
    """2 facettes (une par hémisphère) : points + courbes de statistiques."""
    sides = sorted(df["side"].unique())
    seuils = sorted(df["seuil"].unique())

    fig = make_subplots(
        rows=1,
        cols=len(sides),
        subplot_titles=[f"Side: {side}" for side in sides],
        shared_yaxes=True,
        horizontal_spacing=0.03,
    )

    for col_idx, side in enumerate(sides, start=1):
        pts = df.loc[df["side"] == side]
        pts_hover = pts.reindex(columns=HOVER_COLS)

        # --- points transparents (un point = une paire de bassins d'un sujet) ---
        fig.add_trace(
            go.Scatter(
                x=pts["seuil_jitter"],
                y=pts[METRIC],
                mode="markers",
                name="paires de bassins",
                legendgroup="paires",
                showlegend=(col_idx == 1),
                marker={"size": 5, "color": "#7f7f7f", "opacity": 0.25},
                customdata=pts_hover.to_numpy(),
                hovertemplate=(
                    "<b>Subject:</b> %{customdata[0]}<br>"
                    "<b>Side:</b> %{customdata[1]}<br>"
                    "<b>Label test:</b> %{customdata[2]}<br>"
                    "<b>Label retest:</b> %{customdata[3]}<br>"
                    "<b>Seuil:</b> %{customdata[4]}<br>"
                    "<b>Hausdorff:</b> %{customdata[5]:.3f}<br>"
                    "<b>Mean NN:</b> %{customdata[6]:.3f}<br>"
                    "<b>P95 NN:</b> %{customdata[7]:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )

        # --- courbes de statistiques ---
        st = stats.loc[stats["side"] == side].sort_values("seuil")
        for name, (_, color, dash) in STATS.items():
            fig.add_trace(
                go.Scatter(
                    x=st["seuil"],
                    y=st[name],
                    mode="lines+markers",
                    name=name,
                    legendgroup=name,
                    showlegend=(col_idx == 1),
                    line={"color": color, "dash": dash, "width": 2},
                    marker={"size": 6},
                    hovertemplate=(
                        f"<b>{name}</b> ({side})<br>"
                        "Seuil: %{x}<br>"
                        f"{METRIC}: " "%{y:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_xaxes(
            title="Seuil",
            tickmode="array",
            tickvals=seuils,
            row=1,
            col=col_idx,
        )

    n_subjects = df["subject"].nunique()

    fig.update_layout(
        width=1400,
        height=600,
        margin={"l": 50, "r": 50, "t": 80, "b": 50},
        template="plotly_white",
        hovermode="closest",
        title=(
            f"Évolution de {METRIC} (test vs retest) selon le seuil "
            f"(n = {n_subjects} sujets, {len(df) // max(len(sides), 1)} paires par hémisphère en moyenne)"
        ),
        legend={"title": "Statistique"},
    )
    fig.update_yaxes(title=METRIC, row=1, col=1)

    return fig


if snakemake.log:
    sys.stdout = sys.stderr = open(snakemake.log[0], "w")

df = load_data(snakemake.input.all_methods_tsv)
stats = compute_stats(df)
fig = plot_evol(df, stats)
fig.write_html(snakemake.output.fig)
print(f"Figure écrite : {snakemake.output.fig}")