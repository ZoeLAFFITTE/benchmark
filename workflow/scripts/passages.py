import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def aggregate_by_basin(df):
    """1 ligne = 1 bassin d'un sujet (par hémisphère) pour un nombre de passages donné."""
    return (
        df.groupby(["test", "sub", "hemisphere", "basin", "passages"], as_index=False)
        .agg(length=("length", "mean"), n_segments=("length", "size"))
    )


def plot_segment_lengths(df):
    """Longueur moyenne des segments par bassin, selon le nombre de passages.
    1 point = 1 bassin d'un sujet. Un panneau par session, axe Y partagé.
    """
    sessions = sorted(df["test"].unique())

    stats = df.groupby(["test", "passages"])["length"].agg(
        q1=lambda x: x.quantile(0.25),
        median="median",
        q3=lambda x: x.quantile(0.75),
        mean="mean",
        n="count",
    )

    passages_values = np.sort(df["passages"].unique()).astype(float)
    gap = np.min(np.diff(passages_values)) if len(passages_values) > 1 else 1.0
    half_width = 0.35 * gap

    stat_styles = {
        "Q1": ("q1", "royalblue", "dot"),
        "Médiane": ("median", "red", "solid"),
        "Q3": ("q3", "royalblue", "dot"),
        "Moyenne": ("mean", "green", "dash"),
    }

    fig = make_subplots(
        rows=1, cols=len(sessions),
        shared_yaxes=True,
        subplot_titles=[
            f"{s} ({df[df['test'] == s]['sub'].nunique()} sujets, "
            f"{(df['test'] == s).sum()} points)"
            for s in sessions
        ],
        horizontal_spacing=0.03,
    )

    for col, session in enumerate(sessions, start=1):
        d = df[df["test"] == session]
        first = col == 1

        # Violins
        for passages, group in d.groupby("passages", sort=True):
            fig.add_trace(go.Violin(
                x=group["passages"], y=group["length"], name=str(passages),
                width=0.8, points=False, box_visible=False, meanline_visible=False,
                line_color="gray", fillcolor="rgba(128,128,128,0.25)",
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col)

        # Points : 1 point = 1 bassin d'un sujet
        fig.add_trace(go.Scattergl(
            x=d["passages"], y=d["length"], mode="markers",
            name="Bassins", legendgroup="bassins", showlegend=first,
            marker=dict(size=5, opacity=0.4, color="steelblue"),
            customdata=d[["test", "sub", "hemisphere", "basin", "n_segments"]].to_numpy(),
            hovertemplate=(
                "Session : %{customdata[0]}<br>"
                "Sujet : %{customdata[1]}<br>"
                "Hémisphère : %{customdata[2]}<br>"
                "Bassin : %{customdata[3]}<br>"
                "Segments agrégés : %{customdata[4]}<br>"
                "Passages : %{x}<br>"
                "Longueur moyenne : %{y:.4f}<extra></extra>"
            ),
        ), row=1, col=col)

        # Lignes de statistiques
        for name, (stat_col, color, dash) in stat_styles.items():
            xs, ys = [], []
            for passages, value in stats.loc[session][stat_col].items():
                xs += [passages - half_width, passages + half_width, None]
                ys += [value, value, None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=name, legendgroup=name,
                showlegend=first,
                line=dict(color=color, dash=dash, width=2),
                hovertemplate=f"{name} : %{{y:.4f}}<extra></extra>",
            ), row=1, col=col)

        fig.update_xaxes(
            tickmode="array", tickvals=passages_values,
            title_text="Nombre de passages", row=1, col=col,
        )

    fig.update_yaxes(title_text="Longueur moyenne des segments (par bassin)", row=1, col=1)
    fig.update_layout(
        title="Longueur des segments par bassin selon le nombre de passages : test vs retest",
        violinmode="overlay",
        template="plotly_white",
        height=700,
        legend_title_text="Statistiques",
    )

    return fig, stats


# --- Exécution Snakemake ---
df = pd.concat(
    (pd.read_csv(f, sep="\t") for f in snakemake.input.stats),
    ignore_index=True,
)

if df.empty:
    raise ValueError("Aucun segment trouvé dans les fichiers d'entrée.")

df_basin = aggregate_by_basin(df)

fig, stats = plot_segment_lengths(df_basin)

fig.write_html(snakemake.output.fig)
stats.to_csv(snakemake.output.summary, sep="\t")  # index = (test, passages)