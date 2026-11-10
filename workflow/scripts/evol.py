
#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Métriques analysées (distances : plus petit = mieux).
# p95_nn = 95e centile des distances au plus proche voisin (HD95).
METRICS = ["mean_nn", "p95_nn"]

# Identité d'un bassin : unique par construction (ids propres à chaque sujet).
KEY = ["subject", "side", "label_test", "label_retest"]

TOL = 1e-9  # |Δ| <= TOL : bassin considéré comme inchangé

POINT_COLORS = {"dist": "#6baed6", "len": "#fdae6b"}
CURVE_COLORS = {"dist": "#08306b", "len": "#7f2704"}
SIDE_SYMBOLS = ["circle", "diamond", "square", "cross"]

STATS = {
    "Q1": lambda s: s.quantile(0.25),
    "median": lambda s: s.median(),
    "mean": lambda s: s.mean(),
    "Q3": lambda s: s.quantile(0.75),
}

DASHES = {
    "Q1": "dash",
    "median": "solid",
    "mean": "dashdot",
    "Q3": "dash",
    "improved": "solid",
    "worse": "dashdot",
    "unchanged": "dot",
}

WIDTHS = {
    "Q1": 1.5,
    "median": 3,
    "mean": 3,
    "Q3": 1.5,
    "improved": 2,
    "worse": 2,
    "unchanged": 2,
}


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, sep="\t")

    required = set(KEY) | {"seuil_dist", "seuil_len"} | set(METRICS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: colonnes manquantes : {sorted(missing)}")

    # Éviter les différences numériques dues aux flottants.
    for col in ("seuil_dist", "seuil_len"):
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    for col in METRICS + ["label_test", "label_retest"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["seuil_dist", "seuil_len", "side", "subject"]
    )

    # Retirer les bassins non appariés.
    df = df[
        (df["label_test"] >= 0)
        & (df["label_retest"] >= 0)
    ].copy()

    subset = KEY + ["seuil_dist", "seuil_len"]
    dup = df.duplicated(subset).sum()
    if dup:
        raise ValueError(f"{path}: {dup} doublons sur {subset}")

    return df


def split_curves(df):
    """Extraire les courbes 1D de la grille both.

    dist : seuil_len == 0, en fonction de seuil_dist.
    len  : seuil_dist == 0, en fonction de seuil_len.

    Un seuil égal à zéro désactive le critère correspondant.
    """
    d = df[df["seuil_len"] == 0.0].copy()
    d["seuil"] = d["seuil_dist"]

    l = df[df["seuil_dist"] == 0.0].copy()
    l["seuil"] = l["seuil_len"]

    return {"dist": d, "len": l}


# ----------------------------------------------------------------------------
# Appariement bassin par bassin (référence : seuil 0)
# ----------------------------------------------------------------------------
def make_paired(df, method):
    """Calculer Δ = métrique(seuil) - métrique(seuil=0).

    Ne conserver que les bassins présents à tous les seuils pour
    les deux métriques.
    """
    piv = df.pivot(index=KEY, columns="seuil", values=METRICS)

    for metric in METRICS:
        if 0.0 not in piv[metric].columns:
            raise ValueError(f"{method}: pas de seuil=0 dans les données")

    n_all = len(piv)
    piv = piv.dropna()

    print(
        f"[{method}] bassins : {n_all} -> {len(piv)} "
        "présents à tous les seuils"
    )

    if piv.empty:
        raise ValueError(
            f"{method}: aucun bassin présent à tous les seuils"
        )

    parts = []

    for metric in METRICS:
        wide = piv[metric]
        delta = wide.sub(wide[0.0], axis=0)

        d = wide.reset_index().melt(
            id_vars=KEY,
            var_name="seuil",
            value_name="value",
        )
        dd = delta.reset_index().melt(
            id_vars=KEY,
            var_name="seuil",
            value_name="delta",
        )

        d["delta"] = dd["delta"].to_numpy()
        d["metric"] = metric
        parts.append(d)

    out = pd.concat(parts, ignore_index=True)
    out["method"] = method

    return out


def make_paired_grid(df):
    """Calculer Δ = métrique(sd, sl) - métrique(0, 0).

    Conserver les bassins présents à toutes les combinaisons de seuils.
    """
    piv = df.pivot(
        index=KEY,
        columns=["seuil_dist", "seuil_len"],
        values=METRICS,
    )

    for metric in METRICS:
        if (0.0, 0.0) not in piv[metric].columns:
            raise ValueError(
                "both: pas de combinaison (0, 0) dans les données"
            )

    n_all = len(piv)
    piv = piv.dropna()

    print(
        f"[both] bassins : {n_all} -> {len(piv)} présents à toutes "
        f"les combinaisons ({piv.shape[1] // len(METRICS)} par métrique)"
    )

    if piv.empty:
        raise ValueError(
            "both: aucun bassin présent à toutes les combinaisons"
        )

    return {
        metric: piv[metric].sub(
            piv[metric][(0.0, 0.0)],
            axis=0,
        )
        for metric in METRICS
    }


# ----------------------------------------------------------------------------
# Statistiques des courbes (hémisphères regroupés)
# ----------------------------------------------------------------------------
def compute_stats(df):
    """Calculer les statistiques des deltas et les fractions de bassins.

    Amélioré   : Δ < -TOL
    Dégradé   : Δ > TOL
    Inchangé  : |Δ| <= TOL
    """
    rows = []

    for seuil, group in df.groupby("seuil", observed=True):
        d = group["delta"].dropna()

        rows.append({
            "seuil": seuil,
            "n": len(d),
            **{name: func(d) for name, func in STATS.items()},
            "frac_improved": (d < -TOL).mean(),
            "frac_worse": (d > TOL).mean(),
            "frac_unchanged": (d.abs() <= TOL).mean(),
        })

    return pd.DataFrame(rows).sort_values("seuil")


# ----------------------------------------------------------------------------
# Tracé
# ----------------------------------------------------------------------------
def add_points(fig, sub, metric, method, side, symbol, row, show):
    """Un point représente un bassin à un seuil donné."""
    fig.add_trace(
        go.Scattergl(
            x=sub["seuil"],
            y=sub["delta"],
            mode="markers",
            name=f"{method} — bassins ({side})",
            legendgroup=f"{method}-pts-{side}",
            showlegend=show,
            marker={
                "size": 6,
                "color": POINT_COLORS[method],
                "symbol": symbol,
                "opacity": 0.5,
            },
            customdata=sub[
                ["subject", "side", "label_test", "label_retest", "value"]
            ].to_numpy(),
            hovertemplate=(
                f"Méthode : {method}<br>"
                "Sujet : %{customdata[0]}<br>"
                "Hémisphère : %{customdata[1]}<br>"
                "Bassin : test %{customdata[2]} / "
                "retest %{customdata[3]}<br>"
                "Seuil : %{x}<br>"
                f"Δ {metric} : " "%{y:.4f}<br>"
                f"{metric} (brut) : " "%{customdata[4]:.4f}"
                "<extra></extra>"
            ),
        ),
        row=row,
        col=1,
    )


def add_line(
    fig, x, y, n, label, stat, method, row, show, fmt=".4f"
):
    """Ajouter une courbe statistique."""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=f"{method} — {label}",
            legendgroup=f"{method}-{stat}",
            showlegend=show,
            line={
                "color": CURVE_COLORS[method],
                "dash": DASHES[stat],
                "width": WIDTHS[stat],
            },
            marker={
                "size": 5,
                "color": CURVE_COLORS[method],
            },
            customdata=n,
            hovertemplate=(
                f"Méthode : {method}<br>"
                f"{label}<br>"
                "Seuil : %{x}<br>"
                f"Valeur : %{{y:{fmt}}}<br>"
                "Nombre de bassins : %{customdata}"
                "<extra></extra>"
            ),
        ),
        row=row,
        col=1,
    )


def add_heatmap(fig, delta, k, row):
    """Heatmap de la médiane de Δ sur la grille des deux seuils."""
    med = delta.median().unstack("seuil_dist")
    imp = (delta < -TOL).mean().unstack("seuil_dist")

    xs = [f"{v:.2f}" for v in med.columns]
    ys = [f"{v:.2f}" for v in med.index]

    fig.add_trace(
        go.Heatmap(
            z=med.to_numpy(),
            x=xs,
            y=ys,
            coloraxis=f"coloraxis{k}",
            customdata=np.dstack([
                imp.to_numpy(),
                np.full(med.shape, len(delta)),
            ]),
            texttemplate="%{z:.3f}",
            hovertemplate=(
                "seuil_dist : %{x}<br>"
                "seuil_len : %{y}<br>"
                "Médiane Δ : %{z:.4f}<br>"
                "Fraction améliorés : %{customdata[0]:.2f}<br>"
                "Nombre de bassins : %{customdata[1]}"
                "<extra></extra>"
            ),
        ),
        row=row,
        col=1,
    )


def plot_evolution(df, grids):
    sides = sorted(df["side"].dropna().unique())

    symbols = {
        side: SIDE_SYMBOLS[k % len(SIDE_SYMBOLS)]
        for k, side in enumerate(sides)
    }

    # 3 lignes par métrique : deltas, fractions, heatmap.
    n_rows = 3 * len(METRICS)
    titles = []

    for metric in METRICS:
        titles += [
            f"{metric} — Δ par bassin",
            f"{metric} — fractions améliorés / dégradés / inchangés",
            f"{metric} — heatmap both (médiane de Δ)",
        ]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.05,
    )

    for i, metric in enumerate(METRICS):
        r_val = 3 * i + 1
        r_frac = 3 * i + 2
        r_heat = 3 * i + 3

        for method in POINT_COLORS:
            sub_all = df[
                (df["method"] == method)
                & (df["metric"] == metric)
            ]

            if sub_all.empty:
                print(f"Aucun bassin pour {method} / {metric}")
                continue

            # Légende affichée une seule fois, pour la première métrique.
            show = i == 0

            # Ligne 1 : deltas individuels par bassin.
            for side in sides:
                sub = sub_all[sub_all["side"] == side]

                if sub.empty:
                    continue

                add_points(
                    fig,
                    sub,
                    metric,
                    method,
                    side,
                    symbols[side],
                    r_val,
                    show,
                )

            # Statistiques regroupant les hémisphères.
            stats = compute_stats(sub_all)
            x = stats["seuil"]
            n = stats["n"].to_numpy()

            # Ligne 1 : Q1, médiane, moyenne et Q3.
            for name in STATS:
                add_line(
                    fig,
                    x,
                    stats[name],
                    n,
                    f"Δ {metric} — {name}",
                    name,
                    method,
                    r_val,
                    show,
                )

            # Ligne 2 : fractions de bassins améliorés, dégradés
            # et inchangés.
            add_line(
                fig, x, stats["frac_improved"], n,
                f"Fraction améliorés ({metric})",
                "improved", method, r_frac, False, fmt=".3f",
            )

            add_line(
                fig, x, stats["frac_worse"], n,
                f"Fraction dégradés ({metric})",
                "worse", method, r_frac, False, fmt=".3f",
            )

            add_line(
                fig, x, stats["frac_unchanged"], n,
                f"Fraction inchangés ({metric})",
                "unchanged", method, r_frac, False, fmt=".3f",
            )

        fig.add_hline(
            y=0,
            line={"color": "grey", "width": 1},
            row=r_val,
            col=1,
        )

        # Heatmap de la grille complète.
        add_heatmap(fig, grids[metric], i + 1, r_heat)

        fig.update_xaxes(
            title_text="Seuil (dist : seuil_dist | len : seuil_len)",
            row=r_val,
            col=1,
        )
        fig.update_xaxes(
            title_text="Seuil (dist : seuil_dist | len : seuil_len)",
            row=r_frac,
            col=1,
        )
        fig.update_xaxes(
            title_text="seuil_dist",
            type="category",
            row=r_heat,
            col=1,
        )

        fig.update_yaxes(
            title_text=f"Δ {metric} (seuil − seuil 0)",
            row=r_val,
            col=1,
        )
        fig.update_yaxes(
            title_text=(
                "Fraction — plein : améliorés, tirets-points : "
                "dégradés, pointillés : inchangés"
            ),
            range=[0, 1],
            row=r_frac,
            col=1,
        )
        fig.update_yaxes(
            title_text="seuil_len",
            type="category",
            row=r_heat,
            col=1,
        )

    # Une échelle de couleur par métrique, centrée sur zéro.
    for i, metric in enumerate(METRICS):
        zmax = (
            float(np.nanmax(np.abs(grids[metric].median().to_numpy())))
            or 1.0
        )

        dom = fig.get_subplot(3 * i + 3, 1).yaxis.domain

        fig.update_layout({
            f"coloraxis{i + 1}": {
                "colorscale": "RdBu_r",
                "cmin": -zmax,
                "cmax": zmax,
                "cmid": 0,
                "colorbar": {
                    "title": f"Médiane Δ {metric}",
                    "len": dom[1] - dom[0],
                    "y": (dom[0] + dom[1]) / 2,
                    "yanchor": "middle",
                    "x": 1.02,
                },
            }
        })

    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=420 * n_rows + 120,
        title=(
            "Δ par bassin (1 point = 1 bassin test/retest, 1 hémisphère, "
            "1 sujet ; Δ = seuil − seuil 0 ; couleur : méthode, symbole : "
            "hémisphère) — courbes, fractions et heatmaps : "
            "hémisphères regroupés"
        ),
        hovermode="closest",
        legend_title_text="Méthode — statistique",
        margin={"l": 80, "r": 110, "t": 100, "b": 70},
    )

    return fig


# ----------------------------------------------------------------------------
# Exécution
# ----------------------------------------------------------------------------
if snakemake.log:
    sys.stdout = sys.stderr = open(snakemake.log[0], "w")

df_both = load_data(snakemake.input.both_tsv)

curves = split_curves(df_both)

df = pd.concat(
    [
        make_paired(sub, method)
        for method, sub in curves.items()
    ],
    ignore_index=True,
)

grids = make_paired_grid(df_both)

fig = plot_evolution(df, grids)
fig.write_html(snakemake.output.fig)

print(f"Figure écrite : {snakemake.output.fig}")
