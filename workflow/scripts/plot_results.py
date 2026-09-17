"""
Génère un plot final par métrique (Hausdorff, Mean, P95).

Pour chaque métrique : un point par sujet, une couleur par méthode
(Cachia / Trace / Mindboggle), sur l'ensemble des bassins communs
à tous les sujets.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METHODS = ["cachia", "trace", "mindboggle"]
METRICS = {
    "hausdorff": "Hausdorff",
    "mean": "Mean nearest-neighbor",
    "p95": "P95",
}


def load_all_data(inputs) -> pd.DataFrame:
    """Empile les 9 TSV (3 méthodes x 3 métriques) dans un seul dataframe long.

    Chaque TSV source est au format large : une ligne par sujet, une colonne
    par bassin (label). On le passe au format long (subject, label, distance)
    avant de l'empiler avec les autres.
    """
    frames = []
    for method in METHODS:
        for metric in METRICS:
            path = getattr(inputs, f"{method}_{metric}_tsv")
            wide = pd.read_csv(path, sep="\t")
            long = wide.melt(id_vars="subject", var_name="label", value_name="distance")
            long["method"] = method.capitalize()
            long["metric"] = metric
            frames.append(long)
    return pd.concat(frames, ignore_index=True)


def get_common_labels(data: pd.DataFrame) -> set:
    """Bassins présents chez tous les sujets, toutes méthodes/métriques confondues."""
    labels_by_subject = data.groupby("subject")["label"].apply(set)
    return set.intersection(*labels_by_subject)


def plot_metric(data: pd.DataFrame, metric: str, title: str, output_file: str) -> None:
    """Un plot : x = bassin, y = distance, un point par sujet, couleur par méthode."""
    subset = data.loc[data["metric"] == metric].copy()
    subset["label"] = subset["label"].astype(str)

    n_labels = subset["label"].nunique()
    n_subjects = subset["subject"].nunique()
    fig, ax = plt.subplots(figsize=(max(14, n_labels * 1.3), 7))

    sns.stripplot(
        data=subset,
        x="label",
        y="distance",
        hue="method",
        hue_order=[m.capitalize() for m in METHODS],
        dodge=True,
        jitter=0.25,
        size=3,
        alpha=0.6,
        ax=ax,
    )

    ax.set_xlabel("Basin label")
    ax.set_ylabel("Distance")
    ax.set_title(f"{title} distance per basin (n = {n_subjects} subjects)")
    ax.legend(title="Method")
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(output_file, format="svg")
    plt.close(fig)


data = load_all_data(snakemake.input)
data = data[data["label"].isin(get_common_labels(data))]

for metric, title in METRICS.items():
    plot_metric(data, metric, title, getattr(snakemake.output, f"{metric}_plot"))



