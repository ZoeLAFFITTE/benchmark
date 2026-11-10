#!/usr/bin/env python
"""Aggregate subject-level TSV files into one TSV."""

import os
import re
import sys

import pandas as pd


BOTH_RE = re.compile(r"^d(?P<sdist>[0-9.]+)_l(?P<slen>[0-9.]+)$")


def _check(value, allowed, file_path, label):
    """Vérifie que value (float) appartient à la grille attendue."""
    if allowed is not None and not any(abs(value - float(a)) < 1e-9 for a in allowed):
        raise ValueError(
            f"{file_path}: {label}={value} n'est pas un seuil attendu ({sorted(allowed)})"
        )
    return value


def extract_seuils(file_path, seuils_dist=None, seuils_len=None):
    """Retourne {colonne: valeur} déduit du chemin du fichier."""
    folder = os.path.basename(os.path.dirname(file_path))
    norm = file_path.replace(os.sep, "/")

    if "/adapt/both/" in norm:
        m = BOTH_RE.match(folder)
        if not m:
            raise ValueError(f"{file_path}: dossier '{folder}' ne correspond pas à d<x>_l<y>")
        return {
            "seuil_dist": _check(float(m["sdist"]), seuils_dist, file_path, "seuil_dist"),
            "seuil_len":  _check(float(m["slen"]),  seuils_len,  file_path, "seuil_len"),
        }

    if "/adapt/dist/" in norm:
        return {"seuil_dist": _check(float(folder), seuils_dist, file_path, "seuil_dist")}

    if "/adapt/len/" in norm:
        return {"seuil_len": _check(float(folder), seuils_len, file_path, "seuil_len")}

    return {}  # autres méthodes (trace, mindboggle...) : pas de seuil


def aggregate_results(method_inputs, seuils_dist=None, seuils_len=None):
    """Aggregate subject-level TSV files into one TSV.

    Les colonnes `seuil_dist` et/ou `seuil_len` sont extraites du chemin
    de chaque fichier et validées contre les grilles fournies.
    """
    dataframes = []

    for method, file_paths in method_inputs.items():
        for file_path in file_paths:
            df = pd.read_csv(file_path, sep="\t")

            if "subject" not in df.columns:
                subject = os.path.basename(file_path).split(".")[0]
                df.insert(0, "subject", subject)

            for col, val in extract_seuils(file_path, seuils_dist, seuils_len).items():
                if col not in df.columns:
                    df.insert(1, col, val)

            if "method" not in df.columns and method is not None:
                df.insert(0, "method", method)

            dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


sys.stderr = sys.stdout = open(snakemake.log[0], "w")

candidate_methods = {
    "Zoe": "zoe_results_tsv",
    "Trace": "trace_results_tsv",
    "Mindboggle": "mindboggle_results_tsv",
}

method_inputs = {
    method: list(getattr(snakemake.input, key))
    for method, key in candidate_methods.items()
    if hasattr(snakemake.input, key)
}

# None si la règle ne déclare pas le paramètre
seuils_dist = getattr(snakemake.params, "seuils_dist", None)
seuils_len  = getattr(snakemake.params, "seuils_len", None)

all_methods_df = aggregate_results(
    method_inputs, seuils_dist=seuils_dist, seuils_len=seuils_len
)
all_methods_df.to_csv(snakemake.output.all_methods_tsv, sep="\t", index=False)

print(f"Écrit : {snakemake.output.all_methods_tsv} ({len(all_methods_df)} lignes)")
for col in ("seuil_dist", "seuil_len"):
    if col in all_methods_df.columns:
        print(f"{col} trouvés : {sorted(all_methods_df[col].dropna().unique())}")