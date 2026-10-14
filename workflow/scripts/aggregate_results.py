#!/usr/bin/env python
"""Aggregate subject-level TSV files into one TSV."""

import os
import pandas as pd


def aggregate_results(method_inputs, seuils=None):
    """Aggregate subject-level TSV files into one TSV.

    Si `seuils` est fourni, une colonne `seuil` est ajoutée (extraite du chemin).
    """
    dataframes = []

    for method, file_paths in method_inputs.items():
        for file_path in file_paths:
            df = pd.read_csv(file_path, sep="\t")

            if "subject" not in df.columns:
                subject = os.path.basename(file_path).split(".")[0]
                df.insert(0, "subject", subject)

            if seuils is not None and "seuil" not in df.columns:
                # .../adapt/{seuil}/{sub}.{side}...zoe_dist.tsv
                seuil_str = os.path.basename(os.path.dirname(file_path))
                if seuil_str.isdigit() and int(seuil_str) in seuils:
                    df.insert(1, "seuil", int(seuil_str))

            if "method" not in df.columns and method is not None:
                df.insert(0, "method", method)

            dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


candidate_methods = {
    "Zoe": "zoe_results_tsv",
    "Cachia": "cachia_results_tsv",
    "Trace": "trace_results_tsv",
    "Mindboggle": "mindboggle_results_tsv",
}

method_inputs = {
    method: list(getattr(snakemake.input, key))
    for method, key in candidate_methods.items()
    if hasattr(snakemake.input, key)
}

# None si la règle ne déclare pas params.seuils
seuils = getattr(snakemake.params, "seuils", None)

all_methods_df = aggregate_results(method_inputs, seuils=seuils)
all_methods_df.to_csv(snakemake.output.all_methods_tsv, sep="\t", index=False)