#!/usr/bin/env python
"""Aggregate subject-level TSV files into one TSV."""

import os
import pandas as pd


def aggregate_results(method_inputs):
    """Aggregate subject-level TSV files into one TSV."""

    dataframes = []

    for method, file_paths in method_inputs.items():
        for file_path in file_paths:
            df = pd.read_csv(file_path, sep="\t")

            if "subject" not in df.columns:
                subject = os.path.basename(file_path).split(".")[0]
                df.insert(0, "subject", subject)

            if "method" not in df.columns and method is not None:
                df.insert(0, "method", method)

            dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


method_inputs = {
    "Cachia": snakemake.input.cachia_results_tsv,
    "Trace": snakemake.input.trace_results_tsv,
    "Mindboggle": snakemake.input.mindboggle_results_tsv,
}

print(method_inputs)

intermethod_inputs = {
    None: snakemake.input.intermethod_results_tsv,
}

all_methods_df = aggregate_results(method_inputs)
intermethod_df = aggregate_results(intermethod_inputs)

all_methods_df.to_csv(snakemake.output.all_methods_tsv, sep="\t", index=False)
intermethod_df.to_csv(snakemake.output.intermethod_results_tsv, sep="\t", index=False)