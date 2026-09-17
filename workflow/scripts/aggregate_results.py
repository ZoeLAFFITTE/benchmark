import os
import pandas as pd


def aggregate_results(input_files, output_file):
    """Aggregate subject-level TSV files into one TSV."""

    dataframes = []

    for file_path in input_files:

        df = pd.read_csv(file_path, sep="\t",)
    
        if "subject" not in df.columns:
            subject = os.path.basename(file_path).split(".")[0]
            df.insert(0, "subject", subject)
        dataframes.append(df)

    if dataframes:
        df_all = pd.concat(dataframes, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    df_all.to_csv(output_file, sep="\t", index=False)


aggregate_results(snakemake.input.cachia_results_hausdorff_tsv, snakemake.output.cachia_hausdorff_tsv)
aggregate_results(snakemake.input.cachia_results_mean_tsv, snakemake.output.cachia_mean_tsv)
aggregate_results(snakemake.input.cachia_results_p95_tsv, snakemake.output.cachia_p95_tsv)
aggregate_results(snakemake.input.trace_results_hausdorff_tsv, snakemake.output.trace_hausdorff_tsv)
aggregate_results(snakemake.input.trace_results_mean_tsv, snakemake.output.trace_mean_tsv)
aggregate_results(snakemake.input.trace_results_p95_tsv, snakemake.output.trace_p95_tsv)
aggregate_results(snakemake.input.mindboggle_results_hausdorff_tsv, snakemake.output.mindboggle_hausdorff_tsv)
aggregate_results(snakemake.input.mindboggle_results_mean_tsv, snakemake.output.mindboggle_mean_tsv)
aggregate_results(snakemake.input.mindboggle_results_p95_tsv, snakemake.output.mindboggle_p95_tsv)