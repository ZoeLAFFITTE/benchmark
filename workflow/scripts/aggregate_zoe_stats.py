import sys
import pandas as pd

sys.stderr = sys.stdout = open(snakemake.log[0], "w")

df = pd.concat(
    (pd.read_csv(f, sep="\t") for f in snakemake.input),
    ignore_index=True,
)
df.to_csv(snakemake.output[0], sep="\t", index=False)

print(f"Écrit : {snakemake.output[0]} ({len(df)} lignes)")