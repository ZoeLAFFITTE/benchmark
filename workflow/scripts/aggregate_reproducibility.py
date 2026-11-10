#!/usr/bin/env python
"""
Agrégation des tableaux de reproductibilité individuels par méthode / variante.
Concatène tous les sujets et hémisphères en deux fichiers globaux (standard et alpha).
"""

import sys
from pathlib import Path
import pandas as pd

# Redirection des logs via Snakemake
if hasattr(snakemake, "log") and snakemake.log:
    sys.stderr = sys.stdout = open(snakemake.log[0], "w")

def main():
    inp_files = snakemake.input.tsv_list
    out_tsv = snakemake.output.tsv
    out_alpha_tsv = snakemake.output.alpha_tsv

    dfs, dfs_alpha = [], []

    for f in inp_files:
        p = Path(f)
        if p.exists():
            dfs.append(pd.read_csv(p, sep="\t"))
        
        # Fichier alpha associé correspondant
        p_alpha = Path(f.replace(".reproducibility.tsv", ".reproducibility_alpha.tsv"))
        if p_alpha.exists():
            dfs_alpha.append(pd.read_csv(p_alpha, sep="\t"))

    # Concaténation et écriture du fichier principal global
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(out_tsv, sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(out_tsv, sep="\t", index=False)

    # Concaténation et écriture du fichier alpha global
    if dfs_alpha:
        pd.concat(dfs_alpha, ignore_index=True).to_csv(out_alpha_tsv, sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(out_alpha_tsv, sep="\t", index=False)

if __name__ == "__main__":
    main()