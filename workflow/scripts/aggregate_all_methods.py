#!/usr/bin/env python
"""
Agrégation globale de toutes les méthodes (Trace, Mindboggle, Zoe)
en un seul grand fichier TSV final.
"""

import sys
import traceback
from pathlib import Path
import pandas as pd

# Redirection propre des logs avec capture des erreurs
log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
log_file = open(log_path, "w") if log_path else sys.stdout
sys.stderr = sys.stdout = log_file

def main():
    try:
        inp_files = snakemake.input.tsv_list
        out_tsv = Path(snakemake.output.tsv)
        out_alpha_tsv = Path(snakemake.output.alpha_tsv)

        # S'assurer que les dossiers de sortie existent
        out_tsv.parent.mkdir(parents=True, exist_ok=True)

        dfs, dfs_alpha = [], []

        for f in inp_files:
            p = Path(f)
            if p.exists():
                dfs.append(pd.read_csv(p, sep="\t"))
            else:
                print(f"Attention: Le fichier d'entrée {p} n'existe pas.", file=sys.stderr)
            
            # Fichier alpha associé correspondant
            p_alpha = Path(f.replace("_reproducibility.tsv", "_reproducibility_alpha.tsv"))
            if p_alpha.exists():
                dfs_alpha.append(pd.read_csv(p_alpha, sep="\t"))
            else:
                print(f"Attention: Le fichier alpha {p_alpha} n'existe pas.", file=sys.stderr)

        # Concaténation et écriture du fichier principal global groupé
        if dfs:
            df_combined = pd.concat(dfs, ignore_index=True)
            df_combined.to_csv(out_tsv, sep="\t", index=False)
            print(f"Fichier principal écrit avec succès : {out_tsv} ({len(df_combined)} lignes)")
        else:
            pd.DataFrame().to_csv(out_tsv, sep="\t", index=False)
            print("Aucune donnée principale trouvée, fichier vide créé.", file=sys.stderr)

        # Concaténation et écriture du fichier alpha global groupé
        if dfs_alpha:
            df_alpha_combined = pd.concat(dfs_alpha, ignore_index=True)
            df_alpha_combined.to_csv(out_alpha_tsv, sep="\t", index=False)
            print(f"Fichier alpha écrit avec succès : {out_alpha_tsv} ({len(df_alpha_combined)} lignes)")
        else:
            pd.DataFrame().to_csv(out_alpha_tsv, sep="\t", index=False)
            print("Aucune donnée alpha trouvée, fichier vide créé.", file=sys.stderr)

    except Exception as e:
        print("Erreur critique dans le script d'agrégation globale :", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise e
    finally:
        if log_path:
            log_file.close()

if __name__ == "__main__":
    main()