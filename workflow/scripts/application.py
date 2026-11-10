#!/usr/bin/env python3
"""Nombre de feuilles (n_endpoints) en fonction du volume du mesh white."""
# ---------------------------------------------------------------------------
# À adapter (valeurs de ton config.yaml)
# ---------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import plotly.express as px
import slam.io as sio
import yaml

# ---------------------------------------------------------------------------
# Config (équivalent de `configfile:` dans Snakemake)
# ---------------------------------------------------------------------------
CONFIG_FILE = Path("../config/config.yaml")   # relatif au dossier d'où tu lances le script
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

DATA = Path(config["data_dir"])
DATA_TEST = config["data_test"]
ATLAS = config["atlas_dir"]
DIR = config["dir"]
SOURCES = config["sources"]
MESH = config["mesh_types"]
SIDES = config["side"]

RAW = DIR["rawdata"]
UP = DIR["pitsdata"]

NATIVE = SOURCES["native"]
REG = SOURCES["reg"]
PITS = SOURCES["pits"]

WHITE = MESH["white"]
PIAL = MESH["pial"]
APARC = MESH["aparc"]
CURV = MESH["curv"]
BASIN = MESH["basin"]
SPHEREREG = MESH["sphere.reg"]
SPHERETEMP = MESH["sphere.template"]
WHITETEMP = MESH["white.template"]

# Dans ton Snakefile, OUT est écrasé à la main : on fait pareil ici
OUT = Path("benchmark_final")
TSV = OUT / "aggregated" / "zoe_stats_complex.tsv"


# ---------------------------------------------------------------------------
# 1. Feuilles : une ligne par bassin -> somme par hémisphère
# ---------------------------------------------------------------------------
stats = pd.read_csv(TSV, sep="\t", dtype={"test": str, "sub": str, "hemisphere": str})
print("Colonnes du TSV :", list(stats.columns))   # vérifie test / sub / side / n_endpoints

leaves = (
    stats.groupby(["test", "sub", "hemisphere"], as_index=False)
         .agg(n_leaves=("n_endpoints", "sum"),
              n_basins=("n_endpoints", "size"))
)

def get_volume(row):
    mesh_path = (
        DATA / RAW / row["test"] / row["sub"] / NATIVE
        / f"{row['sub']}.{row['hemisphere']}.{WHITE}.surf.gii"
    )
    if not mesh_path.exists():
        print(f"[skip] mesh manquant : {mesh_path}")
        return float("nan")
    mesh = sio.load_mesh(str(mesh_path))
    return mesh.volume


leaves["volume_mm3"] = leaves.apply(get_volume, axis=1)
df = leaves.dropna(subset=["volume_mm3"])

df.to_csv("leaves_vs_volume.tsv", sep="\t", index=False)
print(df.describe())

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
out_dir = OUT / "test_stat"
out_dir.mkdir(parents=True, exist_ok=True)   # write_html ne crée pas le dossier

fig = px.scatter(
    df, x="volume_mm3", y="n_leaves",
    color="hemisphere", symbol="test",
    hover_data=["sub", "n_basins"],
    labels={"volume": "Volume du mesh", "n_leaves": "Nombre de feuilles"},
    title="Nombre de feuilles du squelette en fonction du volume",
)
fig.write_html(out_dir / "leaves_vs_volume.html")
fig.show()