# -*- coding: utf-8 -*-
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import config

df = pd.read_csv(ROOT / "data/processed/features_checkpoint.csv")
print(f"Raw rows: {len(df)}")

df = df.drop_duplicates(subset=["filename"], keep="last")
df = df[~df["status"].str.startswith("ERROR", na=False)]
print(f"After dedup + error removal: {len(df)}")

on_disk = {p.name for p in config.DATA_RAW.rglob("*.frag.tsv.bgz")}
df = df[df["filename"].isin(on_disk)]
print(f"After removing deleted files: {len(df)}")
print(f"  Cancer : {(df['label']=='cancer').sum()}")
print(f"  Healthy: {(df['label']=='healthy').sum()}")

df.to_csv(ROOT / "data/processed/features_checkpoint.csv", index=False)
df.to_csv(ROOT / "data/processed/features_final.csv", index=False)
print("Saved clean features_checkpoint.csv and features_final.csv")

cancer_sfr  = df[df["label"] == "cancer"]["short_long_ratio"].astype(float)
healthy_sfr = df[df["label"] == "healthy"]["short_long_ratio"].astype(float)
print(f"\nSFR cancer  mean={cancer_sfr.mean():.3f}  std={cancer_sfr.std():.3f}")
print(f"SFR healthy mean={healthy_sfr.mean():.3f}  std={healthy_sfr.std():.3f}")
