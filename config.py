# -*- coding: utf-8 -*-
from pathlib import Path

ROOT        = Path(__file__).parent
DATA_RAW    = ROOT   # .bgz files live in Healthy/ and Cancer/ at root
DATA_PROC   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for d in (DATA_PROC, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Fragment length filter (Cristiano et al. 2019)
FRAG_MIN = 90
FRAG_MAX = 400

# Short / long windows — exact Cristiano et al. definition
SHORT_LO = 100
SHORT_HI = 150
LONG_LO  = 151
LONG_HI  = 220

# Histogram range for 1D CNN input
HIST_MIN = 90
HIST_MAX = 400   # inclusive — 311 bp values

# File columns
COL_NAMES  = ["chrom", "start", "end", "score", "strand"]
COL_DTYPES = {
    "chrom":  "category",
    "start":  "int32",
    "end":    "int32",
    "score":  "int32",
    "strand": "category",
}
