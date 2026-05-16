# -*- coding: utf-8 -*-
"""
Feature extraction from a filtered fragment DataFrame.

Three groups — extracted in one pass, stored in one CSV row:

  Group 1  hist_090 … hist_400  (311 features)
           1 bp resolution normalised density.
           Direct input vector for the 1D CNN.

  Group 2  bin_090_100 … bin_210_220  (13 features)
           10 bp aggregated bins, 90–220 bp.
           Each bin name is [lo, hi] inclusive.
           Used by XGBoost / Random Forest / MLP.

  Group 3  Scalars  (5 features)
           short_long_ratio  — Cristiano et al. exact: count(100–150) / count(151–220)
           mean_length, median_length
           mono_peak_height  — fraction of fragments in 155–180 bp (near 167 bp peak)
           peak_to_flank     — mono_peak_height / fraction(130–154 bp)

  n_fragments is stored in the row for QC but is NOT a modelling feature
  (it reflects sequencing depth, not biology).
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import config


def extract_features(df: pd.DataFrame, label: str) -> dict:
    lengths = df["length"].to_numpy()
    n       = len(lengths)
    feats   = {"label": label, "n_fragments": n}

    # ── Group 1: 1 bp histogram (90–400 bp) ──────────────────────────
    # Vectorised count then single normalisation; dict comprehension to populate.
    hist = np.zeros(config.HIST_MAX - config.HIST_MIN + 1, dtype=np.float64)
    mask = (lengths >= config.HIST_MIN) & (lengths <= config.HIST_MAX)
    np.add.at(hist, lengths[mask] - config.HIST_MIN, 1)
    hist /= n
    feats.update({
        f"hist_{bp:03d}": float(hist[i])
        for i, bp in enumerate(range(config.HIST_MIN, config.HIST_MAX + 1))
    })

    # ── Group 2: 10 bp bins (90–219 bp, 10 values each, consistent width) ───
    # All 13 bins are exactly 10 bp wide: bin_090_100=[90,99], ..., bin_210_220=[210,219].
    # The SFR scalar captures 151–220 correctly via LONG_HI; bins don't need to reach 220.
    for lo in range(90, 220, 10):
        count = int(((lengths >= lo) & (lengths <= lo + 9)).sum())
        feats[f"bin_{lo:03d}_{lo+10:03d}"] = count / n

    # ── Group 3: scalars ─────────────────────────────────────────────
    short = int(((lengths >= config.SHORT_LO) & (lengths <= config.SHORT_HI)).sum())
    long  = int(((lengths >= config.LONG_LO)  & (lengths <= config.LONG_HI)).sum())
    feats["short_long_ratio"] = short / long if long else float("nan")

    feats["mean_length"]   = round(float(lengths.mean()), 4)
    feats["median_length"] = float(np.median(lengths))

    peak  = int(((lengths >= 155) & (lengths <= 180)).sum()) / n
    flank = int(((lengths >= 130) & (lengths <= 154)).sum()) / n
    feats["mono_peak_height"] = round(peak, 6)
    feats["peak_to_flank"]    = round(peak / flank, 6) if flank else float("nan")

    return feats


def feature_groups() -> dict:
    """Return column name lists for each group — used at model training time."""
    hist_cols = [f"hist_{bp:03d}" for bp in range(config.HIST_MIN, config.HIST_MAX + 1)]
    bin_cols  = [f"bin_{lo:03d}_{lo+10:03d}" for lo in range(90, 220, 10)]
    scalar_cols = [
        "short_long_ratio", "mean_length", "median_length",
        "mono_peak_height", "peak_to_flank",
    ]
    return {"hist": hist_cols, "bins": bin_cols, "scalars": scalar_cols}
