# -*- coding: utf-8 -*-
"""Load and filter a single FinaleDB .frag.tsv.bgz file."""

import gzip
from pathlib import Path
from typing import Optional

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_fragment_file(
    path: Path | str,
    chrom: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stream a .frag.tsv.bgz file into a DataFrame and add a 'length' column.

    chrom    : load only this chromosome (e.g. 'chr1') — dev shortcut.
    max_rows : row cap — dev shortcut.
    """
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(
            fh,
            sep="\t",
            header=None,
            names=config.COL_NAMES,
            dtype=config.COL_DTYPES,
            comment="#",
            nrows=max_rows,
        )
    df["length"] = (df["end"] - df["start"]).astype("int32")
    if chrom is not None:
        df = df[df["chrom"] == chrom].reset_index(drop=True)
    return df


def filter_fragments(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only fragments in [FRAG_MIN, FRAG_MAX]."""
    mask = df["length"].between(config.FRAG_MIN, config.FRAG_MAX)
    return df[mask].reset_index(drop=True)


def load_and_filter(
    path: Path | str,
    chrom: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load + filter in one call."""
    return filter_fragments(load_fragment_file(path, chrom=chrom, max_rows=max_rows))
