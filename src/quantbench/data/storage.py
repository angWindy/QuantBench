"""Parquet storage with consistent layout.

Files are written under ``data/processed/{symbol}_{timeframe}.parquet`` with
``pyarrow`` engine + snappy compression. The schema is enforced on read so
that downstream code can rely on column types.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_processed(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist a processed DataFrame to parquet. Returns the resolved path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    return path


def read_processed(path: str | Path) -> pd.DataFrame:
    """Load a processed parquet file. Index is restored as UTC DatetimeIndex."""
    df = pd.read_parquet(Path(path), engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df
