"""Load raw OHLCV CSV files into validated pandas DataFrames.

This module is the *only* place that knows about the on-disk CSV schema
(separator, timestamp format, column names). Everything downstream expects
a standardized DataFrame with:

- DatetimeIndex in UTC, sorted, no duplicates
- Columns: open, high, low, close, volume (+ optional vwap_*)
- Floats for OHLC, integer/uint for volume

If you change the source CSV layout, update ``_read_csv`` and the schema
constants — nothing else.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Source timestamp column name in the raw CSV (kept literal to match file).
_TS_COL = "timestamp ET"

# Standardized output columns.
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
EXTRA_COLUMNS: tuple[str, ...] = ("vwap_rth", "vwap_eth")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame with parsed timestamps.

    The raw file uses ``M/D/YYYY HH:MM`` Eastern Time strings. We parse them
    into tz-aware DatetimeIndex in US/Eastern, then convert to UTC for safe
    downstream work (UTC is the safest single reference for cross-exchange
    data; converting at the boundary is cheaper than carrying ET everywhere).
    """
    df = pd.read_csv(
        path,
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
            "Vwap_RTH": "float64",
            "Vwap_ETH": "float64",
        },
    )

    # Parse ET timestamps ("12/26/2022 18:01") then localize + convert to UTC.
    ts = pd.to_datetime(df[_TS_COL], format="%m/%d/%Y %H:%M", errors="raise")
    ts = ts.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    ts = ts.dt.tz_convert("UTC")

    df = df.assign(timestamp=ts).drop(columns=[_TS_COL])
    df = df.set_index("timestamp").sort_index()

    # Normalize VWAP column names to snake_case for ergonomic access.
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    return df


def load_raw_csv(
    path: str | Path,
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Load a raw CSV file and return a standardized OHLCV DataFrame.

    Parameters
    ----------
    path:
        Path to the raw CSV file.
    symbol:
        Optional symbol tag stored in ``df.attrs["symbol"]`` so downstream
        pipelines can keep multi-symbol frames self-describing.

    Returns
    -------
    pandas.DataFrame
        Indexed by UTC timestamps, columns: open, high, low, close, volume,
        vwap_rth, vwap_eth.
    """
    path = Path(path)
    df = _read_csv(path)
    df.attrs["symbol"] = symbol or path.stem.split("_")[1] if "_" in path.stem else None
    df.attrs["source"] = str(path)
    return df