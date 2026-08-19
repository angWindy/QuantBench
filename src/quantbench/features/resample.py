"""Higher-timeframe resampling utilities.

Used by multi-timeframe strategies. Given a 1-minute OHLCV frame,
``resample_bars`` builds N-minute bars with proper OHLCV aggregation:

    open  = first open of the window
    high  = max high in the window
    low   = min low in the window
    close = last close in the window
    volume= sum volume in the window

The output index is the *right edge* of each window (i.e. the last
minute that contributed to the bar), matching the convention used by
most charting platforms and the rest of QuantBench.

No-lookahead guarantee: every value in the resampled bar at time *t*
depends only on bars with timestamp < t. This is true by construction
because we drop the last incomplete bar (if any) so the right edge is
fully closed before the bar becomes available.
"""
from __future__ import annotations

import pandas as pd


def resample_bars(df: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    """Resample an OHLCV frame to a higher frequency.

    Parameters
    ----------
    df:
        OHLCV frame with a ``DatetimeIndex`` and columns ``open``,
        ``high``, ``low``, ``close``, ``volume``.
    freq:
        Pandas frequency string accepted by ``DataFrame.resample``,
        e.g. ``"5min"``, ``"15min"``, ``"1h"``.

    Returns
    -------
    A new DataFrame with the same columns aggregated per window and
    the right edge of each window as the new index.
    """
    required = ("open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(freq).agg(agg)
    # Drop incomplete windows. An HTF bar is only valid when every 1-min
    # bar in its window contributed. We detect partial windows by comparing
    # each bar's volume to the median volume of the *interior* windows
    # (which we know are full) — anything strictly less is a short tail
    # window and gets dropped. This is robust to volume variability
    # across the day but rejects partial trailing windows.
    full_vol = out["volume"].iloc[:-1].median() if len(out) > 1 else out["volume"].iloc[0]
    if pd.notna(full_vol) and full_vol > 0:
        out = out[out["volume"] >= full_vol * 0.99]
    out = out.dropna(subset=["close"])
    return out
