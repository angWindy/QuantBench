"""Intraday microstructure features.

These features share a property that ``indicators.py`` features don't:
they reset their state at session boundaries. Volume-Weighted Average
Price (VWAP) is the canonical example — it anchors to "the start of the
trading day", not to the start of the data file.

For NQ 1-min data spanning 2022-2025, sessions are aligned to the CME
equity-index futures schedule: RTH (09:30-16:00 ET) is the primary
trading window. Overnight ETH (18:00-09:30 ET) is a separate session
for our purposes because the participant mix and liquidity profile are
different. We treat every day as a fresh session.
"""
from __future__ import annotations

import pandas as pd


def add_vwap(df: pd.DataFrame, *, session_col: str | None = None) -> pd.DataFrame:
    """Append ``vwap`` and ``vwap_z`` columns to ``df``.

    VWAP is the cumulative price*volume divided by cumulative volume
    *within the current session*. By default we use the **calendar date**
    of each bar's UTC timestamp as the session boundary, which lines up
    well with CME's overnight rollover at ~17:00 ET (22:00 UTC during
    DST) — every bar in the same UTC calendar day belongs to the same
    session.

    Parameters
    ----------
    df:
        OHLCV frame with ``DatetimeIndex``. Must contain ``high``,
        ``low``, ``close``, ``volume``.
    session_col:
        Optional pre-computed column whose values define the session id
        (e.g. trading date). If ``None``, we derive it from the index
        date in UTC.

    Returns
    -------
    A copy of ``df`` with two new columns: ``vwap`` (the running
    session VWAP) and ``vwap_z`` (signed z-score of ``close - vwap``
    over a 60-bar rolling window, useful as an over-extension filter).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    required = ("high", "low", "close", "volume")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    out = df.copy()
    if session_col is None:
        session = pd.Series(out.index.date, index=out.index)
    else:
        session = out[session_col]

    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    pv = typical * out["volume"]
    # Group by session so cumsum restarts at each new session.
    grouped = pv.groupby(session)
    cum_pv = grouped.cumsum()
    cum_vol = out["volume"].groupby(session).cumsum().replace(0, pd.NA)
    out["vwap"] = cum_pv / cum_vol
    # 60-bar rolling z-score of (close - vwap) / vol.
    dev = (out["close"] - out["vwap"])
    roll_std = dev.rolling(60, min_periods=60).std()
    out["vwap_z"] = dev / roll_std.replace(0, pd.NA)
    return out


def add_session_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """Append a ``is_session_start`` boolean column marking the first
    bar of each UTC calendar day. Useful for plotting / debugging and
    for time-stop strategies that want to exit at the end of a session.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    out = df.copy()
    dates = out.index.date
    is_start = pd.Series(dates, index=out.index).ne(
        pd.Series(dates, index=out.index).shift(1)
    ).fillna(True)
    out["is_session_start"] = is_start.astype(bool)
    return out
