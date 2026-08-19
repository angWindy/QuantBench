"""Unit tests for VWAP + HTF resample."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbench.features import (
    add_session_boundaries,
    add_vwap,
    resample_bars,
)


def _synth_ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Build a multi-day, multi-bar OHLCV frame for VWAP tests."""
    rng = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
    np.random.seed(seed)
    px = 100 + np.cumsum(np.random.normal(0, 0.05, n))
    return pd.DataFrame(
        {"open": px, "high": px + 0.05, "low": px - 0.05,
         "close": px, "volume": 1000},
        index=rng,
    )


def test_vwap_is_session_reset():
    """VWAP at the first bar of each new session must equal that bar's typical price."""
    df = _synth_ohlcv(n=600)  # ~2 calendar days given 1-min bars
    out = add_vwap(df)
    # For each new calendar day, the first bar of that day must have vwap
    # equal to its own typical price (since cumsum starts from zero).
    days = sorted({t.date() for t in df.index})
    assert len(days) >= 2, "synthetic data should span at least 2 UTC days"
    for day in days:
        mask = out.index.date == day
        first_bar = out[mask].iloc[0]
        typ = (first_bar["high"] + first_bar["low"] + first_bar["close"]) / 3.0
        assert np.isclose(first_bar["vwap"], typ, rtol=1e-9), (
            f"VWAP did not reset at session start on {day}: "
            f"{first_bar['vwap']} vs {typ}"
        )


def test_vwap_weighted_by_volume():
    """VWAP = sum(typical * volume) / sum(volume) within the session."""
    df = _synth_ohlcv(n=120)
    out = add_vwap(df)
    # Day 0 only — use is_session_start
    mask = out.index.date == out.index[0].date()
    sub = out[mask]
    typical = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    pv = typical * sub["volume"]
    expected_vwap = pv.cumsum() / sub["volume"].cumsum()
    pd.testing.assert_series_equal(
        sub["vwap"].reset_index(drop=True),
        expected_vwap.reset_index(drop=True),
        check_names=False,
    )


def test_vwap_rejects_missing_columns():
    df = pd.DataFrame({"close": [1.0]},
                       index=pd.date_range("2024-01-01", periods=1, freq="1min"))
    with pytest.raises(ValueError, match="missing required columns"):
        add_vwap(df)


def test_vwap_rejects_non_datetime_index():
    df = pd.DataFrame({
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]
    })
    with pytest.raises(ValueError, match="DatetimeIndex"):
        add_vwap(df)


def test_session_boundaries_flags_first_bar_of_each_day():
    df = _synth_ohlcv(n=600)
    out = add_session_boundaries(df)
    starts = out.index[out["is_session_start"]]
    # First bar of df is always a session start.
    assert out.iloc[0]["is_session_start"]
    # Every flagged timestamp's date must differ from the previous bar's date.
    if len(starts) > 1:
        for i in range(1, len(starts)):
            assert starts[i].date() != starts[i - 1].date()


def test_resample_bars_aggregates_correctly():
    df = _synth_ohlcv(n=300)  # 300 minutes = 20 fifteen-minute bars exactly
    out_15m = resample_bars(df, freq="15min")
    assert len(out_15m) == 20
    # First 15-minute window: open=df.iloc[0]['open'], close=df.iloc[14]['close']
    assert np.isclose(out_15m.iloc[0]["open"], df.iloc[0]["open"])
    assert np.isclose(out_15m.iloc[0]["close"], df.iloc[14]["close"])
    # High/low/volume aggregated over the 15 bars.
    assert np.isclose(out_15m.iloc[0]["high"], df.iloc[0:15]["high"].max())
    assert np.isclose(out_15m.iloc[0]["low"],  df.iloc[0:15]["low"].min())
    assert out_15m.iloc[0]["volume"] == df.iloc[0:15]["volume"].sum()


def test_resample_bars_drops_incomplete_window():
    df = _synth_ohlcv(n=305)  # 305 minutes = 20 full 15-min bars + 5 trailing
    out_15m = resample_bars(df, freq="15min")
    # Should drop the incomplete trailing window.
    assert len(out_15m) == 20


def test_resample_bars_validates_columns():
    df = pd.DataFrame({"close": [1.0, 2.0]},
                       index=pd.date_range("2024-01-01", periods=2, freq="1min"))
    with pytest.raises(ValueError, match="missing required columns"):
        resample_bars(df, freq="15min")
