"""Unit tests for the strategy layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbench.strategies import MACrossStrategy, RandomStrategy


def _synth(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    px = 100 + np.random.default_rng(seed).normal(0, 0.5, n).cumsum()
    return pd.DataFrame(
        {
            "open": px,
            "high": px + 0.05,
            "low": px - 0.05,
            "close": px,
            "volume": 1000,
        },
        index=rng,
    )


def test_random_strategy_returns_bool_columns():
    df = _synth()
    strat = RandomStrategy(seed=42)
    sig = strat.generate_signals(df)
    assert set(sig.columns) == {"entries", "exits"}
    assert sig["entries"].dtype == bool
    assert sig["exits"].dtype == bool
    assert len(sig) == len(df)
    assert sig.index.equals(df.index)
    # No simultaneous enter+exit on the same bar.
    assert not (sig["entries"] & sig["exits"]).any()


def test_random_strategy_is_deterministic_with_seed():
    df = _synth()
    a = RandomStrategy(seed=7).generate_signals(df)
    b = RandomStrategy(seed=7).generate_signals(df)
    assert a["entries"].equals(b["entries"])
    assert a["exits"].equals(b["exits"])


def test_random_strategy_validates_params():
    with pytest.raises(ValueError):
        RandomStrategy(p_enter=0)
    with pytest.raises(ValueError):
        RandomStrategy(p_exit=1.5)


def test_ma_cross_signals_have_no_lookahead():
    """Adding bars in the future must NOT change earlier MA-cross signals."""
    df = _synth(n=2000)
    strat = MACrossStrategy(fast=10, slow=50)
    sig_partial = strat.generate_signals(df.iloc[:1000])
    sig_full = strat.generate_signals(df)
    common = sig_partial.index
    assert sig_full.loc[common, "entries"].equals(sig_partial["entries"])
    assert sig_full.loc[common, "exits"].equals(sig_partial["exits"])


def test_ma_cross_validates_params():
    with pytest.raises(ValueError):
        MACrossStrategy(fast=50, slow=20)
    with pytest.raises(ValueError):
        MACrossStrategy(fast=0, slow=20)


def test_ma_cross_emits_some_signals_on_trending_data():
    # Sine wave: should produce multiple entry/exit pairs.
    n = 5000
    rng = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    px = 100 + np.sin(np.linspace(0, 30 * np.pi, n)) * 10
    df = pd.DataFrame({"close": px}, index=rng)
    sig = MACrossStrategy(fast=10, slow=50).generate_signals(df)
    assert sig["entries"].sum() > 0
    assert sig["exits"].sum() > 0