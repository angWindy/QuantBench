"""Unit tests for the backtest engine.

The headline tests are ``test_lag_matters`` and ``test_fees_reduce_returns``
— they verify that the three realism knobs (lag, fees, slippage) actually
have the effect you'd expect. Without these, a backtest engine is just
a curve-fitter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbench.backtest.engine import BacktestConfig, run_backtest
from quantbench.strategies import RandomStrategy


def _synth(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    px = 100 + np.random.default_rng(seed).normal(0, 0.1, n).cumsum()
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


def _stats(result) -> float:
    return float(result.portfolio.total_return())


def test_run_backtest_returns_result():
    df = _synth()
    sig = RandomStrategy(seed=1).generate_signals(df)
    result = run_backtest(df, sig)
    assert result.portfolio is not None
    assert result.config.initial_capital == 100_000.0


def test_lag_matters():
    """Same signals must produce different returns under different lags.

    Otherwise the lag knob is cosmetic.
    """
    df = _synth()
    sig = RandomStrategy(seed=2).generate_signals(df)

    r_lag0 = run_backtest(df, sig, BacktestConfig(execution_lag_bars=0, freq="1D"))
    r_lag1 = run_backtest(df, sig, BacktestConfig(execution_lag_bars=1, freq="1D"))

    # Returns need not have a strict inequality (random data + different
    # fill timing), but they MUST differ in at least one of return / sharpe.
    assert not np.isclose(_stats(r_lag0), _stats(r_lag1)) or \
           not np.isclose(float(r_lag0.portfolio.sharpe_ratio()),
                          float(r_lag1.portfolio.sharpe_ratio()))


def test_fees_reduce_returns():
    """Higher fees must reduce realised return."""
    df = _synth()
    sig = RandomStrategy(seed=3).generate_signals(df)

    r_low  = run_backtest(df, sig, BacktestConfig(fee_bps=0.0,    slippage_bps=0.0, freq="1D"))
    r_high = run_backtest(df, sig, BacktestConfig(fee_bps=20.0,   slippage_bps=5.0, freq="1D"))

    assert _stats(r_high) < _stats(r_low)


def test_config_validation():
    with pytest.raises(ValueError):
        BacktestConfig(fee_bps=-1)
    with pytest.raises(ValueError):
        BacktestConfig(execution_lag_bars=-1)
    with pytest.raises(ValueError):
        BacktestConfig(position_size=0)
    with pytest.raises(ValueError):
        BacktestConfig(position_size=1.5)


def test_missing_columns_raise():
    df = _synth().drop(columns=["close"])
    sig = RandomStrategy(seed=0).generate_signals(_synth())
    with pytest.raises(ValueError, match="close"):
        run_backtest(df, sig)


def test_signal_index_mismatch_raises():
    df = _synth()
    sig = RandomStrategy(seed=0).generate_signals(df)
    sig.index = sig.index.shift(1)  # break alignment
    with pytest.raises(ValueError, match="index"):
        run_backtest(df, sig)


def test_apply_lag_idempotent():
    from quantbench.backtest.engine import _apply_lag
    df = _synth(100)
    sig = pd.DataFrame({"entries": [True] * 100, "exits": [False] * 100}, index=df.index)
    lagged = _apply_lag(sig, 0)
    assert lagged["entries"].iloc[0]  # lag=0 returns input unchanged
    lagged2 = _apply_lag(sig, 1)
    assert not lagged2["entries"].iloc[0]  # first bar cannot have a signal (no history)
    assert lagged2["entries"].iloc[1]      # entry shifted from 0 -> 1