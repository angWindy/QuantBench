"""Unit tests for the metrics layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbench.backtest.engine import BacktestConfig, run_backtest
from quantbench.metrics.performance import (
    Metrics,
    _max_drawdown,
    compute_metrics,
    compute_metrics_from_returns,
)
from quantbench.strategies import MACrossStrategy, RandomStrategy


def _synth(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    px = 100 + np.random.default_rng(seed).normal(0, 0.1, n).cumsum()
    return pd.DataFrame(
        {"open": px, "high": px + 0.05, "low": px - 0.05, "close": px, "volume": 1000},
        index=rng,
    )


def test_max_drawdown_is_negative_or_zero():
    eq = pd.Series([100, 110, 120, 90, 95, 105, 130])
    dd = _max_drawdown(eq)
    assert dd <= 0
    # Peak 120, trough 90 -> -25%
    assert np.isclose(dd, -0.25, atol=1e-6)


def test_metrics_dataclass_roundtrip():
    m = Metrics(
        total_return=0.5, cagr=0.2, sharpe=1.5, sortino=2.0,
        max_drawdown=-0.1, calmar=2.0, win_rate=0.55,
        profit_factor=1.5, n_trades=100,
    )
    d = m.as_dict()
    assert d["total_return"] == 0.5
    assert d["n_trades"] == 100


def test_compute_metrics_from_returns_basic():
    rets = pd.Series([0.01, -0.02, 0.03, 0.01, -0.005])
    equity = (1 + rets).cumprod()
    m = compute_metrics_from_returns(rets, periods_per_year=252.0, equity=equity)
    assert isinstance(m, Metrics)
    assert m.max_drawdown <= 0
    assert m.sharpe == m.sharpe  # not NaN


def test_empty_returns_yields_zero_metrics():
    m = compute_metrics_from_returns(pd.Series(dtype=float))
    assert m.total_return == 0
    assert m.n_trades == 0


def test_metrics_for_real_backtest_random():
    df = _synth()
    sig = RandomStrategy(seed=11).generate_signals(df)
    result = run_backtest(df, sig, BacktestConfig(freq="1D"))
    m = compute_metrics(result, periods_per_year=252.0)
    # Random strategy with fees + slippage should be approximately break-even
    # — we just check it doesn't blow up.
    assert m.max_drawdown <= 0
    assert -1.0 < m.total_return < 1.0
    assert m.n_trades >= 0


def test_metrics_for_real_backtest_ma_cross():
    df = _synth(n=3000, seed=5)
    sig = MACrossStrategy(fast=10, slow=30).generate_signals(df)
    result = run_backtest(df, sig, BacktestConfig(freq="1D"))
    m = compute_metrics(result, periods_per_year=252.0)
    assert m.max_drawdown <= 0
    assert np.isfinite(m.sharpe)


def test_sharpe_is_zero_for_constant_returns():
    rets = pd.Series([0.0] * 100)
    m = compute_metrics_from_returns(rets)
    assert m.sharpe == 0.0


def test_high_fees_drag_down_return():
    df = _synth()
    sig = RandomStrategy(seed=42).generate_signals(df)

    r_low  = run_backtest(df, sig, BacktestConfig(fee_bps=0.0,  slippage_bps=0.0, freq="1D"))
    r_high = run_backtest(df, sig, BacktestConfig(fee_bps=50.0, slippage_bps=10.0, freq="1D"))

    m_low  = compute_metrics(r_low,  periods_per_year=252.0)
    m_high = compute_metrics(r_high, periods_per_year=252.0)

    assert m_high.total_return < m_low.total_return