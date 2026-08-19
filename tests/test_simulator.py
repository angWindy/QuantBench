"""Unit tests for the paper trading simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbench.backtest.engine import BacktestConfig
from quantbench.simulator import BarFeed, PaperTrader
from quantbench.strategies import RandomStrategy


def _synth(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    px = 100 + np.random.default_rng(seed).normal(0, 0.1, n).cumsum()
    return pd.DataFrame(
        {"open": px, "high": px + 0.05, "low": px - 0.05, "close": px, "volume": 1000},
        index=rng,
    )


def test_paper_trader_step_returns_state():
    df = _synth(n=10)
    trader = PaperTrader(strategy=RandomStrategy(seed=1),
                         config=BacktestConfig(execution_lag_bars=1))
    state = trader.step(df.iloc[0])
    # First bar: no decision yet because window is too small.
    assert state is None
    state = trader.step(df.iloc[1])
    assert state.bar_time == df.index[1]
    assert state.equity > 0


def test_paper_trader_run_returns_result():
    df = _synth(n=500)
    trader = PaperTrader(strategy=RandomStrategy(seed=2, p_enter=0.01, p_exit=0.02),
                         config=BacktestConfig(execution_lag_bars=1))
    result = trader.run(df)
    assert result.n_bars == len(df)
    assert result.n_fills >= 0
    assert len(result.equity_curve) == len(df)


def test_paper_trader_equity_starts_at_capital():
    df = _synth(n=500)
    cfg = BacktestConfig(initial_capital=50_000.0, execution_lag_bars=1)
    trader = PaperTrader(strategy=RandomStrategy(seed=3, p_enter=0.01, p_exit=0.02),
                         config=cfg)
    result = trader.run(df)
    # First bar's equity is exactly the initial cash (no position yet).
    assert np.isclose(result.equity_curve.iloc[0], 50_000.0, atol=1e-6)


def test_paper_trader_with_no_trades_keeps_capital():
    """If p_enter is tiny enough that no entry ever fires, capital is preserved.

    RandomStrategy validates 0 < p < 1, so we use the lower bound p=1e-6. With
    only 500 bars this gives ~0 expected entries.
    """
    df = _synth(n=500)
    cfg = BacktestConfig(initial_capital=100_000.0, execution_lag_bars=1,
                         fee_bps=10.0, slippage_bps=10.0)
    trader = PaperTrader(strategy=RandomStrategy(seed=4, p_enter=1e-9, p_exit=1e-9),
                         config=cfg)
    result = trader.run(df)
    # No entries at all -> no fills -> equity stays exactly at initial.
    assert result.n_fills == 0
    assert np.isclose(result.final_equity, 100_000.0, rtol=1e-9)


def test_paper_trader_high_fees_reduce_equity():
    """Higher fee_bps must produce lower final equity in the same setup."""
    df = _synth(n=2000)

    trader_low = PaperTrader(
        RandomStrategy(seed=5, p_enter=0.01, p_exit=0.02),
        BacktestConfig(fee_bps=0.0, slippage_bps=0.0, execution_lag_bars=1),
    )
    trader_high = PaperTrader(
        RandomStrategy(seed=5, p_enter=0.01, p_exit=0.02),
        BacktestConfig(fee_bps=20.0, slippage_bps=5.0, execution_lag_bars=1),
    )
    r_low = trader_low.run(df)
    r_high = trader_high.run(df)
    assert r_high.final_equity < r_low.final_equity


def test_paper_trader_reset():
    df = _synth(n=200)
    trader = PaperTrader(strategy=RandomStrategy(seed=6, p_enter=0.01, p_exit=0.02),
                         config=BacktestConfig(execution_lag_bars=1))
    _ = trader.run(df)
    trader.reset()
    # After reset, equity_curve should be empty.
    assert len(trader._equity_curve) == 0
    assert trader._cash == 100_000.0


def test_bar_feed_iterates_rows():
    df = _synth(n=5)
    feed = BarFeed(df)
    out = [b for b in iter(feed)]
    assert len(out) == 5
    assert out[0]["close"] == df.iloc[0]["close"]


def test_bar_feed_validates_index_and_columns():
    bad = pd.DataFrame({"x": [1, 2, 3]})  # no DatetimeIndex
    with pytest.raises(ValueError, match="DatetimeIndex"):
        BarFeed(bad)
    no_close = pd.DataFrame({"open": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, freq="1min"))
    with pytest.raises(ValueError, match="close"):
        BarFeed(no_close)
