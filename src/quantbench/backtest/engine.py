"""Event-driven-style backtest engine, built on top of vectorbt.

vectorbt gives us a fast, well-tested portfolio simulator. We layer three
things on top:

1. **Execution lag** — entries/exits at bar *t* are filled at bar *t+lag*.
   This is the single most important realism knob: in production you
   cannot fill a signal in the same bar that produced it.

2. **Per-side fees + slippage** — applied to every fill, configurable in
   basis points. The default ``fee_bps=2.5`` + ``slippage_bps=1.0`` is
   roughly the retail equity/futures round-trip on liquid US names.

3. **Position sizing** — fixed-fractional (default 100% of equity per
   signal). Vectorbt handles the rest.

The wrapper exposes only ``run_backtest(df, signals, config) -> BacktestResult``
so callers don't depend on vectorbt internals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt


@dataclass(frozen=True)
class BacktestConfig:
    """All knobs in one place. Frozen so a config can be safely hashed."""

    initial_capital: float = 100_000.0
    fee_bps: float = 2.5          # 0.025% per side
    slippage_bps: float = 1.0     # 0.010% per side
    execution_lag_bars: int = 1   # signal at t, fill at t + lag
    position_size: float = 1.0    # fraction of equity per signal
    freq: str | None = None       # bar frequency for annualised metrics (e.g. "1min")

    def __post_init__(self) -> None:
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fees and slippage must be non-negative")
        if self.execution_lag_bars < 0:
            raise ValueError("execution_lag_bars must be non-negative")
        if not 0 < self.position_size <= 1:
            raise ValueError("position_size must be in (0, 1]")


@dataclass(frozen=True)
class BacktestResult:
    """Returned by ``run_backtest``. Wraps a vectorbt Portfolio plus config."""

    portfolio: vbt.Portfolio
    config: BacktestConfig
    signals: pd.DataFrame
    close: pd.Series


def _apply_lag(signals: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Shift entries/exits forward by ``lag`` bars. NaN entries become False."""
    if lag == 0:
        return signals
    out = signals.shift(lag)
    # fillna only when needed (avoids pandas FutureWarning about downcasting
    # on already-bool columns).
    if out.isna().any().any():
        out = out.fillna(False)
    return out.astype(bool)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a backtest with explicit fees, slippage and execution lag.

    Parameters
    ----------
    df:
        OHLCV frame. Must contain ``close``.
    signals:
        DataFrame with ``entries`` and ``exits`` boolean columns, indexed
        the same as ``df``.
    config:
        BacktestConfig. Defaults are sensible for liquid US futures/equities.
    """
    if config is None:
        config = BacktestConfig()
    if "close" not in df.columns:
        raise ValueError("df must contain a 'close' column")
    if not {"entries", "exits"}.issubset(signals.columns):
        raise ValueError("signals must have 'entries' and 'exits' columns")
    if not signals.index.equals(df.index):
        raise ValueError("signals and df must share the same index")

    lagged = _apply_lag(signals[["entries", "exits"]], config.execution_lag_bars)

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=lagged["entries"],
        exits=lagged["exits"],
        init_cash=config.initial_capital,
        size=config.position_size,
        size_type="percent",
        fees=config.fee_bps / 10_000,
        slippage=config.slippage_bps / 10_000,
        freq=config.freq,
    )
    return BacktestResult(portfolio=pf, config=config, signals=signals, close=df["close"])
