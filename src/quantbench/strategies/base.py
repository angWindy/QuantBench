"""Strategy protocol + signal generator contract.

A *strategy* is anything that takes a feature-augmented OHLCV frame and
returns a boolean ``entries`` / ``exits`` pair (or a ``position`` series
in {-1, 0, 1}). It must be **pure**: same input → same output, no I/O,
no randomness seeded by wall-clock time.

The contract deliberately mirrors ``vectorbt.Portfolio.from_signals`` so
the same signals can be backtested either via vectorbt (fast) or the
event-driven engine we'll build next (realistic).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Strategy(Protocol):
    """Protocol every strategy must satisfy."""

    name: str

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with columns ``entries`` and ``exits`` (bool)."""
        ...


def _ensure_bool(s: pd.Series) -> pd.Series:
    return s.astype(bool).fillna(False)


class RandomStrategy:
    """Buy/sell signals chosen uniformly at random.

    Used as a sanity baseline: a correctly-implemented backtest on a random
    strategy should produce ~0 Sharpe and ~50% win rate. If it doesn't,
    one of fees, slippage, or the lag is broken.
    """

    name = "random"

    def __init__(self, p_enter: float = 0.005, p_exit: float = 0.01, seed: int = 0):
        if not 0 < p_enter < 1:
            raise ValueError("p_enter must be in (0, 1)")
        if not 0 < p_exit < 1:
            raise ValueError("p_exit must be in (0, 1)")
        self.p_enter = p_enter
        self.p_exit = p_exit
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        # Use a stateful RNG so that calling generate_signals with a growing
        # window (as the simulator does bar-by-bar) keeps the same sequence
        # as a one-shot call on the full frame. Without this, the simulator
        # would re-seed every step and produce different signals than the
        # pre-computed reference.
        entries = self._rng.random(n) < self.p_enter
        exits = self._rng.random(n) < self.p_exit
        # No simultaneous enter+exit on the same bar.
        both = entries & exits
        if both.any():
            exits = exits & ~both
        return pd.DataFrame(
            {"entries": _ensure_bool(pd.Series(entries, index=df.index)),
             "exits":   _ensure_bool(pd.Series(exits,   index=df.index))}
        )


class MACrossStrategy:
    """Long-only when fast SMA > slow SMA, exit otherwise.

    The crossover is computed using ``shift(1)`` so the signal at bar *t*
    is based on the SMA values at bar *t-1*. This guarantees no
    intrabar lookahead: the strategy decides at the close of bar *t-1*
    and the backtest engine fills at the open of bar *t* (or close of
    *t-1* if you set execution_lag=0).
    """

    name = "ma_cross"

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast <= 0 or slow <= 0:
            raise ValueError("fast/slow must be positive")
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = df["close"].rolling(self.fast, min_periods=self.fast).mean().shift(1)
        slow = df["close"].rolling(self.slow, min_periods=self.slow).mean().shift(1)
        entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        return pd.DataFrame(
            {"entries": _ensure_bool(entries.fillna(False)),
             "exits":   _ensure_bool(exits.fillna(False))}
        )