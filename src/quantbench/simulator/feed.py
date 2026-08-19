"""Mock data feed for paper trading.

The feed wraps a DataFrame and yields one bar (row) at a time on demand.
In a production paper-trading system the same API would wrap a websocket
or a polling endpoint — only ``next()`` matters to the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BarFeed:
    """Iterator-like wrapper over an OHLCV frame."""

    df: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("BarFeed expects a DatetimeIndex")
        if "close" not in self.df.columns:
            raise ValueError("BarFeed df must contain 'close'")

    def __iter__(self) -> "BarFeed":
        self._iter = iter(self.df.iterrows())
        return self

    def __next__(self) -> pd.Series:
        _, bar = next(self._iter)
        return bar
