"""QuantBench strategies."""
from __future__ import annotations

from quantbench.strategies.base import (
    MACrossStrategy,
    RandomStrategy,
    Strategy,
    _ensure_bool,
)
from quantbench.strategies.vwap import VWAPMeanReversionScalper

__all__ = [
    "Strategy",
    "RandomStrategy",
    "MACrossStrategy",
    "VWAPMeanReversionScalper",
    "_ensure_bool",
]