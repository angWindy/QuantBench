"""QuantBench feature engineering layer."""
from __future__ import annotations

from quantbench.features.intraday import add_session_boundaries, add_vwap
from quantbench.features.indicators import (
    DEFAULT_FEATURES,
    FeatureSpec,
    assert_no_lookahead,
    compute_features,
)
from quantbench.features.resample import resample_bars

__all__ = [
    "FeatureSpec",
    "DEFAULT_FEATURES",
    "compute_features",
    "assert_no_lookahead",
    "add_vwap",
    "add_session_boundaries",
    "resample_bars",
]