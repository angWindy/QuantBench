"""Unit tests for the feature engineering layer.

The most important test is ``test_assert_no_lookahead`` — it guarantees
that adding more rows in the future does NOT change a feature value at
an earlier timestamp. If that test ever fails, the entire backtest
pipeline becomes untrustworthy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantbench.data.ingest import load_raw_csv
from quantbench.data.validate import basic_clean
from quantbench.features.indicators import (
    DEFAULT_FEATURES,
    FeatureSpec,
    assert_no_lookahead,
    compute_features,
)

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset_NQ_1min_2022_2025.csv"


@pytest.fixture(scope="module")
def ohlcv():
    if RAW_PATH.exists():
        df = basic_clean(load_raw_csv(RAW_PATH))
        # Subsample for speed in CI — features tests are CPU-heavy.
        return df.iloc[:50_000]
    rng = pd.date_range("2024-01-02 09:30", periods=10_000, freq="1min", tz="UTC")
    px = 100 + np.random.default_rng(0).normal(0, 0.1, 10_000).cumsum()
    return pd.DataFrame(
        {
            "open": px,
            "high": px + 0.05,
            "low": px - 0.05,
            "close": px,
            "volume": np.random.default_rng(1).integers(100, 1000, 10_000),
        },
        index=rng,
    )


def test_compute_features_returns_new_frame(ohlcv):
    out = compute_features(ohlcv)
    assert out is not ohlcv
    assert "sma_20" in out.columns
    assert "rsi_14" in out.columns
    # NaN at the start of warm-up is expected.
    assert out["sma_20"].iloc[:19].isna().all()
    assert out["sma_20"].iloc[20:].notna().all()


def test_default_features_have_no_lookahead(ohlcv):
    """Critical: a feature at t must not change if we append more bars."""
    feat_full = compute_features(ohlcv)
    feat_partial = compute_features(ohlcv.iloc[:5_000])
    # Values computed up to index 5_000 must match between the two frames.
    common_idx = feat_partial.index
    for col in ["sma_20", "rsi_14", "log_return_1", "atr_14", "ema_20"]:
        np.testing.assert_allclose(
            feat_full.loc[common_idx, col].to_numpy(),
            feat_partial[col].to_numpy(),
            equal_nan=True,
            err_msg=f"Feature '{col}' shows lookahead bias",
        )


def test_assert_no_lookahead_passes(ohlcv):
    feat = compute_features(ohlcv)
    assert_no_lookahead(feat, ohlcv)


def test_custom_feature_spec(ohlcv):
    spec = (FeatureSpec("custom_sma", "sma", {"length": 5}),)
    out = compute_features(ohlcv, specs=spec)
    assert "sma_5" in out.columns


def test_unknown_feature_kind_raises(ohlcv):
    bad = (FeatureSpec("bad", "not_a_real_indicator", {}),)
    with pytest.raises(ValueError, match="Unknown feature kind"):
        compute_features(ohlcv, specs=bad)


def test_non_datetime_index_rejected():
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="DatetimeIndex"):
        compute_features(df)