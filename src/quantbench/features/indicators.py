"""Feature engineering for OHLCV bars.

The cardinal rule: every feature at row *t* must depend only on data with
timestamp <= t. Anything else is a lookahead leak and silently inflates
backtest performance. To make this rule enforceable, all transformations
here operate on pandas/numpy primitives only — no groupby on future rows,
no shift with negative offsets, no resampling that crosses t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative description of a feature set.

    Each indicator below uses its own ``append``/``prefix``/``length``
    convention from ``pandas_ta_classic``. Keep the spec data-only so it
    can be serialised to YAML/JSON and replayed exactly.
    """

    name: str
    kind: str  # one of: sma, ema, rsi, macd, bbands, atr, log_return, realized_vol
    params: dict


DEFAULT_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("sma_20", "sma", {"length": 20}),
    FeatureSpec("sma_50", "sma", {"length": 50}),
    FeatureSpec("sma_200", "sma", {"length": 200}),
    FeatureSpec("ema_20", "ema", {"length": 20}),
    FeatureSpec("rsi_14", "rsi", {"length": 14}),
    FeatureSpec("macd", "macd", {"fast": 12, "slow": 26, "signal": 9}),
    FeatureSpec("bbands_20", "bbands", {"length": 20, "std": 2.0}),
    FeatureSpec("atr_14", "atr", {"length": 14}),
    FeatureSpec("log_return_1", "log_return", {"periods": 1}),
    FeatureSpec("log_return_5", "log_return", {"periods": 5}),
    FeatureSpec("realized_vol_60", "realized_vol", {"window": 60}),
)


def _realized_vol(close: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=window).std()


def _add_one(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Append a single indicator. Returns a *new* DataFrame (no in-place edits)."""
    out = df
    if spec.kind == "sma":
        out[f"sma_{spec.params['length']}"] = ta.sma(df["close"], length=spec.params["length"])
    elif spec.kind == "ema":
        out[f"ema_{spec.params['length']}"] = ta.ema(df["close"], length=spec.params["length"])
    elif spec.kind == "rsi":
        out[f"rsi_{spec.params['length']}"] = ta.rsi(df["close"], length=spec.params["length"])
    elif spec.kind == "macd":
        macd = ta.macd(df["close"], fast=spec.params["fast"], slow=spec.params["slow"], signal=spec.params["signal"])
        # macd returns a DataFrame with columns MACD_12_26_9, MACDh_..., MACDs_...
        out = out.join(macd)
    elif spec.kind == "bbands":
        bb = ta.bbands(df["close"], length=spec.params["length"], std=spec.params["std"])
        out = out.join(bb)
    elif spec.kind == "atr":
        out[f"atr_{spec.params['length']}"] = ta.atr(
            df["high"], df["low"], df["close"], length=spec.params["length"]
        )
    elif spec.kind == "log_return":
        out[f"log_return_{spec.params['periods']}"] = np.log(
            df["close"] / df["close"].shift(spec.params["periods"])
        )
    elif spec.kind == "realized_vol":
        out[f"realized_vol_{spec.params['window']}"] = _realized_vol(df["close"], spec.params["window"])
    else:
        raise ValueError(f"Unknown feature kind: {spec.kind}")
    return out


def compute_features(
    df: pd.DataFrame,
    specs: tuple[FeatureSpec, ...] = DEFAULT_FEATURES,
) -> pd.DataFrame:
    """Compute the requested features and return a new DataFrame.

    The returned frame has all original columns plus the feature columns.
    Bars where an indicator is not yet defined (warm-up window) carry NaN
    — this is *correct*. The backtest engine is responsible for skipping
    or filling them as appropriate.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("compute_features expects a DatetimeIndex")
    out = df.copy()
    for spec in specs:
        out = _add_one(out, spec)
    return out


def assert_no_lookahead(
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    sample_cols: tuple[str, ...] = ("sma_20", "rsi_14", "log_return_1"),
) -> None:
    """Sanity-check that features use only past data.

    Compares the feature at index *i* computed on the *full* frame vs.
    computed on a frame truncated at *i*. They MUST match — otherwise the
    feature is using future info. Raises ``AssertionError`` on mismatch.
    """
    for col in sample_cols:
        if col not in df_feat.columns:
            continue
        full = df_feat[col]
        truncated = compute_features(df_raw, specs=DEFAULT_FEATURES)[col]
        # Only check the middle 80% (skip warm-up NaN regions at both ends).
        start = len(full) // 10
        end = -len(full) // 10 or None
        np.testing.assert_allclose(
            full.iloc[start:end].to_numpy(),
            truncated.iloc[start:end].to_numpy(),
            equal_nan=True,
            err_msg=f"Feature '{col}' shows lookahead bias",
        )