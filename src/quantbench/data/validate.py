"""Data quality checks for OHLCV frames.

All checks are *non-destructive*: they never modify the DataFrame. Instead
they return a ``ValidationReport`` describing what they found. Downstream
code decides what to do — typically: log warnings, drop bad rows, or abort.

Why no destructive operations here:
- Lookahead bias can sneak in if we silently "fix" data using future info.
- Reproducibility: the same raw input must yield the same cleaned output.
- Auditing: you want a record of what was rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    """Summary of validation findings. Counts only — no row indices leaked."""

    n_rows: int
    n_duplicates: int = 0
    n_nulls: dict[str, int] = field(default_factory=dict)
    n_zero_volume: int = 0
    n_negative_prices: int = 0
    n_ohlc_violations: int = 0  # rows where high < max(open, close) or low > min(...)
    n_outliers: int = 0
    gap_count: int = 0
    max_gap_minutes: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            self.n_duplicates == 0
            and not any(self.n_nulls.values())
            and self.n_zero_volume == 0
            and self.n_negative_prices == 0
            and self.n_ohlc_violations == 0
            and self.n_outliers == 0
        )


def _ohlc_violations(df: pd.DataFrame) -> int:
    """Rows where OHLC relationships are violated (high < low, etc.)."""
    cond = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"]) \
        | (df["low"] > df["open"]) | (df["low"] > df["close"])
    return int(cond.fillna(False).sum())


def _outlier_returns(df: pd.DataFrame, z_threshold: float = 8.0) -> int:
    """1-minute returns beyond ``z_threshold`` standard deviations.

    Uses a rolling z-score to avoid flagging genuine regime shifts. Bars
    with non-finite values are skipped.
    """
    ret = df["close"].pct_change()
    std = ret.rolling(240, min_periods=60).std()  # ~4 hours of 1-min bars
    z = (ret - ret.rolling(240, min_periods=60).mean()) / std
    return int((z.abs() > z_threshold).fillna(False).sum())


def _detect_gaps(df: pd.DataFrame, expected_freq_minutes: int = 1) -> tuple[int, float]:
    """Count gaps larger than the expected bar frequency."""
    if len(df) < 2:
        return 0, 0.0
    diffs = df.index.to_series().diff().dropna().dt.total_seconds() / 60.0
    expected = float(expected_freq_minutes)
    gaps = diffs[diffs > expected * 1.5]  # tolerate small DST/jitter variance
    if gaps.empty:
        return 0, 0.0
    return int(len(gaps)), float(gaps.max())


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    expected_freq_minutes: int = 1,
    outlier_z: float = 8.0,
) -> ValidationReport:
    """Run the full OHLCV validation suite. Returns a report, never mutates df."""
    n_duplicates = int(df.index.duplicated().sum())
    n_nulls = {col: int(df[col].isna().sum()) for col in df.columns}

    vol = df["volume"]
    n_zero_volume = int(((vol == 0) | vol.isna()).sum())

    prices = df[["open", "high", "low", "close"]]
    n_negative_prices = int((prices < 0).fillna(False).sum().sum())
    n_ohlc_violations = _ohlc_violations(df)
    n_outliers = _outlier_returns(df, z_threshold=outlier_z)
    gap_count, max_gap_minutes = _detect_gaps(df, expected_freq_minutes)

    warnings: list[str] = []
    if gap_count > 0:
        warnings.append(
            f"{gap_count} gaps > {expected_freq_minutes}m "
            f"(max {max_gap_minutes:.0f}m) — expected for futures (overnight, weekends)"
        )

    return ValidationReport(
        n_rows=len(df),
        n_duplicates=n_duplicates,
        n_nulls=n_nulls,
        n_zero_volume=n_zero_volume,
        n_negative_prices=n_negative_prices,
        n_ohlc_violations=n_ohlc_violations,
        n_outliers=n_outliers,
        gap_count=gap_count,
        max_gap_minutes=max_gap_minutes,
        warnings=warnings,
    )


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with duplicates dropped and nulls forward-filled.

    This is the minimum cleanup the rest of the pipeline assumes. It does
    *not* fill weekend/overnight gaps — those are structural for futures and
    must be handled by the backtest engine (bar iteration respects time).
    """
    out = df[~df.index.duplicated(keep="last")].copy()
    # Forward-fill only price/volume; do NOT fill across multi-day gaps —
    # we limit the fill to at most 5 consecutive NaNs to be safe.
    cols = [c for c in OHLCV_COLUMNS if c in out.columns]
    out[cols] = out[cols].ffill(limit=5)
    return out


# Re-export for convenience
from quantbench.data.ingest import OHLCV_COLUMNS  # noqa: E402
