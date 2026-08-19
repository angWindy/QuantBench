"""Unit tests for the data layer (ingest, validate, storage)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantbench.data.ingest import load_raw_csv
from quantbench.data.storage import read_processed, write_processed
from quantbench.data.validate import ValidationReport, basic_clean, validate_ohlcv

# Use the real raw file if available; otherwise synthesize a fixture.
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset_NQ_1min_2022_2025.csv"


@pytest.fixture(scope="module")
def df_raw():
    if RAW_PATH.exists():
        return load_raw_csv(RAW_PATH)
    rng = pd.date_range("2024-01-02 09:30", periods=500, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": 100 + np.random.default_rng(0).normal(0, 0.1, 500).cumsum(),
            "high": 0.0,
            "low": 0.0,
            "close": 0.0,
            "volume": 1000,
            "vwap_rth": 100.0,
            "vwap_eth": 100.0,
        },
        index=rng,
    ).pipe(lambda d: d.assign(high=d["open"] + 0.05, low=d["open"] - 0.05, close=d["open"]))


def test_load_raw_csv_shape_and_index(df_raw):
    assert isinstance(df_raw.index, pd.DatetimeIndex)
    assert df_raw.index.tz is not None  # tz-aware
    assert df_raw.index.is_monotonic_increasing
    assert not df_raw.index.duplicated().any()
    assert {"open", "high", "low", "close", "volume"}.issubset(df_raw.columns)


def test_validate_ohlcv_returns_report(df_raw):
    report = validate_ohlcv(df_raw)
    assert isinstance(report, ValidationReport)
    assert report.n_rows == len(df_raw)
    assert report.n_negative_prices == 0
    assert report.n_ohlc_violations == 0


def test_validate_flags_duplicates(df_raw):
    duped = pd.concat([df_raw, df_raw.iloc[-1:]])
    report = validate_ohlcv(duped)
    assert report.n_duplicates == 1


def test_basic_clean_idempotent(df_raw):
    cleaned = basic_clean(df_raw)
    assert len(cleaned) == len(df_raw)
    # Running again should not change anything.
    assert basic_clean(cleaned).equals(cleaned)


def test_parquet_roundtrip(tmp_path: Path, df_raw):
    cleaned = basic_clean(df_raw)
    out = tmp_path / "test.parquet"
    write_processed(cleaned, out)
    assert out.exists()
    reloaded = read_processed(out)
    assert len(reloaded) == len(cleaned)
    assert reloaded.index.tz is not None
    pd.testing.assert_frame_equal(reloaded, cleaned, check_freq=False)