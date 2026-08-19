"""Ingest raw CSV → validate → write processed parquet.

Usage:
    python scripts/prepare_data.py \\
        --raw data/raw/Dataset_NQ_1min_2022_2025.csv \\
        --out data/processed/NQ_1min.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Ensure `src` is importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantbench.data.ingest import load_raw_csv  # noqa: E402
from quantbench.data.storage import write_processed  # noqa: E402
from quantbench.data.validate import ValidationReport, basic_clean, validate_ohlcv  # noqa: E402


def _log_report(report: ValidationReport) -> None:
    logger.info(f"  rows          : {report.n_rows:,}")
    logger.info(f"  duplicates    : {report.n_duplicates}")
    logger.info(f"  nulls         : {report.n_nulls}")
    logger.info(f"  zero volume   : {report.n_zero_volume}")
    logger.info(f"  negative px   : {report.n_negative_prices}")
    logger.info(f"  OHLC invalid  : {report.n_ohlc_violations}")
    logger.info(f"  outliers (z>8): {report.n_outliers}")
    logger.info(f"  gaps          : {report.gap_count} (max {report.max_gap_minutes:.0f}m)")
    for w in report.warnings:
        logger.warning(w)


def run(raw: Path, out: Path) -> None:
    logger.info(f"Loading {raw} ...")
    df = load_raw_csv(raw)
    logger.info(f"  loaded {len(df):,} rows, tz={df.index.tz}, symbol={df.attrs.get('symbol')}")

    logger.info("Validating ...")
    report = validate_ohlcv(df)
    _log_report(report)

    cleaned = basic_clean(df)
    dropped = len(df) - len(cleaned)
    if dropped:
        logger.warning(f"basic_clean dropped {dropped} rows")

    logger.info(f"Writing parquet to {out} ...")
    write_processed(cleaned, out)
    size_mb = out.stat().st_size / 1e6
    logger.success(f"Done — {len(cleaned):,} rows, {size_mb:.1f} MB")


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest raw CSV -> processed parquet")
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    run(args.raw, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())