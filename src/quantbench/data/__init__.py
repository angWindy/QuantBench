"""QuantBench — Data ingestion, validation, and storage layer.

Public surface lives in submodules:
- `quantbench.data.ingest`: load raw OHLCV CSVs into validated DataFrames
- `quantbench.data.validate`: schema checks, gap detection, outlier flags
- `quantbench.data.storage`: parquet read/write with consistent partitioning
"""
from __future__ import annotations

__all__: list[str] = []