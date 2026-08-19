"""End-to-end CLI runner.

Usage:
    python scripts/run_backtest.py \\
        --data data/processed/NQ_1min.parquet \\
        --strategy random     \\
        --name smoke-random

    python scripts/run_backtest.py \\
        --data data/processed/NQ_1min.parquet \\
        --strategy ma_cross  \\
        --params '{"fast": 20, "slow": 100}' \\
        --name ma_cross_20_100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantbench.backtest.engine import BacktestConfig  # noqa: E402
from quantbench.experiments.runner import (  # noqa: E402
    ExperimentConfig,
    run_experiment,
    write_result,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Run a reproducible backtest experiment")
    p.add_argument("--data", required=True, type=Path, help="Path to processed parquet")
    p.add_argument("--strategy", default="ma_cross", choices=["random", "ma_cross"])
    p.add_argument("--params", default="{}", type=str, help="JSON dict of strategy params")
    p.add_argument("--name", required=True, type=str, help="Experiment name")
    p.add_argument("--fee-bps", type=float, default=2.5)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--ppy", type=float, default=252.0,
                   help="Periods-per-year for annualised metrics")
    p.add_argument("--freq", type=str, default=None,
                   help="Bar frequency string (e.g. 1min, 1D) for vectorbt metrics")
    p.add_argument("--out-dir", type=Path, default=Path("logs"))
    args = p.parse_args()

    cfg = ExperimentConfig(
        name=args.name,
        data_path=str(args.data),
        strategy=args.strategy,
        strategy_params=json.loads(args.params),
        backtest=BacktestConfig(
            initial_capital=args.capital,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            execution_lag_bars=args.lag,
            freq=args.freq,
        ),
        periods_per_year=args.ppy,
    )
    result = run_experiment(cfg)
    write_result(result, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())