"""Run a paper trading session and persist the result.

Reads the same processed parquet the backtest engine uses, runs the
PaperTrader bar-by-bar on a configurable subset, and writes a session
JSON + lightweight CSV artefacts that the Streamlit dashboard reads.

Usage:
    python scripts/run_paper_session.py \\
        --data data/processed/NQ_1min.parquet \\
        --strategy random \\
        --params '{"seed": 42, "p_enter": 0.01, "p_exit": 0.02}' \\
        --name paper_random_42 \\
        --bars 50000          # limit how many bars to replay
        --fee-bps 2.5 --slippage-bps 1.0 --lag 1 \\
        --capital 100000
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantbench.backtest.engine import BacktestConfig  # noqa: E402
from quantbench.data.storage import read_processed  # noqa: E402
from quantbench.metrics.performance import (  # noqa: E402
    Metrics,
    compute_metrics_from_returns,
)
from quantbench.simulator import PaperTrader  # noqa: E402
from quantbench.strategies import MACrossStrategy, RandomStrategy  # noqa: E402

_STRATEGIES = {
    "random": RandomStrategy,
    "ma_cross": MACrossStrategy,
}


def _build_strategy(name: str, params: dict):
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy: {name!r}. "
                         f"Available: {sorted(_STRATEGIES)}")
    return _STRATEGIES[name](**params)


def run_session(
    *,
    df: pd.DataFrame,
    strategy_name: str,
    strategy_params: dict,
    bt_config: BacktestConfig,
    name: str,
    out_dir: Path,
) -> Path:
    df = df.copy()
    strategy = _build_strategy(strategy_name, strategy_params)
    trader = PaperTrader(strategy=strategy, config=bt_config)
    logger.info(f"=== Paper session '{name}' ===")
    logger.info(f"  strategy      : {strategy_name}({strategy_params})")
    logger.info(f"  bars          : {len(df):,}")
    logger.info(f"  execution_lag : {bt_config.execution_lag_bars}")
    logger.info(f"  fees/slippage : {bt_config.fee_bps}/{bt_config.slippage_bps} bps")

    result = trader.run(df)

    metrics = _metrics_from_equity(result.equity_curve, periods_per_year=252.0, trades=result.trades)

    logger.info(f"  final_equity  : {result.final_equity:,.2f}")
    logger.info(f"  total_return  : {metrics.total_return:.2%}")
    logger.info(f"  sharpe        : {metrics.sharpe:.3f}")
    logger.info(f"  max_drawdown  : {metrics.max_drawdown:.2%}")
    logger.info(f"  n_fills       : {result.n_fills}")
    logger.info(f"  n_bars        : {result.n_bars}")

    payload = {
        "name": name,
        "started_at": datetime.now(UTC).isoformat(),
        "n_bars": result.n_bars,
        "n_fills": result.n_fills,
        "n_trades": int(metrics.n_trades),
        "final_equity": result.final_equity,
        "config": {
            "strategy": strategy_name,
            "strategy_params": strategy_params,
            "initial_capital": bt_config.initial_capital,
            "fee_bps": bt_config.fee_bps,
            "slippage_bps": bt_config.slippage_bps,
            "execution_lag_bars": bt_config.execution_lag_bars,
            "position_size": bt_config.position_size,
        },
        "metrics": metrics.as_dict(),
        "data_summary": {
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "rows": len(df),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{name}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"  wrote {path}")

    # Also write CSVs for the dashboard (equity_curve + trades).
    equity_csv = path.with_suffix(".equity.csv")
    trades_csv = path.with_suffix(".trades.csv")
    result.equity_curve.to_frame("equity").to_csv(equity_csv)
    result.trades.to_csv(trades_csv, index=False)
    logger.info(f"  wrote {equity_csv.name}, {trades_csv.name}")
    return path


def _metrics_from_equity(
    equity: pd.Series,
    *,
    periods_per_year: float,
    trades: pd.DataFrame,
) -> Metrics:
    returns = equity.pct_change().dropna()
    return compute_metrics_from_returns(
        returns,
        periods_per_year=periods_per_year,
        equity=equity,
        trades=trades,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Run a paper trading session")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--strategy", default="random", choices=["random", "ma_cross"])
    p.add_argument("--params", default="{}", type=str)
    p.add_argument("--name", required=True, type=str)
    p.add_argument("--bars", type=int, default=20_000,
                   help="How many bars to replay (most recent N)")
    p.add_argument("--fee-bps", type=float, default=2.5)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--ppy", type=float, default=252.0)
    p.add_argument("--out-dir", type=Path, default=Path("reports/sessions"))
    args = p.parse_args()

    df = read_processed(args.data)
    df = df.tail(args.bars)  # last N bars -> realistic paper-trading window

    cfg = BacktestConfig(
        initial_capital=args.capital,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        execution_lag_bars=args.lag,
        freq=None,
    )

    run_session(
        df=df,
        strategy_name=args.strategy,
        strategy_params=json.loads(args.params),
        bt_config=cfg,
        name=args.name,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())