"""Reproducible experiment runner.

An ``ExperimentConfig`` is the single source of truth for a backtest run:
- which dataset
- which strategy + params
- which backtest knobs (fees, slippage, lag)
- which features to compute
- which metrics to report

Given the same config + the same raw data, ``run_experiment`` must produce
the same metrics. No wall-clock time, no environment lookups inside the
hot path — those are passed in explicitly.

Results are written to ``logs/<experiment_name>_<timestamp>.json`` so you
can compare runs over time without bloating git (logs/ is gitignored).
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from quantbench.backtest.engine import BacktestConfig, run_backtest
from quantbench.data.storage import read_processed
from quantbench.data.validate import basic_clean
from quantbench.features.indicators import DEFAULT_FEATURES, FeatureSpec, compute_features
from quantbench.metrics.performance import Metrics, compute_metrics
from quantbench.strategies import MACrossStrategy, RandomStrategy, Strategy


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment specification. Serializable to JSON/YAML."""

    name: str
    data_path: str
    strategy: str = "ma_cross"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    feature_specs: tuple[FeatureSpec, ...] = field(default_factory=lambda: DEFAULT_FEATURES)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    periods_per_year: float = 252.0  # override in caller for intraday data
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    metrics: Metrics
    started_at: str
    finished_at: str
    runtime_seconds: float
    n_bars: int


# --- Strategy registry -------------------------------------------------------

_STRATEGIES: dict[str, type] = {
    "random": RandomStrategy,
    "ma_cross": MACrossStrategy,
}


def get_strategy(cfg: ExperimentConfig) -> Strategy:
    if cfg.strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy: {cfg.strategy!r}. "
                         f"Available: {sorted(_STRATEGIES)}")
    cls = _STRATEGIES[cfg.strategy]
    return cls(**cfg.strategy_params)


# --- Runner ------------------------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, FeatureSpec):
        return {"name": obj.name, "kind": obj.kind, "params": obj.params}
    if isinstance(obj, BacktestConfig):
        return {
            "initial_capital": obj.initial_capital,
            "fee_bps": obj.fee_bps,
            "slippage_bps": obj.slippage_bps,
            "execution_lag_bars": obj.execution_lag_bars,
            "position_size": obj.position_size,
            "freq": obj.freq,
        }
    if isinstance(obj, Metrics):
        return obj.as_dict()
    if isinstance(obj, ExperimentConfig):
        # Manually serialise to avoid dataclasses.asdict recursion.
        return {
            "name": obj.name,
            "data_path": obj.data_path,
            "strategy": obj.strategy,
            "strategy_params": obj.strategy_params,
            "feature_specs": [_to_json_safe(s) for s in obj.feature_specs],
            "backtest": _to_json_safe(obj.backtest),
            "periods_per_year": obj.periods_per_year,
            "tags": obj.tags,
        }
    if isinstance(obj, tuple):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    return obj


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Load data, compute features, generate signals, run backtest, compute metrics."""
    started = datetime.now(UTC)
    logger.info(f"=== Experiment '{cfg.name}' ===")
    logger.info(f"  strategy      : {cfg.strategy}({cfg.strategy_params})")
    logger.info(f"  data          : {cfg.data_path}")
    logger.info(f"  execution_lag : {cfg.backtest.execution_lag_bars} bars")
    logger.info(f"  fees/slippage : {cfg.backtest.fee_bps}/{cfg.backtest.slippage_bps} bps")

    df = read_processed(cfg.data_path)
    df = basic_clean(df)
    df_feat = compute_features(df, specs=cfg.feature_specs)

    strategy = get_strategy(cfg)
    signals = strategy.generate_signals(df_feat)

    result = run_backtest(df_feat, signals, cfg.backtest)
    metrics = compute_metrics(result, periods_per_year=cfg.periods_per_year)

    finished = datetime.now(UTC)
    elapsed = (finished - started).total_seconds()

    logger.info(f"  total_return  : {metrics.total_return:.2%}")
    logger.info(f"  sharpe        : {metrics.sharpe:.3f}")
    logger.info(f"  sortino       : {metrics.sortino:.3f}")
    logger.info(f"  max_drawdown  : {metrics.max_drawdown:.2%}")
    logger.info(f"  calmar        : {metrics.calmar:.3f}")
    logger.info(f"  win_rate      : {metrics.win_rate:.2%}")
    logger.info(f"  n_trades      : {metrics.n_trades}")
    logger.info(f"  runtime       : {elapsed:.2f}s")

    return ExperimentResult(
        config=cfg,
        metrics=metrics,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        runtime_seconds=elapsed,
        n_bars=len(df_feat),
    )


def write_result(result: ExperimentResult, logs_dir: str | Path = "logs") -> Path:
    """Persist the experiment result as JSON. Returns the path written."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir / f"{result.config.name}_{ts}.json"
    payload = {
        "platform": platform.platform(),
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "runtime_seconds": result.runtime_seconds,
        "n_bars": result.n_bars,
        "config": _to_json_safe(result.config),
        "metrics": _to_json_safe(result.metrics),
    }
    path.write_text(json.dumps(payload, indent=2, default=_to_json_safe))
    logger.info(f"  wrote {path}")
    return path
