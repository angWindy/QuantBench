from __future__ import annotations

from dataclasses import dataclass

from .backtest import BacktestConfig, Backtester
from .data import SyntheticDataGenerator
from .features import compute_features
from .metrics import PerformanceReport, evaluate_performance
from .strategies import Strategy


@dataclass(frozen=True)
class ExperimentConfig:
    periods: int = 252
    data_seed: int = 7
    backtest: BacktestConfig = BacktestConfig()
    rsi_period: int = 14


def run_experiment(strategy: Strategy, config: ExperimentConfig | None = None) -> PerformanceReport:
    cfg = config or ExperimentConfig()
    generator = SyntheticDataGenerator(seed=cfg.data_seed)
    market_data = generator.generate_close(periods=cfg.periods)
    features = compute_features(market_data, rsi_period=cfg.rsi_period)
    positions = strategy.generate_positions(features)
    result = Backtester(cfg.backtest).run(market_data, positions)
    return evaluate_performance(result)
