"""QuantBench: modular quantitative research and backtesting framework."""

from .backtest import BacktestConfig, BacktestResult, Backtester
from .data import MarketData, SyntheticDataGenerator
from .experiments import ExperimentConfig, run_experiment
from .features import FeatureSet, compute_features
from .metrics import PerformanceReport, evaluate_performance
from .strategies import RSIStrategy, RandomStrategy, Strategy

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "ExperimentConfig",
    "FeatureSet",
    "MarketData",
    "PerformanceReport",
    "RSIStrategy",
    "RandomStrategy",
    "Strategy",
    "SyntheticDataGenerator",
    "compute_features",
    "evaluate_performance",
    "run_experiment",
]
