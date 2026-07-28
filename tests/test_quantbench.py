import unittest

from quantbench.backtest import BacktestConfig, Backtester
from quantbench.data import SyntheticDataGenerator
from quantbench.experiments import ExperimentConfig, run_experiment
from quantbench.features import compute_features
from quantbench.metrics import evaluate_performance
from quantbench.strategies import RSIStrategy, RandomStrategy


class QuantBenchTests(unittest.TestCase):
    def test_synthetic_data_is_reproducible(self) -> None:
        first = SyntheticDataGenerator(seed=11).generate_close(periods=8).close
        second = SyntheticDataGenerator(seed=11).generate_close(periods=8).close
        self.assertEqual(first, second)

    def test_random_strategy_is_reproducible(self) -> None:
        data = SyntheticDataGenerator(seed=4).generate_close(periods=15)
        features = compute_features(data)
        one = RandomStrategy(seed=2).generate_positions(features)
        two = RandomStrategy(seed=2).generate_positions(features)
        self.assertEqual(one, two)

    def test_rsi_strategy_and_backtest_metrics(self) -> None:
        data = SyntheticDataGenerator(seed=3).generate_close(periods=120)
        features = compute_features(data)
        positions = RSIStrategy().generate_positions(features)
        result = Backtester(BacktestConfig(fee_bps=1, slippage_bps=1, position_size=1)).run(data, positions)
        report = evaluate_performance(result)
        self.assertEqual(len(result.equity_curve), len(data.close))
        self.assertGreaterEqual(report.win_rate, 0.0)
        self.assertLessEqual(report.win_rate, 1.0)
        self.assertGreaterEqual(report.max_drawdown, 0.0)

    def test_experiment_runner_reproducibility(self) -> None:
        strategy = RandomStrategy(seed=101)
        cfg = ExperimentConfig(periods=80, data_seed=8)
        first = run_experiment(strategy, cfg)
        second = run_experiment(strategy, cfg)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
