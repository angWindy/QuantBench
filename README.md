# QuantBench

QuantBench is a modular Python framework for quantitative trading research and backtesting.

## Architecture

- **Market data**: `quantbench.data` provides `SyntheticDataGenerator` and `MarketData`.
- **Feature engineering**: `quantbench.features` computes returns and RSI via `compute_features`.
- **Strategies and signals**: `quantbench.strategies` includes a `Strategy` protocol, plus `RandomStrategy` and `RSIStrategy`.
- **Backtesting engine**: `quantbench.backtest` applies position sizing, fees, and slippage.
- **Performance evaluation**: `quantbench.metrics` reports PnL, Sharpe, drawdown, and win rate.
- **Reproducible experiments**: `quantbench.experiments` provides `ExperimentConfig` and `run_experiment`.

## Run tests

```bash
python -m unittest discover -s tests
```