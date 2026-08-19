# QuantBench

Modular Python framework for quantitative trading research and backtesting.
Built for a single trader iterating from idea → backtest → report.

## Environment

- **Conda env**: `quant`
- **Python**: 3.12

```bash
conda activate quant
pip install -r requirements.txt
```

## Layout

```
QuantBench/
├── data/
│   ├── raw/          # immutable source data (gitignored)
│   ├── processed/    # cleaned/normalized parquet
│   └── external/     # third-party reference data
├── src/quantbench/   # library code
│   ├── data/         # ingest, validation, storage
│   ├── features/     # indicator & feature pipelines (no lookahead)
│   ├── strategies/   # signal generators
│   ├── backtest/     # event-driven engine, fees, slippage
│   ├── metrics/      # Sharpe, Sortino, drawdown, etc.
│   └── experiments/  # reproducible run configs
├── configs/          # YAML strategy & pipeline configs
├── notebooks/        # exploratory research
├── reports/          # backtest tear sheets & equity curves
├── tests/            # unit tests (pytest)
├── scripts/          # CLI entry points (data prep, run_backtest)
├── logs/             # runtime logs (gitignored)
├── requirements.txt
└── .env.example
```

## Pipeline

```
[Data Ingestion] → [Storage (Parquet)] → [Feature Engineering]
       → [Strategy / Model] → [Backtest Engine] → [Report]
```

Each stage writes to disk so any stage can be swapped independently.

## Backtest realism checklist

- Fees + slippage in basis points
- Execution lag: signal at bar *t* fills at bar *t + EXECUTION_LAG_BARS*
- No look-ahead: features only use data available at decision time
- Walk-forward / purged K-Fold validation for any ML model
- Out-of-sample period held out from optimization

## Run tests

```bash
pytest -q
```