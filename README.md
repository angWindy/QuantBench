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
                       ↓
            [Paper Trader (bar-by-bar)] → [Session JSON/CSV]
                       ↓
                 [Streamlit Dashboard]
```

Each stage writes to disk so any stage can be swapped independently.

## Quickstart

```bash
# 1. activate env
conda activate quant

# 2. (optional) regenerate processed data
python scripts/prepare_data.py \
  --raw data/raw/Dataset_NQ_1min_2022_2025.csv \
  --out data/processed/NQ_1min.parquet

# 3. run a paper trading session
python scripts/run_paper_session.py \
  --data data/processed/NQ_1min.parquet \
  --strategy random \
  --params '{"seed": 42, "p_enter": 0.01, "p_exit": 0.02}' \
  --name paper_random_42 \
  --bars 20000

# 4. open the dashboard
streamlit run scripts/dashboard.py
```

Sessions are written to `reports/sessions/*.json` (+ sidecar `.equity.csv`
and `.trades.csv`); the dashboard picks them up automatically.

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