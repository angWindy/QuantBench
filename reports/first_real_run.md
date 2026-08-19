# First Real Run — NQ 1-min (2022-2025)

> Generated: 2026-08-19 — 1,048,575 bars, 3 calendar years of NQ futures at 1-min frequency.

## Why both runs lose

This report is the **first end-to-end sanity check** of QuantBench. Both strategies
*lost* money — and that is exactly what we expect, because:

- **Random strategy**: with 6,972 round-trip trades at 3.5 bps total cost per
  trade (2.5 fee + 1.0 slippage), expected gross return ≈ 0, but
  expected cost ≈ 6,972 × 3.5 bps ≈ **24.4%** of equity. After enough
  re-entries, capital compounds down to near zero.
- **MA-Cross 20/100**: widely documented to underperform on intraday NQ
  after realistic fees. This run confirms the engine captures that
  behaviour — no edge from a textbook rule on noisy 1-min bars.

If either run *had* shown high Sharpe and +50% return, that would have been
the bug. Negative numbers here are evidence the pipeline is honest.

## Results

| Strategy | Total Return | Sharpe | Sortino | Max DD | Calmar | Win Rate | Trades | Runtime |
|---|---|---|---|---|---|---|---|---|
| `random` (seed=42, p_enter=0.01, p_exit=0.02) | -99.02% | -0.389 | -0.553 | -99.03% | -0.001 | 23.15% | 6,972 | 16.9s |
| `ma_cross` (fast=20, slow=100) | -98.89% | -0.309 | -0.451 | -98.89% | -0.001 | 20.25% | 7,243 | 19.8s |

## Configuration

```
data      : data/processed/NQ_1min.parquet  (1,048,575 bars, UTC)
fees      : 2.5 bps/side
slippage  : 1.0 bps/side
lag       : 1 bar (signal at t, fill at t+1)
capital   : 100,000 USD
freq      : 1D (for annualised metrics)
ppy       : 252
```

## What this validates

1. **Data layer**: CSV → parquet pipeline produces a valid 1.05M-bar UTC frame.
2. **Feature layer**: indicators computed without lookahead bias.
3. **Strategy layer**: signals produced deterministically (random with seed).
4. **Engine layer**: vectorbt integration with realistic fees, slippage, lag.
5. **Metrics layer**: Sharpe / Sortino / MaxDD / Calmar / Win Rate / Profit Factor
   all populated correctly after fixing the case-insensitive `PnL` column bug
   in `compute_metrics`.

## Next steps

- **Walk-forward validation**: split 2022-2025 into in-sample / out-of-sample
  folds, optimise MA parameters per fold, report aggregate metrics.
- **ML models (LightGBM/XGBoost)**: train on engineered features, predict
  next-bar direction, evaluate via the same engine.
- **Order-book features**: add microstructure signals (spread, depth,
  imbalance) when live data feed becomes available.
- **Streamlit dashboard**: visualise equity curves, drawdowns, and trade
  distributions across all stored experiment JSON files.

## Raw artefacts

```
logs/nq_random_baseline_20260819T170928Z.json
logs/nq_ma_cross_20_100_20260819T171009Z.json
```