"""QuantBench env verification — run after `conda activate quant`.

Usage:
    python scripts/verify_env.py
"""
from __future__ import annotations

import sys
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd

REQUIRED = [
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "polars",
    "pandas_ta_classic",
    "vectorbt",
    "quantstats",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "pydantic",
    "pyyaml",
    "loguru",
    "matplotlib",
    "plotly",
    "seaborn",
    "mplfinance",
    "pytest",
]


def main() -> int:
    print("=" * 60)
    print(f"Python   : {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"Platform : {platform.system()} {platform.machine()}")
    print("=" * 60)

    missing = []
    for pkg in REQUIRED:
        try:
            print(f"  {pkg:<22} {version(pkg)}")
        except Exception:
            print(f"  {pkg:<22} MISSING")
            missing.append(pkg)

    print("=" * 60)
    if missing:
        print(f"FAIL — missing: {missing}")
        return 1

    # Functional smoke test
    rng = np.random.default_rng(42)
    n = 500
    close = pd.Series(rng.normal(100, 1, n).cumsum() + 10000, name="close")
    df = pd.DataFrame({"close": close})

    import pandas_ta_classic as ta
    import vectorbt as vbt

    sma = ta.sma(df["close"], length=20)
    entries = df["close"] > sma
    exits = df["close"] < sma
    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries.fillna(False),
        exits=exits.fillna(False),
        init_cash=100_000,
        fees=0.00025,
        slippage=0.0001,
        freq="1D",
    )
    stats = pf.stats()

    print("Smoke backtest OK")
    # Print available return / risk keys for transparency
    for key in stats.index:
        if any(t in key for t in ("Return", "Drawdown", "Sharpe", "Sortino")):
            val = stats[key]
            print(f"  {key:<28} {val}")
    print("=" * 60)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())