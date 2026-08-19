"""Tests for the experiment runner."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantbench.backtest.engine import BacktestConfig
from quantbench.experiments.runner import (
    ExperimentConfig,
    get_strategy,
    run_experiment,
    write_result,
)
from quantbench.strategies import MACrossStrategy, RandomStrategy

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "NQ_1min.parquet"


@pytest.fixture(scope="module")
def cfg_random():
    return ExperimentConfig(
        name="test_random",
        data_path=str(PROCESSED) if PROCESSED.exists() else _synth_processed_path(),
        strategy="random",
        strategy_params={"seed": 7},
        backtest=BacktestConfig(freq="1D"),
    )


def _synth_processed_path(tmp_path_factory) -> str:
    """Create a tiny parquet fixture if the real one is missing."""
    import numpy as np
    from quantbench.data.storage import write_processed
    n = 5000
    rng = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    px = 100 + np.random.default_rng(0).normal(0, 0.1, n).cumsum()
    df = pd.DataFrame(
        {"open": px, "high": px + 0.05, "low": px - 0.05, "close": px, "volume": 1000},
        index=rng,
    )
    p = tmp_path_factory.mktemp("data") / "synth.parquet"
    write_processed(df, p)
    return str(p)


def test_get_strategy_random():
    cfg = ExperimentConfig(name="x", data_path="x", strategy="random", strategy_params={"seed": 1})
    s = get_strategy(cfg)
    assert isinstance(s, RandomStrategy)


def test_get_strategy_ma_cross():
    cfg = ExperimentConfig(
        name="x", data_path="x", strategy="ma_cross", strategy_params={"fast": 10, "slow": 30}
    )
    s = get_strategy(cfg)
    assert isinstance(s, MACrossStrategy)


def test_get_strategy_unknown_raises():
    cfg = ExperimentConfig(name="x", data_path="x", strategy="nope")
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy(cfg)


def test_run_experiment_returns_result(cfg_random, tmp_path):
    cfg_random = ExperimentConfig(
        name="test_random", data_path=cfg_random.data_path, strategy="random",
        strategy_params={"seed": 7}, backtest=BacktestConfig(freq="1D"),
    )
    result = run_experiment(cfg_random)
    assert result.metrics is not None
    assert result.n_bars > 0
    assert result.runtime_seconds >= 0


def test_write_result_creates_json(cfg_random, tmp_path):
    cfg_random = ExperimentConfig(
        name="test_random", data_path=cfg_random.data_path, strategy="random",
        strategy_params={"seed": 7}, backtest=BacktestConfig(freq="1D"),
    )
    result = run_experiment(cfg_random)
    path = write_result(result, tmp_path)
    assert path.exists()
    import json
    payload = json.loads(path.read_text())
    assert "metrics" in payload
    assert "config" in payload
    assert "sharpe" in payload["metrics"]