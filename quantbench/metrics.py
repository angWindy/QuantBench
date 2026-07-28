from __future__ import annotations

import math
from dataclasses import dataclass

from .backtest import BacktestResult


@dataclass(frozen=True)
class PerformanceReport:
    pnl: float
    sharpe: float
    max_drawdown: float
    win_rate: float


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
    return max_dd


def evaluate_performance(result: BacktestResult, periods_per_year: int = 252) -> PerformanceReport:
    equity_curve = result.equity_curve
    period_returns = result.period_returns
    pnl = equity_curve[-1] - equity_curve[0]

    mean_r = sum(period_returns) / len(period_returns)
    variance = sum((r - mean_r) ** 2 for r in period_returns) / len(period_returns)
    std = math.sqrt(variance)
    sharpe = 0.0 if std == 0 else math.sqrt(periods_per_year) * mean_r / std

    non_zero = [ret for ret in period_returns if ret != 0.0]
    wins = [ret for ret in non_zero if ret > 0.0]
    win_rate = 0.0 if not non_zero else len(wins) / len(non_zero)

    return PerformanceReport(
        pnl=pnl,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(equity_curve),
        win_rate=win_rate,
    )
