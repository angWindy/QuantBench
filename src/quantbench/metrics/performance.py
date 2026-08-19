"""Performance metrics for backtest results.

All functions are pure: they take a series of returns (or a BacktestResult)
and return a ``Metrics`` dataclass. No plotting, no I/O. Plots live in a
separate module if/when added.

Conventions
-----------
- ``returns`` is a simple arithmetic return series (decimal, not %).
  ``r = close.pct_change()``.
- ``periods_per_year`` depends on bar frequency: 252 for daily, 252*390
  for 1-min equity bars during RTH, 365*24*60 for 1-min crypto, etc.
  Caller is responsible for picking the right number — guessing it
  silently is worse than getting it wrong loudly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantbench.backtest.engine import BacktestResult


@dataclass(frozen=True)
class Metrics:
    """Standard quant performance summary. All values are dimensionless."""

    total_return: float
    cagr: float                # compounded annual growth rate
    sharpe: float              # mean / std, annualised
    sortino: float             # mean / downside_std, annualised
    max_drawdown: float        # negative number, e.g. -0.18
    calmar: float              # cagr / abs(max_drawdown)
    win_rate: float            # fraction of winning trades
    profit_factor: float       # gross_profit / gross_loss
    n_trades: int

    def as_dict(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "n_trades": self.n_trades,
        }


def _annualised_sharpe(returns: pd.Series, ppy: float) -> float:
    if returns.std(ddof=1) == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(ppy))


def _annualised_sortino(returns: pd.Series, ppy: float) -> float:
    downside = returns.clip(upper=0.0)
    if downside.std(ddof=1) == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / downside.std(ddof=1) * np.sqrt(ppy))


def _max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough drawdown, returned as a negative fraction."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_metrics_from_returns(
    returns: pd.Series,
    *,
    periods_per_year: float = 252.0,
    equity: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
) -> Metrics:
    """Compute the full metrics bundle from a return series.

    Parameters
    ----------
    returns:
        Per-bar simple returns (e.g. ``close.pct_change().dropna()``).
    periods_per_year:
        Annualisation factor. Pick deliberately based on bar frequency.
    equity:
        Optional equity curve. If not given, reconstructed from returns
        (slightly slower for very long series).
    trades:
        Optional vectorbt-style trades frame with a ``pnl`` column. Used
        for win rate and profit factor. If absent, both default to 0.0
        and n_trades defaults to 0.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    if equity is None:
        equity = (1.0 + returns).cumprod()

    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    years = len(returns) / periods_per_year if periods_per_year > 0 else np.nan
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0

    sharpe = _annualised_sharpe(returns, periods_per_year)
    sortino = _annualised_sortino(returns, periods_per_year)
    max_dd = _max_drawdown(equity)
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    if trades is not None and "pnl" in trades.columns and len(trades) > 0:
        pnls = trades["pnl"].astype(float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        n_trades = int(len(pnls))
        win_rate = float(len(wins) / n_trades) if n_trades else 0.0
        gross_win = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else np.inf
    else:
        n_trades = 0
        win_rate = 0.0
        profit_factor = 0.0

    return Metrics(
        total_return=total,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=n_trades,
    )


def compute_metrics(result: BacktestResult, *, periods_per_year: float = 252.0) -> Metrics:
    """Compute metrics for a ``BacktestResult``.

    Pulls the returns series, equity curve, and trades frame out of the
    underlying vectorbt portfolio and forwards to ``compute_metrics_from_returns``.
    """
    pf = result.portfolio
    returns = pf.returns()
    equity = pf.value()
    try:
        trades = pf.trades.records_readable
    except Exception:
        trades = None
    return compute_metrics_from_returns(
        returns, periods_per_year=periods_per_year, equity=equity, trades=trades
    )