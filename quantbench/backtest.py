from __future__ import annotations

from dataclasses import dataclass

from .data import MarketData
from .features import compute_returns


@dataclass(frozen=True)
class BacktestConfig:
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    position_size: float = 1.0
    initial_equity: float = 1.0


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: list[float]
    period_returns: list[float]
    positions: list[int]
    turnover: float


class Backtester:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._config = config or BacktestConfig()

    def run(self, data: MarketData, positions: list[int]) -> BacktestResult:
        if len(positions) != data.length:
            raise ValueError("positions length must match market data length")

        market_returns = compute_returns(data.close)
        period_returns = [0.0]
        equity_curve = [self._config.initial_equity]
        prev_position = positions[0]
        turnover = 0.0
        cost_per_turn = (self._config.fee_bps + self._config.slippage_bps) / 10_000.0

        for idx in range(1, data.length):
            position = positions[idx - 1]
            gross = position * self._config.position_size * market_returns[idx]
            delta = abs(positions[idx] - prev_position)
            cost = delta * self._config.position_size * cost_per_turn
            net = gross - cost
            period_returns.append(net)
            equity_curve.append(equity_curve[-1] * (1.0 + net))
            turnover += delta
            prev_position = positions[idx]

        return BacktestResult(
            equity_curve=equity_curve,
            period_returns=period_returns,
            positions=positions,
            turnover=turnover,
        )
