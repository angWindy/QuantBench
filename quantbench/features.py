from __future__ import annotations

from dataclasses import dataclass

from .data import MarketData


@dataclass(frozen=True)
class FeatureSet:
    returns: list[float]
    rsi: list[float | None]


def compute_returns(prices: list[float]) -> list[float]:
    if len(prices) < 2:
        return [0.0 for _ in prices]
    returns = [0.0]
    for idx in range(1, len(prices)):
        prev = prices[idx - 1]
        curr = prices[idx]
        returns.append((curr / prev) - 1.0 if prev else 0.0)
    return returns


def compute_rsi(prices: list[float], period: int = 14) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be > 0")
    if not prices:
        return []

    changes = [0.0]
    for idx in range(1, len(prices)):
        changes.append(prices[idx] - prices[idx - 1])

    rsi: list[float | None] = [None] * len(prices)
    avg_gain = 0.0
    avg_loss = 0.0

    for idx in range(1, len(prices)):
        gain = max(changes[idx], 0.0)
        loss = max(-changes[idx], 0.0)
        if idx <= period:
            avg_gain += gain
            avg_loss += loss
            if idx == period:
                avg_gain /= period
                avg_loss /= period
        else:
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period

        if idx >= period:
            if avg_loss == 0:
                rsi[idx] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[idx] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_features(data: MarketData, rsi_period: int = 14) -> FeatureSet:
    return FeatureSet(
        returns=compute_returns(data.close),
        rsi=compute_rsi(data.close, period=rsi_period),
    )
