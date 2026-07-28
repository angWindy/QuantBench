from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class MarketData:
    close: list[float]

    def __post_init__(self) -> None:
        if len(self.close) < 2:
            raise ValueError("MarketData requires at least 2 close prices.")

    @property
    def length(self) -> int:
        return len(self.close)


class SyntheticDataGenerator:
    """Simple synthetic close-price generator using geometric random walk."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = Random(seed)

    def generate_close(
        self,
        periods: int = 252,
        start_price: float = 100.0,
        drift: float = 0.0002,
        volatility: float = 0.01,
    ) -> MarketData:
        if periods < 2:
            raise ValueError("periods must be >= 2")
        if start_price <= 0:
            raise ValueError("start_price must be > 0")
        if volatility < 0:
            raise ValueError("volatility must be >= 0")

        prices = [start_price]
        for _ in range(periods - 1):
            shock = self._rng.gauss(mu=drift, sigma=volatility)
            next_price = max(1e-8, prices[-1] * (1.0 + shock))
            prices.append(next_price)
        return MarketData(close=prices)
