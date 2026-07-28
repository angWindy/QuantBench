from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from .features import FeatureSet


class Strategy(Protocol):
    def generate_positions(self, features: FeatureSet) -> list[int]:
        """Generate unit positions (-1 short, 0 flat, 1 long)."""


@dataclass
class RandomStrategy:
    seed: int = 0
    shorting: bool = False

    def generate_positions(self, features: FeatureSet) -> list[int]:
        rng = Random(self.seed)
        if self.shorting:
            universe = [-1, 0, 1]
        else:
            universe = [0, 1]
        return [universe[rng.randrange(len(universe))] for _ in features.returns]


@dataclass
class RSIStrategy:
    oversold: float = 30.0
    overbought: float = 70.0

    def generate_positions(self, features: FeatureSet) -> list[int]:
        position = 0
        positions: list[int] = []
        for rsi in features.rsi:
            if rsi is None:
                positions.append(position)
                continue
            if rsi < self.oversold:
                position = 1
            elif rsi > self.overbought:
                position = -1
            positions.append(position)
        return positions
