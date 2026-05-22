"""Small utilities for reproducible rare-event experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def ensure_rng(rng: Optional[np.random.Generator | int] = None) -> np.random.Generator:
    """Return an explicit ``np.random.Generator``.

    Passing generators through the call stack avoids hidden global RNG state.
    Integers are accepted as convenience seeds.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


class RNGStream:
    """Deterministic stream of independent child generators."""

    def __init__(self, seed: Optional[int] = None):
        self.seed_sequence = np.random.SeedSequence(seed)

    def next(self) -> np.random.Generator:
        """Return a fresh child generator."""
        return np.random.default_rng(self.seed_sequence.spawn(1)[0])

    def spawn(self, n: int) -> list[np.random.Generator]:
        """Return ``n`` fresh child generators."""
        return [np.random.default_rng(s) for s in self.seed_sequence.spawn(n)]


def log_product(values: Iterable[float]) -> float:
    """Compute ``log(prod(values))`` and return ``-inf`` if a factor is zero."""
    total = 0.0
    for value in values:
        if value <= 0:
            return -math.inf
        total += math.log(float(value))
    return total


def binomial_standard_error(p: float, n: int) -> float:
    """Standard error of a Bernoulli sample proportion."""
    if n <= 0:
        return math.nan
    p = float(np.clip(p, 0.0, 1.0))
    return math.sqrt(p * (1.0 - p) / n)


def validate_finite_array(name: str, values: np.ndarray) -> None:
    """Raise a clear error if an array contains NaN or infinity."""
    arr = np.asarray(values)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")


@dataclass
class Timer:
    """Lightweight wall-clock timer."""

    start: float = 0.0
    end: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.end = self.start
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        stop = self.end if self.end else time.perf_counter()
        return stop - self.start
