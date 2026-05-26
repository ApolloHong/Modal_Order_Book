"""Petits utilitaires pour des expériences reproductibles d'événements rares."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def ensure_rng(rng: Optional[np.random.Generator | int] = None) -> np.random.Generator:
    """Renvoie un objet ``np.random.Generator`` explicite.

    Passer les générateurs à travers la pile d'appels permet d'éviter l'utilisation
    d'un état global caché (RNG). Les entiers sont acceptés comme graines (seeds) de courtoisie.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


class RNGStream:
    """Flux déterministe de générateurs enfants indépendants."""

    def __init__(self, seed: Optional[int] = None):
        self.seed_sequence = np.random.SeedSequence(seed)

    def next(self) -> np.random.Generator:
        """Renvoie un nouveau générateur enfant indépendant."""
        return np.random.default_rng(self.seed_sequence.spawn(1)[0])

    def spawn(self, n: int) -> list[np.random.Generator]:
        """Renvoie ``n`` nouveaux générateurs enfants indépendants."""
        return [np.random.default_rng(s) for s in self.seed_sequence.spawn(n)]


def log_product(values: Iterable[float]) -> float:
    """Calcule ``log(prod(values))`` et renvoie ``-inf`` si l'un des facteurs est nul."""
    total = 0.0
    for value in values:
        if value <= 0:
            return -math.inf
        total += math.log(float(value))
    return total


def binomial_standard_error(p: float, n: int) -> float:
    """Erreur type de la proportion d'un échantillon de Bernoulli."""
    if n <= 0:
        return math.nan
    p = float(np.clip(p, 0.0, 1.0))
    return math.sqrt(p * (1.0 - p) / n)


def validate_finite_array(name: str, values: np.ndarray) -> None:
    """Lève une erreur explicite si un tableau contient des valeurs NaN ou infinies."""
    arr = np.asarray(values)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} ne doit contenir que des valeurs finies")


@dataclass
class Timer:
    """Chronomètre léger basé sur le temps réel écoulé (wall-clock)."""

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
