"""Cibles d'événements rares et fonctions de score pour les simulations de LOB (carnet d'ordres).

Le thinning d'Ogata simule les trajectoires. Les objets de ce module définissent
quel événement est considéré comme rare et quel score de progression ou règle de frontière
doit guider les estimateurs d'événements rares.
Les scores doivent être monotones, ou du moins refléter une progression le long des trajectoires.
Lorsqu'un score n'est qu'un indicateur approximatif (proxy), les diagnostics doivent être lus avec une attention particulière.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import warnings

import numpy as np

State = np.ndarray
PathMetadata = dict[str, Any]
TargetFunction = Callable[[State, float, PathMetadata], bool]
ScoreFunction = Callable[[State, float, PathMetadata], float]
ObservableFunction = Callable[[State, float, PathMetadata], float]


@dataclass
class RareEventProblem:
    """Définition d'un problème de probabilité d'événement rare à horizon fini.

    La quantité estimée est généralement ``P(tau_A <= T)``. Le simulateur gère
    la dynamique ; cet objet gère l'événement, le score et la condition initiale.
    """

    T: float
    initial_state: State
    target_event: TargetFunction
    score_function: ScoreFunction
    terminal_observable: Optional[ObservableFunction] = None
    event_name: str = "rare_event"
    threshold: float = 1.0
    metadata: PathMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.initial_state = np.asarray(self.initial_state, dtype=float)
        if self.T <= 0:
            raise ValueError("RareEventProblem.T must be positive")
        if self.threshold <= 0:
            raise ValueError("RareEventProblem.threshold must be positive")

    def is_target(self, state: State, time: float, metadata: Optional[PathMetadata] = None) -> bool:
        context = self._context(metadata)
        return bool(self.target_event(np.asarray(state, dtype=float), float(time), context))

    def score(self, state: State, time: float, metadata: Optional[PathMetadata] = None) -> float:
        context = self._context(metadata)
        value = float(self.score_function(np.asarray(state, dtype=float), float(time), context))
        if not np.isfinite(value):
            raise ValueError(f"score_function for {self.event_name!r} returned a non-finite value")
        return value

    def observable(self, state: State, time: float, metadata: Optional[PathMetadata] = None) -> float:
        if self.terminal_observable is None:
            return float(self.is_target(state, time, metadata))
        context = self._context(metadata)
        return float(self.terminal_observable(np.asarray(state, dtype=float), float(time), context))

    def _context(self, metadata: Optional[PathMetadata]) -> PathMetadata:
        context = dict(self.metadata)
        if metadata:
            context.update(metadata)
        return context


def queue_position(metadata: PathMetadata, queue_index: int, default: Optional[int] = None) -> int:
    """Renvoie la position vectorielle d'un indice de file d'attente."""
    indices = metadata.get("queue_indices")
    if indices is None:
        if default is not None:
            return default
        raise KeyError("metadata doit contenir 'queue_indices' ou une position par défaut")
    try:
        return list(indices).index(queue_index)
    except ValueError as exc:
        raise KeyError(f"l'indice de file {queue_index} n'est pas présent dans queue_indices={indices}") from exc


def queue_value(state: State, metadata: PathMetadata, queue_index: int, default: Optional[int] = None) -> float:
    """Lit la valeur d'une file d'attente à partir d'un vecteur d'état."""
    return float(np.asarray(state)[queue_position(metadata, queue_index, default)])


def first_limit_imbalance(state: State, metadata: PathMetadata) -> float:
    """Calcule ``(Q_bid - Q_ask) / (Q_bid + Q_ask)``."""
    q_ask = queue_value(state, metadata, 1, 0)
    q_bid = queue_value(state, metadata, -1, 1 if len(state) > 1 else 0)
    total = q_bid + q_ask
    return 0.0 if total <= 0 else float((q_bid - q_ask) / total)


def distance_to_depletion_score(state: State, time: float, metadata: PathMetadata) -> float:
    """Score de progression qui augmente à mesure que le volume de la file cible approche de zéro."""
    del time
    state = np.asarray(state, dtype=float)
    target_indices = metadata.get("target_indices")
    if target_indices is None:
        target_indices = [metadata.get("queue_index", 1)]
    queues = [queue_value(state, metadata, int(qi), 0 if len(state) == 1 else None) for qi in target_indices]
    current = min(queues)
    initial = float(metadata.get("initial_queue", max(current, 1.0)))
    if "initial_queue" not in metadata and len(target_indices) > 1:
        initial = min(
            queue_value(np.asarray(metadata.get("initial_state", state)), metadata, int(qi), 0)
            for qi in target_indices
        )
    denom = max(initial, 1.0)
    return float(np.clip((initial - current) / denom, 0.0, 1.0))


def imbalance_score(state: State, time: float, metadata: PathMetadata) -> float:
    """Score de progression pour le franchissement d'un déséquilibre signé ou absolu."""
    del time
    imb = first_limit_imbalance(state, metadata)
    threshold = max(float(metadata.get("imbalance_threshold", 0.8)), 1e-12)
    side = metadata.get("side", "abs")
    if side == "bid":
        raw = imb / threshold
    elif side == "ask":
        raw = -imb / threshold
    else:
        raw = abs(imb) / threshold
    return float(np.clip(raw, 0.0, 1.0))


def second_limit_score(state: State, time: float, metadata: PathMetadata) -> float:
    """Score d'approximation pour les événements combinant la déplétion de la première limite et la taille de la seconde.

    La partie déplétion est monotone jusqu'à ce que la première limite atteigne zéro. Le volume de la
    seconde limite peut évoluer dans les deux sens, ce score est donc utile mais pas strictement monotone.
    """
    del time
    side = int(metadata.get("side", 1))
    first_index = side
    second_index = 2 * side
    q_first = queue_value(state, metadata, first_index)
    q_second = queue_value(state, metadata, second_index)
    q_first0 = max(float(metadata.get("initial_first_queue", q_first)), 1.0)
    q_second_threshold = max(float(metadata.get("q2_threshold", 1.0)), 1.0)
    depletion_progress = np.clip((q_first0 - q_first) / q_first0, 0.0, 1.0)
    q2_progress = np.clip(q_second / q_second_threshold, 0.0, 1.0)
    if q_first > 0:
        return float(0.8 * depletion_progress)
    return float(0.8 + 0.2 * q2_progress)


def hawkes_excitation_score(state: State, time: float, metadata: PathMetadata) -> float:
    """Score de progression basé sur l'intensité de Hawkes ou la pression d'excitation."""
    del state, time
    intensity = metadata.get("intensity")
    if intensity is None:
        return 0.0
    arr = np.asarray(intensity, dtype=float)
    positions = metadata.get("intensity_positions")
    selected = arr if positions is None else arr[list(positions)]
    baseline = float(metadata.get("baseline_intensity", 1.0))
    target = float(metadata.get("excitation_threshold", max(baseline, 1.0)))
    if target <= baseline:
        target = baseline + 1.0
    raw = (float(np.max(selected)) - baseline) / (target - baseline)
    return float(np.clip(raw, 0.0, 1.0))


def q1_depletion_problem(
    T: float,
    q1_init: int,
    q_neg1_init: Optional[int] = None,
    queue_index: int = 1,
    event_name: str = "q1_depletion",
) -> RareEventProblem:
    """Crée un problème de déplétion de la meilleure limite vendeuse (best-ask) à une ou deux files."""
    if q_neg1_init is None:
        initial_state = np.array([q1_init], dtype=float)
        queue_indices = [queue_index]
    else:
        initial_state = np.array([q1_init, q_neg1_init], dtype=float)
        queue_indices = [1, -1]
    metadata = {
        "queue_indices": queue_indices,
        "queue_index": queue_index,
        "target_indices": [queue_index],
        "initial_queue": float(q1_init),
        "initial_state": initial_state,
    }

    def target(state: State, time: float, ctx: PathMetadata) -> bool:
        del time
        return queue_value(state, ctx, queue_index, 0) <= 0

    return RareEventProblem(
        T=T,
        initial_state=initial_state,
        target_event=target,
        score_function=distance_to_depletion_score,
        event_name=event_name,
        threshold=1.0,
        metadata=metadata,
    )


def ask_best_depletion_problem(T: float, q1_init: int, q_neg1_init: int) -> RareEventProblem:
    """La meilleure limite vendeuse ``Q+1`` atteint zéro avant ``T``."""
    return q1_depletion_problem(T, q1_init, q_neg1_init, queue_index=1, event_name="ask_best_depletion")


def bid_best_depletion_problem(T: float, q1_init: int, q_neg1_init: int) -> RareEventProblem:
    """La meilleure limite acheteuse ``Q-1`` atteint zéro avant ``T``."""
    initial_state = np.array([q1_init, q_neg1_init], dtype=float)
    metadata = {
        "queue_indices": [1, -1],
        "queue_index": -1,
        "target_indices": [-1],
        "initial_queue": float(q_neg1_init),
        "initial_state": initial_state,
    }

    def target(state: State, time: float, ctx: PathMetadata) -> bool:
        del time
        return queue_value(state, ctx, -1) <= 0

    return RareEventProblem(
        T=T,
        initial_state=initial_state,
        target_event=target,
        score_function=distance_to_depletion_score,
        event_name="bid_best_depletion",
        threshold=1.0,
        metadata=metadata,
    )


def min_best_depletion_problem(T: float, q1_init: int, q_neg1_init: int) -> RareEventProblem:
    """L'une ou l'autre des meilleures limites, ``Q+1`` ou ``Q-1``, atteint zéro avant ``T``."""
    initial_state = np.array([q1_init, q_neg1_init], dtype=float)
    metadata = {
        "queue_indices": [1, -1],
        "target_indices": [1, -1],
        "initial_queue": float(min(q1_init, q_neg1_init)),
        "initial_state": initial_state,
    }

    def target(state: State, time: float, ctx: PathMetadata) -> bool:
        del time
        return queue_value(state, ctx, 1) <= 0 or queue_value(state, ctx, -1) <= 0

    return RareEventProblem(
        T=T,
        initial_state=initial_state,
        target_event=target,
        score_function=distance_to_depletion_score,
        event_name="min_best_depletion",
        threshold=1.0,
        metadata=metadata,
    )


def first_limit_depletion_problem(
    T: float,
    q1_init: int,
    q_neg1_init: int,
    q2_init: int,
    q_neg2_init: int,
    side: Optional[int] = None,
) -> RareEventProblem:
    """Problème à quatre files où une première limite s'épuise avant ``T``.

    Si ``side=None``, vise indifféremment ``Q+1`` ou ``Q-1``. Si ``side=1``, vise uniquement la
    déplétion à la vente (ask), et si ``side=-1``, vise uniquement la déplétion à l'achat (bid). Le score
    reste la progression normalisée vers la déplétion de la première limite, tandis que ``Q+2``
    et ``Q-2`` sont conservés dans l'état afin de pouvoir extraire les observables conditionnels
    des deuxièmes limites à partir des trajectoires d'atteinte.
    """
    initial_state = np.array([q1_init, q_neg1_init, q2_init, q_neg2_init], dtype=float)
    if side is None:
        target_indices = [1, -1]
        initial_queue = float(min(q1_init, q_neg1_init))
        event_name = "first_limit_depletion"
    else:
        side = 1 if side >= 0 else -1
        target_indices = [side]
        initial_queue = float(q1_init if side == 1 else q_neg1_init)
        event_name = "ask_first_limit_depletion" if side == 1 else "bid_first_limit_depletion"

    metadata = {
        "queue_indices": [1, -1, 2, -2],
        "target_indices": target_indices,
        "initial_queue": initial_queue,
        "initial_state": initial_state,
    }

    def target(state: State, time: float, ctx: PathMetadata) -> bool:
        del time
        return any(queue_value(state, ctx, queue_index) <= 0 for queue_index in target_indices)

    return RareEventProblem(
        T=T,
        initial_state=initial_state,
        target_event=target,
        score_function=distance_to_depletion_score,
        event_name=event_name,
        threshold=1.0,
        metadata=metadata,
    )


def imbalance_crossing_problem(
    T: float,
    q1_init: int,
    q_neg1_init: int,
    threshold: float = 0.8,
    side: str = "abs",
) -> RareEventProblem:
    """Le déséquilibre (imbalance) de la première limite franchit un seuil avant ``T``."""
    initial_state = np.array([q1_init, q_neg1_init], dtype=float)
    metadata = {
        "queue_indices": [1, -1],
        "imbalance_threshold": float(threshold),
        "side": side,
        "initial_state": initial_state,
    }

    def target(state: State, time: float, ctx: PathMetadata) -> bool:
        del time
        imb = first_limit_imbalance(state, ctx)
        if side == "bid":
            return imb >= threshold
        if side == "ask":
            return imb <= -threshold
        return abs(imb) >= threshold

    return RareEventProblem(
        T=T,
        initial_state=initial_state,
        target_event=target,
        score_function=imbalance_score,
        event_name="imbalance_crossing",
        threshold=1.0,
        metadata=metadata,
    )


def second_limit_activation_problem(
    T: float,
    initial_state: np.ndarray,
    side: int = 1,
    q2_threshold: int = 8,
) -> RareEventProblem:
    """La première limite s'épuise et la deuxième limite du même côté est suffisamment active."""
    warnings.warn(
        "second_limit_score est de type progression mais n'est pas strictement monotone car la deuxième limite du même côté peut à la fois augmenter et diminuer",
        RuntimeWarning,
        stacklevel=2,
    )
    side = 1 if side >= 0 else -1
    initial_state = np.asarray(initial_state, dtype=float)
    metadata = {
        "queue_indices": [1, -1, 2, -2],
        "side": side,
        "q2_threshold": float(q2_threshold),
        "initial_first_queue": queue_value(initial_state, {"queue_indices": [1, -1, 2, -2]}, side),
        "initial_state": initial_state,
        "non_monotone_score": True,
    }

    def target(state: State, time: float, ctx: PathMetadata) -> bool:
        del time
        return queue_value(state, ctx, side) <= 0 and queue_value(state, ctx, 2 * side) >= q2_threshold

    return RareEventProblem(
        T=T,
        initial_state=initial_state,
        target_event=target,
        score_function=second_limit_score,
        event_name="second_limit_activation",
        threshold=1.0,
        metadata=metadata,
    )


def q2_after_q1_depletion_problem(
    T: float,
    q1_init: int,
    q_neg1_init: int,
    q2_init: int,
    q_neg2_init: int,
    q2_threshold: int,
) -> RareEventProblem:
    """Événement où ``Q+1`` se vide et ``Q+2`` est au moins égal à ``q2_threshold``."""
    initial_state = np.array([q1_init, q_neg1_init, q2_init, q_neg2_init], dtype=float)
    return second_limit_activation_problem(T, initial_state, side=1, q2_threshold=q2_threshold)