"""Continuation-capable path simulators for rare-event estimation.

Ogata thinning is used here only to simulate the underlying point-process path.
The assignment-facing rare-event layer is Markovian Conditional Restart
Splitting in ``model.restart_splitting``.  Legacy Fixed-Level Splitting and AMS
experiments remain in ``model.splitting``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .rare_events import RareEventProblem
from .utils import ensure_rng


@dataclass
class Checkpoint:
    """Sufficient state to continue a path from an intermediate time."""

    time: float
    state: np.ndarray
    hawkes_state: dict[str, np.ndarray | float] = field(default_factory=dict)
    intensity: Optional[np.ndarray] = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Checkpoint":
        return Checkpoint(
            time=float(self.time),
            state=np.asarray(self.state, dtype=float).copy(),
            hawkes_state=_copy_hawkes_state(self.hawkes_state),
            intensity=None if self.intensity is None else np.asarray(self.intensity, dtype=float).copy(),
            score=float(self.score),
            metadata=dict(self.metadata),
        )


@dataclass
class Trajectory:
    """Common representation of a simulated LOB path."""

    times: np.ndarray
    events: np.ndarray
    states: np.ndarray
    intensities: Optional[np.ndarray]
    final_state: np.ndarray
    hit: bool
    hitting_time: Optional[float]
    score_values: np.ndarray
    score_max: float
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[Checkpoint] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return max(0, len(self.events) - 1)

    @property
    def n_candidates(self) -> int:
        return int(self.metadata.get("n_candidates", self.n_events))

    def first_checkpoint_at_level(self, level: float) -> Optional[Checkpoint]:
        for checkpoint in self.checkpoints:
            if checkpoint.score >= level:
                return checkpoint.copy()
        return None


def _copy_hawkes_state(hawkes_state: dict[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    copied: dict[str, np.ndarray | float] = {}
    for key, value in hawkes_state.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = float(value)
    return copied


def _context(
    problem: RareEventProblem,
    intensity: Optional[np.ndarray],
    hawkes_state: dict[str, np.ndarray | float],
    event: int,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata = dict(problem.metadata)
    metadata["intensity"] = intensity
    metadata["hawkes_state"] = hawkes_state
    metadata["event"] = event
    if extra:
        metadata.update(extra)
    return metadata


def _record_point(
    *,
    problem: RareEventProblem,
    times: list[float],
    events: list[int],
    states: list[np.ndarray],
    intensities: list[np.ndarray],
    scores: list[float],
    checkpoints: list[Checkpoint],
    t: float,
    state: np.ndarray,
    event: int,
    intensity: np.ndarray,
    hawkes_state: dict[str, np.ndarray | float],
    keep_checkpoint: bool,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> tuple[float, bool]:
    state_copy = np.asarray(state, dtype=float).copy()
    intensity_copy = np.asarray(intensity, dtype=float).copy()
    hawkes_copy = _copy_hawkes_state(hawkes_state)
    metadata = _context(problem, intensity_copy, hawkes_copy, event, extra_metadata)
    score = problem.score(state_copy, t, metadata)
    hit = problem.is_target(state_copy, t, metadata)

    times.append(float(t))
    events.append(int(event))
    states.append(state_copy)
    intensities.append(intensity_copy)
    scores.append(score)
    if keep_checkpoint:
        checkpoints.append(
            Checkpoint(
                time=float(t),
                state=state_copy,
                hawkes_state=hawkes_copy,
                intensity=intensity_copy,
                score=score,
                metadata=metadata,
            )
        )
    return score, hit


def _build_trajectory(
    *,
    times: list[float],
    events: list[int],
    states: list[np.ndarray],
    intensities: list[np.ndarray],
    scores: list[float],
    hit: bool,
    hitting_time: Optional[float],
    metadata: dict[str, Any],
    checkpoints: list[Checkpoint],
) -> Trajectory:
    score_arr = np.asarray(scores, dtype=float)
    intensity_arr: Optional[np.ndarray]
    if intensities:
        intensity_arr = np.vstack(intensities)
    else:
        intensity_arr = None
    return Trajectory(
        times=np.asarray(times, dtype=float),
        events=np.asarray(events, dtype=int),
        states=np.vstack(states) if states else np.empty((0, 0)),
        intensities=intensity_arr,
        final_state=np.asarray(states[-1], dtype=float).copy() if states else np.empty(0),
        hit=bool(hit),
        hitting_time=None if hitting_time is None else float(hitting_time),
        score_values=score_arr,
        score_max=float(score_arr.max()) if len(score_arr) else 0.0,
        metadata=metadata,
        checkpoints=checkpoints,
    )


class IndependentPoissonSimulator:
    """Birth-death queue simulator with explicit continuation checkpoints."""

    def __init__(
        self,
        lambda_plus: float = 1.2,
        lambda_minus: float = 1.5,
        max_events: int = 1_000_000,
        keep_checkpoints: bool = True,
    ):
        if lambda_plus < 0 or lambda_minus < 0:
            raise ValueError("Poisson rates must be non-negative")
        self.lambda_plus = float(lambda_plus)
        self.lambda_minus = float(lambda_minus)
        self.max_events = int(max_events)
        self.keep_checkpoints = bool(keep_checkpoints)

    def simulate(
        self,
        problem: RareEventProblem,
        rng: Optional[np.random.Generator | int] = None,
        checkpoint: Optional[Checkpoint] = None,
        burn_in: float = 0.0,
        record_path: bool = True,
    ) -> Trajectory:
        rng = ensure_rng(rng)
        if checkpoint is None:
            state = np.asarray(problem.initial_state, dtype=float).copy()
            t = 0.0
            if burn_in > 0:
                state = self._evolve_burn_in(state, float(burn_in), rng)
        else:
            cp = checkpoint.copy()
            state = cp.state.copy()
            t = cp.time
        return self._simulate_from_state(problem, state, t, rng, record_path)

    def continue_from_checkpoint(
        self,
        checkpoint: Checkpoint,
        problem: RareEventProblem,
        rng: Optional[np.random.Generator | int] = None,
        record_path: bool = True,
    ) -> Trajectory:
        return self.simulate(problem, rng=rng, checkpoint=checkpoint, record_path=record_path)

    def checkpoint_at_level(self, trajectory: Trajectory, level: float) -> Optional[Checkpoint]:
        return trajectory.first_checkpoint_at_level(level)

    def _rates(self, state: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
        rates: list[float] = []
        events: list[tuple[int, int]] = []
        for pos, q in enumerate(state):
            rates.append(self.lambda_plus)
            events.append((pos, +1))
            rates.append(self.lambda_minus if q > 0 else 0.0)
            events.append((pos, -1))
        return np.asarray(rates, dtype=float), events

    def _evolve_burn_in(self, state: np.ndarray, horizon: float, rng: np.random.Generator) -> np.ndarray:
        state = state.copy()
        t = 0.0
        for _ in range(self.max_events):
            rates, events = self._rates(state)
            total = float(rates.sum())
            if total <= 0:
                break
            t += rng.exponential(1.0 / total)
            if t > horizon:
                break
            idx = int(rng.choice(len(rates), p=rates / total))
            pos, delta = events[idx]
            state[pos] = max(0.0, state[pos] + delta)
        return state

    def _simulate_from_state(
        self,
        problem: RareEventProblem,
        state: np.ndarray,
        t: float,
        rng: np.random.Generator,
        record_path: bool,
    ) -> Trajectory:
        del record_path
        times: list[float] = []
        events: list[int] = []
        states: list[np.ndarray] = []
        intensities: list[np.ndarray] = []
        scores: list[float] = []
        checkpoints: list[Checkpoint] = []
        n_events = 0
        n_candidates = 0

        intensity, event_defs = self._rates(state)
        _, hit = _record_point(
            problem=problem,
            times=times,
            events=events,
            states=states,
            intensities=intensities,
            scores=scores,
            checkpoints=checkpoints,
            t=t,
            state=state,
            event=-1,
            intensity=intensity,
            hawkes_state={},
            keep_checkpoint=self.keep_checkpoints,
            extra_metadata={"queue_indices": problem.metadata.get("queue_indices")},
        )
        if hit:
            return _build_trajectory(
                times=times,
                events=events,
                states=states,
                intensities=intensities,
                scores=scores,
                hit=True,
                hitting_time=t,
                metadata={"model": "independent_poisson", "n_events": n_events, "n_candidates": n_candidates},
                checkpoints=checkpoints,
            )

        while t < problem.T and n_events < self.max_events:
            intensity, event_defs = self._rates(state)
            total = float(intensity.sum())
            if total <= 0:
                break
            dt = rng.exponential(1.0 / total)
            t += dt
            n_candidates += 1
            if t > problem.T:
                break
            choice = int(rng.choice(len(intensity), p=intensity / total))
            pos, delta = event_defs[choice]
            state[pos] = max(0.0, state[pos] + delta)
            n_events += 1
            new_intensity, _ = self._rates(state)
            _, hit = _record_point(
                problem=problem,
                times=times,
                events=events,
                states=states,
                intensities=intensities,
                scores=scores,
                checkpoints=checkpoints,
                t=t,
                state=state,
                event=choice,
                intensity=new_intensity,
                hawkes_state={},
                keep_checkpoint=self.keep_checkpoints,
                extra_metadata={"n_events": n_events, "n_candidates": n_candidates},
            )
            if hit:
                break

        return _build_trajectory(
            times=times,
            events=events,
            states=states,
            intensities=intensities,
            scores=scores,
            hit=hit,
            hitting_time=t if hit else None,
            metadata={"model": "independent_poisson", "n_events": n_events, "n_candidates": n_candidates},
            checkpoints=checkpoints,
        )


class SingleHawkesSimulator:
    """One-queue Hawkes simulator using Ogata thinning."""

    def __init__(
        self,
        mu_plus: float = 1.2,
        mu_minus: float = 1.5,
        alpha: float = 0.3,
        beta: float = 0.5,
        sign_convention: str = "v4",
        max_events: int = 1_000_000,
        keep_checkpoints: bool = True,
    ):
        self.mu_plus = float(mu_plus)
        self.mu_minus = float(mu_minus)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sign_convention = sign_convention.lower()
        if self.sign_convention not in {"v4", "inverse"}:
            raise ValueError("sign_convention must be either 'v4' or 'inverse'")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        self.max_events = int(max_events)
        self.keep_checkpoints = bool(keep_checkpoints)

    @property
    def add_jump(self) -> float:
        return -self.alpha if self.sign_convention == "v4" else self.alpha

    @property
    def remove_jump(self) -> float:
        return self.alpha if self.sign_convention == "v4" else -self.alpha

    def simulate(
        self,
        problem: RareEventProblem,
        rng: Optional[np.random.Generator | int] = None,
        checkpoint: Optional[Checkpoint] = None,
        burn_in: float = 0.0,
        record_path: bool = True,
    ) -> Trajectory:
        rng = ensure_rng(rng)
        if checkpoint is None:
            state = np.asarray(problem.initial_state, dtype=float).copy()
            H = 0.0
            t = 0.0
            if burn_in > 0:
                state, H = self._evolve_burn_in(state, H, float(burn_in), rng)
        else:
            cp = checkpoint.copy()
            state = cp.state.copy()
            H = float(cp.hawkes_state.get("H", 0.0))
            t = cp.time
        return self._simulate_from_state(problem, state, H, t, rng, record_path)

    def continue_from_checkpoint(
        self,
        checkpoint: Checkpoint,
        problem: RareEventProblem,
        rng: Optional[np.random.Generator | int] = None,
        record_path: bool = True,
    ) -> Trajectory:
        return self.simulate(problem, rng=rng, checkpoint=checkpoint, record_path=record_path)

    def checkpoint_at_level(self, trajectory: Trajectory, level: float) -> Optional[Checkpoint]:
        return trajectory.first_checkpoint_at_level(level)

    def _intensity(self, q: float, H: float) -> np.ndarray:
        lam_minus = max(0.0, self.mu_minus + H) if q > 0 else 0.0
        return np.array([self.mu_plus, lam_minus], dtype=float)

    def _evolve_burn_in(
        self,
        state: np.ndarray,
        H: float,
        horizon: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        state = state.copy()
        t = 0.0
        for _ in range(self.max_events):
            q = state[0]
            lam_minus_max = self.mu_minus + max(H, 0.0) if q > 0 else 0.0
            lam_max = self.mu_plus + lam_minus_max + 0.01
            if lam_max <= 0:
                break
            dt = rng.exponential(1.0 / lam_max)
            t += dt
            if t > horizon:
                break
            H *= np.exp(-self.beta * dt)
            intensity = self._intensity(state[0], H)
            total = float(intensity.sum())
            if total <= 0 or rng.random() > total / lam_max:
                continue
            if rng.random() < self.mu_plus / total:
                state[0] += 1.0
                H += self.add_jump
            else:
                state[0] = max(0.0, state[0] - 1.0)
                H += self.remove_jump
        return state, H

    def _simulate_from_state(
        self,
        problem: RareEventProblem,
        state: np.ndarray,
        H: float,
        t: float,
        rng: np.random.Generator,
        record_path: bool,
    ) -> Trajectory:
        del record_path
        times: list[float] = []
        events: list[int] = []
        states: list[np.ndarray] = []
        intensities: list[np.ndarray] = []
        scores: list[float] = []
        checkpoints: list[Checkpoint] = []
        n_events = 0
        n_candidates = 0

        intensity = self._intensity(state[0], H)
        _, hit = _record_point(
            problem=problem,
            times=times,
            events=events,
            states=states,
            intensities=intensities,
            scores=scores,
            checkpoints=checkpoints,
            t=t,
            state=state,
            event=0,
            intensity=intensity,
            hawkes_state={"H": H},
            keep_checkpoint=self.keep_checkpoints,
        )
        if hit:
            return _build_trajectory(
                times=times,
                events=events,
                states=states,
                intensities=intensities,
                scores=scores,
                hit=True,
                hitting_time=t,
                metadata={"model": "single_hawkes", "n_events": n_events, "n_candidates": n_candidates},
                checkpoints=checkpoints,
            )

        while t < problem.T and n_events < self.max_events:
            q = state[0]
            lam_minus_max = self.mu_minus + max(H, 0.0) if q > 0 else 0.0
            lam_max = self.mu_plus + lam_minus_max + 0.01
            if lam_max <= 0:
                break
            dt = rng.exponential(1.0 / lam_max)
            t += dt
            n_candidates += 1
            if t > problem.T:
                break
            H *= np.exp(-self.beta * dt)
            intensity = self._intensity(state[0], H)
            total = float(intensity.sum())
            if total <= 0 or rng.random() > total / lam_max:
                continue
            if rng.random() < self.mu_plus / total:
                state[0] += 1.0
                H += self.add_jump
                event = +1
            else:
                state[0] = max(0.0, state[0] - 1.0)
                H += self.remove_jump
                event = -1
            n_events += 1
            new_intensity = self._intensity(state[0], H)
            _, hit = _record_point(
                problem=problem,
                times=times,
                events=events,
                states=states,
                intensities=intensities,
                scores=scores,
                checkpoints=checkpoints,
                t=t,
                state=state,
                event=event,
                intensity=new_intensity,
                hawkes_state={"H": H},
                keep_checkpoint=self.keep_checkpoints,
                extra_metadata={"n_events": n_events, "n_candidates": n_candidates},
            )
            if hit:
                break

        return _build_trajectory(
            times=times,
            events=events,
            states=states,
            intensities=intensities,
            scores=scores,
            hit=hit,
            hitting_time=t if hit else None,
            metadata={"model": "single_hawkes", "n_events": n_events, "n_candidates": n_candidates},
            checkpoints=checkpoints,
        )


class CoupledHawkesSimulator:
    """Two-queue coupled Hawkes simulator with checkpointed excitation state."""

    def __init__(
        self,
        mu_plus: float = 1.2,
        mu_minus: float = 1.5,
        alpha: float = 0.3,
        beta: float = 0.5,
        sign_convention: str = "v4",
        max_events: int = 1_000_000,
        keep_checkpoints: bool = True,
    ):
        self.mu_plus = float(mu_plus)
        self.mu_minus = float(mu_minus)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sign_convention = sign_convention.lower()
        if self.sign_convention not in {"v4", "inverse"}:
            raise ValueError("sign_convention must be either 'v4' or 'inverse'")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        self.max_events = int(max_events)
        self.keep_checkpoints = bool(keep_checkpoints)

    def simulate(
        self,
        problem: RareEventProblem,
        rng: Optional[np.random.Generator | int] = None,
        checkpoint: Optional[Checkpoint] = None,
        burn_in: float = 0.0,
        record_path: bool = True,
    ) -> Trajectory:
        rng = ensure_rng(rng)
        if checkpoint is None:
            state = np.asarray(problem.initial_state, dtype=float).copy()
            H = np.zeros(2, dtype=float)
            t = 0.0
            if burn_in > 0:
                state, H = self._evolve_burn_in(state, H, float(burn_in), rng)
        else:
            cp = checkpoint.copy()
            state = cp.state.copy()
            H = np.asarray(cp.hawkes_state.get("H", np.zeros(2)), dtype=float).copy()
            t = cp.time
        return self._simulate_from_state(problem, state, H, t, rng, record_path)

    def continue_from_checkpoint(
        self,
        checkpoint: Checkpoint,
        problem: RareEventProblem,
        rng: Optional[np.random.Generator | int] = None,
        record_path: bool = True,
    ) -> Trajectory:
        return self.simulate(problem, rng=rng, checkpoint=checkpoint, record_path=record_path)

    def checkpoint_at_level(self, trajectory: Trajectory, level: float) -> Optional[Checkpoint]:
        return trajectory.first_checkpoint_at_level(level)

    def _intensity(self, state: np.ndarray, H: np.ndarray) -> np.ndarray:
        lm1 = max(0.0, self.mu_minus + H[0]) if state[0] > 0 else 0.0
        lmn1 = max(0.0, self.mu_minus + H[1]) if state[1] > 0 else 0.0
        return np.array([self.mu_plus, lm1, self.mu_plus, lmn1], dtype=float)

    def _apply_hawkes_jump(self, event: int, H: np.ndarray) -> None:
        if self.sign_convention == "v4":
            if event == 0:
                H[0] -= self.alpha
                H[1] -= self.alpha
            elif event == 1:
                H[0] += self.alpha
                H[1] += self.alpha
            elif event == 2:
                H[1] -= self.alpha
                H[0] -= self.alpha
            elif event == 3:
                H[1] += self.alpha
                H[0] += self.alpha
        else:
            if event == 0:
                H[0] += self.alpha
                H[1] -= self.alpha
            elif event == 1:
                H[0] -= self.alpha
                H[1] += self.alpha
            elif event == 2:
                H[1] += self.alpha
                H[0] -= self.alpha
            elif event == 3:
                H[1] -= self.alpha
                H[0] += self.alpha

    def _apply_queue_event(self, event: int, state: np.ndarray) -> None:
        if event == 0:
            state[0] += 1.0
        elif event == 1:
            state[0] = max(0.0, state[0] - 1.0)
        elif event == 2:
            state[1] += 1.0
        elif event == 3:
            state[1] = max(0.0, state[1] - 1.0)

    def _upper_bound(self, state: np.ndarray, H: np.ndarray) -> float:
        lm1_max = self.mu_minus + max(H[0], 0.0) if state[0] > 0 else 0.0
        lmn1_max = self.mu_minus + max(H[1], 0.0) if state[1] > 0 else 0.0
        return 2.0 * self.mu_plus + lm1_max + lmn1_max + 0.01

    def _evolve_burn_in(
        self,
        state: np.ndarray,
        H: np.ndarray,
        horizon: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = state.copy()
        H = H.copy()
        t = 0.0
        for _ in range(self.max_events):
            lam_max = self._upper_bound(state, H)
            if lam_max <= 0:
                break
            dt = rng.exponential(1.0 / lam_max)
            t += dt
            if t > horizon:
                break
            H *= np.exp(-self.beta * dt)
            intensity = self._intensity(state, H)
            total = float(intensity.sum())
            if total <= 0 or rng.random() > total / lam_max:
                continue
            event = int(rng.choice(4, p=intensity / total))
            self._apply_queue_event(event, state)
            self._apply_hawkes_jump(event, H)
        return state, H

    def _simulate_from_state(
        self,
        problem: RareEventProblem,
        state: np.ndarray,
        H: np.ndarray,
        t: float,
        rng: np.random.Generator,
        record_path: bool,
    ) -> Trajectory:
        del record_path
        times: list[float] = []
        events: list[int] = []
        states: list[np.ndarray] = []
        intensities: list[np.ndarray] = []
        scores: list[float] = []
        checkpoints: list[Checkpoint] = []
        n_events = 0
        n_candidates = 0

        intensity = self._intensity(state, H)
        _, hit = _record_point(
            problem=problem,
            times=times,
            events=events,
            states=states,
            intensities=intensities,
            scores=scores,
            checkpoints=checkpoints,
            t=t,
            state=state,
            event=-1,
            intensity=intensity,
            hawkes_state={"H": H},
            keep_checkpoint=self.keep_checkpoints,
        )
        if hit:
            return _build_trajectory(
                times=times,
                events=events,
                states=states,
                intensities=intensities,
                scores=scores,
                hit=True,
                hitting_time=t,
                metadata={"model": "coupled_hawkes", "n_events": n_events, "n_candidates": n_candidates},
                checkpoints=checkpoints,
            )

        while t < problem.T and n_events < self.max_events:
            lam_max = self._upper_bound(state, H)
            if lam_max <= 0:
                break
            dt = rng.exponential(1.0 / lam_max)
            t += dt
            n_candidates += 1
            if t > problem.T:
                break
            H *= np.exp(-self.beta * dt)
            intensity = self._intensity(state, H)
            total = float(intensity.sum())
            if total <= 0 or rng.random() > total / lam_max:
                continue
            event = int(rng.choice(4, p=intensity / total))
            self._apply_queue_event(event, state)
            self._apply_hawkes_jump(event, H)
            n_events += 1
            new_intensity = self._intensity(state, H)
            _, hit = _record_point(
                problem=problem,
                times=times,
                events=events,
                states=states,
                intensities=intensities,
                scores=scores,
                checkpoints=checkpoints,
                t=t,
                state=state,
                event=event,
                intensity=new_intensity,
                hawkes_state={"H": H},
                keep_checkpoint=self.keep_checkpoints,
                extra_metadata={"n_events": n_events, "n_candidates": n_candidates},
            )
            if hit:
                break

        return _build_trajectory(
            times=times,
            events=events,
            states=states,
            intensities=intensities,
            scores=scores,
            hit=hit,
            hitting_time=t if hit else None,
            metadata={"model": "coupled_hawkes", "n_events": n_events, "n_candidates": n_candidates},
            checkpoints=checkpoints,
        )


class FourQueueHawkesSimulator:
    """Four-queue Hawkes simulator for second-limit experiments.

    State order is ``[Q+1, Q-1, Q+2, Q-2]``.  The Hawkes checkpoint stores
    ``H=[H+1, H-1]`` for first-limit removals and ``G=[G+2, G-2]`` for
    second-limit addition excitation.
    """

    queue_indices = [1, -1, 2, -2]

    def __init__(
        self,
        params: Any,
        max_events: int = 1_000_000,
        record_every: int = 1,
        keep_checkpoints: bool = True,
    ):
        self.params = params
        self.max_events = int(max_events)
        self.record_every = max(1, int(record_every))
        self.keep_checkpoints = bool(keep_checkpoints)

    def simulate(
        self,
        problem: RareEventProblem,
        rng: Optional[np.random.Generator | int] = None,
        checkpoint: Optional[Checkpoint] = None,
        burn_in: float = 0.0,
        record_path: bool = True,
    ) -> Trajectory:
        rng = ensure_rng(rng)
        if checkpoint is None:
            p = self.params
            state = np.array([p.q1_init, p.q_neg1_init, p.q2_init, p.q_neg2_init], dtype=float)
            H = np.zeros(2, dtype=float)
            G = np.zeros(2, dtype=float)
            t = 0.0
            if burn_in > 0:
                state, H, G = self._evolve_burn_in(state, H, G, float(burn_in), rng)
        else:
            cp = checkpoint.copy()
            state = cp.state.copy()
            H = np.asarray(cp.hawkes_state.get("H", np.zeros(2)), dtype=float).copy()
            G = np.asarray(cp.hawkes_state.get("G", np.zeros(2)), dtype=float).copy()
            t = cp.time
        return self._simulate_from_state(problem, state, H, G, t, rng, record_path)

    def continue_from_checkpoint(
        self,
        checkpoint: Checkpoint,
        problem: RareEventProblem,
        rng: Optional[np.random.Generator | int] = None,
        record_path: bool = True,
    ) -> Trajectory:
        return self.simulate(problem, rng=rng, checkpoint=checkpoint, record_path=record_path)

    def checkpoint_at_level(self, trajectory: Trajectory, level: float) -> Optional[Checkpoint]:
        return trajectory.first_checkpoint_at_level(level)

    def _intensity(self, state: np.ndarray, H: np.ndarray, G: np.ndarray) -> np.ndarray:
        p = self.params
        lam_m1 = max(0.01, p.mu_minus_1 + H[0]) if state[0] > 0 else 0.0
        lam_mn1 = max(0.01, p.mu_minus_1 + H[1]) if state[1] > 0 else 0.0
        lam_p1 = p.mu_plus_1 if state[0] >= 0 else 0.0
        lam_pn1 = p.mu_plus_1 if state[1] >= 0 else 0.0
        lam_p2 = max(0.01, p.mu_plus_2 + G[0])
        lam_pn2 = max(0.01, p.mu_plus_2 + G[1])
        lam_m2 = p.mu_minus_2 if state[2] > 0 else 0.0
        lam_mn2 = p.mu_minus_2 if state[3] > 0 else 0.0
        return np.array([lam_p1, lam_m1, lam_pn1, lam_mn1, lam_p2, lam_m2, lam_pn2, lam_mn2], dtype=float)

    def _upper_bound(self, state: np.ndarray, H: np.ndarray, G: np.ndarray) -> float:
        p = self.params
        ub_m1 = p.mu_minus_1 + max(H[0], 0.0) if state[0] > 0 else 0.0
        ub_mn1 = p.mu_minus_1 + max(H[1], 0.0) if state[1] > 0 else 0.0
        ub_p2 = p.mu_plus_2 + max(G[0], 0.0)
        ub_pn2 = p.mu_plus_2 + max(G[1], 0.0)
        lam_p1 = p.mu_plus_1 if state[0] >= 0 else 0.0
        lam_pn1 = p.mu_plus_1 if state[1] >= 0 else 0.0
        lam_m2 = p.mu_minus_2 if state[2] > 0 else 0.0
        lam_mn2 = p.mu_minus_2 if state[3] > 0 else 0.0
        return lam_p1 + ub_m1 + lam_pn1 + ub_mn1 + ub_p2 + lam_m2 + ub_pn2 + lam_mn2 + 0.1

    def _apply_event(self, event: int, state: np.ndarray, H: np.ndarray, G: np.ndarray) -> None:
        p = self.params
        if event == 0:
            state[0] += 1.0
            H[0] -= p.alpha
            H[1] += p.alpha
        elif event == 1:
            state[0] = max(0.0, state[0] - 1.0)
            H[0] += p.alpha
            H[1] += p.alpha
            G[0] += p.a_cross
        elif event == 2:
            state[1] += 1.0
            H[1] -= p.alpha
            H[0] += p.alpha
        elif event == 3:
            state[1] = max(0.0, state[1] - 1.0)
            H[1] += p.alpha
            H[0] += p.alpha
            G[1] += p.a_cross
        elif event == 4:
            state[2] += 1.0
        elif event == 5:
            state[2] = max(0.0, state[2] - 1.0)
        elif event == 6:
            state[3] += 1.0
        elif event == 7:
            state[3] = max(0.0, state[3] - 1.0)

    def _evolve_burn_in(
        self,
        state: np.ndarray,
        H: np.ndarray,
        G: np.ndarray,
        horizon: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = state.copy()
        H = H.copy()
        G = G.copy()
        t = 0.0
        p = self.params
        for _ in range(self.max_events):
            lam_max = self._upper_bound(state, H, G)
            if lam_max <= 0:
                break
            dt = rng.exponential(1.0 / lam_max)
            t += dt
            if t > horizon:
                break
            H *= np.exp(-p.beta * dt)
            if p.a_cross > 0:
                G *= np.exp(-p.b_cross * dt)
            intensity = self._intensity(state, H, G)
            total = float(intensity.sum())
            if total <= 0 or rng.random() > total / lam_max:
                continue
            event = int(rng.choice(8, p=intensity / total))
            self._apply_event(event, state, H, G)
        return state, H, G

    def _simulate_from_state(
        self,
        problem: RareEventProblem,
        state: np.ndarray,
        H: np.ndarray,
        G: np.ndarray,
        t: float,
        rng: np.random.Generator,
        record_path: bool,
    ) -> Trajectory:
        del record_path
        p = self.params
        times: list[float] = []
        events: list[int] = []
        states: list[np.ndarray] = []
        intensities: list[np.ndarray] = []
        scores: list[float] = []
        checkpoints: list[Checkpoint] = []
        n_events = 0
        n_candidates = 0
        first_limit_hit = bool(state[0] <= 0 or state[1] <= 0)
        first_limit_hitting_time = t if first_limit_hit else None
        which_hit = 1 if state[0] <= 0 else (-1 if state[1] <= 0 else 0)

        intensity = self._intensity(state, H, G)
        extra = {
            "queue_indices": self.queue_indices,
            "first_limit_hit": first_limit_hit,
            "first_limit_hitting_time": first_limit_hitting_time,
            "which_hit": which_hit,
        }
        _, hit = _record_point(
            problem=problem,
            times=times,
            events=events,
            states=states,
            intensities=intensities,
            scores=scores,
            checkpoints=checkpoints,
            t=t,
            state=state,
            event=-1,
            intensity=intensity,
            hawkes_state={"H": H, "G": G},
            keep_checkpoint=self.keep_checkpoints,
            extra_metadata=extra,
        )

        while t < problem.T and n_events < self.max_events and not hit:
            lam_max = self._upper_bound(state, H, G)
            if lam_max <= 0:
                break
            dt = rng.exponential(1.0 / lam_max)
            t += dt
            n_candidates += 1
            if t > problem.T:
                break
            H *= np.exp(-p.beta * dt)
            if p.a_cross > 0:
                G *= np.exp(-p.b_cross * dt)
            intensity = self._intensity(state, H, G)
            total = float(intensity.sum())
            if total <= 0 or rng.random() > total / lam_max:
                continue
            event = int(rng.choice(8, p=intensity / total))
            self._apply_event(event, state, H, G)
            n_events += 1
            if not first_limit_hit and (state[0] <= 0 or state[1] <= 0):
                first_limit_hit = True
                first_limit_hitting_time = t
                which_hit = 1 if state[0] <= 0 else -1
            if n_events % self.record_every == 0 or self.keep_checkpoints:
                new_intensity = self._intensity(state, H, G)
                extra = {
                    "queue_indices": self.queue_indices,
                    "first_limit_hit": first_limit_hit,
                    "first_limit_hitting_time": first_limit_hitting_time,
                    "which_hit": which_hit,
                    "n_events": n_events,
                    "n_candidates": n_candidates,
                }
                _, hit = _record_point(
                    problem=problem,
                    times=times,
                    events=events,
                    states=states,
                    intensities=intensities,
                    scores=scores,
                    checkpoints=checkpoints,
                    t=t,
                    state=state,
                    event=event,
                    intensity=new_intensity,
                    hawkes_state={"H": H, "G": G},
                    keep_checkpoint=self.keep_checkpoints,
                    extra_metadata=extra,
                )
            else:
                context = _context(
                    problem,
                    self._intensity(state, H, G),
                    {"H": H.copy(), "G": G.copy()},
                    event,
                    {
                        "queue_indices": self.queue_indices,
                        "first_limit_hit": first_limit_hit,
                        "first_limit_hitting_time": first_limit_hitting_time,
                        "which_hit": which_hit,
                    },
                )
                hit = problem.is_target(state, t, context)
                if hit:
                    final_intensity = self._intensity(state, H, G)
                    final_extra = {
                        "queue_indices": self.queue_indices,
                        "first_limit_hit": first_limit_hit,
                        "first_limit_hitting_time": first_limit_hitting_time,
                        "which_hit": which_hit,
                        "n_events": n_events,
                        "n_candidates": n_candidates,
                    }
                    _record_point(
                        problem=problem,
                        times=times,
                        events=events,
                        states=states,
                        intensities=intensities,
                        scores=scores,
                        checkpoints=checkpoints,
                        t=t,
                        state=state,
                        event=event,
                        intensity=final_intensity,
                        hawkes_state={"H": H, "G": G},
                        keep_checkpoint=self.keep_checkpoints,
                        extra_metadata=final_extra,
                    )

        metadata = {
            "model": "four_queue_hawkes",
            "n_events": n_events,
            "n_candidates": n_candidates,
            "first_limit_hit": first_limit_hit,
            "first_limit_hitting_time": first_limit_hitting_time,
            "which_hit": which_hit,
            "queue_indices": self.queue_indices,
        }
        if states and not np.array_equal(states[-1], state):
            final_intensity = self._intensity(state, H, G)
            final_score = problem.score(
                state,
                min(t, problem.T),
                _context(problem, final_intensity, {"H": H.copy(), "G": G.copy()}, -2, metadata),
            )
            times.append(min(t, problem.T))
            events.append(-2)
            states.append(state.copy())
            intensities.append(final_intensity)
            scores.append(final_score)
        return _build_trajectory(
            times=times,
            events=events,
            states=states,
            intensities=intensities,
            scores=scores,
            hit=hit,
            hitting_time=t if hit else None,
            metadata=metadata,
            checkpoints=checkpoints,
        )
