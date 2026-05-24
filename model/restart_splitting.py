"""Markovian Conditional Restart Splitting for LOB rare events.

This module implements the professor-requested restart method.  Ogata thinning
remains the low-level path simulator; the method here only collects augmented
Markov checkpoints ``(N, S)`` near a boundary and restarts local simulations
from the empirical conditional law of those checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .ogata import Checkpoint
from .rare_events import RareEventProblem, queue_position
from .utils import RNGStream, Timer, binomial_standard_error, ensure_rng


METHOD_NAME = "Markovian Conditional Restart Splitting"
NAIVE_METHOD_NAME = "Naive Ogata Monte Carlo"

QueueCondition = Callable[[np.ndarray, float, dict[str, Any]], bool]
ObservableFunction = Callable[[np.ndarray, float, dict[str, Any]], dict[str, Any]]


@dataclass
class MarkovState:
    """Augmented Markov state ``X=(N,S)`` used for restart splitting."""

    t: float
    queues: np.ndarray
    excitation: np.ndarray
    intensity: Optional[np.ndarray]
    hawkes_state: dict[str, np.ndarray | float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "MarkovState":
        return MarkovState(
            t=float(self.t),
            queues=np.asarray(self.queues, dtype=float).copy(),
            excitation=np.asarray(self.excitation, dtype=float).copy(),
            intensity=None if self.intensity is None else np.asarray(self.intensity, dtype=float).copy(),
            hawkes_state=_copy_hawkes_state(self.hawkes_state),
            metadata=dict(self.metadata),
        )


@dataclass
class BoundaryCheckpoint:
    """A full Markov checkpoint observed at a near-boundary queue level."""

    state: MarkovState
    boundary_name: str
    boundary_level: int
    queue_label: str
    t_hit: float
    observable: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "BoundaryCheckpoint":
        return BoundaryCheckpoint(
            state=self.state.copy(),
            boundary_name=str(self.boundary_name),
            boundary_level=int(self.boundary_level),
            queue_label=str(self.queue_label),
            t_hit=float(self.t_hit),
            observable=dict(self.observable),
        )


@dataclass
class BoundarySample:
    """Empirical boundary sample approximating ``Law(S | boundary)``."""

    checkpoints: list[BoundaryCheckpoint]
    S_samples: np.ndarray
    queue_samples: np.ndarray
    times: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RestartSplittingResult:
    """Result of Markovian Conditional Restart Splitting."""

    method_name: str
    probability_estimate: float
    standard_error: Optional[float]
    confidence_interval: Optional[tuple[float, float]]
    n_restarts: int
    n_successes: int
    hitting_times: np.ndarray
    observables: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultilevelRestartSplittingResult:
    """Result of multilevel Markovian Conditional Restart Splitting."""

    method_name: str
    probability_estimate: float
    log_probability_estimate: float
    level_probabilities: list[float]
    levels: list[int]
    n_particles: int
    n_survivors_per_level: list[int]
    final_checkpoints: list[BoundaryCheckpoint]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def default_hawkes_burn_in(simulator: Any, multiplier: float = 10.0) -> float:
    """Return a conservative Hawkes burn-in horizon for exponential kernels.

    The default is ``multiplier / (beta * (1 - alpha / beta))``.  For the
    four-queue parameters used in nb3, ``alpha=0.3`` and ``beta=0.5`` gives
    ``50``.  Pure Poisson simulators return ``0``.
    """

    params = getattr(simulator, "params", simulator)
    alpha = getattr(params, "alpha", None)
    beta = getattr(params, "beta", None)
    if alpha is None or beta is None:
        return 0.0
    alpha = float(alpha)
    beta = float(beta)
    if beta <= 0:
        raise ValueError("beta must be positive to compute Hawkes burn-in")
    ratio_gap = 1.0 - alpha / beta
    if ratio_gap <= 0:
        raise ValueError("default Hawkes burn-in requires alpha / beta < 1")
    return float(multiplier / (beta * ratio_gap))


def queue_boundary_fn(queue_index: int, boundary_level: int = 1) -> QueueCondition:
    """Return a condition for the first time ``Q_queue_index <= boundary_level``."""

    queue_index = int(queue_index)
    boundary_level = int(boundary_level)

    def condition(state: np.ndarray, time: float, metadata: dict[str, Any]) -> bool:
        del time
        return _queue_value(state, metadata, queue_index) <= boundary_level

    return condition


def local_depletion_target_fn(queue_index: int) -> QueueCondition:
    """Return a local target condition ``Q_queue_index <= 0``."""

    queue_index = int(queue_index)

    def condition(state: np.ndarray, time: float, metadata: dict[str, Any]) -> bool:
        del time
        return _queue_value(state, metadata, queue_index) <= 0

    return condition


def local_recovery_fn(queue_index: int, recovery_level: int = 2) -> QueueCondition:
    """Return a local recovery condition ``Q_queue_index >= recovery_level``."""

    queue_index = int(queue_index)
    recovery_level = int(recovery_level)

    def condition(state: np.ndarray, time: float, metadata: dict[str, Any]) -> bool:
        del time
        return _queue_value(state, metadata, queue_index) >= recovery_level

    return condition


def checkpoint_to_markov_state(
    checkpoint: Checkpoint,
    simulator: Any = None,
    model_name: Optional[str] = None,
) -> MarkovState:
    """Convert an Ogata checkpoint into the augmented Markov state ``(N,S)``."""

    cp = checkpoint.copy()
    excitation, component_names, diagnostics = extract_excitation_vector(cp, simulator=simulator, model_name=model_name)
    metadata = dict(cp.metadata)
    metadata.update(diagnostics)
    metadata["S_component_names"] = component_names
    metadata.setdefault("model_name", diagnostics.get("model_name", model_name))
    return MarkovState(
        t=cp.time,
        queues=cp.state.copy(),
        excitation=excitation.copy(),
        intensity=None if cp.intensity is None else np.asarray(cp.intensity, dtype=float).copy(),
        hawkes_state=_copy_hawkes_state(cp.hawkes_state),
        metadata=metadata,
    )


def markov_state_to_checkpoint(markov_state: MarkovState) -> Checkpoint:
    """Convert a copied Markov state back into a simulator continuation checkpoint."""

    state = markov_state.copy()
    return Checkpoint(
        time=float(state.t),
        state=state.queues.copy(),
        hawkes_state=_copy_hawkes_state(state.hawkes_state),
        intensity=None if state.intensity is None else state.intensity.copy(),
        score=0.0,
        metadata=dict(state.metadata),
    )


def extract_excitation_vector(
    checkpoint: Checkpoint,
    simulator: Any = None,
    model_name: Optional[str] = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Extract the Hawkes excitation vector ``S`` from a checkpoint.

    The checkpoint's ``hawkes_state`` is authoritative.  Reconstructing from
    intensity is only used as a diagnostic fallback because the simulators clip
    some intensities at zero or at ``0.01``.
    """

    state = np.asarray(checkpoint.state, dtype=float)
    hawkes_state = checkpoint.hawkes_state or {}
    model = _infer_model_name(checkpoint, simulator, model_name)
    diagnostics: dict[str, Any] = {
        "model_name": model,
        "S_reconstruction": "hawkes_state",
        "S_reconstruction_warning": None,
    }

    if "G" in hawkes_state:
        H = np.asarray(hawkes_state.get("H", np.zeros(2)), dtype=float)
        G = np.asarray(hawkes_state.get("G", np.zeros(2)), dtype=float)
        excitation = np.array([H[0], H[1], G[0], G[1]], dtype=float)
        component_names = [
            "S^{+1,-}",
            "S^{-1,-}",
            "S^{+1,- -> +2,+}",
            "S^{-1,- -> -2,+}",
        ]
        diagnostics["sign_convention"] = "positive index = ask, negative index = bid"
        diagnostics["queue_state_order"] = [1, -1, 2, -2]
        diagnostics["cross_component_note"] = (
            "G[0] is the ask-side cross excitation Q+1 removal -> Q+2 addition; "
            "G[1] is the bid-side cross excitation Q-1 removal -> Q-2 addition."
        )
        return excitation, component_names, diagnostics

    if "H" in hawkes_state:
        H_raw = hawkes_state["H"]
        H = np.asarray(H_raw, dtype=float)
        if H.ndim == 0:
            return np.array([float(H)], dtype=float), ["S^{+1,-}"], diagnostics
        if H.size == 2:
            return H.astype(float).copy(), ["S^{+1,-}", "S^{-1,-}"], diagnostics
        names = [f"S_{idx}" for idx in range(H.size)]
        return H.astype(float).copy(), names, diagnostics

    fallback = _reconstruct_excitation_from_intensity(checkpoint, simulator, model, state)
    if fallback is not None:
        excitation, component_names = fallback
        diagnostics["S_reconstruction"] = "intensity_minus_baseline"
        diagnostics["S_reconstruction_warning"] = (
            "Excitation was reconstructed from intensity and may be lossy under clipping."
        )
        return excitation, component_names, diagnostics

    return np.empty(0, dtype=float), [], diagnostics


def collect_boundary_states(
    simulator: Any,
    initial_state: Optional[Sequence[float]] = None,
    boundary_fn: Optional[QueueCondition] = None,
    horizon: float = 100.0,
    n_paths: int = 1_000,
    rng: Optional[np.random.Generator | int] = None,
    first_hit_only: bool = True,
    burn_in: Optional[float] = None,
    record_path: bool = False,
    queue_index: Optional[int] = None,
    boundary_level: int = 1,
    boundary_name: Optional[str] = None,
    queue_label: Optional[str] = None,
    queue_indices: Optional[Sequence[int]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> BoundarySample:
    """Collect empirical Markov checkpoints at a near-boundary queue level.

    If ``burn_in`` is ``None``, Hawkes simulators use
    ``10 / (beta * (1 - alpha / beta))`` before the boundary clock starts.
    Pass ``burn_in=0.0`` explicitly to disable this.
    """

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not first_hit_only:
        raise NotImplementedError("collect_boundary_states currently records the first boundary hit only")
    if boundary_fn is None:
        if queue_index is None:
            raise ValueError("Either boundary_fn or queue_index must be provided")
        boundary_fn = queue_boundary_fn(queue_index, boundary_level)

    initial = _infer_initial_state(simulator, initial_state)
    resolved_burn_in = _resolve_burn_in(simulator, burn_in)
    indices = list(queue_indices) if queue_indices is not None else _default_queue_indices(initial, queue_index)
    q_index = int(queue_index) if queue_index is not None else int(indices[0])
    q_label = queue_label or _queue_label(q_index)
    b_name = boundary_name or f"{q_label}={boundary_level}"
    base_metadata = dict(metadata or {})
    base_metadata.update(
        {
            "queue_indices": indices,
            "queue_index": q_index,
            "boundary_level": int(boundary_level),
            "boundary_name": b_name,
            "queue_label": q_label,
        }
    )

    def target(state: np.ndarray, time: float, ctx: dict[str, Any]) -> bool:
        return boundary_fn(state, time, ctx)

    def score(state: np.ndarray, time: float, ctx: dict[str, Any]) -> float:
        del time
        if queue_index is None:
            return 0.0
        q0 = max(float(base_metadata.get("initial_queue", _queue_value(initial, base_metadata, q_index))), 1.0)
        q = _queue_value(state, ctx, q_index)
        return float(np.clip((q0 - q) / q0, 0.0, 1.0))

    problem = RareEventProblem(
        T=float(horizon),
        initial_state=np.asarray(initial, dtype=float),
        target_event=target,
        score_function=score,
        event_name=f"boundary_{b_name}",
        threshold=1.0,
        metadata=base_metadata,
    )

    stream = _make_rng_stream(rng)
    checkpoints: list[BoundaryCheckpoint] = []
    n_events = 0
    n_candidates = 0
    warnings: list[str] = []
    model_name = _infer_model_name_from_simulator(simulator)

    with Timer() as timer:
        for _ in range(int(n_paths)):
            trajectory = simulator.simulate(problem, rng=stream.next(), burn_in=resolved_burn_in, record_path=record_path)
            n_events += trajectory.n_events
            n_candidates += trajectory.n_candidates
            if not trajectory.hit or not trajectory.checkpoints:
                continue
            cp = trajectory.checkpoints[-1].copy()
            markov_state = checkpoint_to_markov_state(cp, simulator=simulator, model_name=model_name)
            observable = {
                "final_queue_state": markov_state.queues.copy(),
                "n_events": trajectory.n_events,
                "n_candidates": trajectory.n_candidates,
            }
            warning = markov_state.metadata.get("S_reconstruction_warning")
            if warning:
                warnings.append(str(warning))
            checkpoints.append(
                BoundaryCheckpoint(
                    state=markov_state,
                    boundary_name=b_name,
                    boundary_level=int(boundary_level),
                    queue_label=q_label,
                    t_hit=markov_state.t,
                    observable=observable,
                )
            )

    if checkpoints:
        S_samples = _stack_or_empty([cp.state.excitation for cp in checkpoints])
        queue_samples = np.vstack([cp.state.queues for cp in checkpoints])
        times = np.asarray([cp.t_hit for cp in checkpoints], dtype=float)
        component_names = list(checkpoints[0].state.metadata.get("S_component_names", []))
    else:
        S_samples = np.empty((0, 0), dtype=float)
        queue_samples = np.empty((0, len(initial)), dtype=float)
        times = np.empty(0, dtype=float)
        component_names = []

    return BoundarySample(
        checkpoints=checkpoints,
        S_samples=S_samples,
        queue_samples=queue_samples,
        times=times,
        metadata={
            **base_metadata,
            "method_name": METHOD_NAME,
            "n_paths": int(n_paths),
            "n_boundary_hits": len(checkpoints),
            "boundary_hit_rate": len(checkpoints) / int(n_paths),
            "burn_in": float(resolved_burn_in),
            "burn_in_default_used": burn_in is None,
            "horizon": float(horizon),
            "cpu_seconds": timer.elapsed,
            "n_events": int(n_events),
            "n_candidates": int(n_candidates),
            "S_component_names": component_names,
            "warnings": sorted(set(warnings)),
            "model_name": model_name,
        },
    )


def restart_from_boundary_distribution(
    checkpoints: Sequence[BoundaryCheckpoint] | BoundarySample,
    simulator: Any,
    local_target_fn: QueueCondition,
    recovery_fn: QueueCondition,
    horizon_local: float,
    n_restarts: int,
    rng: Optional[np.random.Generator | int] = None,
    sample_with_replacement: bool = True,
    record_path: bool = False,
    observable_fn: Optional[ObservableFunction] = None,
    reset_excitation: bool = False,
    method_name: str = METHOD_NAME,
) -> RestartSplittingResult:
    """Restart local simulations from empirical boundary Markov states."""

    if isinstance(checkpoints, BoundarySample):
        checkpoint_list = checkpoints.checkpoints
        sample_metadata = dict(checkpoints.metadata)
    else:
        checkpoint_list = list(checkpoints)
        sample_metadata = {}
    if not checkpoint_list:
        raise ValueError("At least one boundary checkpoint is required")
    if n_restarts <= 0:
        raise ValueError("n_restarts must be positive")
    if horizon_local <= 0:
        raise ValueError("horizon_local must be positive")
    if not sample_with_replacement and n_restarts > len(checkpoint_list):
        raise ValueError("n_restarts cannot exceed checkpoint count without replacement")

    rng_main = ensure_rng(rng)
    stream = _make_rng_stream(rng_main)
    if sample_with_replacement:
        selected = rng_main.integers(0, len(checkpoint_list), size=int(n_restarts))
    else:
        selected = rng_main.choice(len(checkpoint_list), size=int(n_restarts), replace=False)

    successes: list[bool] = []
    recoveries: list[bool] = []
    timeouts: list[bool] = []
    outcomes: list[str] = []
    local_times: list[float] = []
    hitting_times: list[float] = []
    final_queues: list[np.ndarray] = []
    start_queues: list[np.ndarray] = []
    start_S: list[np.ndarray] = []
    end_S: list[np.ndarray] = []
    q_neg2_values: list[float] = []
    q_neg2_success_values: list[float] = []
    user_observables: list[dict[str, Any]] = []
    total_events = 0
    total_candidates = 0

    with Timer() as timer:
        for idx in selected:
            boundary_checkpoint = checkpoint_list[int(idx)].copy()
            state = boundary_checkpoint.state.copy()
            if reset_excitation:
                state = _zero_excitation_state(state)
            start_checkpoint = markov_state_to_checkpoint(state)
            start_meta = dict(state.metadata)
            start_meta.setdefault("queue_indices", sample_metadata.get("queue_indices"))
            start_t = float(state.t)
            local_T = start_t + float(horizon_local)

            def target_or_recovery(x: np.ndarray, time: float, ctx: dict[str, Any]) -> bool:
                return local_target_fn(x, time, ctx) or recovery_fn(x, time, ctx)

            def score(x: np.ndarray, time: float, ctx: dict[str, Any]) -> float:
                del time
                return 1.0 if local_target_fn(x, start_t, ctx) else 0.0

            problem = RareEventProblem(
                T=local_T,
                initial_state=state.queues.copy(),
                target_event=target_or_recovery,
                score_function=score,
                event_name="local_restart_depletion_or_recovery",
                threshold=1.0,
                metadata=start_meta,
            )
            trajectory = simulator.continue_from_checkpoint(
                start_checkpoint,
                problem,
                rng=stream.next(),
                record_path=record_path,
            )
            final_state = np.asarray(trajectory.final_state, dtype=float)
            final_time = float(trajectory.times[-1]) if len(trajectory.times) else local_T
            context = dict(start_meta)
            if trajectory.intensities is not None and len(trajectory.intensities):
                context["intensity"] = trajectory.intensities[-1]
            if trajectory.checkpoints:
                context["hawkes_state"] = trajectory.checkpoints[-1].hawkes_state
            success = bool(local_target_fn(final_state, final_time, context))
            recovery = bool(recovery_fn(final_state, final_time, context)) and not success
            timeout = not success and not recovery
            elapsed_local = max(0.0, final_time - start_t)

            successes.append(success)
            recoveries.append(recovery)
            timeouts.append(timeout)
            outcomes.append("success" if success else ("recovery" if recovery else "timeout"))
            local_times.append(elapsed_local)
            if success:
                hitting_times.append(elapsed_local)
            final_queues.append(final_state.copy())
            start_queues.append(state.queues.copy())
            start_S.append(state.excitation.copy())
            end_state = (
                checkpoint_to_markov_state(trajectory.checkpoints[-1], simulator=simulator)
                if trajectory.checkpoints
                else state
            )
            end_S.append(end_state.excitation.copy())
            total_events += trajectory.n_events
            total_candidates += trajectory.n_candidates
            if len(final_state) >= 4:
                q_neg2 = float(final_state[3])
                q_neg2_values.append(q_neg2)
                if success:
                    q_neg2_success_values.append(q_neg2)
            if observable_fn is not None:
                user_observables.append(observable_fn(final_state, final_time, context))

    success_arr = np.asarray(successes, dtype=bool)
    recovery_arr = np.asarray(recoveries, dtype=bool)
    timeout_arr = np.asarray(timeouts, dtype=bool)
    selected_arr = np.asarray(selected, dtype=int)
    usage_ids, usage_counts = np.unique(selected_arr, return_counts=True)
    usage_frequency = {int(idx): int(count) for idx, count in zip(usage_ids, usage_counts)}
    usage_ess = _usage_effective_sample_size(usage_counts)
    p_hat = float(success_arr.mean()) if len(success_arr) else np.nan
    se = binomial_standard_error(p_hat, len(success_arr)) if len(success_arr) else np.nan
    ci = _normal_confidence_interval(p_hat, se)
    observables: dict[str, Any] = {
        "success": success_arr,
        "recovery": recovery_arr,
        "timeout": timeout_arr,
        "outcome": np.asarray(outcomes, dtype=object),
        "local_times": np.asarray(local_times, dtype=float),
        "final_queues": np.vstack(final_queues) if final_queues else np.empty((0, 0)),
        "start_queues": np.vstack(start_queues) if start_queues else np.empty((0, 0)),
        "start_S": _stack_or_empty(start_S),
        "end_S": _stack_or_empty(end_S),
        "sampled_checkpoint_indices": selected_arr,
        "checkpoint_usage_frequency": usage_frequency,
        "user_observables": user_observables,
    }
    if q_neg2_values:
        observables["q_neg2"] = np.asarray(q_neg2_values, dtype=float)
        observables["q_neg2_success"] = np.asarray(q_neg2_success_values, dtype=float)

    return RestartSplittingResult(
        method_name=method_name,
        probability_estimate=p_hat,
        standard_error=se,
        confidence_interval=ci,
        n_restarts=int(n_restarts),
        n_successes=int(success_arr.sum()),
        hitting_times=np.asarray(hitting_times, dtype=float),
        observables=observables,
        diagnostics={
            "method_name": method_name,
            "cpu_seconds": timer.elapsed,
            "n_boundary_checkpoints": len(checkpoint_list),
            "sample_with_replacement": bool(sample_with_replacement),
            "reset_excitation": bool(reset_excitation),
            "horizon_local": float(horizon_local),
            "n_events": int(total_events),
            "n_candidates": int(total_candidates),
            "acceptance_candidate_ratio": (
                float(total_events / total_candidates) if total_candidates > 0 else np.nan
            ),
            "success_probability": p_hat,
            "recovery_probability": float(recovery_arr.mean()) if len(recovery_arr) else np.nan,
            "timeout_probability": float(timeout_arr.mean()) if len(timeout_arr) else np.nan,
            "average_local_simulation_time": (
                float(np.mean(local_times)) if local_times else np.nan
            ),
            "unique_checkpoint_usage_frequency": usage_frequency,
            "checkpoint_usage_effective_sample_size": usage_ess,
            "checkpoint_usage_ess": usage_ess,
            "effective_sample_size": usage_ess,
            "boundary_metadata": sample_metadata,
        },
    )


def run_naive_boundary_mc(
    simulator: Any,
    initial_state: Sequence[float],
    queue_index: int,
    horizon: float,
    n_paths: int,
    horizon_local: float,
    rng: Optional[np.random.Generator | int] = None,
    boundary_level: int = 1,
    recovery_level: int = 2,
    burn_in: Optional[float] = None,
    queue_indices: Optional[Sequence[int]] = None,
) -> RestartSplittingResult:
    """Naive full-path boundary sampling with one local continuation per hit."""

    stream = _make_rng_stream(rng)
    sample = collect_boundary_states(
        simulator=simulator,
        initial_state=initial_state,
        horizon=horizon,
        n_paths=n_paths,
        rng=stream.next(),
        burn_in=burn_in,
        queue_index=queue_index,
        boundary_level=boundary_level,
        queue_indices=queue_indices,
    )
    if not sample.checkpoints:
        return RestartSplittingResult(
            method_name=NAIVE_METHOD_NAME,
            probability_estimate=0.0,
            standard_error=np.nan,
            confidence_interval=(0.0, 0.0),
            n_restarts=0,
            n_successes=0,
            hitting_times=np.empty(0),
            observables={"success": np.empty(0, dtype=bool)},
            diagnostics={
                "n_paths": int(n_paths),
                "n_boundary_checkpoints": 0,
                "boundary_metadata": sample.metadata,
                "warning": "No boundary checkpoint collected.",
            },
        )
    return restart_from_boundary_distribution(
        checkpoints=sample,
        simulator=simulator,
        local_target_fn=local_depletion_target_fn(queue_index),
        recovery_fn=local_recovery_fn(queue_index, recovery_level),
        horizon_local=horizon_local,
        n_restarts=len(sample.checkpoints),
        rng=stream.next(),
        sample_with_replacement=False,
        method_name=NAIVE_METHOD_NAME,
    )


def run_naive_depletion_monte_carlo(
    simulator: Any,
    initial_state: Optional[Sequence[float]],
    queue_index: int,
    horizon: float,
    n_paths: int,
    rng: Optional[np.random.Generator | int] = None,
    burn_in: Optional[float] = None,
    queue_indices: Optional[Sequence[int]] = None,
    record_path: bool = False,
) -> RestartSplittingResult:
    """Naive Ogata Monte Carlo estimate of first-limit depletion by horizon."""

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    stream = _make_rng_stream(rng)
    initial = _infer_initial_state(simulator, initial_state)
    indices = list(queue_indices) if queue_indices is not None else _default_queue_indices(initial, queue_index)
    q_index = int(queue_index)
    resolved_burn_in = _resolve_burn_in(simulator, burn_in)

    def target(state: np.ndarray, time: float, ctx: dict[str, Any]) -> bool:
        del time
        return _queue_value(state, ctx, q_index) <= 0

    def score(state: np.ndarray, time: float, ctx: dict[str, Any]) -> float:
        del time
        q0 = max(float(_queue_value(initial, {"queue_indices": indices}, q_index)), 1.0)
        q = _queue_value(state, ctx, q_index)
        return float(np.clip((q0 - q) / q0, 0.0, 1.0))

    problem = RareEventProblem(
        T=float(horizon),
        initial_state=np.asarray(initial, dtype=float),
        target_event=target,
        score_function=score,
        event_name=f"naive_depletion_{_queue_label(q_index)}",
        threshold=1.0,
        metadata={"queue_indices": indices, "queue_index": q_index},
    )

    successes: list[bool] = []
    hitting_times: list[float] = []
    final_queues: list[np.ndarray] = []
    q_neg2_success: list[float] = []
    n_events = 0
    n_candidates = 0

    with Timer() as timer:
        for _ in range(int(n_paths)):
            trajectory = simulator.simulate(
                problem,
                rng=stream.next(),
                burn_in=resolved_burn_in,
                record_path=record_path,
            )
            hit = bool(trajectory.hit)
            successes.append(hit)
            final_state = np.asarray(trajectory.final_state, dtype=float)
            final_queues.append(final_state.copy())
            if hit and trajectory.hitting_time is not None:
                hitting_times.append(float(trajectory.hitting_time))
            if hit and len(final_state) >= 4:
                q_neg2_success.append(float(final_state[3]))
            n_events += trajectory.n_events
            n_candidates += trajectory.n_candidates

    success_arr = np.asarray(successes, dtype=bool)
    p_hat = float(success_arr.mean())
    se = binomial_standard_error(p_hat, int(n_paths))
    return RestartSplittingResult(
        method_name=NAIVE_METHOD_NAME,
        probability_estimate=p_hat,
        standard_error=se,
        confidence_interval=_normal_confidence_interval(p_hat, se),
        n_restarts=int(n_paths),
        n_successes=int(success_arr.sum()),
        hitting_times=np.asarray(hitting_times, dtype=float),
        observables={
            "success": success_arr,
            "timeout": ~success_arr,
            "recovery": np.zeros_like(success_arr, dtype=bool),
            "final_queues": np.vstack(final_queues) if final_queues else np.empty((0, 0)),
            "q_neg2_success": np.asarray(q_neg2_success, dtype=float),
        },
        diagnostics={
            "method_name": NAIVE_METHOD_NAME,
            "cpu_seconds": timer.elapsed,
            "n_paths": int(n_paths),
            "burn_in": float(resolved_burn_in),
            "burn_in_default_used": burn_in is None,
            "n_events": int(n_events),
            "n_candidates": int(n_candidates),
            "acceptance_candidate_ratio": float(n_events / n_candidates) if n_candidates > 0 else np.nan,
            "effective_sample_size": int(n_paths),
        },
    )


def summarize_conditional_S(
    S_samples: np.ndarray,
    component_names: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Summarize the empirical conditional distribution of ``S`` at boundary."""

    S = np.asarray(S_samples, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    n, dim = S.shape if S.size else (S.shape[0], S.shape[1] if S.ndim == 2 else 0)
    names = list(component_names or [f"S_{i}" for i in range(dim)])
    if dim == 0:
        return {
            "n_samples": int(n),
            "component_names": names,
            "mean": np.empty(0),
            "std": np.empty(0),
            "covariance": np.empty((0, 0)),
            "correlation": np.empty((0, 0)),
            "quantiles": {},
        }
    if n == 0:
        mean = np.full(dim, np.nan)
        std = np.full(dim, np.nan)
        covariance = np.full((dim, dim), np.nan)
        correlation = np.full((dim, dim), np.nan)
        quantiles = {q: np.full(dim, np.nan) for q in [0.05, 0.25, 0.50, 0.75, 0.95]}
    else:
        mean = S.mean(axis=0)
        std = S.std(axis=0, ddof=1) if n > 1 else np.zeros(dim)
        covariance = np.cov(S, rowvar=False) if n > 1 else np.zeros((dim, dim))
        covariance = np.atleast_2d(covariance)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = covariance / np.outer(np.sqrt(np.diag(covariance)), np.sqrt(np.diag(covariance)))
        quantiles = {q: np.quantile(S, q, axis=0) for q in [0.05, 0.25, 0.50, 0.75, 0.95]}
    return {
        "n_samples": int(n),
        "component_names": names,
        "mean": mean,
        "std": std,
        "covariance": covariance,
        "correlation": correlation,
        "quantiles": quantiles,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator | int] = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a sample mean."""

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng_obj = ensure_rng(rng)
    draws = rng_obj.choice(x, size=(int(n_bootstrap), len(x)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1.0 - alpha)),
    )


def bootstrap_difference_ci(
    left: Sequence[float],
    right: Sequence[float],
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator | int] = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``mean(left) - mean(right)``."""

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return (np.nan, np.nan)
    rng_obj = ensure_rng(rng)
    x_draws = rng_obj.choice(x, size=(int(n_bootstrap), len(x)), replace=True).mean(axis=1)
    y_draws = rng_obj.choice(y, size=(int(n_bootstrap), len(y)), replace=True).mean(axis=1)
    diff = x_draws - y_draws
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(diff, alpha)),
        float(np.quantile(diff, 1.0 - alpha)),
    )


def multilevel_markovian_restart_splitting(
    simulator: Any,
    initial_state: Optional[Sequence[float]],
    queue_index: int,
    levels: Sequence[int],
    horizon: float,
    n_particles: int,
    rng: Optional[np.random.Generator | int] = None,
    burn_in: Optional[float] = None,
    queue_indices: Optional[Sequence[int]] = None,
    record_path: bool = False,
) -> MultilevelRestartSplittingResult:
    """Propagate particles through decreasing queue levels with MCRS.

    ``levels`` should be decreasing, for example ``[8, 6, 4, 2, 1, 0]`` for
    an initial queue near ``10``.  At each transition, particles that first hit
    the next level before the horizon survive; survivors are resampled with
    replacement and continued from their full Markov checkpoints.
    """

    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    level_list = [int(level) for level in levels]
    if len(level_list) < 2:
        raise ValueError("levels must contain at least two levels")
    if any(level_list[i] <= level_list[i + 1] for i in range(len(level_list) - 1)):
        raise ValueError("levels must be strictly decreasing")

    stream = _make_rng_stream(rng)
    resolved_burn_in = _resolve_burn_in(simulator, burn_in)
    q_index = int(queue_index)
    initial = _infer_initial_state(simulator, initial_state)
    indices = list(queue_indices) if queue_indices is not None else _default_queue_indices(initial, q_index)

    with Timer() as timer:
        first_sample = collect_boundary_states(
            simulator=simulator,
            initial_state=initial,
            horizon=horizon,
            n_paths=n_particles,
            rng=stream.next(),
            burn_in=resolved_burn_in,
            queue_index=q_index,
            boundary_level=level_list[0],
            queue_indices=indices,
            boundary_name=f"{_queue_label(q_index)}<={level_list[0]}",
            queue_label=_queue_label(q_index),
        )
        current = [checkpoint.copy() for checkpoint in first_sample.checkpoints]
        level_probabilities = [len(current) / float(n_particles)]
        n_survivors = [len(current)]
        n_events = int(first_sample.metadata.get("n_events", 0))
        n_candidates = int(first_sample.metadata.get("n_candidates", 0))
        resampling_usage: list[dict[int, int]] = []

        if not current:
            return MultilevelRestartSplittingResult(
                method_name="Multilevel Markovian Conditional Restart Splitting",
                probability_estimate=0.0,
                log_probability_estimate=-np.inf,
                level_probabilities=level_probabilities,
                levels=level_list,
                n_particles=int(n_particles),
                n_survivors_per_level=n_survivors,
                final_checkpoints=[],
                diagnostics={
                    "cpu_seconds": timer.elapsed,
                    "burn_in": float(resolved_burn_in),
                    "burn_in_default_used": burn_in is None,
                    "n_events": n_events,
                    "n_candidates": n_candidates,
                    "warning": "No particle reached the first level.",
                },
            )

        current, usage = _resample_boundary_checkpoints(current, n_particles, stream.next())
        resampling_usage.append(usage)

        for next_level in level_list[1:]:
            survivors: list[BoundaryCheckpoint] = []
            for parent in current:
                checkpoint = markov_state_to_checkpoint(parent.state)

                def target(state: np.ndarray, time: float, ctx: dict[str, Any]) -> bool:
                    del time
                    return _queue_value(state, ctx, q_index) <= next_level

                def score(state: np.ndarray, time: float, ctx: dict[str, Any]) -> float:
                    del time
                    q0 = max(float(level_list[0]), 1.0)
                    q = _queue_value(state, ctx, q_index)
                    return float(np.clip((q0 - q) / q0, 0.0, 1.0))

                problem = RareEventProblem(
                    T=float(horizon),
                    initial_state=parent.state.queues.copy(),
                    target_event=target,
                    score_function=score,
                    event_name=f"hit_{_queue_label(q_index)}_{next_level}",
                    threshold=1.0,
                    metadata={
                        **parent.state.metadata,
                        "queue_indices": indices,
                        "queue_index": q_index,
                        "boundary_level": next_level,
                    },
                )
                trajectory = simulator.continue_from_checkpoint(
                    checkpoint,
                    problem,
                    rng=stream.next(),
                    record_path=record_path,
                )
                n_events += trajectory.n_events
                n_candidates += trajectory.n_candidates
                if trajectory.hit and trajectory.checkpoints:
                    cp = trajectory.checkpoints[-1].copy()
                    markov_state = checkpoint_to_markov_state(cp, simulator=simulator)
                    survivors.append(
                        BoundaryCheckpoint(
                            state=markov_state,
                            boundary_name=f"{_queue_label(q_index)}<={next_level}",
                            boundary_level=int(next_level),
                            queue_label=_queue_label(q_index),
                            t_hit=markov_state.t,
                            observable={
                                "n_events": trajectory.n_events,
                                "n_candidates": trajectory.n_candidates,
                                "final_queue_state": markov_state.queues.copy(),
                            },
                        )
                    )

            level_probabilities.append(len(survivors) / float(n_particles))
            n_survivors.append(len(survivors))
            if not survivors:
                current = []
                break
            current, usage = _resample_boundary_checkpoints(survivors, n_particles, stream.next())
            resampling_usage.append(usage)

    log_probability = (
        float(np.sum(np.log(level_probabilities)))
        if all(prob > 0 for prob in level_probabilities)
        else -np.inf
    )
    probability = float(np.exp(log_probability)) if np.isfinite(log_probability) else 0.0
    log_variance_terms = [
        (1.0 - prob) / (float(n_particles) * prob)
        for prob in level_probabilities
        if prob > 0
    ]
    probability_se = (
        float(probability * np.sqrt(np.sum(log_variance_terms)))
        if probability > 0 and log_variance_terms
        else np.nan
    )
    return MultilevelRestartSplittingResult(
        method_name="Multilevel Markovian Conditional Restart Splitting",
        probability_estimate=probability,
        log_probability_estimate=log_probability,
        level_probabilities=[float(prob) for prob in level_probabilities],
        levels=level_list,
        n_particles=int(n_particles),
        n_survivors_per_level=[int(count) for count in n_survivors],
        final_checkpoints=current,
        diagnostics={
            "cpu_seconds": timer.elapsed,
            "burn_in": float(resolved_burn_in),
            "burn_in_default_used": burn_in is None,
            "n_events": int(n_events),
            "n_candidates": int(n_candidates),
            "acceptance_candidate_ratio": float(n_events / n_candidates) if n_candidates > 0 else np.nan,
            "probability_se_delta": probability_se,
            "confidence_interval_delta": _normal_confidence_interval(probability, probability_se),
            "resampling_usage": resampling_usage,
            "effective_sample_size": (
                min(
                    _usage_effective_sample_size(np.asarray(list(usage.values()), dtype=float))
                    for usage in resampling_usage
                )
                if resampling_usage
                else np.nan
            ),
            "checkpoint_usage_ess": (
                min(
                    _usage_effective_sample_size(np.asarray(list(usage.values()), dtype=float))
                    for usage in resampling_usage
                )
                if resampling_usage
                else np.nan
            ),
            "effective_sample_size_by_level": [
                _usage_effective_sample_size(np.asarray(list(usage.values()), dtype=float))
                for usage in resampling_usage
            ],
        },
    )


def plot_conditional_S_marginals(
    S_samples: np.ndarray,
    component_names: Sequence[str],
    bins: int = 50,
):
    """Plot marginal histograms for the conditional excitation sample."""

    import matplotlib.pyplot as plt

    S = np.asarray(S_samples, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    dim = S.shape[1] if S.ndim == 2 else 0
    if dim == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No Hawkes excitation S", ha="center", va="center")
        ax.set_axis_off()
        return fig, np.asarray([ax])
    fig, axes = plt.subplots(dim, 1, figsize=(8, max(2.2, 2.2 * dim)), squeeze=False)
    for j in range(dim):
        ax = axes[j, 0]
        ax.hist(S[:, j], bins=bins, alpha=0.75)
        ax.set_title(component_names[j] if j < len(component_names) else f"S_{j}")
        ax.set_ylabel("count")
    axes[-1, 0].set_xlabel("excitation value")
    fig.tight_layout()
    return fig, axes.ravel()


def plot_conditional_S_pairwise(
    S_samples: np.ndarray,
    component_names: Sequence[str],
    pairs: Optional[Sequence[tuple[int, int]]] = None,
    bins: int = 50,
):
    """Plot pairwise conditional excitation clouds or 2D histograms."""

    import matplotlib.pyplot as plt

    S = np.asarray(S_samples, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    dim = S.shape[1] if S.ndim == 2 else 0
    if dim < 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Need at least two S components", ha="center", va="center")
        ax.set_axis_off()
        return fig, np.asarray([ax])
    if pairs is None:
        pairs = [(0, j) for j in range(1, dim)]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(7, max(3, 3 * len(pairs))), squeeze=False)
    for ax, (i, j) in zip(axes.ravel(), pairs):
        ax.hist2d(S[:, i], S[:, j], bins=bins)
        ax.set_xlabel(component_names[i] if i < len(component_names) else f"S_{i}")
        ax.set_ylabel(component_names[j] if j < len(component_names) else f"S_{j}")
    fig.tight_layout()
    return fig, axes.ravel()


def plot_conditional_S_heatmap(
    S_samples: np.ndarray,
    x_component: int,
    y_component: int,
    bins: int = 50,
    component_names: Optional[Sequence[str]] = None,
):
    """Plot a two-dimensional heatmap for two components of ``S``."""

    import matplotlib.pyplot as plt

    S = np.asarray(S_samples, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if S.shape[1] <= max(x_component, y_component):
        raise ValueError("Requested S component is not available")
    names = list(component_names or [f"S_{i}" for i in range(S.shape[1])])
    fig, ax = plt.subplots(figsize=(6, 5))
    hist = ax.hist2d(S[:, x_component], S[:, y_component], bins=bins)
    ax.set_xlabel(names[x_component])
    ax.set_ylabel(names[y_component])
    fig.colorbar(hist[3], ax=ax, label="count")
    fig.tight_layout()
    return fig, ax


def _copy_hawkes_state(hawkes_state: dict[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    copied: dict[str, np.ndarray | float] = {}
    for key, value in hawkes_state.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = float(value)
    return copied


def _infer_model_name(checkpoint: Checkpoint, simulator: Any = None, model_name: Optional[str] = None) -> str:
    if model_name:
        return str(model_name)
    if simulator is not None:
        return _infer_model_name_from_simulator(simulator)
    if "G" in checkpoint.hawkes_state:
        return "four_queue_hawkes"
    H = checkpoint.hawkes_state.get("H")
    if isinstance(H, np.ndarray) and H.size == 2:
        return "coupled_hawkes"
    if H is not None:
        return "single_hawkes"
    return "independent_poisson"


def _infer_model_name_from_simulator(simulator: Any) -> str:
    name = simulator.__class__.__name__.lower()
    if "fourqueue" in name:
        return "four_queue_hawkes"
    if "coupled" in name:
        return "coupled_hawkes"
    if "single" in name:
        return "single_hawkes"
    if "poisson" in name:
        return "independent_poisson"
    return name


def _reconstruct_excitation_from_intensity(
    checkpoint: Checkpoint,
    simulator: Any,
    model: str,
    state: np.ndarray,
) -> Optional[tuple[np.ndarray, list[str]]]:
    del state
    if checkpoint.intensity is None or simulator is None:
        return None
    intensity = np.asarray(checkpoint.intensity, dtype=float)
    if model == "single_hawkes" and hasattr(simulator, "mu_minus") and len(intensity) >= 2:
        return np.array([intensity[1] - simulator.mu_minus]), ["S^{+1,-}"]
    if model == "coupled_hawkes" and hasattr(simulator, "mu_minus") and len(intensity) >= 4:
        return np.array([intensity[1] - simulator.mu_minus, intensity[3] - simulator.mu_minus]), [
            "S^{+1,-}",
            "S^{-1,-}",
        ]
    params = getattr(simulator, "params", None)
    if model == "four_queue_hawkes" and params is not None and len(intensity) >= 8:
        return (
            np.array(
                [
                    intensity[1] - params.mu_minus_1,
                    intensity[3] - params.mu_minus_1,
                    intensity[4] - params.mu_plus_2,
                    intensity[6] - params.mu_plus_2,
                ],
                dtype=float,
            ),
            ["S^{+1,-}", "S^{-1,-}", "S^{+1,- -> +2,+}", "S^{-1,- -> -2,+}"],
        )
    return None


def _infer_initial_state(simulator: Any, initial_state: Optional[Sequence[float]]) -> np.ndarray:
    if initial_state is not None:
        return np.asarray(initial_state, dtype=float)
    params = getattr(simulator, "params", None)
    if params is not None and all(hasattr(params, attr) for attr in ["q1_init", "q_neg1_init", "q2_init", "q_neg2_init"]):
        return np.array([params.q1_init, params.q_neg1_init, params.q2_init, params.q_neg2_init], dtype=float)
    raise ValueError("initial_state is required for this simulator")


def _default_queue_indices(initial_state: np.ndarray, queue_index: Optional[int]) -> list[int]:
    n = len(initial_state)
    if n == 1:
        return [int(queue_index) if queue_index is not None else 1]
    if n == 2:
        return [1, -1]
    if n == 4:
        return [1, -1, 2, -2]
    return list(range(n))


def _queue_value(state: np.ndarray, metadata: dict[str, Any], queue_index: int) -> float:
    indices = metadata.get("queue_indices")
    if indices is None:
        if len(state) == 1:
            pos = 0
        else:
            raise KeyError("metadata must contain queue_indices")
    else:
        pos = queue_position({"queue_indices": indices}, int(queue_index))
    return float(np.asarray(state, dtype=float)[pos])


def _queue_label(queue_index: int) -> str:
    return f"Q{queue_index:+d}".replace("+", "+")


def _make_rng_stream(rng: Optional[np.random.Generator | int]) -> RNGStream:
    if isinstance(rng, np.random.Generator):
        return RNGStream(int(rng.integers(0, np.iinfo(np.uint32).max)))
    return RNGStream(rng)


def _resolve_burn_in(simulator: Any, burn_in: Optional[float]) -> float:
    if burn_in is None:
        return default_hawkes_burn_in(simulator)
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    return float(burn_in)


def _usage_effective_sample_size(counts: Sequence[float]) -> float:
    count_arr = np.asarray(counts, dtype=float)
    total = float(count_arr.sum())
    denom = float(np.sum(count_arr**2))
    if total <= 0 or denom <= 0:
        return np.nan
    return float(total**2 / denom)


def _resample_boundary_checkpoints(
    checkpoints: Sequence[BoundaryCheckpoint],
    n_particles: int,
    rng: Optional[np.random.Generator | int],
) -> tuple[list[BoundaryCheckpoint], dict[int, int]]:
    rng_obj = ensure_rng(rng)
    selected = rng_obj.integers(0, len(checkpoints), size=int(n_particles))
    ids, counts = np.unique(selected, return_counts=True)
    usage = {int(idx): int(count) for idx, count in zip(ids, counts)}
    return [checkpoints[int(idx)].copy() for idx in selected], usage


def _stack_or_empty(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0, 0), dtype=float)
    arrays = [np.asarray(value, dtype=float).reshape(1, -1) for value in values]
    width = arrays[0].shape[1]
    if any(arr.shape[1] != width for arr in arrays):
        raise ValueError("All excitation vectors must have the same dimension")
    return np.vstack(arrays)


def _normal_confidence_interval(p_hat: float, se: float) -> Optional[tuple[float, float]]:
    if not np.isfinite(p_hat) or not np.isfinite(se):
        return None
    return (float(max(0.0, p_hat - 1.96 * se)), float(min(1.0, p_hat + 1.96 * se)))


def _zero_excitation_state(state: MarkovState) -> MarkovState:
    copied = state.copy()
    copied.excitation = np.zeros_like(copied.excitation)
    for key, value in copied.hawkes_state.items():
        if isinstance(value, np.ndarray):
            copied.hawkes_state[key] = np.zeros_like(value)
        else:
            copied.hawkes_state[key] = 0.0
    copied.metadata["excitation_reset"] = True
    return copied
