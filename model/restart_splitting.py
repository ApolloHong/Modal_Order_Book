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
        excitation = np.array([H[0], H[1], G[0], G[1], G[1]], dtype=float)
        component_names = [
            "S^{1,+}",
            "S^{1,-}",
            "S^{2,+}",
            "S^{2,-}",
            "S^{1,+ -> 2,-}",
        ]
        diagnostics["sign_convention"] = "positive index = ask, negative index = bid"
        diagnostics["queue_state_order"] = [1, -1, 2, -2]
        diagnostics["cross_component_note"] = (
            "The fifth component duplicates G[-2] in the current code because "
            "Q-1 removals excite Q-2 additions through a_cross."
        )
        return excitation, component_names, diagnostics

    if "H" in hawkes_state:
        H_raw = hawkes_state["H"]
        H = np.asarray(H_raw, dtype=float)
        if H.ndim == 0:
            return np.array([float(H)], dtype=float), ["S^{1,-}"], diagnostics
        if H.size == 2:
            return H.astype(float).copy(), ["S^{1,+}", "S^{1,-}"], diagnostics
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
    burn_in: float = 0.0,
    record_path: bool = False,
    queue_index: Optional[int] = None,
    boundary_level: int = 1,
    boundary_name: Optional[str] = None,
    queue_label: Optional[str] = None,
    queue_indices: Optional[Sequence[int]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> BoundarySample:
    """Collect empirical Markov checkpoints at a near-boundary queue level."""

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

    for _ in range(int(n_paths)):
        trajectory = simulator.simulate(problem, rng=stream.next(), burn_in=burn_in, record_path=record_path)
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
            "burn_in": float(burn_in),
            "horizon": float(horizon),
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
            elapsed_local = max(0.0, final_time - start_t)

            successes.append(success)
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
    p_hat = float(success_arr.mean()) if len(success_arr) else np.nan
    se = binomial_standard_error(p_hat, len(success_arr)) if len(success_arr) else np.nan
    ci = _normal_confidence_interval(p_hat, se)
    observables: dict[str, Any] = {
        "success": success_arr,
        "local_times": np.asarray(local_times, dtype=float),
        "final_queues": np.vstack(final_queues) if final_queues else np.empty((0, 0)),
        "start_queues": np.vstack(start_queues) if start_queues else np.empty((0, 0)),
        "start_S": _stack_or_empty(start_S),
        "end_S": _stack_or_empty(end_S),
        "sampled_checkpoint_indices": np.asarray(selected, dtype=int),
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
    burn_in: float = 0.0,
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
        return np.array([intensity[1] - simulator.mu_minus]), ["S^{1,-}"]
    if model == "coupled_hawkes" and hasattr(simulator, "mu_minus") and len(intensity) >= 4:
        return np.array([intensity[1] - simulator.mu_minus, intensity[3] - simulator.mu_minus]), [
            "S^{1,+}",
            "S^{1,-}",
        ]
    params = getattr(simulator, "params", None)
    if model == "four_queue_hawkes" and params is not None and len(intensity) >= 8:
        G_minus = intensity[6] - params.mu_plus_2
        return (
            np.array(
                [
                    intensity[1] - params.mu_minus_1,
                    intensity[3] - params.mu_minus_1,
                    intensity[4] - params.mu_plus_2,
                    G_minus,
                    G_minus,
                ],
                dtype=float,
            ),
            ["S^{1,+}", "S^{1,-}", "S^{2,+}", "S^{2,-}", "S^{1,+ -> 2,-}"],
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
