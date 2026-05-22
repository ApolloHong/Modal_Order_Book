"""Rare-event splitting estimators for LOB trajectory simulators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .ogata import Checkpoint, Trajectory
from .rare_events import RareEventProblem
from .utils import RNGStream, log_product


@dataclass
class SplittingResult:
    """Result of a fixed-level splitting run."""

    probability_estimate: float
    log_probability_estimate: float
    level_probabilities: list[float]
    levels: list[float]
    n_particles_per_level: list[int]
    n_survivors_per_level: list[int]
    trajectories: list[Trajectory]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AMSResult:
    """Result of an Adaptive Multilevel Splitting run."""

    probability_estimate: float
    log_probability_estimate: float
    adaptive_levels: list[float]
    survival_factors: list[float]
    n_iterations: int
    n_particles: int
    final_hit_ratio: float
    trajectories: list[Trajectory]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class FixedLevelSplitting:
    """Fixed-level rare-event splitting estimator.

    The simulator must expose ``simulate``, ``continue_from_checkpoint`` and
    ``checkpoint_at_level``.  Checkpoints must include Hawkes excitation state
    for Hawkes models; restarting from queue sizes alone is not valid.
    """

    def __init__(
        self,
        simulator: Any,
        problem: RareEventProblem,
        levels: list[float],
        n_particles: int = 100,
        seed: Optional[int] = None,
        burn_in: float = 0.0,
        store_trajectories: bool = True,
    ):
        if n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if not levels:
            raise ValueError("levels must contain at least one level")
        levels = [float(x) for x in levels]
        if any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError("levels must be strictly increasing")
        if levels[-1] < problem.threshold:
            levels = [*levels, float(problem.threshold)]
        self.simulator = simulator
        self.problem = problem
        self.levels = levels
        self.n_particles = int(n_particles)
        self.seed = seed
        self.burn_in = float(burn_in)
        self.store_trajectories = bool(store_trajectories)

    def run(self) -> SplittingResult:
        stream = RNGStream(self.seed)
        selector_rng = stream.next()
        warnings: list[str] = []
        if self.problem.metadata.get("non_monotone_score"):
            warnings.append(
                "Problem score is marked non-monotone; splitting remains valid only if level crossings are meaningful."
            )

        current_checkpoints: list[Optional[Checkpoint]] = [None] * self.n_particles
        level_probabilities: list[float] = []
        n_particles_per_level: list[int] = []
        n_survivors_per_level: list[int] = []
        final_trajectories: list[Trajectory] = []
        clone_counts: list[int] = []
        total_events = 0
        total_candidates = 0

        for level_index, level in enumerate(self.levels):
            trajectories = self._simulate_particles(current_checkpoints, stream)
            total_events += int(sum(traj.n_events for traj in trajectories))
            total_candidates += int(sum(traj.n_candidates for traj in trajectories))
            n_particles = len(trajectories)
            is_final_level = level_index == len(self.levels) - 1
            is_target_level = level >= self.problem.threshold
            if is_target_level:
                survivor_indices = [i for i, traj in enumerate(trajectories) if traj.hit]
            else:
                survivor_indices = [i for i, traj in enumerate(trajectories) if traj.score_max >= level]

            n_survivors = len(survivor_indices)
            level_probability = n_survivors / n_particles if n_particles else 0.0
            level_probabilities.append(level_probability)
            n_particles_per_level.append(n_particles)
            n_survivors_per_level.append(n_survivors)

            if n_survivors == 0:
                warnings.append(
                    f"No survivor at level {level:.6g}; estimator is zero. Consider easier or more closely spaced levels."
                )
                final_trajectories = trajectories if self.store_trajectories else []
                break

            if is_final_level:
                final_trajectories = trajectories if self.store_trajectories else []
                break

            survivor_checkpoints: list[Checkpoint] = []
            for idx in survivor_indices:
                checkpoint = self.simulator.checkpoint_at_level(trajectories[idx], level)
                if checkpoint is not None:
                    survivor_checkpoints.append(checkpoint)

            if not survivor_checkpoints:
                warnings.append(
                    f"Survivors reached level {level:.6g}, but no continuation checkpoint was available."
                )
                final_trajectories = trajectories if self.store_trajectories else []
                level_probabilities[-1] = 0.0
                break

            choices = selector_rng.integers(0, len(survivor_checkpoints), size=self.n_particles)
            current_checkpoints = [survivor_checkpoints[int(choice)].copy() for choice in choices]
            clone_counts.append(self.n_particles)

        log_estimate = log_product(level_probabilities)
        probability = 0.0 if math.isinf(log_estimate) and log_estimate < 0 else float(math.exp(log_estimate))
        log_variance_delta = 0.0
        for p, n in zip(level_probabilities, n_particles_per_level):
            if p > 0 and n > 0:
                log_variance_delta += (1.0 - p) / (n * p)
        diagnostics = {
            "warnings": warnings,
            "log_se_delta": math.sqrt(log_variance_delta) if level_probabilities else math.nan,
            "probability_se_delta": probability * math.sqrt(log_variance_delta) if probability > 0 else 0.0,
            "clone_counts": clone_counts,
            "n_events": total_events,
            "n_candidates": total_candidates,
            "event_name": self.problem.event_name,
        }
        return SplittingResult(
            probability_estimate=probability,
            log_probability_estimate=log_estimate,
            level_probabilities=level_probabilities,
            levels=self.levels[: len(level_probabilities)],
            n_particles_per_level=n_particles_per_level,
            n_survivors_per_level=n_survivors_per_level,
            trajectories=final_trajectories,
            diagnostics=diagnostics,
        )

    def _simulate_particles(
        self,
        checkpoints: list[Optional[Checkpoint]],
        stream: RNGStream,
    ) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        for checkpoint in checkpoints:
            rng = stream.next()
            if checkpoint is None:
                trajectories.append(
                    self.simulator.simulate(
                        self.problem,
                        rng=rng,
                        checkpoint=None,
                        burn_in=self.burn_in,
                        record_path=True,
                    )
                )
            else:
                trajectories.append(
                    self.simulator.continue_from_checkpoint(
                        checkpoint,
                        self.problem,
                        rng=rng,
                        record_path=True,
                    )
                )
        return trajectories


class AdaptiveMultilevelSplitting:
    """Interacting-particle AMS estimator."""

    def __init__(
        self,
        simulator: Any,
        problem: RareEventProblem,
        n_particles: int = 100,
        kill_fraction: float = 0.1,
        n_kill: Optional[int] = None,
        max_iterations: int = 100,
        seed: Optional[int] = None,
        burn_in: float = 0.0,
        store_trajectories: bool = True,
    ):
        if n_particles <= 1:
            raise ValueError("n_particles must be greater than one")
        if not (0 < kill_fraction < 1):
            raise ValueError("kill_fraction must be in (0, 1)")
        self.simulator = simulator
        self.problem = problem
        self.n_particles = int(n_particles)
        self.n_kill = int(n_kill) if n_kill is not None else max(1, int(np.floor(kill_fraction * n_particles)))
        self.n_kill = min(self.n_kill, self.n_particles - 1)
        self.max_iterations = int(max_iterations)
        self.seed = seed
        self.burn_in = float(burn_in)
        self.store_trajectories = bool(store_trajectories)

    def run(self) -> AMSResult:
        stream = RNGStream(self.seed)
        selector_rng = stream.next()
        warnings: list[str] = []
        if self.problem.metadata.get("non_monotone_score"):
            warnings.append(
                "Problem score is marked non-monotone; AMS adaptive levels may be noisy."
            )

        particles = [
            self.simulator.simulate(self.problem, rng=stream.next(), burn_in=self.burn_in, record_path=True)
            for _ in range(self.n_particles)
        ]
        total_events = int(sum(traj.n_events for traj in particles))
        total_candidates = int(sum(traj.n_candidates for traj in particles))
        adaptive_levels: list[float] = []
        survival_factors: list[float] = []
        killed_counts: list[int] = []
        lineage: list[dict[str, int]] = []
        previous_level = -math.inf
        stagnation_warned = False

        for iteration in range(self.max_iterations):
            scores = np.asarray([traj.score_max for traj in particles], dtype=float)
            final_hit_ratio = float(np.mean([traj.hit for traj in particles]))
            if final_hit_ratio >= 1.0:
                break

            order = np.argsort(scores, kind="mergesort")
            killed = order[: self.n_kill]
            level = float(scores[killed[-1]])
            adaptive_levels.append(level)

            if level >= self.problem.threshold:
                break
            if level <= previous_level and iteration > 0 and not stagnation_warned:
                warnings.append(
                    f"Adaptive level stagnated at {level:.6g}; estimator may have high variance."
                )
                stagnation_warned = True
            previous_level = level

            killed_set = set(int(i) for i in killed)
            survivors = [i for i in range(self.n_particles) if i not in killed_set and scores[i] >= level]
            if not survivors:
                warnings.append("AMS degeneracy: no survivor available for cloning.")
                break

            survival_factor = (self.n_particles - len(killed)) / self.n_particles
            survival_factors.append(float(survival_factor))
            killed_counts.append(len(killed))

            for killed_idx in killed:
                parent_idx = int(selector_rng.choice(survivors))
                parent = particles[parent_idx]
                checkpoint = self.simulator.checkpoint_at_level(parent, level)
                if checkpoint is None:
                    warnings.append(
                        f"AMS could not find parent checkpoint at level {level:.6g}; stopped early."
                    )
                    final_hit_ratio = float(np.mean([traj.hit for traj in particles]))
                    return self._result(
                        particles,
                        adaptive_levels,
                        survival_factors,
                        iteration + 1,
                        final_hit_ratio,
                        warnings,
                        killed_counts,
                        lineage,
                        total_events,
                        total_candidates,
                    )
                child = self.simulator.continue_from_checkpoint(
                    checkpoint,
                    self.problem,
                    rng=stream.next(),
                    record_path=True,
                )
                particles[int(killed_idx)] = child
                total_events += child.n_events
                total_candidates += child.n_candidates
                lineage.append({"iteration": iteration, "child": int(killed_idx), "parent": parent_idx})

        final_hit_ratio = float(np.mean([traj.hit for traj in particles]))
        return self._result(
            particles,
            adaptive_levels,
            survival_factors,
            len(survival_factors),
            final_hit_ratio,
            warnings,
            killed_counts,
            lineage,
            total_events,
            total_candidates,
        )

    def _result(
        self,
        particles: list[Trajectory],
        adaptive_levels: list[float],
        survival_factors: list[float],
        n_iterations: int,
        final_hit_ratio: float,
        warnings: list[str],
        killed_counts: list[int],
        lineage: list[dict[str, int]],
        total_events: int,
        total_candidates: int,
    ) -> AMSResult:
        log_estimate = log_product(survival_factors)
        if final_hit_ratio <= 0:
            log_estimate = -math.inf
            probability = 0.0
        else:
            log_estimate += math.log(final_hit_ratio)
            probability = float(math.exp(log_estimate))
        trajectories = particles if self.store_trajectories else []
        diagnostics = {
            "warnings": warnings,
            "killed_counts": killed_counts,
            "lineage": lineage,
            "n_events": int(total_events),
            "n_candidates": int(total_candidates),
            "event_name": self.problem.event_name,
            "degenerate": final_hit_ratio == 0.0,
        }
        return AMSResult(
            probability_estimate=probability,
            log_probability_estimate=log_estimate,
            adaptive_levels=adaptive_levels,
            survival_factors=survival_factors,
            n_iterations=int(n_iterations),
            n_particles=self.n_particles,
            final_hit_ratio=float(final_hit_ratio),
            trajectories=trajectories,
            diagnostics=diagnostics,
        )
