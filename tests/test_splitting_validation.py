import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import (
    AdaptiveMultilevelSplitting,
    CoupledHawkesSimulator,
    FixedLevelSplitting,
    IndependentPoissonSimulator,
    SingleHawkesSimulator,
    min_best_depletion_problem,
    q1_depletion_problem,
)
from model.hawkes_4q import FourQueueParams
from model.ogata import Checkpoint, FourQueueHawkesSimulator, Trajectory
from model.rare_events import RareEventProblem
from model.analysis import compare_estimators, run_ams_replications
from model.analysis import extract_four_queue_depletion_samples


class PoissonCountSimulator:
    """Minimal simulator for validating against an exact Poisson tail."""

    def __init__(self, rate: float):
        self.rate = float(rate)

    def simulate(self, problem, rng=None, checkpoint=None, burn_in=0.0, record_path=True):
        del burn_in, record_path
        rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        if checkpoint is None:
            t = 0.0
            count = int(problem.initial_state[0])
        else:
            t = checkpoint.time
            count = int(checkpoint.state[0])

        times = [t]
        events = [-1]
        states = [np.array([count], dtype=float)]
        intensities = [np.array([self.rate], dtype=float)]
        scores = [problem.score(states[-1], t, {"intensity": intensities[-1]})]
        checkpoints = [
            Checkpoint(
                time=t,
                state=states[-1].copy(),
                intensity=intensities[-1].copy(),
                score=scores[-1],
                metadata={"intensity": intensities[-1].copy()},
            )
        ]
        hit = problem.is_target(states[-1], t, {"intensity": intensities[-1]})
        while t < problem.T and not hit:
            t += rng.exponential(1.0 / self.rate)
            if t > problem.T:
                break
            count += 1
            state = np.array([count], dtype=float)
            intensity = np.array([self.rate], dtype=float)
            score = problem.score(state, t, {"intensity": intensity})
            hit = problem.is_target(state, t, {"intensity": intensity})
            times.append(t)
            events.append(1)
            states.append(state)
            intensities.append(intensity)
            scores.append(score)
            checkpoints.append(
                Checkpoint(
                    time=t,
                    state=state.copy(),
                    intensity=intensity.copy(),
                    score=score,
                    metadata={"intensity": intensity.copy()},
                )
            )
        return Trajectory(
            times=np.asarray(times),
            events=np.asarray(events),
            states=np.vstack(states),
            intensities=np.vstack(intensities),
            final_state=states[-1].copy(),
            hit=hit,
            hitting_time=t if hit else None,
            score_values=np.asarray(scores),
            score_max=float(np.max(scores)),
            metadata={"n_events": len(events) - 1, "n_candidates": len(events) - 1},
            checkpoints=checkpoints,
        )

    def continue_from_checkpoint(self, checkpoint, problem, rng=None, record_path=True):
        return self.simulate(problem, rng=rng, checkpoint=checkpoint, record_path=record_path)

    def checkpoint_at_level(self, trajectory, level):
        return trajectory.first_checkpoint_at_level(level)


def poisson_tail(mean: float, k: int) -> float:
    pmf = math.exp(-mean)
    cdf = pmf
    for n in range(1, k):
        pmf *= mean / n
        cdf += pmf
    return 1.0 - cdf


def test_poisson_count_splitting_against_exact_tail():
    rate = 1.0
    horizon = 5.0
    threshold = 8
    exact = poisson_tail(rate * horizon, threshold)

    def target(state, time, metadata):
        del time, metadata
        return state[0] >= threshold

    def score(state, time, metadata):
        del time, metadata
        return min(float(state[0]) / threshold, 1.0)

    problem = RareEventProblem(
        T=horizon,
        initial_state=np.array([0]),
        target_event=target,
        score_function=score,
        event_name="poisson_count_tail",
    )
    simulator = PoissonCountSimulator(rate)
    fls = FixedLevelSplitting(
        simulator, problem, levels=[0.25, 0.50, 0.75, 1.0], n_particles=600, seed=123
    ).run()
    ams = AdaptiveMultilevelSplitting(
        simulator, problem, n_particles=500, kill_fraction=0.1, max_iterations=30, seed=123
    ).run()

    assert abs(fls.probability_estimate - exact) < 0.08
    assert abs(ams.probability_estimate - exact) < 0.10


def test_fixed_level_splitting_reproducible_with_same_seed():
    problem = q1_depletion_problem(T=8.0, q1_init=8)
    simulator = IndependentPoissonSimulator(lambda_plus=1.4, lambda_minus=1.1)
    kwargs = dict(levels=[0.25, 0.50, 0.75, 1.0], n_particles=120, seed=7)
    first = FixedLevelSplitting(simulator, problem, **kwargs).run()
    second = FixedLevelSplitting(simulator, problem, **kwargs).run()
    assert first.probability_estimate == second.probability_estimate
    assert first.level_probabilities == second.level_probabilities


def test_replicated_ams_summary_has_empirical_error_bar():
    def target(state, time, metadata):
        del time, metadata
        return state[0] >= 4

    def score(state, time, metadata):
        del time, metadata
        return min(float(state[0]) / 4.0, 1.0)

    problem = RareEventProblem(
        T=4.0,
        initial_state=np.array([0]),
        target_event=target,
        score_function=score,
        event_name="poisson_count_tail",
    )
    simulator = PoissonCountSimulator(rate=1.0)

    summary = run_ams_replications(
        simulator,
        problem,
        n_particles=60,
        n_replications=3,
        kill_fraction=0.1,
        max_iterations=10,
        seed=99,
    )
    table = compare_estimators([summary])

    assert len(summary["replication_estimates"]) == 3
    assert np.isfinite(summary["standard_error"])
    assert table.loc[0, "n_replications"] == 3


def test_four_queue_depletion_sample_extraction():
    ask_hit = Trajectory(
        times=np.array([0.0, 1.0]),
        events=np.array([-1, 1]),
        states=np.array([[5.0, 5.0, 5.0, 5.0], [0.0, 5.0, 8.0, 3.0]]),
        intensities=None,
        final_state=np.array([0.0, 5.0, 8.0, 3.0]),
        hit=True,
        hitting_time=1.0,
        score_values=np.array([0.0, 1.0]),
        score_max=1.0,
        metadata={"which_hit": 1, "first_limit_hitting_time": 1.0},
    )
    bid_hit = Trajectory(
        times=np.array([0.0, 1.0]),
        events=np.array([-1, 3]),
        states=np.array([[5.0, 5.0, 5.0, 5.0], [5.0, 0.0, 4.0, 9.0]]),
        intensities=None,
        final_state=np.array([5.0, 0.0, 4.0, 9.0]),
        hit=True,
        hitting_time=1.0,
        score_values=np.array([0.0, 1.0]),
        score_max=1.0,
        metadata={"which_hit": -1, "first_limit_hitting_time": 1.0},
    )

    samples = extract_four_queue_depletion_samples([ask_hit, bid_hit])

    np.testing.assert_array_equal(samples["q_plus2_when_plus1_zero"], np.array([8.0]))
    np.testing.assert_array_equal(samples["q_same"], np.array([8.0, 9.0]))
    np.testing.assert_array_equal(samples["q_opp"], np.array([3.0, 4.0]))
    np.testing.assert_array_equal(samples["q_neg2_when_plus1_zero"], np.array([3.0]))
    np.testing.assert_array_equal(samples["q_neg2_when_neg1_zero"], np.array([9.0]))
    np.testing.assert_array_equal(samples["q_neg2_same"], np.array([9.0]))
    np.testing.assert_array_equal(samples["q_neg2_opp"], np.array([3.0]))
    assert samples["n_valid"] == 2


def test_hawkes_checkpoint_preserves_excitation_state():
    problem = min_best_depletion_problem(T=30.0, q1_init=8, q_neg1_init=8)
    simulator = CoupledHawkesSimulator(mu_plus=1.2, mu_minus=1.4, alpha=0.2, beta=0.5)
    trajectory = simulator.simulate(problem, rng=np.random.default_rng(4))
    checkpoint = simulator.checkpoint_at_level(trajectory, 0.25)
    assert checkpoint is not None
    child = simulator.continue_from_checkpoint(checkpoint, problem, rng=np.random.default_rng(5))
    assert "H" in checkpoint.hawkes_state
    np.testing.assert_allclose(child.checkpoints[0].hawkes_state["H"], checkpoint.hawkes_state["H"])


def test_hawkes_intensities_are_non_negative():
    problem = q1_depletion_problem(T=20.0, q1_init=10)
    simulator = SingleHawkesSimulator(mu_plus=1.2, mu_minus=1.5, alpha=0.4, beta=0.5)
    trajectory = simulator.simulate(problem, rng=np.random.default_rng(11))
    assert trajectory.intensities is not None
    assert np.all(trajectory.intensities >= 0.0)


def test_four_queue_cross_excitation_updates_same_side_second_limit():
    params = FourQueueParams(a_cross=0.7)
    simulator = FourQueueHawkesSimulator(params)
    state = np.array([5.0, 5.0, 5.0, 5.0])
    H = np.zeros(2)
    G = np.zeros(2)
    simulator._apply_event(1, state, H, G)
    assert G[0] == 0.7
    assert G[1] == 0.0


def test_public_imports():
    from model.Hawkes import simulate_hawkes_queue
    from model.hawkes import simulate_coupled_hawkes
    from model import FixedLevelSplitting, IndependentPoissonSimulator, RareEventProblem

    assert simulate_hawkes_queue is not None
    assert simulate_coupled_hawkes is not None
    assert FixedLevelSplitting is not None
    assert IndependentPoissonSimulator is not None
    assert RareEventProblem is not None
