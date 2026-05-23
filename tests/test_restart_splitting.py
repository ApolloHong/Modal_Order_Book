import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.hawkes_4q import FourQueueParams
from model.ogata import Checkpoint, CoupledHawkesSimulator, FourQueueHawkesSimulator, IndependentPoissonSimulator
from model.restart_splitting import (
    METHOD_NAME,
    BoundaryCheckpoint,
    MarkovState,
    collect_boundary_states,
    extract_excitation_vector,
    local_depletion_target_fn,
    local_recovery_fn,
    multilevel_markovian_restart_splitting,
    restart_from_boundary_distribution,
    summarize_conditional_S,
)


def test_poisson_boundary_restart_has_empty_excitation_and_is_reproducible():
    simulator = IndependentPoissonSimulator(lambda_plus=0.4, lambda_minus=2.0)
    sample = collect_boundary_states(
        simulator=simulator,
        initial_state=[3],
        queue_index=1,
        horizon=20.0,
        n_paths=80,
        rng=123,
    )

    assert len(sample.checkpoints) > 0
    assert sample.S_samples.shape == (len(sample.checkpoints), 0)

    kwargs = dict(
        checkpoints=sample,
        simulator=simulator,
        local_target_fn=local_depletion_target_fn(1),
        recovery_fn=local_recovery_fn(1, 2),
        horizon_local=5.0,
        n_restarts=120,
        rng=99,
    )
    first = restart_from_boundary_distribution(**kwargs)
    second = restart_from_boundary_distribution(**kwargs)

    assert first.method_name == METHOD_NAME
    assert 0.0 <= first.probability_estimate <= 1.0
    assert first.probability_estimate == second.probability_estimate
    assert first.n_successes == second.n_successes


def test_coupled_hawkes_boundary_sample_has_two_dimensional_S():
    simulator = CoupledHawkesSimulator(mu_plus=0.8, mu_minus=2.0, alpha=0.25, beta=0.7)
    sample = collect_boundary_states(
        simulator=simulator,
        initial_state=[4, 4],
        queue_index=-1,
        horizon=20.0,
        n_paths=80,
        rng=321,
        queue_indices=[1, -1],
    )

    assert len(sample.checkpoints) > 0
    assert sample.S_samples.shape[1] == 2
    assert np.all(np.isfinite(sample.S_samples))
    assert np.all(np.isfinite(np.cov(sample.S_samples, rowvar=False)))
    summary = summarize_conditional_S(sample.S_samples, sample.metadata["S_component_names"])
    assert summary["n_samples"] == len(sample.checkpoints)


def test_four_queue_S_maps_ask_and_bid_cross_components_without_duplication():
    checkpoint = Checkpoint(
        time=1.0,
        state=np.array([5.0, 1.0, 7.0, 9.0]),
        hawkes_state={"H": np.array([0.2, 0.4]), "G": np.array([0.6, 0.8])},
        intensity=np.ones(8),
        metadata={"queue_indices": [1, -1, 2, -2]},
    )

    S, names, diagnostics = extract_excitation_vector(checkpoint, model_name="four_queue_hawkes")

    np.testing.assert_allclose(S, np.array([0.2, 0.4, 0.6, 0.8]))
    assert names[-2] == "S^{+1,- -> +2,+}"
    assert names[-1] == "S^{-1,- -> -2,+}"
    assert "Q-1 removal -> Q-2 addition" in diagnostics["cross_component_note"]


def test_restart_from_boundary_preserves_or_resets_excitation_explicitly():
    params = FourQueueParams(mu_plus_1=0.8, mu_minus_1=1.8, alpha=0.2, beta=0.6, a_cross=0.5)
    simulator = FourQueueHawkesSimulator(params)
    state = MarkovState(
        t=2.0,
        queues=np.array([5.0, 1.0, 5.0, 5.0]),
        excitation=np.array([0.1, 0.3, 0.2, 0.7]),
        intensity=np.ones(8),
        hawkes_state={"H": np.array([0.1, 0.3]), "G": np.array([0.2, 0.7])},
        metadata={"queue_indices": [1, -1, 2, -2]},
    )
    boundary = BoundaryCheckpoint(
        state=state,
        boundary_name="Q-1=1",
        boundary_level=1,
        queue_label="Q-1",
        t_hit=2.0,
    )

    correct = restart_from_boundary_distribution(
        [boundary],
        simulator,
        local_target_fn=local_depletion_target_fn(-1),
        recovery_fn=local_recovery_fn(-1, 2),
        horizon_local=2.0,
        n_restarts=1,
        rng=7,
    )
    reset = restart_from_boundary_distribution(
        [boundary],
        simulator,
        local_target_fn=local_depletion_target_fn(-1),
        recovery_fn=local_recovery_fn(-1, 2),
        horizon_local=2.0,
        n_restarts=1,
        rng=7,
        reset_excitation=True,
    )

    np.testing.assert_allclose(correct.observables["start_S"][0], state.excitation)
    np.testing.assert_allclose(reset.observables["start_S"][0], np.zeros_like(state.excitation))


def test_hawkes_decay_jump_consistency_formula():
    S_old = np.array([0.4, -0.2])
    beta = 0.5
    dt = 1.7
    jump = np.array([0.3, 0.3])

    S_new = S_old * np.exp(-beta * dt) + jump

    np.testing.assert_allclose(S_new, np.array([0.4, -0.2]) * np.exp(-0.5 * 1.7) + jump)


def test_multilevel_restart_splitting_runs_and_preserves_checkpoints():
    params = FourQueueParams(
        mu_plus_1=0.8,
        mu_minus_1=1.8,
        alpha=0.2,
        beta=0.6,
        q1_init=6,
        q_neg1_init=6,
        a_cross=0.3,
    )
    simulator = FourQueueHawkesSimulator(params)
    result = multilevel_markovian_restart_splitting(
        simulator=simulator,
        initial_state=[6, 6, 5, 5],
        queue_index=-1,
        levels=[4, 2, 1, 0],
        horizon=20.0,
        n_particles=40,
        rng=123,
        burn_in=0.0,
        queue_indices=[1, -1, 2, -2],
    )

    assert 0.0 <= result.probability_estimate <= 1.0
    assert len(result.level_probabilities) >= 1
    for checkpoint in result.final_checkpoints:
        assert "H" in checkpoint.state.hawkes_state
        assert "G" in checkpoint.state.hawkes_state
