"""
Analysis utilities for QR model simulations.

Computes key metrics from simulation results:
    - Mean reversion ratio η = N_c / (2 N_a)
    - Volatility (at various sampling frequencies)
    - Empirical stationary distribution
    - Inter-event time statistics
    - Imbalance trajectory
"""

import numpy as np
from typing import Any, Optional


# ====================================================================== #
#  Mean Reversion Ratio
# ====================================================================== #

def mean_reversion_ratio(result: Any) -> dict:
    """
    Compute the mean reversion ratio η = N_c / (2 N_a).
    
    From Robert & Rosenbaum (2011):
        N_c = number of CONTINUATIONS (consecutive moves in same direction)
        N_a = number of ALTERNATIONS (consecutive moves in opposite directions)
        η = N_c / (2 N_a)
    
    η < 0.5 → strong mean reversion (price tends to reverse)
    η = 0.5 → random walk (no memory)
    η > 0.5 → trending / momentum
    
    Returns
    -------
    dict with keys: 'eta', 'N_c', 'N_a', 'n_moves', 'directions'
    """
    # Extract price moves (non-zero changes only)
    _, prices = result.price_array()
    dp = np.diff(prices)
    moves = dp[dp != 0]

    if len(moves) < 2:
        return {'eta': np.nan, 'N_c': 0, 'N_a': 0, 'n_moves': len(moves)}

    # Direction of each move: +1 or -1
    directions = np.sign(moves)

    # Count continuations and alternations
    N_c = 0  # continuations (same direction twice)
    N_a = 0  # alternations (opposite directions)
    for k in range(1, len(directions)):
        if directions[k] == directions[k - 1]:
            N_c += 1
        else:
            N_a += 1

    eta = N_c / (2 * N_a) if N_a > 0 else np.inf

    return {
        'eta': eta,
        'N_c': N_c,
        'N_a': N_a,
        'n_moves': len(moves),
        'directions': directions,
    }


# ====================================================================== #
#  Volatility
# ====================================================================== #

def compute_volatility(result: Any,
                       dt: float = 10.0,
                       unit: str = 'bps') -> dict:
    """
    Compute realized volatility at sampling frequency dt.
    
    Parameters
    ----------
    dt : float
        Sampling interval in seconds (default: 10s).
    unit : str
        'bps' for basis points, 'pct' for percent, 'raw' for raw.
    
    Returns
    -------
    dict with 'vol', 'returns', 'sample_prices', 'sample_times'
    """
    t_arr, p_arr = result.price_array()
    if len(p_arr) < 2:
        return {'vol': np.nan, 'returns': np.array([])}

    T = t_arr[-1]
    sample_times = np.arange(0, T, dt)
    sample_prices = np.interp(sample_times, t_arr, p_arr)

    returns = np.diff(sample_prices) / sample_prices[:-1]

    scale = {'bps': 1e4, 'pct': 100, 'raw': 1}.get(unit, 1e4)
    vol = returns.std() * scale

    return {
        'vol': vol,
        'returns': returns * scale,
        'sample_prices': sample_prices,
        'sample_times': sample_times,
        'unit': unit,
    }


# ====================================================================== #
#  Stationary Distribution
# ====================================================================== #

def stationary_distribution(result: Any,
                            queue_index: int,
                            burn_in_frac: float = 0.1,
                            max_n: int = 50) -> dict:
    """
    Estimate the empirical stationary distribution of queue i.
    
    Parameters
    ----------
    queue_index : int
        Which queue (-1 for bid, 1 for ask, etc.)
    burn_in_frac : float
        Fraction of snapshots to discard as burn-in.
    max_n : int
        Maximum queue size to consider.
    
    Returns
    -------
    dict with 'pmf', 'ns', 'mean', 'std', 'samples'
    """
    snaps = result.snapshots
    if not snaps:
        raise ValueError("No snapshots available. Run simulation with snapshot_interval.")

    start = int(burn_in_frac * len(snaps))
    samples = np.array([s.q(queue_index) for _, s in snaps[start:]])

    ns = np.arange(0, max_n + 1)
    counts, _ = np.histogram(samples, bins=np.arange(-0.5, max_n + 1.5, 1))
    pmf = counts / counts.sum()

    return {
        'pmf': pmf,
        'ns': ns,
        'mean': samples.mean(),
        'std': samples.std(),
        'samples': samples,
    }


# ====================================================================== #
#  Imbalance Trajectory
# ====================================================================== #

def imbalance_trajectory(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the imbalance trajectory over time.
    
    imb(t) = (Q_{-1}(t) - Q_1(t)) / (Q_{-1}(t) + Q_1(t))
    
    Returns (times, imbalances) arrays.
    """
    if not result.queue_path:
        return np.array([]), np.array([])

    times = []
    imbs = []
    for t, qdict in result.queue_path:
        qb = qdict.get(-1, 0.0)
        qa = qdict.get(1, 0.0)
        total = qb + qa
        imb = (qb - qa) / total if total > 0 else 0.0
        times.append(t)
        imbs.append(imb)

    return np.array(times), np.array(imbs)


# ====================================================================== #
#  Price at Infinity (starting price impact)
# ====================================================================== #

def price_at_infinity(results_batch: list[Any]) -> dict:
    """
    Estimate the final price distribution from a batch of simulations.
    
    Useful for studying: given an initial price and LOB state,
    what is the distribution of p(+∞)?
    
    Returns
    -------
    dict with 'final_prices', 'mean', 'std', 'ci_95'
    """
    finals = []
    for r in results_batch:
        _, p = r.price_array()
        if len(p) > 0:
            finals.append(p[-1])

    finals = np.array(finals)
    n = len(finals)
    mean = finals.mean()
    std = finals.std()
    ci = 1.96 * std / np.sqrt(n) if n > 0 else np.nan

    return {
        'final_prices': finals,
        'mean': mean,
        'std': std,
        'ci_95': (mean - ci, mean + ci),
        'n_runs': n,
    }


# ====================================================================== #
#  Batch Summary
# ====================================================================== #

def batch_summary(results: list[Any], dt_vol: float = 10.0) -> dict:
    """
    Compute summary statistics across a batch of simulation runs.
    
    Returns dict with arrays of per-run metrics.
    """
    etas = []
    vols = []
    mean_q_bid = []
    mean_q_ask = []
    n_events_list = []
    n_price_changes_list = []

    for r in results:
        mr = mean_reversion_ratio(r)
        etas.append(mr['eta'])

        v = compute_volatility(r, dt=dt_vol)
        vols.append(v['vol'])

        if r.snapshots:
            sd_bid = stationary_distribution(r, -1)
            sd_ask = stationary_distribution(r, 1)
            mean_q_bid.append(sd_bid['mean'])
            mean_q_ask.append(sd_ask['mean'])

        n_events_list.append(r.n_events)
        n_price_changes_list.append(r.n_price_changes)

    def _stats(arr):
        a = np.array(arr)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return {'mean': np.nan, 'std': np.nan, 'ci_95': np.nan}
        m, s, n = a.mean(), a.std(), len(a)
        return {'mean': m, 'std': s, 'ci_95': 1.96 * s / np.sqrt(n), 'values': a}

    return {
        'eta': _stats(etas),
        'volatility': _stats(vols),
        'mean_q_bid': _stats(mean_q_bid),
        'mean_q_ask': _stats(mean_q_ask),
        'n_events': _stats(n_events_list),
        'n_price_changes': _stats(n_price_changes_list),
    }


# ====================================================================== #
#  Rare-event estimator comparison
# ====================================================================== #

def run_mc_baseline(
    simulator: Any,
    problem: Any,
    n_runs: int,
    seed: int = 42,
    burn_in: Optional[float] = None,
    store_trajectories: bool = False,
) -> dict:
    """Run classical Monte Carlo using the baseline path simulator."""
    from .utils import RNGStream, Timer, binomial_standard_error

    stream = RNGStream(seed)
    trajectories = []
    hits = np.zeros(n_runs, dtype=bool)
    n_events = 0
    n_candidates = 0
    with Timer() as timer:
        for k in range(n_runs):
            traj = simulator.simulate(problem, rng=stream.next(), burn_in=burn_in, record_path=True)
            hits[k] = traj.hit
            n_events += traj.n_events
            n_candidates += traj.n_candidates
            if store_trajectories:
                trajectories.append(traj)
    p_hat = float(hits.mean()) if n_runs else np.nan
    return {
        "method": "Ogata MC",
        "probability_estimate": p_hat,
        "standard_error": binomial_standard_error(p_hat, n_runs),
        "relative_error": (
            binomial_standard_error(p_hat, n_runs) / p_hat if p_hat > 0 else np.inf
        ),
        "cpu_seconds": timer.elapsed,
        "n_runs": int(n_runs),
        "n_events": int(n_events),
        "n_candidates": int(n_candidates),
        "hits": int(hits.sum()),
        "trajectories": trajectories,
        "diagnostics": {"event_name": problem.event_name},
    }


def run_fixed_level_splitting(
    simulator: Any,
    problem: Any,
    levels: list[float],
    n_particles: int,
    seed: int = 42,
    burn_in: Optional[float] = None,
    store_trajectories: bool = True,
) -> dict:
    """Run Fixed-Level Splitting and return a table-friendly summary."""
    from .splitting import FixedLevelSplitting
    from .utils import Timer

    estimator = FixedLevelSplitting(
        simulator=simulator,
        problem=problem,
        levels=levels,
        n_particles=n_particles,
        seed=seed,
        burn_in=burn_in,
        store_trajectories=store_trajectories,
    )
    with Timer() as timer:
        result = estimator.run()
    se = result.diagnostics.get("probability_se_delta", np.nan)
    p_hat = result.probability_estimate
    return {
        "method": "Fixed-Level Splitting",
        "probability_estimate": p_hat,
        "standard_error": se,
        "relative_error": se / p_hat if p_hat > 0 else np.inf,
        "cpu_seconds": timer.elapsed,
        "n_runs": int(sum(result.n_particles_per_level)),
        "n_particles": int(n_particles),
        "n_events": result.diagnostics.get("n_events", 0),
        "n_candidates": result.diagnostics.get("n_candidates", 0),
        "result": result,
        "diagnostics": result.diagnostics,
    }


def run_ams(
    simulator: Any,
    problem: Any,
    n_particles: int,
    kill_fraction: float = 0.1,
    max_iterations: int = 100,
    seed: int = 42,
    burn_in: float = 0.0,
    store_trajectories: bool = True,
) -> dict:
    """Run Adaptive Multilevel Splitting and return a table-friendly summary."""
    from .splitting import AdaptiveMultilevelSplitting
    from .utils import Timer

    estimator = AdaptiveMultilevelSplitting(
        simulator=simulator,
        problem=problem,
        n_particles=n_particles,
        kill_fraction=kill_fraction,
        max_iterations=max_iterations,
        seed=seed,
        burn_in=burn_in,
        store_trajectories=store_trajectories,
    )
    with Timer() as timer:
        result = estimator.run()
    p_hat = result.probability_estimate
    # A single AMS run does not provide an honest standard error; use repeated
    # macro-replications for publication-quality error bars.
    return {
        "method": "AMS",
        "probability_estimate": p_hat,
        "standard_error": np.nan,
        "relative_error": np.nan,
        "cpu_seconds": timer.elapsed,
        "n_runs": int(result.n_particles * max(1, result.n_iterations)),
        "n_particles": int(n_particles),
        "n_events": result.diagnostics.get("n_events", 0),
        "n_candidates": result.diagnostics.get("n_candidates", 0),
        "result": result,
        "diagnostics": result.diagnostics,
    }


def run_ams_replications(
    simulator: Any,
    problem: Any,
    n_particles: int,
    n_replications: int = 10,
    kill_fraction: float = 0.1,
    max_iterations: int = 100,
    seed: int = 42,
    burn_in: float = 0.0,
) -> dict:
    """Run independent AMS macro-replications and estimate empirical error bars.

    A single AMS population gives a rare-event probability estimate but does not
    expose a simple, model-free standard error.  Independent macro-replications
    provide an honest empirical standard error for the mean estimate.
    """
    from .utils import RNGStream, Timer

    if n_replications <= 0:
        raise ValueError("n_replications must be positive")

    stream = RNGStream(seed)
    estimates: list[float] = []
    results = []
    n_runs = 0
    n_events = 0
    n_candidates = 0

    with Timer() as timer:
        for _ in range(int(n_replications)):
            replicated = run_ams(
                simulator=simulator,
                problem=problem,
                n_particles=n_particles,
                kill_fraction=kill_fraction,
                max_iterations=max_iterations,
                seed=int(stream.next().integers(0, np.iinfo(np.uint32).max)),
                burn_in=burn_in,
                store_trajectories=False,
            )
            estimates.append(float(replicated["probability_estimate"]))
            results.append(replicated["result"])
            n_runs += int(replicated.get("n_runs", 0))
            n_events += int(replicated.get("n_events", 0))
            n_candidates += int(replicated.get("n_candidates", 0))

    estimate_arr = np.asarray(estimates, dtype=float)
    p_hat = float(estimate_arr.mean())
    replication_std = (
        float(estimate_arr.std(ddof=1)) if len(estimate_arr) > 1 else np.nan
    )
    standard_error = (
        replication_std / np.sqrt(len(estimate_arr))
        if len(estimate_arr) > 1
        else np.nan
    )
    warnings = [
        warning
        for result in results
        for warning in result.diagnostics.get("warnings", [])
    ]
    return {
        "method": f"AMS ({n_replications} reps)",
        "probability_estimate": p_hat,
        "standard_error": standard_error,
        "relative_error": (
            standard_error / p_hat
            if p_hat > 0 and np.isfinite(standard_error)
            else np.nan
        ),
        "cpu_seconds": timer.elapsed,
        "n_runs": int(n_runs),
        "n_particles": int(n_particles),
        "n_replications": int(n_replications),
        "n_events": int(n_events),
        "n_candidates": int(n_candidates),
        "replication_estimates": estimate_arr,
        "replication_std": replication_std,
        "results": results,
        "diagnostics": {
            "event_name": problem.event_name,
            "warnings": warnings,
            "n_replications": int(n_replications),
            "replication_std": replication_std,
        },
    }


def extract_four_queue_depletion_samples(trajectories: list[Any]) -> dict:
    """Extract second-limit samples from four-queue depletion trajectories.

    The returned shape mirrors the notebook's Ogata collector: ``q_same`` is the
    second limit on the side that depleted, and ``q_opp`` is the second limit on
    the opposite side.  Only trajectories that hit the rare event are used.
    """
    q_plus2_when_plus1_zero = []
    q_plus2_when_neg1_zero = []
    q_neg2_when_plus1_zero = []
    q_neg2_when_neg1_zero = []
    q_same = []
    q_opp = []
    q_neg2_same = []
    q_neg2_opp = []
    which = []
    hitting_times = []

    for trajectory in trajectories:
        if not getattr(trajectory, "hit", False):
            continue
        state = np.asarray(trajectory.final_state, dtype=float)
        if len(state) < 4:
            continue
        side = int(trajectory.metadata.get("which_hit", 0))
        if side == 0:
            if state[0] <= 0:
                side = 1
            elif state[1] <= 0:
                side = -1
        if side not in (1, -1):
            continue

        q_plus2 = float(state[2])
        q_neg2 = float(state[3])
        which.append(side)
        hitting_time = trajectory.metadata.get(
            "first_limit_hitting_time",
            trajectory.hitting_time,
        )
        hitting_times.append(np.nan if hitting_time is None else float(hitting_time))

        if side == 1:
            q_plus2_when_plus1_zero.append(q_plus2)
            q_neg2_when_plus1_zero.append(q_neg2)
            q_same.append(q_plus2)
            q_opp.append(q_neg2)
            q_neg2_opp.append(q_neg2)
        else:
            q_plus2_when_neg1_zero.append(q_plus2)
            q_neg2_when_neg1_zero.append(q_neg2)
            q_same.append(q_neg2)
            q_opp.append(q_plus2)
            q_neg2_same.append(q_neg2)

    return {
        "q_plus2_when_plus1_zero": np.asarray(q_plus2_when_plus1_zero, dtype=float),
        "q_plus2_when_neg1_zero": np.asarray(q_plus2_when_neg1_zero, dtype=float),
        "q_neg2_when_plus1_zero": np.asarray(q_neg2_when_plus1_zero, dtype=float),
        "q_neg2_when_neg1_zero": np.asarray(q_neg2_when_neg1_zero, dtype=float),
        "q_same": np.asarray(q_same, dtype=float),
        "q_opp": np.asarray(q_opp, dtype=float),
        "q_neg2_same": np.asarray(q_neg2_same, dtype=float),
        "q_neg2_opp": np.asarray(q_neg2_opp, dtype=float),
        "which": np.asarray(which, dtype=int),
        "hitting_times": np.asarray(hitting_times, dtype=float),
        "n_valid": len(q_same),
    }


def run_fixed_level_conditional_q2(
    simulator: Any,
    problem: Any,
    levels: list[float],
    n_particles: int,
    seed: int = 42,
    burn_in: float = 0.0,
) -> dict:
    """Run FLS and extract conditional four-queue second-limit samples."""
    summary = run_fixed_level_splitting(
        simulator=simulator,
        problem=problem,
        levels=levels,
        n_particles=n_particles,
        seed=seed,
        burn_in=burn_in,
        store_trajectories=True,
    )
    samples = extract_four_queue_depletion_samples(summary["result"].trajectories)
    summary = dict(summary)
    summary["method"] = "FLS conditional Q2"
    summary["samples"] = samples
    summary["n_conditioned"] = samples["n_valid"]
    return summary


def run_ams_conditional_q2(
    simulator: Any,
    problem: Any,
    n_particles: int,
    kill_fraction: float = 0.1,
    max_iterations: int = 100,
    seed: int = 42,
    burn_in: float = 0.0,
) -> dict:
    """Run one AMS population and extract conditional second-limit samples."""
    summary = run_ams(
        simulator=simulator,
        problem=problem,
        n_particles=n_particles,
        kill_fraction=kill_fraction,
        max_iterations=max_iterations,
        seed=seed,
        burn_in=burn_in,
        store_trajectories=True,
    )
    samples = extract_four_queue_depletion_samples(summary["result"].trajectories)
    summary = dict(summary)
    summary["method"] = "AMS conditional Q2"
    summary["samples"] = samples
    summary["n_conditioned"] = samples["n_valid"]
    return summary


def run_markovian_conditional_restart_splitting(
    simulator: Any,
    initial_state: Any,
    queue_index: int,
    horizon: float,
    n_boundary_paths: int,
    horizon_local: float,
    n_restarts: int,
    seed: int = 42,
    boundary_level: int = 1,
    recovery_level: int = 2,
    burn_in: float = 0.0,
    queue_indices: Optional[list[int]] = None,
    reset_excitation: bool = False,
) -> dict:
    """Run Markovian Conditional Restart Splitting and return a table summary."""

    from .restart_splitting import (
        METHOD_NAME,
        collect_boundary_states,
        local_depletion_target_fn,
        local_recovery_fn,
        restart_from_boundary_distribution,
    )
    from .utils import RNGStream

    stream = RNGStream(seed)
    boundary_sample = collect_boundary_states(
        simulator=simulator,
        initial_state=initial_state,
        horizon=horizon,
        n_paths=n_boundary_paths,
        rng=stream.next(),
        burn_in=burn_in,
        queue_index=queue_index,
        boundary_level=boundary_level,
        queue_indices=queue_indices,
    )
    result = restart_from_boundary_distribution(
        checkpoints=boundary_sample,
        simulator=simulator,
        local_target_fn=local_depletion_target_fn(queue_index),
        recovery_fn=local_recovery_fn(queue_index, recovery_level),
        horizon_local=horizon_local,
        n_restarts=n_restarts,
        rng=stream.next(),
        sample_with_replacement=True,
        reset_excitation=reset_excitation,
        method_name=METHOD_NAME,
    )
    p_hat = result.probability_estimate
    se = result.standard_error
    return {
        "method": METHOD_NAME,
        "probability_estimate": p_hat,
        "standard_error": se,
        "relative_error": se / p_hat if p_hat > 0 and se is not None else np.nan,
        "confidence_interval": result.confidence_interval,
        "cpu_seconds": result.diagnostics.get("cpu_seconds", np.nan),
        "n_boundary_paths": int(n_boundary_paths),
        "n_boundary_samples": len(boundary_sample.checkpoints),
        "n_restarts": int(result.n_restarts),
        "n_successes": int(result.n_successes),
        "n_events": result.diagnostics.get("n_events", 0) + boundary_sample.metadata.get("n_events", 0),
        "n_candidates": result.diagnostics.get("n_candidates", 0) + boundary_sample.metadata.get("n_candidates", 0),
        "boundary_sample": boundary_sample,
        "result": result,
        "diagnostics": {
            "boundary_metadata": boundary_sample.metadata,
            "restart_diagnostics": result.diagnostics,
            "reset_excitation": bool(reset_excitation),
        },
    }


def run_naive_boundary_mc_comparison(
    simulator: Any,
    initial_state: Any,
    queue_index: int,
    horizon: float,
    n_paths: int,
    horizon_local: float,
    seed: int = 42,
    boundary_level: int = 1,
    recovery_level: int = 2,
    burn_in: float = 0.0,
    queue_indices: Optional[list[int]] = None,
) -> dict:
    """Run naive Ogata boundary sampling with one continuation per boundary hit."""

    from .restart_splitting import NAIVE_METHOD_NAME, run_naive_boundary_mc
    from .utils import Timer

    with Timer() as timer:
        result = run_naive_boundary_mc(
            simulator=simulator,
            initial_state=initial_state,
            queue_index=queue_index,
            horizon=horizon,
            n_paths=n_paths,
            horizon_local=horizon_local,
            rng=seed,
            boundary_level=boundary_level,
            recovery_level=recovery_level,
            burn_in=burn_in,
            queue_indices=queue_indices,
        )
    p_hat = result.probability_estimate
    se = result.standard_error
    boundary_meta = result.diagnostics.get("boundary_metadata", {})
    return {
        "method": NAIVE_METHOD_NAME,
        "probability_estimate": p_hat,
        "standard_error": se,
        "relative_error": se / p_hat if p_hat > 0 and se is not None and np.isfinite(se) else np.nan,
        "confidence_interval": result.confidence_interval,
        "cpu_seconds": timer.elapsed,
        "n_boundary_paths": int(n_paths),
        "n_boundary_samples": int(result.diagnostics.get("n_boundary_checkpoints", 0)),
        "n_restarts": int(result.n_restarts),
        "n_successes": int(result.n_successes),
        "n_events": int(result.diagnostics.get("n_events", 0) + boundary_meta.get("n_events", 0)),
        "n_candidates": int(result.diagnostics.get("n_candidates", 0) + boundary_meta.get("n_candidates", 0)),
        "result": result,
        "diagnostics": result.diagnostics,
    }


def compare_restart_results(results: list[dict]) -> "Any":
    """Build a compact table for naive Ogata and restart-splitting summaries."""

    import pandas as pd

    rows = []
    for result in results:
        p_hat = result.get("probability_estimate", np.nan)
        se = result.get("standard_error", np.nan)
        rows.append(
            {
                "method": result.get("method"),
                "probability": p_hat,
                "std_error": se,
                "relative_error": result.get("relative_error", np.nan),
                "ci_low": (
                    result.get("confidence_interval", (np.nan, np.nan))[0]
                    if result.get("confidence_interval") is not None
                    else np.nan
                ),
                "ci_high": (
                    result.get("confidence_interval", (np.nan, np.nan))[1]
                    if result.get("confidence_interval") is not None
                    else np.nan
                ),
                "cpu_seconds": result.get("cpu_seconds", np.nan),
                "n_boundary_paths": result.get("n_boundary_paths", np.nan),
                "n_boundary_samples": result.get("n_boundary_samples", np.nan),
                "n_restarts": result.get("n_restarts", np.nan),
                "n_successes": result.get("n_successes", np.nan),
                "n_events": result.get("n_events", np.nan),
                "n_candidates": result.get("n_candidates", np.nan),
            }
        )
    table = pd.DataFrame(rows)
    if "relative_error" in table:
        table["cost_normalized_rel_error"] = table["relative_error"] * np.sqrt(table["cpu_seconds"])
    return table


def extract_q_neg2_restart_observables(
    same_side_result: Any = None,
    opposite_side_result: Any = None,
) -> dict:
    """Extract nb3 bid second-limit observables from restart results.

    ``same_side_result`` should correspond to restarting from ``Q-1=1`` and
    targeting ``Q-1=0``.  ``opposite_side_result`` should correspond to
    restarting from ``Q+1=1`` and targeting ``Q+1=0`` while observing ``Q-2``.
    """

    same = _unwrap_restart_result(same_side_result)
    opp = _unwrap_restart_result(opposite_side_result)
    q_same = _q_neg2_success(same)
    q_opp = _q_neg2_success(opp)
    return {
        "q_neg2_same": q_same,
        "q_neg2_opp": q_opp,
        "q_neg2_when_neg1_zero": q_same,
        "q_neg2_when_plus1_zero": q_opp,
        "n_same": len(q_same),
        "n_opp": len(q_opp),
    }


def _unwrap_restart_result(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict) and "result" in value:
        return value["result"]
    return value


def _q_neg2_success(result: Any) -> np.ndarray:
    if result is None:
        return np.empty(0, dtype=float)
    values = result.observables.get("q_neg2_success", np.empty(0, dtype=float))
    return np.asarray(values, dtype=float)


def compare_estimators(results: list[dict]) -> "Any":
    """Build a compact comparison table from estimator summary dictionaries."""
    import pandas as pd

    rows = []
    include_replications = any("n_replications" in result for result in results)
    for result in results:
        row = {
            "method": result.get("method"),
            "probability": result.get("probability_estimate"),
            "std_error": result.get("standard_error"),
            "relative_error": result.get("relative_error"),
            "cpu_seconds": result.get("cpu_seconds"),
            "n_runs": result.get("n_runs"),
            "n_particles": result.get("n_particles", np.nan),
            "n_events": result.get("n_events"),
            "n_candidates": result.get("n_candidates"),
        }
        if include_replications:
            row["n_replications"] = result.get("n_replications", np.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    if "relative_error" in table:
        table["cost_normalized_rel_error"] = table["relative_error"] * np.sqrt(table["cpu_seconds"])
    return table
