"""Utilitaires d'analyse pour les simulations de modèles QR (Queue Resilience).

This project is conducted by Lizhan Hong and Tom Zhang under the supervision of Professor Charles-Albert Lehalle.

Calcule des métriques clés à partir des résultats de simulation :
    - Ratio de retour à la moyenne η = N_c / (2 N_a)
    - Volatilité (à diverses fréquences d'échantillonnage)
    - Distribution stationnaire empirique
    - Statistiques de temps inter-événements
    - Trajectoire du déséquilibre (imbalance)
"""

import numpy as np
from typing import Any, Optional


# ====================================================================== #
#  Ratio de retour à la moyenne
# ====================================================================== #

def mean_reversion_ratio(result: Any) -> dict:
    """
    Calcule le ratio de retour à la moyenne η = N_c / (2 N_a).
    
    D'après Robert & Rosenbaum (2011) :
        N_c = nombre de CONTINUATIONS (mouvements consécutifs dans la même direction)
        N_a = nombre d'ALTERNANCES (mouvements consécutifs dans des directions opposées)
        η = N_c / (2 N_a)
    
    η < 0.5 → fort retour à la moyenne (le prix tend à s'inverser)
    η = 0.5 → marche aléatoire (pas de mémoire)
    η > 0.5 → tendance / momentum
    
    Retours
    -------
    dict avec les clés : 'eta', 'N_c', 'N_a', 'n_moves', 'directions'
    """
    # Extraire les mouvements de prix (changements non nuls uniquement)
    _, prices = result.price_array()
    dp = np.diff(prices)
    moves = dp[dp != 0]

    if len(moves) < 2:
        return {'eta': np.nan, 'N_c': 0, 'N_a': 0, 'n_moves': len(moves)}

    # Direction de chaque mouvement : +1 ou -1
    directions = np.sign(moves)

    # Compter les continuations et alternances
    N_c = 0  # continuations (même direction deux fois)
    N_a = 0  # alternances (directions opposées)
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
#  Volatilité
# ====================================================================== #

def compute_volatility(result: Any,
                       dt: float = 10.0,
                       unit: str = 'bps') -> dict:
    """
    Calcule la volatilité réalisée à une fréquence d'échantillonnage dt.
    
    Paramètres
    ----------
    dt : float
        Intervalle d'échantillonnage en secondes (défaut : 10s).
    unit : str
        'bps' pour points de base, 'pct' pour pourcentage, 'raw' pour brut.
    
    Retours
    -------
    dict avec 'vol', 'returns', 'sample_prices', 'sample_times'
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
#  Distribution stationnaire
# ====================================================================== #

def stationary_distribution(result: Any,
                            queue_index: int,
                            burn_in_frac: float = 0.1,
                            max_n: int = 50) -> dict:
    """
    Estime la distribution stationnaire empirique de la file i.
    
    Paramètres
    ----------
    queue_index : int
        Quelle file (-1 pour bid, 1 pour ask, etc.)
    burn_in_frac : float
        Fraction des instantanés (snapshots) à ignorer en tant que période de chauffe.
    max_n : int
        Taille de file maximale à considérer.
    
    Retours
    -------
    dict avec 'pmf', 'ns', 'mean', 'std', 'samples'
    """
    snaps = result.snapshots
    if not snaps:
        raise ValueError("Aucun instantané disponible. Exécutez la simulation avec snapshot_interval.")

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
#  Trajectoire du déséquilibre
# ====================================================================== #

def imbalance_trajectory(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule la trajectoire du déséquilibre au cours du temps.
    
    imb(t) = (Q_{-1}(t) - Q_1(t)) / (Q_{-1}(t) + Q_1(t))
    
    Retourne les tableaux (temps, déséquilibres).
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
#  Prix à l'infini (impact initial du prix)
# ====================================================================== #

def price_at_infinity(results_batch: list[Any]) -> dict:
    """
    Estime la distribution finale des prix à partir d'un lot de simulations.
    
    Utile pour étudier : étant donné un prix initial et l'état du LOB,
    quelle est la distribution de p(+∞) ?
    
    Retours
    -------
    dict avec 'final_prices', 'mean', 'std', 'ci_95'
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
#  Résumé par lot (batch)
# ====================================================================== #

def batch_summary(results: list[Any], dt_vol: float = 10.0) -> dict:
    """
    Calcule des statistiques récapitulatives sur un lot de simulations.
    
    Retourne un dict avec les tableaux des métriques par exécution.
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
#  Comparaison d'estimateurs d'événements rares
# ====================================================================== #

def run_mc_baseline(
    simulator: Any,
    problem: Any,
    n_runs: int,
    seed: int = 42,
    burn_in: Optional[float] = None,
    store_trajectories: bool = False,
) -> dict:
    """Exécute une Monte Carlo classique en utilisant le simulateur de trajectoire de référence."""
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
    """Exécute le fractionnement à niveaux fixes (Fixed-Level Splitting) et retourne un résumé."""
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
    """Exécute le fractionnement adaptatif multiniveaux (AMS) et retourne un résumé."""
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
    # Une seule exécution AMS ne fournit pas d'erreur type honnête ; utilisez des
    # macro-réplications répétées pour des barres d'erreur de qualité publication.
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
    """Exécute des macro-réplications AMS indépendantes et estime les barres d'erreur empiriques.

    Une seule population AMS donne une estimation de probabilité d'événement rare mais ne
    fournit pas d'erreur type simple et sans modèle. Les macro-réplications indépendantes
    fournissent une erreur type empirique honnête pour l'estimation moyenne.
    """
    from .utils import RNGStream, Timer

    if n_replications <= 0:
        raise ValueError("n_replications doit être positif")

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
    """Extrait des échantillons de limites secondaires à partir de trajectoires d'épuisement à quatre files.

    La forme retournée reflète le collecteur Ogata du notebook : ``q_same`` est la
    limite secondaire du côté qui a été épuisé, et ``q_opp`` est la limite secondaire
    du côté opposé. Seules les trajectoires qui atteignent l'événement rare sont utilisées.
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
    """Exécute FLS et extrait des échantillons de limites secondaires conditionnelles à quatre files."""
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
    """Exécute une population AMS et extrait des échantillons de limites secondaires conditionnelles."""
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
    """Exécute le fractionnement par redémarrage conditionnel markovien et retourne un résumé."""

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
    """Exécute un échantillonnage de limite Ogata naïf avec une continuation par passage de limite."""

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
    """Construit un tableau compact pour les résumés Ogata naïfs et de fractionnement par redémarrage."""

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
    """Extrait les observables de limite secondaire nb3 bid à partir des résultats de redémarrage.

    ``same_side_result`` doit correspondre au redémarrage à partir de ``Q-1=1`` et
    ciblant ``Q-1=0``. ``opposite_side_result`` doit correspondre au redémarrage à
    partir de ``Q+1=1`` et ciblant ``Q+1=0`` tout en observant ``Q-2``.
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
    """Construit un tableau de comparaison compact à partir des dictionnaires de résumé d'estimateurs."""
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
