"""
First Passage Time Analysis for Independent Birth-Death Queues.

Specialized simulation and analytics for Questions 1.2.3 of the MODAL project:
    - Simulate two independent queues until one hits zero
    - Brownian motion approximation
    - Hitting time distributions (simulated vs theoretical)
    - Mean hitting time as function of initial conditions
    - Probability that Q_1 hits zero before Q_{-1}

Mathematical background:
    Q_i(t) is a birth-death process: +1 at rate λ⁺, -1 at rate λ⁻.
    
    Diffusion approximation (CLT):
        Q_i(t) ≈ Q_i(0) + μt + σW(t)
    where μ = λ⁺ - λ⁻ (drift), σ² = λ⁺ + λ⁻ (variance rate).
    
    First hitting time of 0 for BM(μ, σ²) starting at x > 0:
        T₀ ~ Inverse Gaussian(μ_IG, λ_IG)
        with μ_IG = x/|μ|, λ_IG = x²/σ²
    
    PDF: f(t) = (x / (σ√(2πt³))) exp(-(x + μt)² / (2σ²t))
    Mean: E[T₀] = x / |μ|  (when μ < 0)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HittingTimeResult:
    """Result of one simulation run (two queues until one hits zero)."""
    hitting_time: float          # time when first queue hits 0
    which_hit: int               # +1 if Q_1 hit first, -1 if Q_{-1} hit first
    q1_path: np.ndarray          # trajectory of Q_1
    q_neg1_path: np.ndarray      # trajectory of Q_{-1}
    times: np.ndarray            # time points
    q1_final: int                # Q_1 at stopping time
    q_neg1_final: int            # Q_{-1} at stopping time


def simulate_until_hit_zero(
    q1_init: int = 10,
    q_neg1_init: int = 10,
    lambda_plus: float = 1.2,
    lambda_minus: float = 1.5,
    rng: Optional[np.random.Generator] = None,
    max_events: int = 1_000_000,
    record_path: bool = True,
) -> HittingTimeResult:
    """
    Simulate two independent birth-death queues until one hits zero.
    
    Each queue independently:
        +1 at rate λ⁺ (Poisson process of additions)
        -1 at rate λ⁻ (Poisson process of removals)
    
    The two queues are INDEPENDENT (Hypothèse: premières limites indépendantes).
    We stop when min(Q_1, Q_{-1}) = 0.
    
    Parameters
    ----------
    q1_init, q_neg1_init : int
        Initial queue sizes.
    lambda_plus, lambda_minus : float
        Intensities (per ms in the MODAL specification).
    record_path : bool
        If True, record full trajectory (for plotting).
    
    Returns
    -------
    HittingTimeResult
    """
    if rng is None:
        rng = np.random.default_rng()

    q1 = q1_init
    q_neg1 = q_neg1_init
    t = 0.0

    # Total rate for the system: each queue has λ⁺ + λ⁻, two queues
    rate_per_queue = lambda_plus + lambda_minus
    total_rate = 2 * rate_per_queue

    # Probability of each event type
    # 4 independent Poisson processes:
    #   Q_1 add:    λ⁺ / total_rate
    #   Q_1 remove: λ⁻ / total_rate
    #   Q_{-1} add:    λ⁺ / total_rate
    #   Q_{-1} remove: λ⁻ / total_rate
    p_q1_add = lambda_plus / total_rate
    p_q1_rem = lambda_minus / total_rate
    p_qn1_add = lambda_plus / total_rate
    # p_qn1_rem = lambda_minus / total_rate  (implicit: 1 - sum of others)

    cumprobs = np.array([p_q1_add, p_q1_add + p_q1_rem,
                         p_q1_add + p_q1_rem + p_qn1_add])

    if record_path:
        times_list = [0.0]
        q1_list = [q1]
        qn1_list = [q_neg1]

    for _ in range(max_events):
        # Draw inter-event time
        dt = rng.exponential(1.0 / total_rate)
        t += dt

        # Choose event
        u = rng.random()
        if u < cumprobs[0]:
            q1 += 1                  # Q_1 add
        elif u < cumprobs[1]:
            q1 = max(0, q1 - 1)     # Q_1 remove
        elif u < cumprobs[2]:
            q_neg1 += 1             # Q_{-1} add
        else:
            q_neg1 = max(0, q_neg1 - 1)  # Q_{-1} remove

        if record_path:
            times_list.append(t)
            q1_list.append(q1)
            qn1_list.append(q_neg1)

        # Check stopping condition
        if q1 <= 0 or q_neg1 <= 0:
            which = 1 if q1 <= 0 else -1
            if record_path:
                return HittingTimeResult(
                    hitting_time=t, which_hit=which,
                    q1_path=np.array(q1_list), q_neg1_path=np.array(qn1_list),
                    times=np.array(times_list),
                    q1_final=q1, q_neg1_final=q_neg1,
                )
            else:
                return HittingTimeResult(
                    hitting_time=t, which_hit=which,
                    q1_path=np.array([]), q_neg1_path=np.array([]),
                    times=np.array([]),
                    q1_final=q1, q_neg1_final=q_neg1,
                )

    # If we hit max_events without reaching 0 (shouldn't happen with μ < 0)
    which = 1 if q1 <= q_neg1 else -1
    if record_path:
        return HittingTimeResult(
            t, which, np.array(q1_list), np.array(qn1_list),
            np.array(times_list), q1, q_neg1)
    return HittingTimeResult(t, which, np.array([]), np.array([]),
                             np.array([]), q1, q_neg1)


def simulate_brownian_until_hit_zero(
    x_init: float = 10.0,
    mu: float = -0.3,
    sigma: float = np.sqrt(2.7),
    dt: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    max_steps: int = 1_000_000,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Simulate arithmetic Brownian motion X(t) = x + μt + σW(t)
    until it hits zero.
    
    Returns (hitting_time, times_array, path_array)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = x_init
    t = 0.0
    times = [0.0]
    path = [x]

    for _ in range(max_steps):
        dW = rng.normal(0, np.sqrt(dt))
        x += mu * dt + sigma * dW
        t += dt
        times.append(t)
        path.append(x)

        if x <= 0:
            return t, np.array(times), np.array(path)

    return t, np.array(times), np.array(path)


# ====================================================================== #
#  Theoretical formulas
# ====================================================================== #

def hitting_time_pdf_brownian(t: np.ndarray, x0: float,
                              mu: float, sigma: float) -> np.ndarray:
    """
    PDF of first hitting time of 0 for BM starting at x0 > 0.
    
    X(t) = x0 + μt + σW(t), T₀ = inf{t: X(t) ≤ 0}
    
    This is the Inverse Gaussian distribution:
        f(t) = (x0 / (σ √(2πt³))) exp(-(x0 + μt)² / (2σ²t))
    
    Valid when μ < 0 (negative drift, so hitting 0 is certain).
    """
    t = np.asarray(t, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        pdf = (x0 / (sigma * np.sqrt(2 * np.pi * t**3))) * \
              np.exp(-(x0 + mu * t)**2 / (2 * sigma**2 * t))
    pdf = np.where(t > 0, pdf, 0.0)
    return pdf


def hitting_time_cdf_brownian(t: np.ndarray, x0: float,
                              mu: float, sigma: float) -> np.ndarray:
    """CDF of the Inverse Gaussian hitting time distribution."""
    from scipy.stats import norm
    t = np.asarray(t, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_t = np.sqrt(t)
        term1 = norm.cdf((-x0 - mu * t) / (sigma * sqrt_t))
        term2 = np.exp(-2 * mu * x0 / sigma**2) * \
                norm.cdf((-x0 + mu * t) / (sigma * sqrt_t))
    cdf = np.where(t > 0, term1 + term2, 0.0)
    return cdf


def hitting_time_mean_brownian(x0: float, mu: float) -> float:
    """
    Mean hitting time E[T₀] for BM with drift μ < 0 starting at x0.
    
    E[T₀] = x0 / |μ|
    """
    if mu >= 0:
        return np.inf  # never hits 0 if drift is non-negative
    return x0 / abs(mu)


def hitting_time_variance_brownian(x0: float, mu: float,
                                    sigma: float) -> float:
    """
    Variance of T₀ for BM with drift μ < 0 starting at x0.
    
    Var[T₀] = x0 σ² / |μ|³
    """
    if mu >= 0:
        return np.inf
    return x0 * sigma**2 / abs(mu)**3


def prob_q1_hits_first_brownian(x1: float, x2: float,
                                mu: float, sigma: float) -> float:
    """
    Approximate probability that queue 1 (starting at x1) hits zero
    before queue 2 (starting at x2), assuming independent BMs.
    
    Uses the exact Inverse Gaussian CDF:
        P(T₁ < T₂) = ∫₀^∞ f_{T₁}(t) · (1 - F_{T₂}(t)) dt
    
    Computed numerically.
    """
    t_max = max(hitting_time_mean_brownian(x1, mu),
                hitting_time_mean_brownian(x2, mu)) * 5
    t_grid = np.linspace(0.001, t_max, 5000)
    dt = t_grid[1] - t_grid[0]

    f1 = hitting_time_pdf_brownian(t_grid, x1, mu, sigma)
    F2 = hitting_time_cdf_brownian(t_grid, x2, mu, sigma)

    # P(T1 < T2) = ∫ f₁(t) · [1 - F₂(t)] dt
    prob = np.sum(f1 * (1 - F2) * dt)
    return float(np.clip(prob, 0, 1))


# ====================================================================== #
#  Batch simulation helpers
# ====================================================================== #

def batch_hitting_times(
    n_runs: int,
    q1_init: int = 10,
    q_neg1_init: int = 10,
    lambda_plus: float = 1.2,
    lambda_minus: float = 1.5,
    seed: int = 42,
) -> dict:
    """
    Run n_runs independent simulations, collect hitting times.
    
    Returns dict with:
        'hitting_times': array of T₀ values
        'which_hit': array of ±1 (which queue hit first)
        'mean_T': mean hitting time
        'std_T': std of hitting time
        'ci_95': 95% CI for mean
        'prob_q1_first': fraction of runs where Q_1 hit first
    """
    rng = np.random.default_rng(seed)
    times = np.zeros(n_runs)
    which = np.zeros(n_runs, dtype=int)

    for k in range(n_runs):
        res = simulate_until_hit_zero(
            q1_init, q_neg1_init, lambda_plus, lambda_minus,
            rng=rng, record_path=False)
        times[k] = res.hitting_time
        which[k] = res.which_hit

    mean_T = times.mean()
    std_T = times.std()
    ci = 1.96 * std_T / np.sqrt(n_runs)

    return {
        'hitting_times': times,
        'which_hit': which,
        'mean_T': mean_T,
        'std_T': std_T,
        'ci_95': (mean_T - ci, mean_T + ci),
        'prob_q1_first': float((which == 1).mean()),
        'n_runs': n_runs,
    }


def scan_initial_conditions(
    q1_range: np.ndarray,
    q_neg1_range: np.ndarray,
    n_runs: int = 500,
    lambda_plus: float = 1.2,
    lambda_minus: float = 1.5,
    seed: int = 42,
) -> dict:
    """
    Scan mean hitting time and P(Q1 first) over a grid of initial conditions.
    
    Returns dict with 2D arrays:
        'mean_T_grid': shape (len(q_neg1_range), len(q1_range))
        'prob_q1_grid': same shape
    """
    rng = np.random.default_rng(seed)
    n1, n2 = len(q1_range), len(q_neg1_range)
    mean_T_grid = np.zeros((n2, n1))
    prob_q1_grid = np.zeros((n2, n1))

    for j, qn1 in enumerate(q_neg1_range):
        for i, q1 in enumerate(q1_range):
            times = []
            q1_firsts = 0
            for _ in range(n_runs):
                res = simulate_until_hit_zero(
                    int(q1), int(qn1), lambda_plus, lambda_minus,
                    rng=rng, record_path=False)
                times.append(res.hitting_time)
                if res.which_hit == 1:
                    q1_firsts += 1
            mean_T_grid[j, i] = np.mean(times)
            prob_q1_grid[j, i] = q1_firsts / n_runs

    return {
        'q1_range': q1_range,
        'q_neg1_range': q_neg1_range,
        'mean_T_grid': mean_T_grid,
        'prob_q1_grid': prob_q1_grid,
    }
