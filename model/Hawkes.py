"""
Hawkes Process for Queue Dynamics + Theoretical Complements.

Section 1.2.4: Non-independent first limits via Hawkes processes.
Section 2: Discrete vs Brownian hitting probabilities.

Hawkes Model (Hypothèse 3, convention corrigée v4):
    λ⁺(t) = μ⁺                              (constant birth rate)
    λ⁻(t) = μ⁻ - ∫ α·e^{-β(t-s)} dN⁺_s
                 + ∫ α·e^{-β(t-s)} dN⁻_s

    Interpretation: additions decrease the future removal intensity (inertia),
    while removals increase it (self-excitation of depletion).

    Stationary mean, v4 convention:
        m⁻ = (μ⁻ - (α/β)μ⁺) / (1 - α/β)
    Stationary mean, inverse historical convention:
        m⁻ = (μ⁻ + (α/β)μ⁺) / (1 + α/β)
    Stationarity condition: α/β < 1
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ====================================================================== #
#  Section 2.1: Discrete vs Brownian hitting probability
# ====================================================================== #

def prob_reach_zero_discrete(a: int, lambda_plus: float,
                              lambda_minus: float) -> float:
    """
    Exact probability of reaching 0 for a birth-death process
    starting at a > 0.
    
    P = 1                       if λ⁻ ≥ λ⁺
    P = (λ⁻/λ⁺)^a              if λ⁻ < λ⁺
    """
    if lambda_minus >= lambda_plus:
        return 1.0
    rho = lambda_minus / lambda_plus
    return rho ** a


def prob_reach_zero_brownian(a: float, lambda_plus: float,
                              lambda_minus: float) -> float:
    """
    Probability of reaching 0 for Brownian motion X(t) = a + μt + σW(t).
    
    P = 1                                   if μ ≤ 0
    P = exp(-2μa/σ²)                        if μ > 0
    """
    mu = lambda_plus - lambda_minus
    if mu <= 0:
        return 1.0
    sigma2 = lambda_plus + lambda_minus
    return np.exp(-2 * mu * a / sigma2)


def compare_hitting_probabilities(
    a_range: np.ndarray,
    lambda_plus: float,
    lambda_minus: float,
) -> dict:
    """
    Compare discrete vs Brownian hitting probabilities over a range of a.
    
    Returns dict with arrays for discrete and BM probabilities.
    """
    p_discrete = np.array([prob_reach_zero_discrete(int(a), lambda_plus, lambda_minus)
                           for a in a_range])
    p_bm = np.array([prob_reach_zero_brownian(float(a), lambda_plus, lambda_minus)
                      for a in a_range])
    return {
        'a_range': a_range,
        'p_discrete': p_discrete,
        'p_brownian': p_bm,
        'rho': lambda_minus / lambda_plus,
        'mu': lambda_plus - lambda_minus,
        'sigma2': lambda_plus + lambda_minus,
    }


# ====================================================================== #
#  Section 2.2: Hawkes 1D — stationary intensity
# ====================================================================== #

def _validate_sign_convention(sign_convention: str) -> str:
    """Normalize and validate a Hawkes sign convention name."""
    convention = sign_convention.lower()
    if convention not in {"v4", "inverse"}:
        raise ValueError("sign_convention must be either 'v4' or 'inverse'")
    return convention


def hawkes_stationary_intensity(mu_plus: float, mu_minus: float,
                                 alpha: float, beta: float,
                                 sign_convention: str = "v4") -> float:
    """
    Stationary mean of λ⁻(t) for the Hawkes model.

    v4 corrected signs:
        λ⁻ = μ⁻ - φ*dN⁺ + φ*dN⁻
        m⁻ = (μ⁻ - (α/β)·μ⁺) / (1 - α/β)

    inverse historical signs:
        λ⁻ = μ⁻ + φ*dN⁺ - φ*dN⁻
        m⁻ = (μ⁻ + (α/β)·μ⁺) / (1 + α/β)

    Requires α/β < 1 for stationarity.
    """
    convention = _validate_sign_convention(sign_convention)
    ratio = alpha / beta
    # if ratio >= 1:
    #     raise ValueError(f"α/β = {ratio:.3f} ≥ 1: process is non-stationary!")
    # if convention == "v4":
    #     return (mu_minus - ratio * mu_plus) / (1 - ratio)
    return (mu_minus - ratio * mu_plus) / (1 - ratio)



# ====================================================================== #
#  Section 1.2.4: Hawkes queue simulation (single queue)
# ====================================================================== #

@dataclass
class HawkesQueueResult:
    """Result of a Hawkes queue simulation."""
    times: np.ndarray           # event times
    queue_path: np.ndarray      # queue size after each event
    lambda_minus_path: np.ndarray  # λ⁻(t) at each event
    event_types: np.ndarray     # +1 for add, -1 for remove
    hitting_time: float         # time when queue hits 0 (or T_max)
    hit_zero: bool


def simulate_hawkes_queue(
    q_init: int = 10,
    mu_plus: float = 1.2,
    mu_minus: float = 1.5,
    alpha: float = 0.3,
    beta: float = 0.5,
    T_max: float = 1000.0,
    rng: Optional[np.random.Generator] = None,
    stop_at_zero: bool = True,
    sign_convention: str = "v4",
) -> HawkesQueueResult:
    """
    Simulate a single queue with Hawkes removal intensity.

    λ⁺(t) = μ⁺  (constant)
    v4:
        λ⁻(t) = μ⁻ - Σ_{add} α·e^{-β(t-s)} + Σ_{rem} α·e^{-β(t-s)}
    inverse:
        λ⁻(t) = μ⁻ + Σ_{add} α·e^{-β(t-s)} - Σ_{rem} α·e^{-β(t-s)}

    Uses Ogata's thinning algorithm:
        1. Compute upper bound λ_max = λ⁺ + max(λ⁻(t), μ⁻)
        2. Draw candidate Δt ~ Exp(λ_max)
        3. Accept with probability (λ⁺ + λ⁻(t)) / λ_max
        4. If accepted, choose add vs remove proportionally

    When the queue is empty and stop_at_zero=False, removal events are
    disabled until an addition refills the queue.
    """
    if rng is None:
        rng = np.random.default_rng()
    convention = _validate_sign_convention(sign_convention)
    add_jump = -alpha if convention == "v4" else alpha
    remove_jump = alpha if convention == "v4" else -alpha

    q = q_init
    t = 0.0

    times_list = [0.0]
    queue_list = [q]
    lam_minus_list = [mu_minus]
    event_types_list = [0]

    # Track the Hawkes state H(t). It decays exponentially between events
    # and jumps according to the chosen sign convention at accepted events.
    H = 0.0  # current excitation level

    while t < T_max:
        # Current λ⁻(t) = μ⁻ + H  (clamped to ≥ 0)
        lam_minus_current = max(0.0, mu_minus + H) if q > 0 else 0.0
        lam_plus_current = mu_plus

        total_rate = lam_plus_current + lam_minus_current

        if total_rate <= 0:
            break

        # Upper bound for thinning:
        # If H > 0, λ⁻ = μ⁻ + H decays → current value is max.
        # If H < 0, λ⁻ increases toward μ⁻ → μ⁻ is the max.
        lam_minus_max = mu_minus + max(H, 0) if q > 0 else 0.0
        lam_max = lam_plus_current + lam_minus_max + 0.01

        # Draw candidate inter-event time
        dt = rng.exponential(1.0 / lam_max)
        t += dt

        if t > T_max:
            break

        # Decay H to current time
        H *= np.exp(-beta * dt)

        # Accept/reject (thinning)
        lam_minus_now = max(0.0, mu_minus + H) if q > 0 else 0.0
        total_now = lam_plus_current + lam_minus_now

        if rng.random() > total_now / lam_max:
            continue  # reject (thinning)

        # Event accepted — choose type
        if rng.random() < lam_plus_current / total_now:
            # Addition event
            q += 1
            H += add_jump
            event_type = +1
        else:
            # Removal event
            q = max(0, q - 1)
            H += remove_jump
            event_type = -1

        times_list.append(t)
        queue_list.append(q)
        lam_minus_list.append(max(0.0, mu_minus + H) if q > 0 else 0.0)
        event_types_list.append(event_type)

        if stop_at_zero and q <= 0:
            return HawkesQueueResult(
                times=np.array(times_list),
                queue_path=np.array(queue_list),
                lambda_minus_path=np.array(lam_minus_list),
                event_types=np.array(event_types_list),
                hitting_time=t, hit_zero=True,
            )

    return HawkesQueueResult(
        times=np.array(times_list),
        queue_path=np.array(queue_list),
        lambda_minus_path=np.array(lam_minus_list),
        event_types=np.array(event_types_list),
        hitting_time=t, hit_zero=False,
    )


# ====================================================================== #
#  Section 1.2.4 Q2: Two coupled Hawkes queues
# ====================================================================== #

@dataclass
class CoupledHawkesResult:
    """Result of two coupled Hawkes queues simulation."""
    times: np.ndarray
    q1_path: np.ndarray        # ask queue
    q_neg1_path: np.ndarray    # bid queue
    lam_minus_1_path: np.ndarray   # λ⁻ for ask
    lam_minus_neg1_path: np.ndarray  # λ⁻ for bid
    hitting_time: float
    which_hit: int             # +1 or -1
    hit_zero: bool


def simulate_coupled_hawkes(
    q1_init: int = 10,
    q_neg1_init: int = 10,
    mu_plus: float = 1.2,
    mu_minus: float = 1.5,
    alpha: float = 0.3,
    beta: float = 0.5,
    T_max: float = 1000.0,
    rng: Optional[np.random.Generator] = None,
    stop_at_zero: bool = True,
    sign_convention: str = "v4",
) -> CoupledHawkesResult:
    """
    Simulate two coupled Hawkes queues.
    
    Each queue has:
        λ⁺_i(t) = μ⁺
        λ⁻_i(t) = μ⁻ + H_i(t)

    v4 convention:
        any addition event contributes -α to the future removal states;
        any removal event contributes +α to the future removal states.

    inverse convention:
        reproduces the historical implementation: own additions excite own
        removals, own removals inhibit them, while cross effects retain the
        depletion-pressure interpretation.

    When a queue is empty and stop_at_zero=False, its removal intensity is
    set to zero until it is refilled by an addition.
    """
    if rng is None:
        rng = np.random.default_rng()
    convention = _validate_sign_convention(sign_convention)

    q1, qn1 = q1_init, q_neg1_init
    t = 0.0
    H1, Hn1 = 0.0, 0.0  # excitation states

    times_list = [0.0]
    q1_list, qn1_list = [q1], [qn1]
    lm1_list = [mu_minus]
    lmn1_list = [mu_minus]

    while t < T_max:
        # Current intensities
        lm1 = max(0.0, mu_minus + H1) if q1 > 0 else 0.0
        lmn1 = max(0.0, mu_minus + Hn1) if qn1 > 0 else 0.0

        # Upper bound: if H > 0, it will decay → current is max.
        # If H < 0, it will decay toward 0 → μ⁻ is the max.
        lm1_max = mu_minus + max(H1, 0) if q1 > 0 else 0.0
        lmn1_max = mu_minus + max(Hn1, 0) if qn1 > 0 else 0.0
        lam_max = 2 * mu_plus + lm1_max + lmn1_max + 0.01

        if lam_max <= 0:
            break

        dt = rng.exponential(1.0 / lam_max)
        t += dt
        if t > T_max:
            break

        # Decay excitations
        decay = np.exp(-beta * dt)
        H1 *= decay
        Hn1 *= decay

        # Thinning
        lm1 = max(0.0, mu_minus + H1) if q1 > 0 else 0.0
        lmn1 = max(0.0, mu_minus + Hn1) if qn1 > 0 else 0.0
        total_now = 2 * mu_plus + lm1 + lmn1

        if rng.random() > total_now / lam_max:
            continue

        # Choose which event: 4 types
        # Q1 add (μ⁺), Q1 remove (lm1), Q-1 add (μ⁺), Q-1 remove (lmn1)
        u = rng.random() * total_now
        if u < mu_plus:
            # Q1 add
            q1 += 1
            if convention == "v4":
                H1 -= alpha
                Hn1 -= alpha
            else:
                H1 += alpha
                Hn1 -= alpha
        elif u < mu_plus + lm1:
            # Q1 remove
            q1 = max(0, q1 - 1)
            if convention == "v4":
                H1 += alpha
                Hn1 += alpha
            else:
                H1 -= alpha
                Hn1 += alpha
        elif u < 2 * mu_plus + lm1:
            # Q-1 add
            qn1 += 1
            if convention == "v4":
                Hn1 -= alpha
                H1 -= alpha
            else:
                Hn1 += alpha
                H1 -= alpha
        else:
            # Q-1 remove
            qn1 = max(0, qn1 - 1)
            if convention == "v4":
                Hn1 += alpha
                H1 += alpha
            else:
                Hn1 -= alpha
                H1 += alpha

        times_list.append(t)
        q1_list.append(q1)
        qn1_list.append(qn1)
        lm1_list.append(max(0.0, mu_minus + H1) if q1 > 0 else 0.0)
        lmn1_list.append(max(0.0, mu_minus + Hn1) if qn1 > 0 else 0.0)

        if stop_at_zero and (q1 <= 0 or qn1 <= 0):
            which = 1 if q1 <= 0 else -1
            return CoupledHawkesResult(
                np.array(times_list), np.array(q1_list), np.array(qn1_list),
                np.array(lm1_list), np.array(lmn1_list),
                t, which, True)

    return CoupledHawkesResult(
        np.array(times_list), np.array(q1_list), np.array(qn1_list),
        np.array(lm1_list), np.array(lmn1_list),
        t, 0, False)


# ====================================================================== #
#  Batch helpers
# ====================================================================== #

def estimate_hawkes_stationary_intensity(
    n_runs: int = 200, T_long: float = 5000.0,
    mu_plus: float = 1.2, mu_minus: float = 1.5,
    alpha: float = 0.3, beta: float = 0.5,
    seed: int = 42,
    sign_convention: str = "v4",
) -> dict:
    """
    Estimate stationary λ⁻ by running long simulations (without hitting zero).
    
    Returns dict with empirical mean, std, and theoretical value.
    """
    rng = np.random.default_rng(seed)
    lam_means = []

    for _ in range(n_runs):
        res = simulate_hawkes_queue(
            q_init=50,  # large initial to avoid hitting zero
            mu_plus=mu_plus, mu_minus=mu_minus,
            alpha=alpha, beta=beta, sign_convention=sign_convention,
            T_max=T_long, rng=rng, stop_at_zero=False)
        # Use second half to avoid burn-in
        half = len(res.lambda_minus_path) // 2
        if half > 10:
            lam_means.append(res.lambda_minus_path[half:].mean())

    arr = np.array(lam_means)
    m_theo = hawkes_stationary_intensity(
        mu_plus, mu_minus, alpha, beta, sign_convention=sign_convention)

    return {
        'empirical_mean': arr.mean(),
        'empirical_std': arr.std(),
        'ci_95': (arr.mean() - 1.96*arr.std()/np.sqrt(len(arr)),
                  arr.mean() + 1.96*arr.std()/np.sqrt(len(arr))),
        'theoretical': m_theo,
        'n_runs': len(arr),
    }
