"""
Four-Queue Hawkes LOB System (I=2).

Implements Section 1.2.4 (corrected signs) and Section 1.2.5 from MODAL v4.

Four queues: N^{+1}, N^{+2} (bid), N^{-1}, N^{-2} (ask).

First limits (i ∈ {+1, -1}) — Hawkes dynamics (eq. 3, corrected signs):
    λ^{i,+} = μ^{i,+}                                          (constant)
    λ^{i,-} = μ^{i,-}  - ∫ φ(t-s)(dN^{i,+} - dN^{i,-})        (self)
                        + ∫ φ(t-s)(dN^{-i,+} - dN^{-i,-})      (cross)

    Corrected interpretation (v4):
      Own additions    → DECREASE removal rate (inertia: growing queue keeps growing)
      Own removals     → INCREASE removal rate (self-excitation of removals)
      Opposite adds    → INCREASE removal rate (opposite grows → I feel pressure)
      Opposite removals→ INCREASE removal rate (opposite shrinks → contagion/panic)

Second limits (j ∈ {+2, -2}):
    Q1.2.5.1: constant λ^{j,+} and λ^{j,-}
    Q1.2.5.2: λ^{-2,+}(t) = μ^{-2,+} + ∫ a·e^{-b(t-s)} dN^{-1,-}   (eq. 4)
              i.e., first limit removals EXCITE second limit additions
              ("rush to queue" when first limit is being depleted)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FourQueueResult:
    """Result of a 4-queue Hawkes simulation."""
    times: np.ndarray
    # Queue sizes: q[i] for i in {+1, -1, +2, -2}
    q_paths: dict[int, np.ndarray]
    # Removal intensities for first limits
    lam_minus_paths: dict[int, np.ndarray]
    # Addition intensities for second limits (if Hawkes-excited)
    lam_plus_2_paths: dict[int, np.ndarray]
    hitting_time: float
    which_hit: int       # queue index that hit zero first (±1)
    hit_zero: bool
    n_events: int


@dataclass
class FourQueueParams:
    """Parameters for the 4-queue system."""
    # First limits: birth (constant) and death (Hawkes)
    mu_plus_1: float = 1.2      # λ^{±1,+} (constant addition rate)
    mu_minus_1: float = 1.5     # μ^{±1,-} (baseline removal rate)
    alpha: float = 0.3          # Hawkes kernel amplitude
    beta: float = 0.5           # Hawkes kernel decay rate

    # Second limits: birth and death
    mu_plus_2: float = 0.8      # λ^{±2,+} baseline addition rate
    mu_minus_2: float = 0.6     # λ^{±2,-} (constant removal rate)

    # Second limit Hawkes excitation (eq. 4): removals of 1st → additions of 2nd
    a_cross: float = 0.0        # 0 = constant (Q1.2.5.1), >0 = excited (Q1.2.5.2)
    b_cross: float = 0.5        # decay rate for cross-excitation

    # Initial conditions
    q1_init: int = 10
    q_neg1_init: int = 10
    q2_init: int = 5
    q_neg2_init: int = 5

    @property
    def ratio(self):
        return self.alpha / self.beta

    @property
    def stationary_lam_minus(self):
        """Stationary mean of λ^{i,-} for first limits (approximate)."""
        r = self.ratio
        if r >= 1:
            return float('inf')
        return (self.mu_minus_1 + r * self.mu_plus_1) / (1 + r)


def simulate_4queue(
    params: FourQueueParams,
    T_max: float = 1000.0,
    rng: Optional[np.random.Generator] = None,
    stop_at_first_limit_zero: bool = True,
    record_every: int = 1,
) -> FourQueueResult:
    """
    Simulate the 4-queue Hawkes LOB system.

    Uses Ogata's thinning algorithm with the corrected v4 sign convention.

    The excitation state H_i tracks the Hawkes contribution to λ^{i,-}:
        H_{+1} updates:
            Own add (N^{+1,+}):     H_{+1} -= α   (own adds decrease removal)
            Own remove (N^{+1,-}):  H_{+1} += α   (own removals increase removal)
            Opp add (N^{-1,+}):     H_{+1} += α   (opp adds increase my removal)
            Opp remove (N^{-1,-}):  H_{+1} += α   (opp removals increase my removal)
        Symmetric for H_{-1}.

    The excitation state G_j tracks Hawkes contribution to λ^{j,+} (2nd limit):
        G_{+2} updates when N^{+1,-} fires: G_{+2} += a_cross
        G_{-2} updates when N^{-1,-} fires: G_{-2} += a_cross
    """
    if rng is None:
        rng = np.random.default_rng()

    p = params
    q = {1: p.q1_init, -1: p.q_neg1_init, 2: p.q2_init, -2: p.q_neg2_init}
    t = 0.0

    # Hawkes excitation states
    H = {1: 0.0, -1: 0.0}     # for λ^{±1,-}
    G = {2: 0.0, -2: 0.0}     # for λ^{±2,+}

    # Recording
    times_list = [0.0]
    q_rec = {i: [q[i]] for i in [1, -1, 2, -2]}
    lm_rec = {1: [p.mu_minus_1], -1: [p.mu_minus_1]}
    lp2_rec = {2: [p.mu_plus_2], -2: [p.mu_plus_2]}

    n_events = 0
    evt_count = 0

    while t < T_max:
        # ── Current intensities ──────────────────────────────────
        # First limits: removal rates (Hawkes)
        lam_m1 = max(0.01, p.mu_minus_1 + H[1])
        lam_mn1 = max(0.01, p.mu_minus_1 + H[-1])

        # First limits: addition rates (constant)
        lam_p1 = p.mu_plus_1 if q[1] >= 0 else 0.0
        lam_pn1 = p.mu_plus_1 if q[-1] >= 0 else 0.0

        # Second limits: addition rates (possibly Hawkes-excited)
        lam_p2 = max(0.01, p.mu_plus_2 + G[2])
        lam_pn2 = max(0.01, p.mu_plus_2 + G[-2])

        # Second limits: removal rates (constant)
        lam_m2 = p.mu_minus_2 if q[2] > 0 else 0.0
        lam_mn2 = p.mu_minus_2 if q[-2] > 0 else 0.0

        # ── Thinning upper bound ─────────────────────────────────
        # H decays toward 0, G decays toward 0
        # Upper bound: use max(current, baseline)
        ub_m1 = p.mu_minus_1 + max(H[1], 0)
        ub_mn1 = p.mu_minus_1 + max(H[-1], 0)
        ub_p2 = p.mu_plus_2 + max(G[2], 0)
        ub_pn2 = p.mu_plus_2 + max(G[-2], 0)

        lam_max = (lam_p1 + ub_m1 + lam_pn1 + ub_mn1 +
                   ub_p2 + lam_m2 + ub_pn2 + lam_mn2 + 0.1)

        # ── Draw candidate ───────────────────────────────────────
        dt = rng.exponential(1.0 / lam_max)
        t += dt
        if t > T_max:
            break

        # Decay all excitation states
        decay = np.exp(-p.beta * dt)
        H[1] *= decay
        H[-1] *= decay
        if p.a_cross > 0:
            decay_cross = np.exp(-p.b_cross * dt)
            G[2] *= decay_cross
            G[-2] *= decay_cross

        # Recompute after decay
        lam_m1 = max(0.01, p.mu_minus_1 + H[1])
        lam_mn1 = max(0.01, p.mu_minus_1 + H[-1])
        lam_p2 = max(0.01, p.mu_plus_2 + G[2])
        lam_pn2 = max(0.01, p.mu_plus_2 + G[-2])
        lam_m2 = p.mu_minus_2 if q[2] > 0 else 0.0
        lam_mn2 = p.mu_minus_2 if q[-2] > 0 else 0.0

        total = (lam_p1 + lam_m1 + lam_pn1 + lam_mn1 +
                 lam_p2 + lam_m2 + lam_pn2 + lam_mn2)

        # Thinning acceptance
        if rng.random() > total / lam_max:
            continue

        # ── Select event ─────────────────────────────────────────
        # 8 possible events:
        rates = [
            lam_p1,    # 0: Q1 add
            lam_m1,    # 1: Q1 remove
            lam_pn1,   # 2: Q-1 add
            lam_mn1,   # 3: Q-1 remove
            lam_p2,    # 4: Q2 add
            lam_m2,    # 5: Q2 remove
            lam_pn2,   # 6: Q-2 add
            lam_mn2,   # 7: Q-2 remove
        ]
        rates = np.array(rates)
        probs = rates / rates.sum()
        event_idx = rng.choice(8, p=probs)

        # ── Apply event + update Hawkes states ───────────────────
        if event_idx == 0:    # Q1 add (N^{+1,+})
            q[1] += 1
            H[1] -= p.alpha     # own add → decrease own removal
            H[-1] += p.alpha    # opp sees my add → increase opp removal
        elif event_idx == 1:  # Q1 remove (N^{+1,-})
            q[1] = max(0, q[1] - 1)
            H[1] += p.alpha     # own remove → increase own removal
            H[-1] += p.alpha    # opp sees my remove → increase opp removal
            G[2] += p.a_cross   # first limit removal → excite second limit addition
        elif event_idx == 2:  # Q-1 add (N^{-1,+})
            q[-1] += 1
            H[-1] -= p.alpha
            H[1] += p.alpha
        elif event_idx == 3:  # Q-1 remove (N^{-1,-})
            q[-1] = max(0, q[-1] - 1)
            H[-1] += p.alpha
            H[1] += p.alpha
            G[-2] += p.a_cross
        elif event_idx == 4:  # Q2 add
            q[2] += 1
        elif event_idx == 5:  # Q2 remove
            q[2] = max(0, q[2] - 1)
        elif event_idx == 6:  # Q-2 add
            q[-2] += 1
        elif event_idx == 7:  # Q-2 remove
            q[-2] = max(0, q[-2] - 1)

        n_events += 1
        evt_count += 1

        # Record
        if evt_count % record_every == 0:
            times_list.append(t)
            for i in [1, -1, 2, -2]:
                q_rec[i].append(q[i])
            lm_rec[1].append(max(0.01, p.mu_minus_1 + H[1]))
            lm_rec[-1].append(max(0.01, p.mu_minus_1 + H[-1]))
            lp2_rec[2].append(max(0.01, p.mu_plus_2 + G[2]))
            lp2_rec[-2].append(max(0.01, p.mu_plus_2 + G[-2]))

        # Check stopping
        if stop_at_first_limit_zero and (q[1] <= 0 or q[-1] <= 0):
            which = 1 if q[1] <= 0 else -1
            times_list.append(t)
            for i in [1, -1, 2, -2]:
                q_rec[i].append(q[i])
            lm_rec[1].append(max(0.01, p.mu_minus_1 + H[1]))
            lm_rec[-1].append(max(0.01, p.mu_minus_1 + H[-1]))
            lp2_rec[2].append(max(0.01, p.mu_plus_2 + G[2]))
            lp2_rec[-2].append(max(0.01, p.mu_plus_2 + G[-2]))
            return FourQueueResult(
                times=np.array(times_list),
                q_paths={i: np.array(q_rec[i]) for i in [1,-1,2,-2]},
                lam_minus_paths={i: np.array(lm_rec[i]) for i in [1,-1]},
                lam_plus_2_paths={i: np.array(lp2_rec[i]) for i in [2,-2]},
                hitting_time=t, which_hit=which, hit_zero=True,
                n_events=n_events)

    return FourQueueResult(
        times=np.array(times_list),
        q_paths={i: np.array(q_rec[i]) for i in [1,-1,2,-2]},
        lam_minus_paths={i: np.array(lm_rec[i]) for i in [1,-1]},
        lam_plus_2_paths={i: np.array(lp2_rec[i]) for i in [2,-2]},
        hitting_time=t, which_hit=0, hit_zero=False,
        n_events=n_events)


# ====================================================================== #
#  Q1.2.5: conditional distributions at first-limit depletion
# ====================================================================== #

def conditional_q2_at_depletion(
    params: FourQueueParams,
    n_runs: int = 1000,
    T_max: float = 5000.0,
    seed: int = 42,
) -> dict:
    """
    Collect the distribution of Q^{+2} and Q^{-2} at the moment
    when the first queue to hit zero does so.

    Returns dict with:
        'q2_when_same_depleted': Q^{+2} when Q^{+1}=0 (or Q^{-2} when Q^{-1}=0)
        'q2_when_opp_depleted':  Q^{+2} when Q^{-1}=0 (or Q^{-2} when Q^{+1}=0)
    """
    rng = np.random.default_rng(seed)
    q2_same = []   # second limit on the SAME side as depletion
    q2_opp = []    # second limit on the OPPOSITE side

    for _ in range(n_runs):
        res = simulate_4queue(params, T_max, rng, stop_at_first_limit_zero=True,
                              record_every=10)
        if not res.hit_zero:
            continue
        w = res.which_hit  # +1 or -1
        # Same side second limit
        same_2 = 2 * np.sign(w)   # +2 if +1 hit, -2 if -1 hit
        opp_2 = -same_2

        q2_same.append(res.q_paths[int(same_2)][-1])
        q2_opp.append(res.q_paths[int(opp_2)][-1])

    return {
        'q2_same': np.array(q2_same),
        'q2_opp': np.array(q2_opp),
        'n_valid': len(q2_same),
    }


# ====================================================================== #
#  Q1.2.4.5: four-model comparison
# ====================================================================== #

def four_model_comparison(
    n_runs: int = 500,
    q_init: int = 10,
    mu_plus: float = 1.2,
    mu_minus: float = 1.5,
    alpha: float = 0.3,
    beta: float = 0.5,
    T_max: float = 2000.0,
    seed: int = 42,
) -> dict:
    """
    Compare hitting times across four models (Q1.2.4.5):
        1. Single Poisson queue
        2. Two independent Poisson queues (min)
        3. Single Hawkes queue
        4. Two coupled Hawkes queues

    Expected ordering: E[T_coupled] < E[T_single_H] < E[T_two_P] < E[T_single_P]
    """
    from model.hawkes import simulate_hawkes_queue, simulate_coupled_hawkes
    from model.hitting_times import simulate_until_hit_zero

    rng = np.random.default_rng(seed)
    results = {}

    # 1. Single Poisson
    ht = []
    for _ in range(n_runs):
        q, t = q_init, 0.0
        while q > 0 and t < T_max:
            dt = rng.exponential(1.0 / (mu_plus + mu_minus))
            t += dt
            q += 1 if rng.random() < mu_plus / (mu_plus + mu_minus) else -1
        ht.append(t if q <= 0 else np.nan)
    results['single_poisson'] = np.array([x for x in ht if not np.isnan(x)])

    # 2. Two independent Poisson
    ht = []
    for _ in range(n_runs):
        r = simulate_until_hit_zero(q_init, q_init, mu_plus, mu_minus,
                                     rng=rng, record_path=False)
        ht.append(r.hitting_time)
    results['two_poisson'] = np.array(ht)

    # 3. Single Hawkes
    ht = []
    for _ in range(n_runs):
        r = simulate_hawkes_queue(q_init, mu_plus, mu_minus, alpha, beta,
                                   T_max, rng, stop_at_zero=True)
        if r.hit_zero:
            ht.append(r.hitting_time)
    results['single_hawkes'] = np.array(ht)

    # 4. Two coupled Hawkes
    ht = []
    for _ in range(n_runs):
        r = simulate_coupled_hawkes(q_init, q_init, mu_plus, mu_minus,
                                     alpha, beta, T_max, rng, stop_at_zero=True)
        if r.hit_zero:
            ht.append(r.hitting_time)
    results['two_hawkes'] = np.array(ht)

    # Summary
    summary = {}
    for name, arr in results.items():
        if len(arr) > 0:
            summary[name] = {
                'mean': arr.mean(),
                'std': arr.std(),
                'ci_95': 1.96 * arr.std() / np.sqrt(len(arr)),
                'n': len(arr),
            }
    return {'raw': results, 'summary': summary}
