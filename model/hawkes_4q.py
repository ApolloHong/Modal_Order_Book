"""Système LOB (Limit Order Book) à quatre files d'attente de type Hawkes (I=2).

This project is conducted by Lizhan Hong and Tom Zhang under the supervision of Professor Charles-Albert Lehalle.

Implémente la section 1.2.4 (signes corrigés) et la section 1.2.5 du MODAL v4.

Quatre files : N^{+1}, N^{+2} (ask), N^{-1}, N^{-2} (bid).

Premières limites (i ∈ {+1, -1}) — dynamique de Hawkes (éq. 3, signes corrigés) :
    λ^{i,+} = μ^{i,+}                                      (constante)
    λ^{i,-} = μ^{i,-}  - ∫ φ(t-s)(dN^{i,+} - dN^{i,-})     (propre)
                      + ∫ φ(t-s)(dN^{-i,+} + dN^{-i,-})    (croisée)

    Interprétation :
      Ajouts propres      → DIMINUENT le taux de retrait (inertie : une file qui grandit continue de grandir)
      Retraits propres    → AUGMENTENT le taux de retrait (auto-excitation des retraits)
      Ajouts opposés      → AUGMENTENT le taux de retrait (l'opposé grandit → je ressens une pression)
      Retraits opposés    → AUGMENTENT le taux de retrait (l'opposé rétrécit → contagion/panique)

Secondes limites (j ∈ {+2, -2}) :
    Q1.2.5.1 : λ^{j,+} et λ^{j,-} constants
    Q1.2.5.2 : λ^{-2,+}(t) = μ^{-2,+} + ∫ a·e^{-b(t-s)} dN^{-1,-}   (éq. 4)
               λ^{+2,+}(t) = μ^{+2,+} + ∫ a·e^{-b(t-s)} dN^{+1,-}
               c.-à-d., les retraits de la première limite EXCITENT les ajouts de la seconde limite
               ("ruée vers la file" quand la première limite est épuisée)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FourQueueResult:
    """Résultat d'une simulation de Hawkes à 4 files."""
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
    """Paramètres pour le système à 4 files."""
    # Premières limites : naissance (constante) et décès (Hawkes)
    mu_plus_1: float = 1.2      # λ^{±1,+} (taux d'ajout constant)
    mu_minus_1: float = 1.5     # μ^{±1,-} (taux de retrait de base)
    alpha: float = 0.3          # amplitude du noyau de Hawkes
    beta: float = 0.5           # taux de décroissance de Hawkes

    # Secondes limites : naissance et décès
    mu_plus_2: float = 0.8      # λ^{±2,+} taux d'ajout de base
    mu_minus_2: float = 0.6     # λ^{±2,-} (taux de retrait constant)

    # Excitation Hawkes de la seconde limite (éq. 4) : retraits de la 1ère → ajouts de la 2nde
    a_cross: float = 0.0        # 0 = constant (Q1.2.5.1), >0 = excité (Q1.2.5.2)
    b_cross: float = 0.5        # taux de décroissance pour l'excitation croisée

    # Conditions initiales
    q1_init: int = 10       # Q^{+1} : ask première limite
    q_neg1_init: int = 10   # Q^{-1} : bid première limite
    q2_init: int = 5        # Q^{+2} : ask seconde limite
    q_neg2_init: int = 5    # Q^{-2} : bid seconde limite

    @property
    def ratio(self):
        return self.alpha / self.beta

    @property
    def stationary_lam_minus(self):
        """Moyenne stationnaire de λ^{i,-}.

        Avec r = alpha / beta, la moyenne de l'intensité de retrait de la première limite résout

            (1 - r) m^- = mu^- - r mu^+,

        d'où

            m^- = (mu^- - r mu^+) / (1 - r).
        """
        r = self.ratio
        if r >= 1:
            return float('inf')
        return (self.mu_minus_1 - r * self.mu_plus_1) / (1 - r)


def simulate_4queue(
    params: FourQueueParams,
    T_max: float = 1000.0,
    rng: Optional[np.random.Generator] = None,
    stop_at_first_limit_zero: bool = True,
    record_every: int = 1,
) -> FourQueueResult:
    """
    Simule le système LOB à 4 files de Hawkes.

    Utilise l'algorithme de thinning (amincissement) d'Ogata.

    L'état d'excitation H_i suit la contribution Hawkes à λ^{i,-} :
        Mises à jour de H_{+1} :
            Ajout propre (N^{+1,+}) :    H_{+1} -= α   (les ajouts propres diminuent le retrait)
            Retrait propre (N^{+1,-}) :  H_{+1} += α   (les retraits propres augmentent le retrait)
            Ajout opposé (N^{-1,+}) :    H_{+1} += α   (les ajouts opposés augmentent mon retrait)
            Retrait opposé (N^{-1,-}) :  H_{+1} += α   (les retraits opposés augmentent mon retrait)
        Symétrique pour H_{-1}.

    L'état d'excitation G_j suit la contribution Hawkes à λ^{j,+} (2ème limite) :
        G_{+2} est mis à jour quand N^{+1,-} se produit : G_{+2} += a_cross
        G_{-2} est mis à jour quand N^{-1,-} se produit : G_{-2} += a_cross
    """
    if rng is None:
        rng = np.random.default_rng()

    p = params
    q = {1: p.q1_init, -1: p.q_neg1_init, 2: p.q2_init, -2: p.q_neg2_init}
    t = 0.0

    # États d'excitation de Hawkes
    H = {1: 0.0, -1: 0.0}     # for λ^{±1,-}
    G = {2: 0.0, -2: 0.0}     # for λ^{±2,+}

    # Enregistrement
    times_list = [0.0]
    q_rec = {i: [q[i]] for i in [1, -1, 2, -2]}
    lm_rec = {1: [p.mu_minus_1], -1: [p.mu_minus_1]}
    lp2_rec = {2: [p.mu_plus_2], -2: [p.mu_plus_2]}

    n_events = 0
    evt_count = 0
    hit_zero = False
    hitting_time = t
    which_hit = 0

    while t < T_max:
        # ── Intensités actuelles ──────────────────────────────────
        # Premières limites : taux de retrait (Hawkes)
        lam_m1 = max(0.01, p.mu_minus_1 + H[1]) if q[1] > 0 else 0.0
        lam_mn1 = max(0.01, p.mu_minus_1 + H[-1]) if q[-1] > 0 else 0.0

        # Premières limites : taux d'ajout (constante)
        lam_p1 = p.mu_plus_1 if q[1] >= 0 else 0.0
        lam_pn1 = p.mu_plus_1 if q[-1] >= 0 else 0.0

        # Secondes limites : taux d'ajout (possiblement excités par Hawkes)
        lam_p2 = max(0.01, p.mu_plus_2 + G[2])
        lam_pn2 = max(0.01, p.mu_plus_2 + G[-2])

        # Secondes limites : taux de retrait (constante)
        lam_m2 = p.mu_minus_2 if q[2] > 0 else 0.0
        lam_mn2 = p.mu_minus_2 if q[-2] > 0 else 0.0

        # ── Borne supérieure pour le thinning ────────────────────
        # H décroît vers 0, G décroît vers 0
        # Borne supérieure : utiliser max(actuel, base)
        ub_m1 = p.mu_minus_1 + max(H[1], 0) if q[1] > 0 else 0.0
        ub_mn1 = p.mu_minus_1 + max(H[-1], 0) if q[-1] > 0 else 0.0
        ub_p2 = p.mu_plus_2 + max(G[2], 0)
        ub_pn2 = p.mu_plus_2 + max(G[-2], 0)

        lam_max = (lam_p1 + ub_m1 + lam_pn1 + ub_mn1 +
                   ub_p2 + lam_m2 + ub_pn2 + lam_mn2 + 0.1)

        # ── Tirage du candidat ───────────────────────────────────
        dt = rng.exponential(1.0 / lam_max)
        t += dt
        if t > T_max:
            break

        # Décroissance de tous les états d'excitation
        decay = np.exp(-p.beta * dt)
        H[1] *= decay
        H[-1] *= decay
        if p.a_cross > 0:
            decay_cross = np.exp(-p.b_cross * dt)
            G[2] *= decay_cross
            G[-2] *= decay_cross

        # Recalcul après décroissance
        lam_m1 = max(0.01, p.mu_minus_1 + H[1]) if q[1] > 0 else 0.0
        lam_mn1 = max(0.01, p.mu_minus_1 + H[-1]) if q[-1] > 0 else 0.0
        lam_p2 = max(0.01, p.mu_plus_2 + G[2])
        lam_pn2 = max(0.01, p.mu_plus_2 + G[-2])
        lam_m2 = p.mu_minus_2 if q[2] > 0 else 0.0
        lam_mn2 = p.mu_minus_2 if q[-2] > 0 else 0.0

        total = (lam_p1 + lam_m1 + lam_pn1 + lam_mn1 +
                 lam_p2 + lam_m2 + lam_pn2 + lam_mn2)

        # Acceptation du rejet
        if rng.random() > total / lam_max:
            continue

        # ── Sélection de l'événement ─────────────────────────────
        # 8 événements possibles :
        rates = [
            lam_p1,    # 0 : Q+1 ajout
            lam_m1,    # 1 : Q+1 retrait
            lam_pn1,   # 2 : Q-1 ajout
            lam_mn1,   # 3 : Q-1 retrait
            lam_p2,    # 4 : Q+2 ajout
            lam_m2,    # 5 : Q+2 retrait
            lam_pn2,   # 6 : Q-2 ajout
            lam_mn2,   # 7 : Q-2 retrait
        ]
        rates = np.array(rates)
        probs = rates / rates.sum()
        event_idx = rng.choice(8, p=probs)

        # ── Application de l'événement + mise à jour des états Hawkes ──
        if event_idx == 0:    # Q+1 ajout (N^{+1,+})
            q[1] += 1
            H[1] -= p.alpha     # ajout propre → diminue retrait propre
            H[-1] += p.alpha    # l'opposé voit mon ajout → augmente retrait opposé
        elif event_idx == 1:  # Q+1 retrait (N^{+1,-})
            q[1] = max(0, q[1] - 1)
            H[1] += p.alpha     # retrait propre → augmente retrait propre
            H[-1] += p.alpha    # l'opposé voit mon retrait → augmente retrait opposé
            G[2] += p.a_cross   # retrait 1ère limite → excite ajout 2nde limite
        elif event_idx == 2:  # Q-1 ajout (N^{-1,+})
            q[-1] += 1
            H[-1] -= p.alpha
            H[1] += p.alpha
        elif event_idx == 3:  # Q-1 retrait (N^{-1,-})
            q[-1] = max(0, q[-1] - 1)
            H[-1] += p.alpha
            H[1] += p.alpha
            G[-2] += p.a_cross
        elif event_idx == 4:  # Q+2 ajout
            q[2] += 1
        elif event_idx == 5:  # Q+2 retrait
            q[2] = max(0, q[2] - 1)
        elif event_idx == 6:  # Q-2 ajout
            q[-2] += 1
        elif event_idx == 7:  # Q-2 retrait
            q[-2] = max(0, q[-2] - 1)

        n_events += 1
        evt_count += 1

        if not hit_zero and (q[1] <= 0 or q[-1] <= 0):
            hit_zero = True
            hitting_time = t
            which_hit = 1 if q[1] <= 0 else -1

        # Enregistrement
        if evt_count % record_every == 0:
            times_list.append(t)
            for i in [1, -1, 2, -2]:
                q_rec[i].append(q[i])
            lm_rec[1].append(max(0.01, p.mu_minus_1 + H[1]))
            lm_rec[-1].append(max(0.01, p.mu_minus_1 + H[-1]))
            lp2_rec[2].append(max(0.01, p.mu_plus_2 + G[2]))
            lp2_rec[-2].append(max(0.01, p.mu_plus_2 + G[-2]))

        # Vérification de l'arrêt
        if stop_at_first_limit_zero and hit_zero:
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
                hitting_time=hitting_time, which_hit=which_hit, hit_zero=True,
                n_events=n_events)

    return FourQueueResult(
        times=np.array(times_list),
        q_paths={i: np.array(q_rec[i]) for i in [1,-1,2,-2]},
        lam_minus_paths={i: np.array(lm_rec[i]) for i in [1,-1]},
        lam_plus_2_paths={i: np.array(lp2_rec[i]) for i in [2,-2]},
        hitting_time=hitting_time if hit_zero else t,
        which_hit=which_hit,
        hit_zero=hit_zero,
        n_events=n_events)


# ====================================================================== #
#  Q1.2.5 : distributions conditionnelles à l'épuisement de la 1ère limite
# ====================================================================== #

def conditional_q2_at_depletion(
    params: FourQueueParams,
    n_runs: int = 1000,
    T_max: float = 5000.0,
    seed: int = 42,
) -> dict:
    """
    Collecte la distribution de Q^{+2} et Q^{-2} au moment
    où la première file à atteindre zéro le fait.

    Retourne un dictionnaire avec :
        'q2_when_same_depleted' : Q^{+2} quand Q^{+1}=0 (ou Q^{-2} quand Q^{-1}=0)
        'q2_when_opp_depleted'  : Q^{+2} quand Q^{-1}=0 (ou Q^{-2} quand Q^{+1}=0)
    """
    rng = np.random.default_rng(seed)
    q2_same = []   # seconde limite du MÊME côté que l'épuisement
    q2_opp = []    # seconde limite du côté OPPOSÉ

    for _ in range(n_runs):
        res = simulate_4queue(params, T_max, rng, stop_at_first_limit_zero=True,
                              record_every=10)
        if not res.hit_zero:
            continue
        w = res.which_hit  # +1 ou -1
        # Seconde limite du même côté
        same_2 = 2 * np.sign(w)   # +2 si +1 a atteint, -2 si -1 a atteint
        opp_2 = -same_2

        q2_same.append(res.q_paths[int(same_2)][-1])
        q2_opp.append(res.q_paths[int(opp_2)][-1])

    return {
        'q2_same': np.array(q2_same),
        'q2_opp': np.array(q2_opp),
        'n_valid': len(q2_same),
    }


# ====================================================================== #
#  Q1.2.4.5 : comparaison des quatre modèles
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
    Compare les temps d'atteinte (hitting times) entre quatre modèles (Q1.2.4.5) :
        1. Une seule file de Poisson
        2. Deux files de Poisson indépendantes (min)
        3. Une seule file de Hawkes
        4. Deux files de Hawkes couplées

    Ordre attendu : E[T_couplé] < E[T_seule_H] < E[T_deux_P] < E[T_seule_P]
    """
    from model.hawkes import simulate_hawkes_queue, simulate_coupled_hawkes
    from model.hitting_times import simulate_until_hit_zero

    rng = np.random.default_rng(seed)
    results = {}

    # 1. Poisson simple
    ht = []
    for _ in range(n_runs):
        q, t = q_init, 0.0
        while q > 0 and t < T_max:
            dt = rng.exponential(1.0 / (mu_plus + mu_minus))
            t += dt
            q += 1 if rng.random() < mu_plus / (mu_plus + mu_minus) else -1
        ht.append(t if q <= 0 else np.nan)
    results['single_poisson'] = np.array([x for x in ht if not np.isnan(x)])

    # 2. Deux Poissons indépendants
    ht = []
    for _ in range(n_runs):
        r = simulate_until_hit_zero(q_init, q_init, mu_plus, mu_minus,
                                     rng=rng, record_path=False)
        ht.append(r.hitting_time)
    results['two_poisson'] = np.array(ht)

    # 3.  Hawkes simple
    ht = []
    for _ in range(n_runs):
        r = simulate_hawkes_queue(q_init, mu_plus, mu_minus, alpha, beta,
                                   T_max, rng, stop_at_zero=True)
        if r.hit_zero:
            ht.append(r.hitting_time)
    results['single_hawkes'] = np.array(ht)

    # 4.  Hawkes couplées
    ht = []
    for _ in range(n_runs):
        r = simulate_coupled_hawkes(q_init, q_init, mu_plus, mu_minus,
                                     alpha, beta, T_max, rng, stop_at_zero=True)
        if r.hit_zero:
            ht.append(r.hitting_time)
    results['two_hawkes'] = np.array(ht)

    # résumé
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
