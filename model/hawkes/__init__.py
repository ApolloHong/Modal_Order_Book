"""
Processus de Hawkes pour la dynamique des files d'attente + compléments théoriques.

Section 1.2.4 : Premières limites non indépendantes via des processus de Hawkes.
Section 2 : Probabilités d'atteinte discrètes vs browniennes.

Modèle de Hawkes (Hypothèse 3, convention corrigée v4) :
    λ⁺(t) = μ⁺                                      (taux de naissance constant)
    λ⁻(t) = μ⁻ - ∫ α·e^{-β(t-s)} dN⁺_s
               + ∫ α·e^{-β(t-s)} dN⁻_s

    Interprétation : les ajouts diminuent l'intensité de retrait future (inertie),
    tandis que les retraits l'augmentent (auto-excitation de l'épuisement).

    Moyenne stationnaire, convention v4 corrigée :
        m⁻ = (μ⁻ - (α/β)μ⁺) / (1 - α/β)
    Condition de stationnarité : α/β < 1
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ====================================================================== #
#  Section 2.1 : Probabilité d'atteinte discrète vs brownienne
# ====================================================================== #

def prob_reach_zero_discrete(a: int, lambda_plus: float,
                             lambda_minus: float) -> float:
    """
    Probabilité exacte d'atteindre 0 pour un processus de naissance-décès
    commençant à a > 0.
    
    P = 1                      si λ⁻ ≥ λ⁺
    P = (λ⁻/λ⁺)^a             si λ⁻ < λ⁺
    """
    if lambda_minus >= lambda_plus:
        return 1.0
    rho = lambda_minus / lambda_plus
    return rho ** a


def prob_reach_zero_brownian(a: float, lambda_plus: float,
                             lambda_minus: float) -> float:
    """
    Probabilité d'atteindre 0 pour un mouvement brownien X(t) = a + μt + σW(t).
    
    P = 1                           si μ ≤ 0
    P = exp(-2μa/σ²)                si μ > 0
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
    Compare les probabilités d'atteinte discrètes vs browniennes sur une plage de a.
    
    Retourne un dictionnaire avec des tableaux pour les probabilités discrètes et MB.
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
#  Section 2.2 : Hawkes 1D — intensité stationnaire
# ====================================================================== #

def _validate_sign_convention(sign_convention: str) -> str:
    """Normalise et valide le nom d'une convention de signes pour Hawkes."""
    convention = sign_convention.lower()
    if convention not in {"v4", "inverse"}:
        raise ValueError("sign_convention doit être 'v4' ou 'inverse'")
    return convention


def hawkes_stationary_intensity(mu_plus: float, mu_minus: float,
                                 alpha: float, beta: float,
                                 sign_convention: str = "v4") -> float:
    """
    Moyenne stationnaire de λ⁻(t) pour le modèle de Hawkes.

    Signes v4 corrigés :
        λ⁻ = μ⁻ - φ*dN⁺ + φ*dN⁻
        m⁻ = (μ⁻ - (α/β)·μ⁺) / (1 - α/β)

    L'argument sign_convention est conservé pour les appels rétrocompatibles, mais
    la convention d'affectation est unifiée à la formule v4 ci-dessus.

    Nécessite α/β < 1 pour la stationnarité.
    """
    _validate_sign_convention(sign_convention)
    ratio = alpha / beta
    return (mu_minus - ratio * mu_plus) / (1 - ratio)



# ====================================================================== #
#  Section 1.2.4 : Simulation de file d'attente Hawkes (file unique)
# ====================================================================== #

@dataclass
class HawkesQueueResult:
    """Résultat d'une simulation de file d'attente Hawkes."""
    times: np.ndarray             # temps des événements
    queue_path: np.ndarray        # taille de la file après chaque événement
    lambda_minus_path: np.ndarray  # λ⁻(t) à chaque événement
    event_types: np.ndarray        # +1 pour ajout, -1 pour retrait
    hitting_time: float           # temps où la file atteint 0 (ou T_max)
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
    Simule une file unique avec une intensité de retrait de Hawkes.

    λ⁺(t) = μ⁺  (constante)
    v4 :
        λ⁻(t) = μ⁻ - Σ_{add} α·e^{-β(t-s)} + Σ_{rem} α·e^{-β(t-s)}
    inverse :
        λ⁻(t) = μ⁻ + Σ_{add} α·e^{-β(t-s)} - Σ_{rem} α·e^{-β(t-s)}

    Utilise l'algorithme de thinning (amincissement) d'Ogata :
        1. Calculer la borne supérieure λ_max = λ⁺ + max(λ⁻(t), μ⁻)
        2. Tirer Δt ~ Exp(λ_max)
        3. Accepter avec une probabilité (λ⁺ + λ⁻(t)) / λ_max
        4. Si accepté, choisir ajout ou retrait proportionnellement

    Quand la file est vide et stop_at_zero=False, les événements de retrait sont
    désactivés jusqu'à ce qu'un ajout remplisse la file.
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

    # Suivre l'état H(t) de Hawkes. Il décroît exponentiellement entre les événements
    # et saute selon la convention de signes choisie aux événements acceptés.
    H = 0.0  # niveau d'excitation actuel

    while t < T_max:
        # λ⁻(t) actuel = μ⁻ + H (limité à ≥ 0)
        lam_minus_current = max(0.0, mu_minus + H) if q > 0 else 0.0
        lam_plus_current = mu_plus

        total_rate = lam_plus_current + lam_minus_current

        if total_rate <= 0:
            break

        # Borne supérieure pour le thinning :
        # Si H > 0, λ⁻ = μ⁻ + H décroît → valeur actuelle est le max.
        # Si H < 0, λ⁻ augmente vers μ⁻ → μ⁻ est le max.
        lam_minus_max = mu_minus + max(H, 0) if q > 0 else 0.0
        lam_max = lam_plus_current + lam_minus_max + 0.01

        # Tirer le temps inter-événement candidat
        dt = rng.exponential(1.0 / lam_max)
        t += dt

        if t > T_max:
            break

        # Décroître H au temps actuel
        H *= np.exp(-beta * dt)

        # Acceptation/rejet (thinning)
        lam_minus_now = max(0.0, mu_minus + H) if q > 0 else 0.0
        total_now = lam_plus_current + lam_minus_now

        if rng.random() > total_now / lam_max:
            continue  # rejet (thinning)

        # Événement accepté — choisir le type
        if rng.random() < lam_plus_current / total_now:
            # Événement d'ajout
            q += 1
            H += add_jump
            event_type = +1
        else:
            # Événement de retrait
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
#  Section 1.2.4 Q2 : Deux files de Hawkes couplées
# ====================================================================== #

@dataclass
class CoupledHawkesResult:
    """Résultat d'une simulation de deux files de Hawkes couplées."""
    times: np.ndarray
    q1_path: np.ndarray        # file ask
    q_neg1_path: np.ndarray    # file bid
    lam_minus_1_path: np.ndarray   # λ⁻ pour ask
    lam_minus_neg1_path: np.ndarray  # λ⁻ pour bid
    hitting_time: float
    which_hit: int             # +1 ou -1
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
    Simule deux files de Hawkes couplées.
    
    Chaque file a :
        λ⁺_i(t) = μ⁺
        λ⁻_i(t) = μ⁻ + H_i(t)

    Convention v4 :
        tout événement d'ajout contribue pour -α aux états de retrait futurs ;
        tout événement de retrait contribue pour +α aux états de retrait futurs.

    Convention inverse :
        reproduit l'implémentation historique : les ajouts propres excitent les
        retraits propres, les retraits propres les inhibent, tandis que les effets
        croisés conservent l'interprétation de pression d'épuisement.

    Quand une file est vide et stop_at_zero=False, son intensité de retrait est
    mise à zéro jusqu'à ce qu'elle soit remplie par un ajout.
    """
    if rng is None:
        rng = np.random.default_rng()
    convention = _validate_sign_convention(sign_convention)

    q1, qn1 = q1_init, q_neg1_init
    t = 0.0
    H1, Hn1 = 0.0, 0.0  # états d'excitation

    times_list = [0.0]
    q1_list, qn1_list = [q1], [qn1]
    lm1_list = [mu_minus]
    lmn1_list = [mu_minus]

    while t < T_max:
        # Intensités actuelles
        lm1 = max(0.0, mu_minus + H1) if q1 > 0 else 0.0
        lmn1 = max(0.0, mu_minus + Hn1) if qn1 > 0 else 0.0

        # Borne supérieure : si H > 0, il décroîtra → actuel est le max.
        # Si H < 0, il décroîtra vers 0 → μ⁻ est le max.
        lm1_max = mu_minus + max(H1, 0) if q1 > 0 else 0.0
        lmn1_max = mu_minus + max(Hn1, 0) if qn1 > 0 else 0.0
        lam_max = 2 * mu_plus + lm1_max + lmn1_max + 0.01

        if lam_max <= 0:
            break

        dt = rng.exponential(1.0 / lam_max)
        t += dt
        if t > T_max:
            break

        # Décroître les excitations
        decay = np.exp(-beta * dt)
        H1 *= decay
        Hn1 *= decay

        # Thinning
        lm1 = max(0.0, mu_minus + H1) if q1 > 0 else 0.0
        lmn1 = max(0.0, mu_minus + Hn1) if qn1 > 0 else 0.0
        total_now = 2 * mu_plus + lm1 + lmn1

        if rng.random() > total_now / lam_max:
            continue

        # Choisir quel événement : 4 types
        # Q1 ajout (μ⁺), Q1 retrait (lm1), Q-1 ajout (μ⁺), Q-1 retrait (lmn1)
        u = rng.random() * total_now
        if u < mu_plus:
            # Q1 ajout
            q1 += 1
            if convention == "v4":
                H1 -= alpha
                Hn1 -= alpha
            else:
                H1 += alpha
                Hn1 -= alpha
        elif u < mu_plus + lm1:
            # Q1 retrait
            q1 = max(0, q1 - 1)
            if convention == "v4":
                H1 += alpha
                Hn1 += alpha
            else:
                H1 -= alpha
                Hn1 += alpha
        elif u < 2 * mu_plus + lm1:
            # Q-1 ajout
            qn1 += 1
            if convention == "v4":
                Hn1 -= alpha
                H1 -= alpha
            else:
                Hn1 += alpha
                H1 -= alpha
        else:
            # Q-1 retrait
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
#  Helpers par lots
# ====================================================================== #

def estimate_hawkes_stationary_intensity(
    n_runs: int = 200, T_long: float = 5000.0,
    mu_plus: float = 1.2, mu_minus: float = 1.5,
    alpha: float = 0.3, beta: float = 0.5,
    seed: int = 42,
    sign_convention: str = "v4",
) -> dict:
    """
    Estimer λ⁻ stationnaire en exécutant de longues simulations (sans atteindre zéro).
    
    Retourne un dictionnaire avec la moyenne empirique, l'écart-type et la valeur théorique.
    """
    rng = np.random.default_rng(seed)
    lam_means = []

    for _ in range(n_runs):
        res = simulate_hawkes_queue(
            q_init=50,  # grand initial pour éviter d'atteindre zéro
            mu_plus=mu_plus, mu_minus=mu_minus,
            alpha=alpha, beta=beta, sign_convention=sign_convention,
            T_max=T_long, rng=rng, stop_at_zero=False)
        # Utiliser la seconde moitié pour éviter le burn-in
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