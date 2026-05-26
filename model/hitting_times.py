"""Analyse des temps d'atteinte pour deux files birth-death indépendantes.

This project is conducted by Lizhan Hong and Tom Zhang under the supervision of Professor Charles-Albert Lehalle.

Ce module regroupe les outils utilisés dans la question 1.2.3 du projet MODAL :
    - simulation de deux files indépendantes jusqu'à ce que l'une atteigne 0 ;
    - approximation brownienne ;
    - densité et fonction de répartition du temps d'atteinte brownien ;
    - estimation Monte Carlo de \\mathbb{E}[T_min] et d'intervalles de confiance ;
    - probabilité que Q_1 atteigne 0 avant Q_{-1}.

Rappel mathématique :
    Q_i(t) est un processus birth-death : +1 au taux λ⁺, -1 au taux λ⁻.

    Approximation par diffusion (TCL fonctionnel) :
        Q_i(t) ≈ Q_i(0) + μt + σW(t)
    avec μ = λ⁺ - λ⁻ et σ² = λ⁺ + λ⁻.

    Si μ < 0, le temps d'atteinte de 0 du brownien partant de x > 0 suit
    une loi inverse gaussienne.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HittingTimeResult:
    """Résultat d'une simulation de deux files jusqu'à la première atteinte de 0."""
    hitting_time: float          # temps où la première file atteint 0
    which_hit: int               # +1 si Q_1 atteint 0 en premier, -1 pour Q_{-1}
    q1_path: np.ndarray          # trajectoire de Q_1
    q_neg1_path: np.ndarray      # trajectoire de Q_{-1}
    times: np.ndarray            # instants enregistrés
    q1_final: int                # valeur finale de Q_1
    q_neg1_final: int            # valeur finale de Q_{-1}


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
    Simule deux files birth-death indépendantes jusqu'à ce que l'une atteigne 0.

    Pour chaque file :
        +1 au taux λ⁺, ce qui représente un ajout dans la file ;
        -1 au taux λ⁻, ce qui représente un retrait.

    Les deux files sont indépendantes. Le critère d'arrêt est
    min(Q_1, Q_{-1}) = 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    q1 = q1_init
    q_neg1 = q_neg1_init
    t = 0.0

    # Taux total du système : chaque file a le taux λ⁺ + λ⁻.
    rate_per_queue = lambda_plus + lambda_minus
    total_rate = 2 * rate_per_queue

    # Probabilités des quatre types d'événements :
    #   ajout dans Q_1, retrait dans Q_1,
    #   ajout dans Q_{-1}, retrait dans Q_{-1}.
    p_q1_add = lambda_plus / total_rate
    p_q1_rem = lambda_minus / total_rate
    p_qn1_add = lambda_plus / total_rate
    # p_qn1_rem = lambda_minus / total_rate, implicite en fait.

    cumprobs = np.array([p_q1_add, p_q1_add + p_q1_rem,
                         p_q1_add + p_q1_rem + p_qn1_add])

    if record_path:
        times_list = [0.0]
        q1_list = [q1]
        qn1_list = [q_neg1]

    for _ in range(max_events):
        # Temps d'attente jusqu'au prochain événement.
        dt = rng.exponential(1.0 / total_rate)
        t += dt

        # Choix du type d'événement.
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

        # Condition d'arrêt : l'une des deux files est vide.
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

    # Cas de secours si max_events est atteint avant 0.
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
    Simule le mouvement brownien arithmétique X(t) = x + μt + σW(t)
    jusqu'à l'atteinte de 0.

    Renvoie (temps_d_atteinte, grille_de_temps, trajectoire).
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
#  Formules théoriques
# ====================================================================== #

def hitting_time_pdf_brownian(t: np.ndarray, x0: float,
                              mu: float, sigma: float) -> np.ndarray:
    """
    Densité du temps d'atteinte de 0 pour X(t) = x0 + μt + σW(t).

    La formule correspond à une loi inverse gaussienne. Elle est utilisée ici
    lorsque μ < 0, cas où l'atteinte de 0 est presque sûre.
    """
    t = np.asarray(t, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        pdf = (x0 / (sigma * np.sqrt(2 * np.pi * t**3))) * \
              np.exp(-(x0 + mu * t)**2 / (2 * sigma**2 * t))
    pdf = np.where(t > 0, pdf, 0.0)
    return pdf


def hitting_time_cdf_brownian(t: np.ndarray, x0: float,
                              mu: float, sigma: float) -> np.ndarray:
    """Fonction de répartition du temps d'atteinte brownien."""
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
    Espérance \\mathbb{E}[T₀] du temps d'atteinte brownien, pour μ < 0.

    \\mathbb{E}[T₀] = x0 / |μ|.
    """
    if mu >= 0:
        return np.inf  # le temps moyen est infini si le drift est non négatif
    return x0 / abs(mu)


def hitting_time_variance_brownian(x0: float, mu: float,
                                    sigma: float) -> float:
    """
    Variance de T₀ pour un brownien avec drift μ < 0.

    Var[T₀] = x0 σ² / |μ|³.
    """
    if mu >= 0:
        return np.inf
    return x0 * sigma**2 / abs(mu)**3


def prob_q1_hits_first_brownian(x1: float, x2: float,
                                mu: float, sigma: float) -> float:
    """
    Approxime P(T_1 < T_2) pour deux approximations browniennes indépendantes.

    On utilise l'identité :
        P(T_1 < T_2) = ∫ f_{T_1}(t) · [1 - F_{T_2}(t)] dt.
    L'intégrale est calculée numériquement sur une grille de temps.
    """
    t_max = max(hitting_time_mean_brownian(x1, mu),
                hitting_time_mean_brownian(x2, mu)) * 5
    t_grid = np.linspace(0.001, t_max, 5000)
    dt = t_grid[1] - t_grid[0]

    f1 = hitting_time_pdf_brownian(t_grid, x1, mu, sigma)
    F2 = hitting_time_cdf_brownian(t_grid, x2, mu, sigma)

    # P(T1 < T2) = ∫ f1(t) · [1 - F2(t)] dt.
    prob = np.sum(f1 * (1 - F2) * dt)
    return float(np.clip(prob, 0, 1))


# ====================================================================== #
#  Outils de simulation Monte Carlo
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
    Lance n_runs simulations indépendantes et agrège les temps d'atteinte.

    Le dictionnaire retourné contient les temps simulés, la file qui atteint 0
    en premier, la moyenne empirique, l'écart-type, un IC à 95 %, et la
    fréquence empirique de l'événement {Q_1 atteint 0 en premier}.
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
    Étudie \\mathbb{E}[T_min] et P(Q_1 en premier) sur une grille de conditions initiales.

    Les matrices retournées sont indexées par Q_{-1}(0) en ligne et Q_1(0)
    en colonne.
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
