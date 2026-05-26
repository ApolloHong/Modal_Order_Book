"""
Représentation de l'état du carnet d'ordres limites (LOB - Limit Order Book).

This project is conducted by Lizhan Hong and Tom Zhang under the supervision of Professor Charles-Albert Lehalle.

Le carnet d'ordres est modélisé par 2I files d'attente : Q_{-I}, ..., Q_{-1}, Q_1, ..., Q_I
où les indices négatifs correspondent au côté achat (bid) et les indices positifs au côté vente (ask).

Le prix moyen (mid-price) est p_0 = (p_1 + p_{-1}) / 2.
Chaque file d'attente Q_i est associée à un prix p_i, espacé par le pas de cotation (tick size) δ^p.
"""

import numpy as np
from typing import Optional
from copy import deepcopy


class LOBState:
    """
    État d'un carnet d'ordres limites avec I files d'attente de chaque côté.
    
    Attributs
    ----------
    I : int
        Nombre de files d'attente de chaque côté (total = 2I files).
    queues : dict[int, float]
        Tailles des files d'attente indexées par i ∈ {-I,...,-1, 1,...,I}.
        Q_i ≥ 0 pour tout i.
    mid_price : float
        Prix moyen actuel p_0 = (p_1 + p_{-1}) / 2.
    tick_size : float
        Incrément de prix δ^p entre deux limites consécutives.
    """

    def __init__(self, I: int, queues: dict[int, float],
                 mid_price: float = 100.0, tick_size: float = 0.01):
        self.I = I
        self.queues = queues  # {-I: q_{-I}, ..., -1: q_{-1}, 1: q_1, ..., I: q_I}
        self.mid_price = mid_price
        self.tick_size = tick_size

    # ------------------------------------------------------------------ #
    #  Accesseurs pratiques
    # ------------------------------------------------------------------ #

    @property
    def bid_indices(self) -> list[int]:
        """Indices des files d'achat (bid) : -1, -2, ..., -I."""
        return list(range(-1, -self.I - 1, -1))

    @property
    def ask_indices(self) -> list[int]:
        """Indices des files de vente (ask) : 1, 2, ..., I."""
        return list(range(1, self.I + 1))

    @property
    def all_indices(self) -> list[int]:
        """Tous les indices des files d'attente : -I, ..., -1, 1, ..., I."""
        return list(range(-self.I, 0)) + list(range(1, self.I + 1))

    def q(self, i: int) -> float:
        """Obtient la taille de la file d'attente à l'indice i."""
        return self.queues.get(i, 0.0)

    def set_q(self, i: int, value: float):
        """Définit la taille de la file d'attente à l'indice i (contrainte à ≥ 0)."""
        self.queues[i] = max(0.0, value)

    @property
    def best_bid(self) -> float:
        """Meilleur prix d'achat = p_{-1} = mid_price - tick_size / 2."""
        return self.mid_price - self.tick_size / 2

    @property
    def best_ask(self) -> float:
        """Meilleur prix de vente = p_1 = mid_price + tick_size / 2."""
        return self.mid_price + self.tick_size / 2

    def price_at(self, i: int) -> float:
        """Niveau de prix à l'indice de file i."""
        if i > 0:
            return self.mid_price + (i - 0.5) * self.tick_size
        else:
            return self.mid_price + (i + 0.5) * self.tick_size

    # ------------------------------------------------------------------ #
    #  Mesures de déséquilibre (utiles pour les modèles ultérieurs)
    # ------------------------------------------------------------------ #

    def first_limit_imbalance(self) -> float:
        """
        Déséquilibre (imbalance) aux premières limites :
            (Q_{-1} - Q_1) / (Q_{-1} + Q_1)
        Renvoie 0 si les deux files sont vides.
        """
        qb, qa = self.q(-1), self.q(1)
        total = qb + qa
        if total == 0:
            return 0.0
        return (qb - qa) / total

    def relative_size(self, i: int) -> float:
        """
        Taille relative de la file i au sein de son côté :
            Q_i / Σ_{j=1}^{I} Q_{±j}
        """
        sign = 1 if i > 0 else -1
        total = sum(self.q(sign * j) for j in range(1, self.I + 1))
        if total == 0:
            return 0.0
        return self.q(i) / total

    # ------------------------------------------------------------------ #
    #  Mécanique de décalage des prix
    # ------------------------------------------------------------------ #

    def shift_right(self, new_queue_law: Optional[callable] = None):
        """
        Le prix augmente d'un pas (événement acheteur sur première limite vendeuse vide).
        Toutes les files se décalent : Q_i ← Q_{i+1}.
        Q_I est tiré selon new_queue_law ; Q_{-I} est perdu.
        """
        new_queues = {}
        for i in self.all_indices:
            if i == self.I:
                # File la plus à droite : tirage selon la loi ou initialisation à 0
                new_queues[i] = new_queue_law() if new_queue_law else 0.0
            else:
                # i ← i+1 (file suivante vers la droite)
                next_i = i + 1 if i + 1 != 0 else 1  # sauter le 0
                new_queues[i] = self.q(next_i)
        self.queues = new_queues
        self.mid_price += self.tick_size

    def shift_left(self, new_queue_law: Optional[callable] = None):
        """
        Le prix diminue d'un pas (événement vendeur sur première limite acheteuse vide).
        Toutes les files se décalent : Q_i ← Q_{i-1}.
        Q_{-I} est tiré selon new_queue_law ; Q_I est perdu.
        """
        new_queues = {}
        for i in self.all_indices:
            if i == -self.I:
                new_queues[i] = new_queue_law() if new_queue_law else 0.0
            else:
                prev_i = i - 1 if i - 1 != 0 else -1  # sauter le 0
                new_queues[i] = self.q(prev_i)
        self.queues = new_queues
        self.mid_price -= self.tick_size

    # ------------------------------------------------------------------ #
    #  Affichage
    # ------------------------------------------------------------------ #

    def copy(self) -> "LOBState":
        return LOBState(self.I, dict(self.queues), self.mid_price, self.tick_size)

    def __repr__(self):
        bid_str = " ".join(f"Q{i}={self.q(i):.0f}" for i in self.bid_indices[::-1])
        ask_str = " ".join(f"Q{i}={self.q(i):.0f}" for i in self.ask_indices)
        return (f"LOB(I={self.I}, mid={self.mid_price:.4f}) "
                f"[BID: {bid_str}] | [ASK: {ask_str}]")
