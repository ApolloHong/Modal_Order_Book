"""
Limit Order Book (LOB) state representation.

The LOB is modeled as 2I queues: Q_{-I}, ..., Q_{-1}, Q_1, ..., Q_I
where negative indices = bid (buy) side, positive = ask (sell) side.

The mid-price is p_0 = (p_1 + p_{-1}) / 2.
Each queue Q_i is attached to price p_i, spaced by tick size δ^p.
"""

import numpy as np
from typing import Optional
from copy import deepcopy


class LOBState:
    """
    State of a limit order book with I queues on each side.
    
    Attributes
    ----------
    I : int
        Number of queues on each side (total = 2I queues).
    queues : dict[int, float]
        Queue sizes keyed by index i ∈ {-I,...,-1, 1,...,I}.
        Q_i ≥ 0 for all i.
    mid_price : float
        Current mid-price p_0 = (p_1 + p_{-1}) / 2.
    tick_size : float
        Price increment δ^p between consecutive limits.
    """

    def __init__(self, I: int, queues: dict[int, float],
                 mid_price: float = 100.0, tick_size: float = 0.01):
        self.I = I
        self.queues = queues  # {-I: q_{-I}, ..., -1: q_{-1}, 1: q_1, ..., I: q_I}
        self.mid_price = mid_price
        self.tick_size = tick_size

    # ------------------------------------------------------------------ #
    #  Convenience accessors
    # ------------------------------------------------------------------ #

    @property
    def bid_indices(self) -> list[int]:
        """Indices of bid (buy) queues: -1, -2, ..., -I."""
        return list(range(-1, -self.I - 1, -1))

    @property
    def ask_indices(self) -> list[int]:
        """Indices of ask (sell) queues: 1, 2, ..., I."""
        return list(range(1, self.I + 1))

    @property
    def all_indices(self) -> list[int]:
        """All queue indices: -I, ..., -1, 1, ..., I."""
        return list(range(-self.I, 0)) + list(range(1, self.I + 1))

    def q(self, i: int) -> float:
        """Get queue size at index i."""
        return self.queues.get(i, 0.0)

    def set_q(self, i: int, value: float):
        """Set queue size at index i (clamped to ≥ 0)."""
        self.queues[i] = max(0.0, value)

    @property
    def best_bid(self) -> float:
        """Best bid price = p_{-1} = mid_price - tick_size / 2."""
        return self.mid_price - self.tick_size / 2

    @property
    def best_ask(self) -> float:
        """Best ask price = p_1 = mid_price + tick_size / 2."""
        return self.mid_price + self.tick_size / 2

    def price_at(self, i: int) -> float:
        """Price level at queue index i."""
        if i > 0:
            return self.mid_price + (i - 0.5) * self.tick_size
        else:
            return self.mid_price + (i + 0.5) * self.tick_size

    # ------------------------------------------------------------------ #
    #  Imbalance measures (useful for later models)
    # ------------------------------------------------------------------ #

    def first_limit_imbalance(self) -> float:
        """
        Imbalance at first limits:
            (Q_{-1} - Q_1) / (Q_{-1} + Q_1)
        Returns 0 if both queues are empty.
        """
        qb, qa = self.q(-1), self.q(1)
        total = qb + qa
        if total == 0:
            return 0.0
        return (qb - qa) / total

    def relative_size(self, i: int) -> float:
        """
        Relative size of queue i among its side:
            Q_i / Σ_{j=1}^{I} Q_{±j}
        """
        sign = 1 if i > 0 else -1
        total = sum(self.q(sign * j) for j in range(1, self.I + 1))
        if total == 0:
            return 0.0
        return self.q(i) / total

    # ------------------------------------------------------------------ #
    #  Price shift mechanics
    # ------------------------------------------------------------------ #

    def shift_right(self, new_queue_law: Optional[callable] = None):
        """
        Price increases by one tick (bid event on empty ask first limit).
        All queues shift: Q_i ← Q_{i+1}.
        Q_I is drawn from new_queue_law; Q_{-I} is lost.
        """
        new_queues = {}
        for i in self.all_indices:
            if i == self.I:
                # Rightmost queue: draw from law or set to 0
                new_queues[i] = new_queue_law() if new_queue_law else 0.0
            else:
                # i ← i+1 (next queue to the right)
                next_i = i + 1 if i + 1 != 0 else 1  # skip 0
                new_queues[i] = self.q(next_i)
        self.queues = new_queues
        self.mid_price += self.tick_size

    def shift_left(self, new_queue_law: Optional[callable] = None):
        """
        Price decreases by one tick (ask event on empty bid first limit).
        All queues shift: Q_i ← Q_{i-1}.
        Q_{-I} is drawn from new_queue_law; Q_I is lost.
        """
        new_queues = {}
        for i in self.all_indices:
            if i == -self.I:
                new_queues[i] = new_queue_law() if new_queue_law else 0.0
            else:
                prev_i = i - 1 if i - 1 != 0 else -1  # skip 0
                new_queues[i] = self.q(prev_i)
        self.queues = new_queues
        self.mid_price -= self.tick_size

    # ------------------------------------------------------------------ #
    #  Display
    # ------------------------------------------------------------------ #

    def copy(self) -> "LOBState":
        return LOBState(self.I, dict(self.queues), self.mid_price, self.tick_size)

    def __repr__(self):
        bid_str = " ".join(f"Q{i}={self.q(i):.0f}" for i in self.bid_indices[::-1])
        ask_str = " ".join(f"Q{i}={self.q(i):.0f}" for i in self.ask_indices)
        return (f"LOB(I={self.I}, mid={self.mid_price:.4f}) "
                f"[BID: {bid_str}] | [ASK: {ask_str}]")
