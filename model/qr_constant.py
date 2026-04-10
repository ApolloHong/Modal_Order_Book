"""
Constant Queue Reactive Model (QR Constant).

The simplest QR model: intensities depend ONLY on the queue index,
not on the actual queue sizes or LOB state.

For I=1 (two queues: Q_{-1} and Q_1):
    - Each non-empty queue has events: ADD (λ+), CANCEL (λ-), TRADE (λT)
    - Each empty first limit has events: BID (λB), ASK (λA)

The "feature function" is trivial: f_c(Q, i) = ℓ_i (just the index).

Parameters
----------
For each queue index i, we store 5 intensities:
    λ_plus[i]  : intensity of limit order arrival (ADD)
    λ_minus[i] : intensity of cancellation (CANCEL)
    λ_trade[i] : intensity of market order (TRADE) — first limits only
    λ_bid[i]   : intensity of bid filling on empty limit
    λ_ask[i]   : intensity of ask filling on empty limit
"""

import numpy as np
from .base import BaseQRModel
from .lob import LOBState
from .events import EventType


class QRConstant(BaseQRModel):
    """
    Constant-intensity QR model.
    
    Example usage:
    >>> model = QRConstant(I=1, params={
    ...     "lambda_plus":  {-1: 1.5, 1: 1.5},   # limit order arrival rate
    ...     "lambda_minus": {-1: 0.6, 1: 0.6},   # cancellation rate
    ...     "lambda_trade": {-1: 0.3, 1: 0.3},   # market order rate
    ...     "lambda_bid":   1.0,                   # bid fill rate (empty ask)
    ...     "lambda_ask":   1.0,                   # ask fill rate (empty bid)
    ...     "new_queue_mean": 5.0,                 # mean of new queue law
    ... })
    """

    def __init__(self, I: int = 1, params: dict = None, tick_size: float = 0.01):
        super().__init__(I, tick_size)

        if params is None:
            # Reasonable defaults inspired by the HLR paper (France Telecom)
            params = self._default_params()

        self.lambda_plus = params["lambda_plus"]    # {i: float}
        self.lambda_minus = params["lambda_minus"]  # {i: float}
        self.lambda_trade = params["lambda_trade"]  # {i: float}
        self.lambda_bid = params.get("lambda_bid", 1.0)    # scalar
        self.lambda_ask = params.get("lambda_ask", 1.0)    # scalar
        self.new_queue_mean = params.get("new_queue_mean", 5.0)

    def _default_params(self) -> dict:
        """
        Default parameters for I=1 constant QR model.
        
        These are "toy" values for getting started.
        In the project you will calibrate them from data or explore
        different parameter regimes.
        """
        indices = list(range(-self.I, 0)) + list(range(1, self.I + 1))
        return {
            "lambda_plus":  {i: 0.6 for i in indices},  # ≈ 0.6 orders/sec
            "lambda_minus": {i: 0.5 for i in indices},  # ≈ 0.5 cancels/sec
            "lambda_trade": {i: 0.3 for i in indices},  # ≈ 0.3 trades/sec
            "lambda_bid":   1.5,   # bid fill rate when first limit empty
            "lambda_ask":   1.5,   # ask fill rate when first limit empty
            "new_queue_mean": 5.0,
        }
        # drift = 0.6 - 0.5 - 0.3 = -0.2 < 0 → queue shrinks → prices move

    # ------------------------------------------------------------------ #
    #  Interface implementation
    # ------------------------------------------------------------------ #

    def feature(self, state: LOBState, i: int):
        """For constant model, feature is just the queue index itself."""
        return i

    def intensities(self, state: LOBState, i: int) -> dict[EventType, float]:
        """
        Compute intensities for queue i given the current state.
        
        Logic:
        - If Q_i > 0 (non-empty queue):
            → ADD, CANCEL are always possible
            → TRADE is possible only for first limits (|i| = 1)
            → CANCEL intensity is 0 if Q_i = 0 (safety)
        - If Q_i = 0 AND |i| = 1 (empty first limit):
            → BID and ASK events can refill the limit
        - If Q_i = 0 AND |i| > 1:
            → Only ADD is possible
        """
        qi = state.q(i)
        result = {}

        if qi > 0:
            # Non-empty queue: orders can arrive and leave
            result[EventType.ADD] = self.lambda_plus.get(i, 0.0)

            # Cancel: only if queue has at least 1 order
            result[EventType.CANCEL] = self.lambda_minus.get(i, 0.0)

            # Trade: only at first limits (|i| = 1)
            if abs(i) == 1:
                result[EventType.TRADE] = self.lambda_trade.get(i, 0.0)

        elif abs(i) == 1:
            # Empty first limit → can be refilled by BID or ASK event
            # A BID event on the ask side (i=1 empty) → price goes UP
            # An ASK event on the bid side (i=-1 empty) → price goes DOWN
            if i == 1:
                # Ask first limit is empty
                result[EventType.ASK] = self.lambda_ask   # sell order fills it
                result[EventType.BID] = self.lambda_bid   # buy order → price shift right
            else:  # i == -1
                # Bid first limit is empty
                result[EventType.BID] = self.lambda_bid   # buy order fills it
                result[EventType.ASK] = self.lambda_ask   # sell order → price shift left

        else:
            # Empty non-first limit: can still receive new orders
            result[EventType.ADD] = self.lambda_plus.get(i, 0.0)

        return result

    def new_queue_law(self) -> callable:
        """
        Returns a callable that generates random queue sizes
        for queues that "appear" when price shifts.
        
        Uses a geometric distribution (discrete, ≥ 1) as a simple model.
        """
        mean = self.new_queue_mean
        p = 1.0 / (1.0 + mean)  # geometric distribution parameter

        def draw():
            return float(np.random.geometric(p))

        return draw

    # ------------------------------------------------------------------ #
    #  Analytical formulas (for validation)
    # ------------------------------------------------------------------ #

    def next_event_probabilities(self, state: LOBState, i: int) -> dict[EventType, float]:
        """
        Probability that the next event on queue i is of each type.
        
        For the constant model with non-empty queue:
            P(ADD)    = λ+ / (λ+ + λ- + λT)
            P(CANCEL) = λ- / (λ+ + λ- + λT)
            P(TRADE)  = λT / (λ+ + λ- + λT)
        """
        rates = self.intensities(state, i)
        total = sum(rates.values())
        if total == 0:
            return {}
        return {etype: lam / total for etype, lam in rates.items()}

    def expected_queue_drift(self, i: int) -> float:
        """
        Expected drift of queue i per unit time:
            drift = λ+ - λ- - λT
        If negative → queue tends to shrink (stability condition).
        """
        lp = self.lambda_plus.get(i, 0.0)
        lm = self.lambda_minus.get(i, 0.0)
        lt = self.lambda_trade.get(i, 0.0) if abs(i) == 1 else 0.0
        return lp - lm - lt

    def __repr__(self):
        return (f"QRConstant(I={self.I}, "
                f"λ+={list(self.lambda_plus.values())}, "
                f"λ-={list(self.lambda_minus.values())}, "
                f"λT={list(self.lambda_trade.values())})")
