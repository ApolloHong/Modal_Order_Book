"""
Abstract base class for Queue Reactive (QR) models.

A QR model defines:
    1. The "feature function" f(Q, i) that maps (LOB state, queue index) → discrete indicator
    2. The intensity functions λ^+, λ^-, λ^T, λ^B, λ^A for each queue,
       parameterized by the indicator f(Q, i)
    3. The laws L^Q_{-I} and L^Q_I for drawing new queue sizes when price shifts

All concrete models (constant, imbalance, ...) inherit from this class.
"""

from abc import ABC, abstractmethod
import numpy as np
from .lob import LOBState
from .events import EventType


class BaseQRModel(ABC):
    """
    Abstract base class for Queue Reactive models.

    Subclasses must implement:
        - feature(state, i): the indicator function f(Q, i)
        - intensities(state, i): returns dict {EventType: λ} for queue i
        - new_queue_law(): callable returning a random queue size
    """

    def __init__(self, I: int, tick_size: float = 0.01):
        """
        Parameters
        ----------
        I : int
            Number of queues on each side.
        tick_size : float
            Price increment between consecutive limits.
        """
        self.I = I
        self.tick_size = tick_size

    @abstractmethod
    def feature(self, state: LOBState, i: int):
        """
        Compute the discrete indicator f(Q, i).
        
        For constant model: just returns the queue index.
        For imbalance model: returns discretized imbalance.
        """
        pass

    @abstractmethod
    def intensities(self, state: LOBState, i: int) -> dict[EventType, float]:
        """
        Return all event intensities for queue i given the LOB state.
        
        Returns
        -------
        dict mapping EventType → intensity (float ≥ 0)
        Only include event types that are possible for this queue in this state.
        """
        pass

    @abstractmethod
    def new_queue_law(self) -> callable:
        """
        Return a callable that draws a random queue size
        for newly appearing queues at the boundary.
        """
        pass

    # ------------------------------------------------------------------ #
    #  Derived methods (shared by all models)
    # ------------------------------------------------------------------ #

    def all_intensities(self, state: LOBState) -> list[tuple[int, EventType, float]]:
        """
        Compute all non-zero intensities across all queues.
        
        Returns
        -------
        list of (queue_index, event_type, intensity) tuples
        """
        result = []
        for i in state.all_indices:
            for etype, lam in self.intensities(state, i).items():
                if lam > 0:
                    result.append((i, etype, lam))
        return result

    def total_intensity(self, state: LOBState) -> float:
        """Sum of all intensities Λ = Σ λ."""
        return sum(lam for _, _, lam in self.all_intensities(state))

    def create_initial_state(self, q_bid: float = 5.0, q_ask: float = 5.0,
                             mid_price: float = 100.0) -> LOBState:
        """
        Create a simple initial LOB state for I=1.
        Override for more complex initializations.
        """
        queues = {}
        for i in range(-self.I, 0):
            queues[i] = q_bid
        for i in range(1, self.I + 1):
            queues[i] = q_ask
        return LOBState(self.I, queues, mid_price, self.tick_size)
