"""
Event types for the Queue Reactive model.

Events that can occur on each queue:
    ADD     (A)  : A new limit order arrives → queue size +1
    CANCEL  (C)  : An existing order is cancelled → queue size -1
    TRADE   (T)  : A market order consumes one unit → queue size -1
                   (only on first limits Q_{±1})

Special events when a first limit is empty:
    BID     (B)  : A buy order fills the empty ask-side limit
                   → triggers price shift LEFT (price decreases)
    ASK     (A)  : A sell order fills the empty bid-side limit
                   → triggers price shift RIGHT (price increases)

Extended event (for more complex models):
    TOTAL_CONSUMPTION (TC) : A market order depletes the entire first limit
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class EventType(Enum):
    ADD = auto()       # Limit order insertion: +1
    CANCEL = auto()    # Order cancellation: -1
    TRADE = auto()     # Market order (first limits only): -1
    BID = auto()       # Fill empty limit from bid side → price shift
    ASK = auto()       # Fill empty limit from ask side → price shift
    TOTAL_CONSUMPTION = auto()  # Full depletion of first limit (extension)


@dataclass
class Event:
    """
    A single event in the LOB simulation.
    
    Attributes
    ----------
    time : float
        Absolute time when the event occurs.
    event_type : EventType
        Type of the event (ADD, CANCEL, TRADE, BID, ASK).
    queue_index : int
        Index of the queue affected (e.g., -1 for best bid, 1 for best ask).
    size : float
        Size change (usually +1 or -1; can be different for extensions).
    """
    time: float
    event_type: EventType
    queue_index: int
    size: float = 1.0

    @property
    def is_price_changing(self) -> bool:
        """Does this event trigger a reference price change?"""
        return self.event_type in (EventType.BID, EventType.ASK)

    def __repr__(self):
        return (f"Event(t={self.time:.6f}, {self.event_type.name}, "
                f"Q[{self.queue_index}], Δ={self.size:+.0f})")
