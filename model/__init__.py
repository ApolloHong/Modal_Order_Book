"""
Queue Reactive (QR) Order Book Models
======================================

A modular framework for simulating limit order book dynamics
using Queue Reactive models.

Quick start:
    from model import QRConstant, Simulator, LOBState

    model = QRConstant(I=1)
    state = model.create_initial_state(q_bid=5, q_ask=5)
    sim = Simulator(model, seed=42)
    result = sim.run(state, T=100.0)
"""

from model.lob import LOBState
from model.events import Event, EventType
from model.base import BaseQRModel
from model.qr_constant import QRConstant
from model.qr_imbalance import QRImbalance
from model.simulator import Simulator, SimulationResult
from model import analysis

__all__ = [
    "LOBState",
    "Event",
    "EventType",
    "BaseQRModel",
    "QRConstant",
    "QRImbalance",
    "Simulator",
    "SimulationResult",
    "analysis",
]
