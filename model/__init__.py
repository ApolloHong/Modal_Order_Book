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
from model import hitting_times
from model.hitting_times import (
    HittingTimeResult,
    simulate_until_hit_zero,
    simulate_brownian_until_hit_zero,
    hitting_time_pdf_brownian,
    hitting_time_cdf_brownian,
    hitting_time_mean_brownian,
    hitting_time_variance_brownian,
    prob_q1_hits_first_brownian,
    batch_hitting_times,
    scan_initial_conditions,
)

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
    "hitting_times",
    "HittingTimeResult",
    "simulate_until_hit_zero",
    "simulate_brownian_until_hit_zero",
    "hitting_time_pdf_brownian",
    "hitting_time_cdf_brownian",
    "hitting_time_mean_brownian",
    "hitting_time_variance_brownian",
    "prob_q1_hits_first_brownian",
    "batch_hitting_times",
    "scan_initial_conditions",
]
