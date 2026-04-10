"""
Simulation engine for Queue Reactive models.

Uses the Gillespie algorithm (exact stochastic simulation):
    1. Compute all event intensities given the current LOB state
    2. Total intensity Λ = Σ λ_k
    3. Draw inter-event time Δt ~ Exp(Λ)
    4. Choose event type k with probability λ_k / Λ
    5. Apply the event to the LOB state
    6. Handle price shifts if first limit is depleted
    7. Repeat

This is mathematically exact: the resulting process is the correct
continuous-time Markov chain specified by the intensity functions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .base import BaseQRModel
from .lob import LOBState
from .events import Event, EventType


@dataclass
class SimulationResult:
    """
    Container for simulation output.
    
    Stores the full event history and periodic LOB snapshots
    for later analysis.
    """
    events: list[Event] = field(default_factory=list)
    # Snapshots: list of (time, LOBState copy) taken at regular intervals
    snapshots: list[tuple[float, LOBState]] = field(default_factory=list)
    # Price trajectory: list of (time, mid_price)
    price_path: list[tuple[float, float]] = field(default_factory=list)
    # Queue trajectory: list of (time, {i: q_i})
    queue_path: list[tuple[float, dict[int, float]]] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def n_price_changes(self) -> int:
        return sum(1 for e in self.events if e.is_price_changing)

    def price_array(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, prices) as numpy arrays."""
        if not self.price_path:
            return np.array([]), np.array([])
        t, p = zip(*self.price_path)
        return np.array(t), np.array(p)

    def queue_array(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, queue_sizes) for queue i as numpy arrays."""
        if not self.queue_path:
            return np.array([]), np.array([])
        t = np.array([x[0] for x in self.queue_path])
        q = np.array([x[1].get(i, 0.0) for x in self.queue_path])
        return t, q

    def event_counts(self) -> dict[EventType, int]:
        """Count events by type."""
        counts = {}
        for e in self.events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts

    def event_counts_by_queue(self) -> dict[int, dict[EventType, int]]:
        """Count events by (queue_index, event_type)."""
        counts = {}
        for e in self.events:
            if e.queue_index not in counts:
                counts[e.queue_index] = {}
            d = counts[e.queue_index]
            d[e.event_type] = d.get(e.event_type, 0) + 1
        return counts


class Simulator:
    """
    Gillespie-based simulator for QR models.
    
    Parameters
    ----------
    model : BaseQRModel
        The QR model defining intensity functions.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, model: BaseQRModel, seed: Optional[int] = None):
        self.model = model
        self.rng = np.random.default_rng(seed)

    def run(self, initial_state: LOBState, T: float,
            snapshot_interval: Optional[float] = None,
            max_events: int = 10_000_000,
            record_full_path: bool = True,
            verbose: bool = False) -> SimulationResult:
        """
        Run the simulation from initial_state until time T.
        
        Parameters
        ----------
        initial_state : LOBState
            Starting configuration of the order book.
        T : float
            Total simulation time (in seconds).
        snapshot_interval : float, optional
            If given, record LOB snapshots at this interval.
        max_events : int
            Safety limit on number of events.
        record_full_path : bool
            If True, record every (time, price) and (time, queues) pair.
            Set False for very long simulations to save memory.
        verbose : bool
            Print progress every 10% of simulation time.
        
        Returns
        -------
        SimulationResult
        """
        state = initial_state.copy()
        result = SimulationResult()
        t = 0.0

        # Record initial state
        if record_full_path:
            result.price_path.append((t, state.mid_price))
            result.queue_path.append((t, dict(state.queues)))

        # Snapshot bookkeeping
        next_snapshot_time = snapshot_interval if snapshot_interval else None
        if next_snapshot_time is not None:
            result.snapshots.append((t, state.copy()))

        new_queue_draw = self.model.new_queue_law()
        n_events = 0

        while t < T and n_events < max_events:
            # ---------------------------------------------------------- #
            # Step 1: Compute all intensities
            # ---------------------------------------------------------- #
            all_rates = self.model.all_intensities(state)

            if not all_rates:
                # No possible events (deadlock) — shouldn't happen
                # with well-parameterized models
                if verbose:
                    print(f"  [t={t:.4f}] No events possible, stopping.")
                break

            indices, etypes, lambdas = zip(*all_rates)
            lambdas = np.array(lambdas)
            total_lambda = lambdas.sum()

            if total_lambda <= 0:
                break

            # ---------------------------------------------------------- #
            # Step 2: Draw inter-event time ~ Exp(Λ)
            # ---------------------------------------------------------- #
            dt = self.rng.exponential(1.0 / total_lambda)
            t += dt

            if t > T:
                break

            # ---------------------------------------------------------- #
            # Step 3: Choose which event occurs (proportional to λ_k / Λ)
            # ---------------------------------------------------------- #
            probs = lambdas / total_lambda
            choice = self.rng.choice(len(all_rates), p=probs)
            qi = indices[choice]
            etype = etypes[choice]

            # ---------------------------------------------------------- #
            # Step 4: Apply the event
            # ---------------------------------------------------------- #
            event = self._apply_event(state, t, qi, etype, new_queue_draw)
            result.events.append(event)
            n_events += 1

            # ---------------------------------------------------------- #
            # Step 5: Record state
            # ---------------------------------------------------------- #
            if record_full_path:
                result.price_path.append((t, state.mid_price))
                result.queue_path.append((t, dict(state.queues)))

            # Snapshots at regular intervals
            if next_snapshot_time is not None:
                while next_snapshot_time <= t and next_snapshot_time <= T:
                    result.snapshots.append((next_snapshot_time, state.copy()))
                    next_snapshot_time += snapshot_interval

            # Progress reporting
            if verbose and n_events % (max_events // 10) == 0:
                print(f"  [t={t:.2f}/{T:.0f}] {n_events} events, "
                      f"price={state.mid_price:.4f}")

        if verbose:
            print(f"  Simulation done: {n_events} events in {t:.2f}s, "
                  f"final price={state.mid_price:.4f}")

        return result

    def _apply_event(self, state: LOBState, t: float, qi: int,
                     etype: EventType, new_queue_draw: callable) -> Event:
        """
        Apply a single event to the LOB state (mutates state in-place).
        
        Handles:
        - ADD: Q_i += 1
        - CANCEL: Q_i -= 1 (clamped to 0)
        - TRADE: Q_i -= 1 (first limits only)
        - BID on empty ask: refill or trigger price shift right
        - ASK on empty bid: refill or trigger price shift left
        """
        if etype == EventType.ADD:
            state.set_q(qi, state.q(qi) + 1)
            return Event(t, etype, qi, size=+1)

        elif etype == EventType.CANCEL:
            state.set_q(qi, state.q(qi) - 1)
            return Event(t, etype, qi, size=-1)

        elif etype == EventType.TRADE:
            old_q = state.q(qi)
            state.set_q(qi, old_q - 1)

            # Check if this trade emptied the first limit
            if state.q(qi) <= 0 and abs(qi) == 1:
                # The first limit is now empty.
                # In the simplest I=1 model, this doesn't immediately
                # shift the price — we wait for BID/ASK events.
                pass

            return Event(t, etype, qi, size=-1)

        elif etype == EventType.BID:
            # BID event on an empty first limit
            if qi == 1 and state.q(1) == 0:
                # Empty ask first limit, bid (buy) arrives → price shift RIGHT
                state.shift_right(new_queue_draw)
                return Event(t, etype, qi, size=0)
            elif qi == -1 and state.q(-1) == 0:
                # Empty bid first limit, bid fills it → just refill
                state.set_q(-1, new_queue_draw())
                return Event(t, etype, qi, size=+1)
            else:
                # Shouldn't happen with correct intensity logic
                return Event(t, etype, qi, size=0)

        elif etype == EventType.ASK:
            if qi == -1 and state.q(-1) == 0:
                # Empty bid first limit, ask (sell) arrives → price shift LEFT
                state.shift_left(new_queue_draw)
                return Event(t, etype, qi, size=0)
            elif qi == 1 and state.q(1) == 0:
                # Empty ask first limit, ask fills it → just refill
                state.set_q(1, new_queue_draw())
                return Event(t, etype, qi, size=+1)
            else:
                return Event(t, etype, qi, size=0)

        return Event(t, etype, qi, size=0)

    # ------------------------------------------------------------------ #
    #  Batch simulation (for Monte Carlo studies)
    # ------------------------------------------------------------------ #

    def run_batch(self, initial_state: LOBState, T: float, n_runs: int,
                  seed_offset: int = 0,
                  record_full_path: bool = False,
                  snapshot_interval: Optional[float] = None) -> list[SimulationResult]:
        """
        Run multiple independent simulations (for confidence intervals, etc.).
        
        Parameters
        ----------
        n_runs : int
            Number of independent simulation runs.
        seed_offset : int
            Each run uses seed = seed_offset + run_index.
        
        Returns
        -------
        list of SimulationResult
        """
        results = []
        for k in range(n_runs):
            self.rng = np.random.default_rng(seed_offset + k)
            sim_result = self.run(
                initial_state.copy(), T,
                snapshot_interval=snapshot_interval,
                record_full_path=record_full_path,
            )
            results.append(sim_result)
        return results
