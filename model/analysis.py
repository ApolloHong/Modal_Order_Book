"""
Analysis utilities for QR model simulations.

Computes key metrics from simulation results:
    - Mean reversion ratio η = N_c / (2 N_a)
    - Volatility (at various sampling frequencies)
    - Empirical stationary distribution
    - Inter-event time statistics
    - Imbalance trajectory
"""

import numpy as np
from typing import Optional
from .simulator import SimulationResult
from .events import EventType


# ====================================================================== #
#  Mean Reversion Ratio
# ====================================================================== #

def mean_reversion_ratio(result: SimulationResult) -> dict:
    """
    Compute the mean reversion ratio η = N_c / (2 N_a).
    
    From Robert & Rosenbaum (2011):
        N_c = number of CONTINUATIONS (consecutive moves in same direction)
        N_a = number of ALTERNATIONS (consecutive moves in opposite directions)
        η = N_c / (2 N_a)
    
    η < 0.5 → strong mean reversion (price tends to reverse)
    η = 0.5 → random walk (no memory)
    η > 0.5 → trending / momentum
    
    Returns
    -------
    dict with keys: 'eta', 'N_c', 'N_a', 'n_moves', 'directions'
    """
    # Extract price moves (non-zero changes only)
    _, prices = result.price_array()
    dp = np.diff(prices)
    moves = dp[dp != 0]

    if len(moves) < 2:
        return {'eta': np.nan, 'N_c': 0, 'N_a': 0, 'n_moves': len(moves)}

    # Direction of each move: +1 or -1
    directions = np.sign(moves)

    # Count continuations and alternations
    N_c = 0  # continuations (same direction twice)
    N_a = 0  # alternations (opposite directions)
    for k in range(1, len(directions)):
        if directions[k] == directions[k - 1]:
            N_c += 1
        else:
            N_a += 1

    eta = N_c / (2 * N_a) if N_a > 0 else np.inf

    return {
        'eta': eta,
        'N_c': N_c,
        'N_a': N_a,
        'n_moves': len(moves),
        'directions': directions,
    }


# ====================================================================== #
#  Volatility
# ====================================================================== #

def compute_volatility(result: SimulationResult,
                       dt: float = 10.0,
                       unit: str = 'bps') -> dict:
    """
    Compute realized volatility at sampling frequency dt.
    
    Parameters
    ----------
    dt : float
        Sampling interval in seconds (default: 10s).
    unit : str
        'bps' for basis points, 'pct' for percent, 'raw' for raw.
    
    Returns
    -------
    dict with 'vol', 'returns', 'sample_prices', 'sample_times'
    """
    t_arr, p_arr = result.price_array()
    if len(p_arr) < 2:
        return {'vol': np.nan, 'returns': np.array([])}

    T = t_arr[-1]
    sample_times = np.arange(0, T, dt)
    sample_prices = np.interp(sample_times, t_arr, p_arr)

    returns = np.diff(sample_prices) / sample_prices[:-1]

    scale = {'bps': 1e4, 'pct': 100, 'raw': 1}.get(unit, 1e4)
    vol = returns.std() * scale

    return {
        'vol': vol,
        'returns': returns * scale,
        'sample_prices': sample_prices,
        'sample_times': sample_times,
        'unit': unit,
    }


# ====================================================================== #
#  Stationary Distribution
# ====================================================================== #

def stationary_distribution(result: SimulationResult,
                            queue_index: int,
                            burn_in_frac: float = 0.1,
                            max_n: int = 50) -> dict:
    """
    Estimate the empirical stationary distribution of queue i.
    
    Parameters
    ----------
    queue_index : int
        Which queue (-1 for bid, 1 for ask, etc.)
    burn_in_frac : float
        Fraction of snapshots to discard as burn-in.
    max_n : int
        Maximum queue size to consider.
    
    Returns
    -------
    dict with 'pmf', 'ns', 'mean', 'std', 'samples'
    """
    snaps = result.snapshots
    if not snaps:
        raise ValueError("No snapshots available. Run simulation with snapshot_interval.")

    start = int(burn_in_frac * len(snaps))
    samples = np.array([s.q(queue_index) for _, s in snaps[start:]])

    ns = np.arange(0, max_n + 1)
    counts, _ = np.histogram(samples, bins=np.arange(-0.5, max_n + 1.5, 1))
    pmf = counts / counts.sum()

    return {
        'pmf': pmf,
        'ns': ns,
        'mean': samples.mean(),
        'std': samples.std(),
        'samples': samples,
    }


# ====================================================================== #
#  Imbalance Trajectory
# ====================================================================== #

def imbalance_trajectory(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the imbalance trajectory over time.
    
    imb(t) = (Q_{-1}(t) - Q_1(t)) / (Q_{-1}(t) + Q_1(t))
    
    Returns (times, imbalances) arrays.
    """
    if not result.queue_path:
        return np.array([]), np.array([])

    times = []
    imbs = []
    for t, qdict in result.queue_path:
        qb = qdict.get(-1, 0.0)
        qa = qdict.get(1, 0.0)
        total = qb + qa
        imb = (qb - qa) / total if total > 0 else 0.0
        times.append(t)
        imbs.append(imb)

    return np.array(times), np.array(imbs)


# ====================================================================== #
#  Price at Infinity (starting price impact)
# ====================================================================== #

def price_at_infinity(results_batch: list[SimulationResult]) -> dict:
    """
    Estimate the final price distribution from a batch of simulations.
    
    Useful for studying: given an initial price and LOB state,
    what is the distribution of p(+∞)?
    
    Returns
    -------
    dict with 'final_prices', 'mean', 'std', 'ci_95'
    """
    finals = []
    for r in results_batch:
        _, p = r.price_array()
        if len(p) > 0:
            finals.append(p[-1])

    finals = np.array(finals)
    n = len(finals)
    mean = finals.mean()
    std = finals.std()
    ci = 1.96 * std / np.sqrt(n) if n > 0 else np.nan

    return {
        'final_prices': finals,
        'mean': mean,
        'std': std,
        'ci_95': (mean - ci, mean + ci),
        'n_runs': n,
    }


# ====================================================================== #
#  Batch Summary
# ====================================================================== #

def batch_summary(results: list[SimulationResult], dt_vol: float = 10.0) -> dict:
    """
    Compute summary statistics across a batch of simulation runs.
    
    Returns dict with arrays of per-run metrics.
    """
    etas = []
    vols = []
    mean_q_bid = []
    mean_q_ask = []
    n_events_list = []
    n_price_changes_list = []

    for r in results:
        mr = mean_reversion_ratio(r)
        etas.append(mr['eta'])

        v = compute_volatility(r, dt=dt_vol)
        vols.append(v['vol'])

        if r.snapshots:
            sd_bid = stationary_distribution(r, -1)
            sd_ask = stationary_distribution(r, 1)
            mean_q_bid.append(sd_bid['mean'])
            mean_q_ask.append(sd_ask['mean'])

        n_events_list.append(r.n_events)
        n_price_changes_list.append(r.n_price_changes)

    def _stats(arr):
        a = np.array(arr)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return {'mean': np.nan, 'std': np.nan, 'ci_95': np.nan}
        m, s, n = a.mean(), a.std(), len(a)
        return {'mean': m, 'std': s, 'ci_95': 1.96 * s / np.sqrt(n), 'values': a}

    return {
        'eta': _stats(etas),
        'volatility': _stats(vols),
        'mean_q_bid': _stats(mean_q_bid),
        'mean_q_ask': _stats(mean_q_ask),
        'n_events': _stats(n_events_list),
        'n_price_changes': _stats(n_price_changes_list),
    }
