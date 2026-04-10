"""
Imbalance Queue Reactive Model (QR Imbalance).

Intensities depend on the **bid-ask imbalance** at first limits:

    f_imb(Q, ±1) = ± floor_{1/10}( (Q_{-1} - Q_1) / (Q_{-1} + Q_1) )

This captures the "balance of power" between buyers and sellers.
When imbalance > 0 (bid side larger), buying pressure is strong,
which affects order flows on both sides.

For I=1, the imbalance is the ONLY feature (since we only have first limits).

Empirical findings from the paper (Model IIb, Figure 6):
  - Limit order insertion at Q_1 INCREASES when opposite (bid) is large
    (imbalance > 0 → price likely closer to ask → profitable to post)
  - Cancellation at Q_1 INCREASES when opposite is small/empty
    (imbalance < 0 → market may move unfavorably → flee)
  - Market orders at Q_1 INCREASE when opposite is large
    (imbalance > 0 → ask price relatively cheap → buy aggressively)

The model stores a table:
    intensities_table[imb_value] = {EventType: λ}
for each discretized imbalance value.
"""

import numpy as np
from .base import BaseQRModel
from .lob import LOBState
from .events import EventType


def _round_tenth(x: float) -> float:
    """Round to nearest 0.1 (the ⌊·⌋_{1/10} operator)."""
    return round(round(x * 10) / 10, 1)


class QRImbalance(BaseQRModel):
    """
    Imbalance-dependent QR model for I=1.
    
    The intensities for queue i=±1 depend on the discretized imbalance:
        imb = ±⌊(Q_{-1} - Q_1) / (Q_{-1} + Q_1)⌋_{1/10}
    
    Parameters
    ----------
    I : int
        Number of queues per side (typically 1).
    intensity_profiles : dict, optional
        Custom intensity profiles. If None, uses parametric defaults.
    base_params : dict, optional
        Base rates around which imbalance modulates.
    """

    # All possible discretized imbalance values
    IMB_VALUES = [round(x * 0.1, 1) for x in range(-10, 11)]  # -1.0, ..., 1.0

    def __init__(self, I: int = 1, intensity_profiles: dict = None,
                 base_params: dict = None, tick_size: float = 0.01):
        super().__init__(I, tick_size)

        if base_params is None:
            base_params = {
                'lambda_plus_base':  0.6,
                'lambda_minus_base': 0.5,
                'lambda_trade_base': 0.3,
                'lambda_bid':        1.5,
                'lambda_ask':        1.5,
                'new_queue_mean':    5.0,
            }
        self.base_params = base_params

        if intensity_profiles is not None:
            self._profiles = intensity_profiles
        else:
            self._profiles = self._build_default_profiles()

    # ------------------------------------------------------------------ #
    #  Default parametric profiles (inspired by HLR paper Figure 6)
    # ------------------------------------------------------------------ #

    def _build_default_profiles(self) -> dict:
        """
        Build intensity lookup table indexed by imbalance value.
        
        For the ASK side (i=1), imbalance f is defined as:
            f = +(Q_{-1} - Q_1) / (Q_{-1} + Q_1)
        
        f > 0: bid is larger → buying pressure
        f < 0: ask is larger → selling pressure
        f = 0: balanced
        
        By symmetry, BID side (i=-1) mirrors the ASK side with f → -f.
        So we only need to define profiles for one side.
        
        Effects of imbalance on ASK queue (i=1):
        ─────────────────────────────────────────
        f >> 0 (strong bid):
            → λ+ ↑  (more sellers post: price attractive for them)
            → λ- ↓  (less cancellation: orders well-positioned)
            → λT ↑  (more buyers hit ask: price seems cheap)
        
        f << 0 (strong ask):
            → λ+ ↓  (fewer sellers post: market moving away)
            → λ- ↑  (more cancellation: flee before price drops)
            → λT ↓  (fewer buyers: price seems expensive)
        """
        bp = self.base_params
        lp0 = bp['lambda_plus_base']
        lm0 = bp['lambda_minus_base']
        lt0 = bp['lambda_trade_base']

        profiles = {}
        for f in self.IMB_VALUES:
            # Modulation factors (smooth sigmoid-like functions of imbalance)
            # λ+ increases with imbalance (more posting when opposite is large)
            mod_plus = 1.0 + 0.4 * f  # range: [0.6, 1.4] × base

            # λ- decreases with imbalance (less cancellation when opposite large)
            mod_minus = 1.0 - 0.3 * f  # range: [0.7, 1.3] × base

            # λT increases with imbalance (more aggressive trading when opp. large)
            mod_trade = 1.0 + 0.5 * f  # range: [0.5, 1.5] × base

            profiles[f] = {
                EventType.ADD:    max(0.01, lp0 * mod_plus),
                EventType.CANCEL: max(0.01, lm0 * mod_minus),
                EventType.TRADE:  max(0.01, lt0 * mod_trade),
            }

        return profiles

    # ------------------------------------------------------------------ #
    #  Interface implementation
    # ------------------------------------------------------------------ #

    def feature(self, state: LOBState, i: int) -> float:
        """
        Compute the discretized imbalance feature.
        
        For i=±1:
            f_imb(Q, ±1) = ± ⌊(Q_{-1} - Q_1) / (Q_{-1} + Q_1)⌋_{1/10}
        
        Convention: positive f means the SAME side as i has less volume
        (i.e., the queue at i is "under pressure").
        """
        qb = state.q(-1)
        qa = state.q(1)
        total = qb + qa

        if total == 0:
            return 0.0

        raw_imb = (qb - qa) / total  # > 0 when bid is larger

        if i > 0:
            # Ask side: f = +imb (positive when bid larger = buying pressure)
            return _round_tenth(np.clip(raw_imb, -1.0, 1.0))
        else:
            # Bid side: f = -imb (by symmetry, mirror)
            return _round_tenth(np.clip(-raw_imb, -1.0, 1.0))

    def intensities(self, state: LOBState, i: int) -> dict[EventType, float]:
        """
        Return event intensities for queue i, modulated by imbalance.
        """
        qi = state.q(i)

        if qi > 0:
            # Non-empty queue: look up imbalance-dependent rates
            f = self.feature(state, i)
            profile = self._profiles.get(f, self._profiles[0.0])
            result = dict(profile)  # copy

            # Cancel only if queue non-empty
            if qi <= 0:
                result.pop(EventType.CANCEL, None)

            # Trade only at first limits
            if abs(i) != 1:
                result.pop(EventType.TRADE, None)

            return result

        elif abs(i) == 1:
            # Empty first limit → BID/ASK events
            bp = self.base_params
            if i == 1:
                return {
                    EventType.ASK: bp['lambda_ask'],
                    EventType.BID: bp['lambda_bid'],
                }
            else:
                return {
                    EventType.BID: bp['lambda_bid'],
                    EventType.ASK: bp['lambda_ask'],
                }
        else:
            # Empty non-first limit
            f = self.feature(state, i)
            profile = self._profiles.get(f, self._profiles[0.0])
            return {EventType.ADD: profile[EventType.ADD]}

    def new_queue_law(self) -> callable:
        mean = self.base_params.get('new_queue_mean', 5.0)
        p = 1.0 / (1.0 + mean)

        def draw():
            return float(np.random.geometric(p))
        return draw

    # ------------------------------------------------------------------ #
    #  Utility: inspect the profile
    # ------------------------------------------------------------------ #

    def profile_table(self) -> dict:
        """Return the full intensity profile table for inspection/plotting."""
        return {f: {k.name: v for k, v in prof.items()}
                for f, prof in self._profiles.items()}

    def expected_drift(self, f: float) -> float:
        """Expected queue drift at a given imbalance value."""
        prof = self._profiles.get(_round_tenth(f), self._profiles[0.0])
        return prof[EventType.ADD] - prof[EventType.CANCEL] - prof[EventType.TRADE]

    def __repr__(self):
        bp = self.base_params
        return (f"QRImbalance(I={self.I}, "
                f"base: λ+={bp['lambda_plus_base']}, "
                f"λ-={bp['lambda_minus_base']}, "
                f"λT={bp['lambda_trade_base']})")
