"""Compatibility wrapper for the historical uppercase Hawkes module.

New code should import from ``model.hawkes``.  This file is kept so existing
notebooks using ``from model.Hawkes import ...`` continue to run unchanged.
"""

from .hawkes import *  # noqa: F401,F403
