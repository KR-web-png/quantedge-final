"""Shared portfolio helpers."""
import numpy as np
import pandas as pd

CANDIDATES = {
    "A Equal weight": {"SPY": 1/7, "EFA": 1/7, "EEM": 1/7, "AGG": 1/7, "GLD": 1/7, "XLE": 1/7, "XLK": 1/7},
    "B Conservative": {"AGG": 0.40, "GLD": 0.15, "SPY": 0.20, "EFA": 0.10, "EEM": 0.05, "XLE": 0.05, "XLK": 0.05},
    "C Balanced":     {"AGG": 0.30, "GLD": 0.10, "SPY": 0.25, "EFA": 0.15, "EEM": 0.05, "XLE": 0.05, "XLK": 0.10},
    "D Defensive":    {"AGG": 0.50, "GLD": 0.20, "SPY": 0.25, "EFA": 0.05},
}


def portfolio_simple_returns(weights, r):
    """Daily SIMPLE return of the portfolio (for fixed weights, reset daily)."""
    w = pd.Series(weights).reindex(r.columns).fillna(0.0)
    return (np.exp(r) - 1) @ w