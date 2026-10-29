"""Stage 2: compare candidate portfolios (illustrative weights, to be tested)."""
import numpy as np
import pandas as pd
from src.data import load_returns

r = load_returns()
RF = 0.02   # PLACEHOLDER annual risk-free rate; we will replace it with the FRED T-bill

CANDIDATES = {
    "A Equal weight": {"SPY": 1/7, "EFA": 1/7, "EEM": 1/7, "AGG": 1/7, "GLD": 1/7, "XLE": 1/7, "XLK": 1/7},
    "B Conservative": {"AGG": 0.40, "GLD": 0.15, "SPY": 0.20, "EFA": 0.10, "EEM": 0.05, "XLE": 0.05, "XLK": 0.05},
    "C Balanced":     {"AGG": 0.30, "GLD": 0.10, "SPY": 0.25, "EFA": 0.15, "EEM": 0.05, "XLE": 0.05, "XLK": 0.10},
    "D Defensive":    {"AGG": 0.50, "GLD": 0.20, "SPY": 0.25, "EFA": 0.05},
}
WINDOWS = {
    "GFC": ("2007-10-01", "2009-03-09"),
    "COVID": ("2020-02-19", "2020-03-23"),
    "2022": ("2022-01-01", "2022-12-31"),
}


def portfolio_returns(weights, r):
    """Daily portfolio return for fixed weights (assumes daily rebalancing)."""
    w = pd.Series(weights).reindex(r.columns).fillna(0.0)
    simple = np.exp(r) - 1            # convert log -> simple: portfolio returns add up as SIMPLE returns
    return np.log1p(simple @ w)       # back to log returns for the statistics


def summarise(p):
    cum = np.exp(p.cumsum())
    ann_ret = np.exp(p.mean() * 252) - 1
    vol = p.std() * np.sqrt(252)
    loss = -p
    var99 = np.quantile(loss, 0.99)           # a first, rough historical VaR (the real lesson comes next)
    return {
        "annual_return": ann_ret,
        "volatility": vol,
        "sharpe": (ann_ret - RF) / vol,
        "max_drawdown": (cum / cum.cummax() - 1).min(),
        "VaR99_1d": var99,
        "ES99_1d": loss[loss >= var99].mean(),
    }


rows = {}
for name, w in CANDIDATES.items():
    p = portfolio_returns(w, r)
    d = summarise(p)
    for label, (a, b) in WINDOWS.items():
        d[label] = np.exp(p.loc[a:b].sum()) - 1
    rows[name] = d

table = pd.DataFrame(rows).T
print(table.round(3))
print("\n99% 1-day ES in dollars on USD 10m:")
print((table["ES99_1d"] * 10_000_000).round(0))