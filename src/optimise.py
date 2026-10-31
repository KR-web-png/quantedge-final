"""Layer 1: Markowitz mean-variance optimisation (long-only, fully invested) and the efficient frontier.

Expected returns come from history, which is NOISY: always show the optimiser's answer next to the team's
own judgement, and cap single-fund weights (max_w) so it does not put everything in one fund.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

DAYS = 252


def inputs(r):
    """Annual expected simple returns (mu) and annual covariance (cov) from daily LOG returns r."""
    simple = np.exp(r) - 1
    return simple.mean() * DAYS, simple.cov() * DAYS


def _solve(obj, n, max_w, extra=None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}] + (extra or [])
    res = minimize(obj, np.full(n, 1 / n), method="SLSQP", bounds=[(0, max_w)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def min_variance(mu, cov, max_w=1.0):
    return pd.Series(_solve(lambda w: w @ cov.values @ w, len(mu), max_w), index=mu.index)


def max_sharpe(mu, cov, rf, max_w=1.0):
    f = lambda w: -(w @ mu.values - rf) / np.sqrt(w @ cov.values @ w)
    return pd.Series(_solve(f, len(mu), max_w), index=mu.index)


def efficient_frontier(mu, cov, max_w=1.0, points=25):
    """Lowest-risk portfolio for each target return between the min-variance and the best single-fund return."""
    lo = float(min_variance(mu, cov, max_w) @ mu)
    hi = float(mu.max()) if max_w >= 1 else float(np.sort(mu.values)[::-1][:int(np.ceil(1 / max_w))] @
                                                  np.full(int(np.ceil(1 / max_w)), 1 / np.ceil(1 / max_w)))
    rows = []
    for tgt in np.linspace(lo, hi * 0.999, points):
        extra = [{"type": "eq", "fun": lambda w, t=tgt: w @ mu.values - t}]
        w = _solve(lambda w: w @ cov.values @ w, len(mu), max_w, extra)
        if abs(w @ mu.values - tgt) < 1e-4:
            rows.append({"return": float(w @ mu.values), "vol": float(np.sqrt(w @ cov.values @ w)),
                         **{k: float(v) for k, v in zip(mu.index, w)}})
    return pd.DataFrame(rows)


def stats_of(w, mu, cov, rf):
    w = pd.Series(w).reindex(mu.index).fillna(0).values
    ret, vol = float(w @ mu.values), float(np.sqrt(w @ cov.values @ w))
    return {"return": ret, "vol": vol, "sharpe": (ret - rf) / vol}


if __name__ == "__main__":
    from src.data import load_returns
    from src.kpis import risk_free_rate
    from src.portfolio import CANDIDATES
    r = load_returns(); mu, cov = inputs(r); rf, _ = risk_free_rate()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print("Expected returns (history, annual):\n", (mu * 100).round(2).to_string())
    for label, w in [("Min variance (max 50% each)", min_variance(mu, cov, 0.5)),
                     ("Max Sharpe (max 50% each)", max_sharpe(mu, cov, rf, 0.5))]:
        print(f"\n{label}:", (w * 100).round(1).to_dict(), stats_of(w, mu, cov, rf))
    print("\nYour candidates:")
    for n, w in CANDIDATES.items():
        print(n, {k: round(v, 3) for k, v in stats_of(w, mu, cov, rf).items()})