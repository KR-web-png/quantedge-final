"""Stage 5: Extreme Value Theory (peaks over threshold).

Idea: instead of modelling ALL days, model only the extreme tail. Take every day whose loss is above a
threshold u, look at HOW FAR above u it goes (the "excess"), and fit a Generalised Pareto Distribution (GPD)
to those excesses. The GPD has two numbers:
    xi   (shape)  - how heavy the tail is. Bigger xi = more extreme disasters are possible.
    beta (scale)  - the typical size of the excesses.
Convention: a LOSS is a positive number (0.02 = lose 2%), 1-day horizon.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data import load_returns
from src.portfolio import CANDIDATES, portfolio_simple_returns


def fit_gpd(losses, q=0.90):
    """Fit the GPD to losses above the q-quantile. Returns threshold u, shape xi, scale beta, count."""
    losses = np.asarray(losses, dtype=float)
    u = np.quantile(losses, q)
    excess = losses[losses > u] - u
    xi, _, beta = stats.genpareto.fit(excess, floc=0)   # location fixed at 0
    return u, xi, beta, len(excess)


def evt_var_es(losses, conf, q=0.90):
    """EVT-based VaR and ES at confidence `conf` (needs conf > q). Returns (VaR, ES, details)."""
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    u, xi, beta, n_u = fit_gpd(losses, q)
    ratio = (n / n_u) * (1 - conf)
    if abs(xi) < 1e-6:
        var = u - beta * np.log(ratio)
        es = var + beta
    else:
        var = u + beta / xi * (ratio ** (-xi) - 1)
        es = var / (1 - xi) + (beta - xi * u) / (1 - xi)
    return float(var), float(es), {"u": u, "xi": xi, "beta": beta, "n_exceed": n_u}


def threshold_table(losses, qs=(0.85, 0.90, 0.925, 0.95, 0.975), conf=0.99):
    """How sensitive are the answers to the threshold? Stable results = a defensible threshold."""
    rows = []
    for q in qs:
        var, es, d = evt_var_es(losses, conf, q)
        rows.append({"threshold_q": q, "threshold_u": d["u"], "n_exceed": d["n_exceed"],
                     "xi": d["xi"], "beta": d["beta"], f"VaR{int(conf*100)}": var, f"ES{int(conf*100)}": es})
    return pd.DataFrame(rows).set_index("threshold_q")


def mean_excess_plot(losses, path="report/figures/mean_excess.png"):
    """Mean-excess plot: for a GPD tail this curve should be roughly a straight line above a good threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    losses = np.asarray(losses, dtype=float)
    us = np.quantile(losses, np.linspace(0.80, 0.995, 60))
    me = [losses[losses > u].mean() - u for u in us]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(us * 100, np.array(me) * 100, marker="o", ms=3)
    plt.xlabel("Threshold u (% loss)")
    plt.ylabel("Mean excess over u (%)")
    plt.title("Mean excess plot")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    r = load_returns()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    for name, w in CANDIDATES.items():
        loss = -portfolio_simple_returns(w, r).values
        print(f"\n=== {name}: threshold sensitivity at 99% (values in % of portfolio) ===")
        t = threshold_table(loss)
        print((t.assign(threshold_u=t.threshold_u * 100, beta=t.beta * 100,
                        VaR99=t.VaR99 * 100, ES99=t.ES99 * 100)).round(3))
        for conf in (0.95, 0.99, 0.995):
            v, e, d = evt_var_es(loss, conf, q=0.90)
            print(f"  EVT (q=0.90) conf={conf}: VaR={v*100:.2f}%  ES={e*100:.2f}%   (xi={d['xi']:.3f})")
        mean_excess_plot(loss, f"report/figures/mean_excess_{name[0]}.png")