"""Stage 3: Value-at-Risk (three methods) and Expected Shortfall.
Convention: a LOSS is a positive number (0.02 = lose 2%), 1-day horizon."""
import numpy as np
import pandas as pd
from scipy import stats
from src.data import load_returns
from src.portfolio import CANDIDATES, portfolio_simple_returns


# ---- 1. Historical simulation -------------------------------------------
def var_historical(s, conf):
    """The loss exceeded on only (1 - conf) of past days."""
    return float(np.quantile(-s, conf))


def es_historical(s, conf):
    loss = -s
    v = np.quantile(loss, conf)
    return float(loss[loss >= v].mean())


# ---- 2. Parametric (normal) ---------------------------------------------
def var_parametric(s, conf):
    mu, sigma = s.mean(), s.std()
    return float(-mu + stats.norm.ppf(conf) * sigma)


def es_parametric(s, conf):
    mu, sigma = s.mean(), s.std()
    z = stats.norm.ppf(conf)
    return float(-mu + sigma * stats.norm.pdf(z) / (1 - conf))


# ---- 3. Monte Carlo -------------------------------------------------------
def simulate_losses(r, weights, n=200_000, df=None, seed=42):
    """Simulate many possible next-day portfolio losses.
    df=None -> normal shocks; df=5 -> fatter-tailed Student-t shocks (same covariance)."""
    rng = np.random.default_rng(seed)
    mu = r.mean().values
    cov = r.cov().values
    z = rng.multivariate_normal(np.zeros(len(mu)), cov, size=n)
    if df is not None:
        z = z / np.sqrt(rng.chisquare(df, size=(n, 1)) / df) * np.sqrt((df - 2) / df)
    w = pd.Series(weights).reindex(r.columns).fillna(0.0).values
    return -(np.expm1(mu + z) @ w)


def var_monte_carlo(r, weights, conf, **kw):
    return float(np.quantile(simulate_losses(r, weights, **kw), conf))


def es_monte_carlo(r, weights, conf, **kw):
    L = simulate_losses(r, weights, **kw)
    v = np.quantile(L, conf)
    return float(L[L >= v].mean())


if __name__ == "__main__":
    r = load_returns()
    rows = []
    for name, w in CANDIDATES.items():
        s = portfolio_simple_returns(w, r)
        for conf in (0.95, 0.99):
            rows.append({
                "portfolio": name, "conf": conf,
                "VaR_hist": var_historical(s, conf),
                "VaR_param": var_parametric(s, conf),
                "VaR_MC_normal": var_monte_carlo(r, w, conf),
                "VaR_MC_t5": var_monte_carlo(r, w, conf, df=5),
                "ES_hist": es_historical(s, conf),
                "ES_param": es_parametric(s, conf),
                "ES_MC_t5": es_monte_carlo(r, w, conf, df=5),
            })
    out = pd.DataFrame(rows).set_index(["portfolio", "conf"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print((out * 100).round(2))        # shown as % of the portfolio