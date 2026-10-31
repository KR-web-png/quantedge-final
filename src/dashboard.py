"""Streamlit-free helpers for app.py (so the logic can be tested without a browser)."""
import numpy as np
import pandas as pd

from src.backtest import ewma_sigma
from src.evt import evt_var_es
from src.portfolio import portfolio_simple_returns

PORTFOLIO_USD = 10_000_000


def clean_weights(w):
    """Drop zeros, normalise to sum to 1."""
    s = pd.Series(w, dtype=float)
    s = s[s > 0]
    return (s / s.sum()).to_dict()


def risk_table(weights, r, confs=(0.95, 0.99), value=PORTFOLIO_USD):
    """1-day VaR and ES for four methods. Loss = positive number. Returns tidy DataFrame."""
    loss = -portfolio_simple_returns(weights, r).values
    sigma = ewma_sigma(loss)
    z = loss / sigma
    rows = []
    for c in confs:
        v_h = np.quantile(loss, c); e_h = loss[loss >= v_h].mean()
        from scipy import stats
        m, s = loss.mean(), loss.std(ddof=1)
        zc = stats.norm.ppf(c)
        v_p = m + zc * s; e_p = m + s * stats.norm.pdf(zc) / (1 - c)
        v_e, e_e, _ = evt_var_es(loss, c, q=0.90)
        zq = np.quantile(z, c); v_f = zq * sigma[-1]; e_f = z[z >= zq].mean() * sigma[-1]
        for name, v, e in [("Historical", v_h, e_h), ("Parametric (normal)", v_p, e_p),
                           ("EVT (GPD tail)", v_e, e_e), ("Filtered historical (today's vol)", v_f, e_f)]:
            rows.append({"method": name, "confidence": c, "VaR_%": 100 * v, "ES_%": 100 * e,
                         "VaR_USD": v * value, "ES_USD": e * value})
    return pd.DataFrame(rows)


def wealth_and_drawdown(weights, r):
    ret = portfolio_simple_returns(weights, r)
    wealth = (1 + ret).cumprod()
    return wealth, wealth / wealth.cummax() - 1


def custom_scenario(weights, shocks, value=PORTFOLIO_USD):
    """Loss from one-off fund shocks (decimal, e.g. -0.15). Funds missing from `shocks` count as 0."""
    w = pd.Series(weights, dtype=float)
    s = pd.Series(shocks, dtype=float).reindex(w.index).fillna(0.0)
    ret = float((w * s).sum())
    return {"return_%": 100 * ret, "loss_USD": -ret * value}