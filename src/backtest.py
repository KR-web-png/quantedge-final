"""Stage 6: out-of-sample backtesting of VaR models.

For every day t we forecast VaR using ONLY the previous `window` days (so no peeking), then check whether
the real loss on day t exceeded it (a "violation" or "exception"). At 99% confidence we expect violations on
about 1% of days. Convention: a LOSS is a positive number, 1-day horizon.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy, xlog1py

from src.data import load_returns
from src.evt import evt_var_es
from src.portfolio import CANDIDATES, portfolio_simple_returns


def ewma_sigma(loss, lam=0.94):
    """EWMA volatility forecast. sigma[t] = forecast for day t made with information up to day t-1."""
    loss = np.asarray(loss, dtype=float)
    sigma = np.empty(len(loss))
    var = np.var(loss[:250]) if len(loss) >= 250 else np.var(loss)
    sigma[0] = np.sqrt(var)
    for t in range(1, len(loss)):
        var = lam * var + (1 - lam) * loss[t - 1] ** 2
        sigma[t] = np.sqrt(var)
    return sigma


# ---- Rolling one-day-ahead VaR forecasts -----------------------------------
def rolling_var(loss, conf, method, window=1000, refit=21):
    """Forecast for day t uses days t-window ... t-1 only. EVT is refitted every `refit` days (it is slow)."""
    loss = np.asarray(loss, dtype=float)
    out = np.full(len(loss), np.nan)
    last = None
    if method == "filtered":
        sigma = ewma_sigma(loss)          # sigma[t] is known BEFORE day t (uses days up to t-1 only)
    for t in range(window, len(loss)):
        past = loss[t - window:t]
        if method == "historical":
            out[t] = np.quantile(past, conf)
        elif method == "parametric":
            out[t] = past.mean() + stats.norm.ppf(conf) * past.std(ddof=1)
        elif method == "evt":
            if last is None or (t - window) % refit == 0:
                last = evt_var_es(past, conf, q=0.90)[0]
            out[t] = last
        elif method == "filtered":
            # volatility-filtered historical simulation: rescale past losses to today's volatility level
            z = loss[t - window:t] / sigma[t - window:t]
            out[t] = np.quantile(z, conf) * sigma[t]
        else:
            raise ValueError(method)
    return out


# ---- Test 1: Kupiec proportion-of-failures -----------------------------------
def kupiec_pof(hits, conf):
    """H0: the violation rate equals 1 - conf. Returns (LR statistic, p-value). Small p = model rejected."""
    n, x, p = len(hits), int(np.sum(hits)), 1 - conf
    pi = x / n
    lr = -2 * ((xlog1py(n - x, -p) + xlogy(x, p)) - (xlog1py(n - x, -pi) + xlogy(x, pi)))
    return float(lr), float(stats.chi2.sf(lr, 1))


# ---- Test 2: Christoffersen independence --------------------------------------
def christoffersen_independence(hits):
    """H0: violations do not cluster (one violation does not make another more likely tomorrow)."""
    h = np.asarray(hits, dtype=int)
    a, b = h[:-1], h[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_ind = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_dep = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(float(-2 * (ll_ind - ll_dep)), 0.0)
    return lr, float(stats.chi2.sf(lr, 1))


# ---- Test 3: Basel traffic light ----------------------------------------------
def basel_zone(n_violations, n_days=250, p=0.01):
    """Green / Yellow / Red from the binomial distribution (Basel: cumulative prob < 95% green, < 99.99% yellow).
    For 250 days at 99% this gives green 0-4, yellow 5-9, red 10+ violations."""
    cum = stats.binom.cdf(n_violations, n_days, p)
    return "GREEN" if cum < 0.95 else ("YELLOW" if cum < 0.9999 else "RED")


def basel_summary(hits, n_days=250, p=0.01):
    """Zone of the most recent 250 days, and the share of all rolling 250-day windows in each zone."""
    hits = np.asarray(hits, dtype=int)
    counts = pd.Series(hits).rolling(n_days).sum().dropna().astype(int)
    zones = counts.map(lambda c: basel_zone(c, n_days, p))
    return {"last_window_violations": int(counts.iloc[-1]), "last_window_zone": zones.iloc[-1],
            "share_green": float((zones == "GREEN").mean()),
            "share_yellow": float((zones == "YELLOW").mean()),
            "share_red": float((zones == "RED").mean())}


def run_backtest(loss, dates, conf, method, window=1000):
    var = rolling_var(loss, conf, method, window)
    ok = ~np.isnan(var)
    hits = (np.asarray(loss)[ok] > var[ok]).astype(int)
    lr_uc, p_uc = kupiec_pof(hits, conf)
    lr_ind, p_ind = christoffersen_independence(hits)
    res = {"method": method, "conf": conf, "days": int(ok.sum()), "violations": int(hits.sum()),
           "expected": round((1 - conf) * ok.sum(), 1), "hit_rate_%": round(100 * hits.mean(), 2),
           "kupiec_p": round(p_uc, 3), "christoffersen_p": round(p_ind, 3)}
    if conf >= 0.99:
        b = basel_summary(hits)
        res.update({"light_last_250d": b["last_window_zone"],
                    "%windows_yellow": round(100 * b["share_yellow"], 1),
                    "%windows_red": round(100 * b["share_red"], 1)})
    return res, pd.Series(var, index=dates), hits


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "B Conservative"   # python -m src.backtest "D Defensive"
    r = load_returns()
    loss = -portfolio_simple_returns(CANDIDATES[name], r).values
    rows = []
    for conf in (0.95, 0.99):
        for method in ("historical", "parametric", "evt", "filtered"):
            if method == "evt" and conf < 0.95:
                continue
            res, _, _ = run_backtest(loss, r.index, conf, method)
            rows.append(res)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(f"\nBacktest for {name} (window 1000 days, one-day VaR)")
    print(pd.DataFrame(rows).to_string(index=False))