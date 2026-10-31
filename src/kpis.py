"""Layer 1: performance KPIs (Sharpe, Sortino, Treynor, Jensen alpha, information ratio, drawdown, Calmar).

All inputs are DAILY SIMPLE returns (pandas Series). 252 trading days per year.
Risk-free rate: US 3-month T-bill from FRED (approved source). If the download fails we fall back to
FALLBACK_RF and say so on screen - never hide a fallback.
"""
import numpy as np
import pandas as pd

FALLBACK_RF = 0.02          # 2% a year, used only if FRED cannot be reached
DAYS = 252


def risk_free_rate(start="2005-01-01"):
    """Average annual risk-free rate (decimal) and a note saying where it came from."""
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
        s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s = s[s.index >= start]
        return float(s.mean() / 100), "FRED DTB3 (US 3-month T-bill), average since " + start
    except Exception:
        return FALLBACK_RF, f"FALLBACK {FALLBACK_RF:.0%} (FRED not reachable) - state this in the report"


def max_drawdown(ret):
    wealth = (1 + ret).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def kpi_table(ret, bench, rf):
    """KPIs of portfolio `ret` against benchmark `bench` (e.g. SPY). Returns a one-column DataFrame."""
    ret, bench = ret.align(bench, join="inner")
    ann_ret = float((1 + ret).prod() ** (DAYS / len(ret)) - 1)
    ann_vol = float(ret.std(ddof=1) * np.sqrt(DAYS))
    rf_d = rf / DAYS
    ex = ret - rf_d
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(DAYS)
    beta = float(np.cov(ret, bench, ddof=1)[0, 1] / bench.var(ddof=1))
    bench_ann = float((1 + bench).prod() ** (DAYS / len(bench)) - 1)
    jensen = ann_ret - (rf + beta * (bench_ann - rf))            # CAPM alpha, annual
    active = ret - bench
    te = float(active.std(ddof=1) * np.sqrt(DAYS))
    mdd = max_drawdown(ret)
    out = {
        "Annual return": ann_ret, "Annual volatility": ann_vol,
        "Sharpe": (ann_ret - rf) / ann_vol, "Sortino": (ann_ret - rf) / downside,
        "Beta vs SPY": beta, "Treynor": (ann_ret - rf) / beta if abs(beta) > 1e-9 else np.nan,
        "Jensen alpha": jensen, "Information ratio": (ann_ret - bench_ann) / te if te > 0 else np.nan,
        "Max drawdown": mdd, "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
    }
    return pd.DataFrame({"value": out})


if __name__ == "__main__":
    from src.data import load_returns
    from src.portfolio import CANDIDATES, portfolio_simple_returns
    r = load_returns()
    rf, note = risk_free_rate()
    print(f"Risk-free {rf:.2%}: {note}")
    spy = (np.exp(r["SPY"]) - 1)
    cols = {n: kpi_table(portfolio_simple_returns(w, r), spy, rf)["value"] for n, w in CANDIDATES.items()}
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print(pd.DataFrame(cols).round(3))