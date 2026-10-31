"""Stage 7b: stress tests - historical crises plus one hypothetical scenario designed by the team.

Historical: replay the actual fund returns of a crisis window through the portfolio.
Hypothetical: an instant one-off shock to each fund (the team must design AND justify the numbers).
"""
import numpy as np
import pandas as pd

from src.data import load_returns
from src.portfolio import CANDIDATES, portfolio_simple_returns

PORTFOLIO_USD = 10_000_000

HISTORICAL = {
    "2008 Global Financial Crisis": ("2007-10-01", "2009-03-09"),
    "March 2020 COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate-hiking drawdown": ("2022-01-01", "2022-12-31"),
}

# TEAM TO DESIGN: replace the story and every number below with your own, and justify them in the report.
HYPOTHETICAL = {
    "name": "Global risk-off with an oil and dollar shock (EXAMPLE - replace with your own)",
    "story": "Placeholder story: a sudden geopolitical shock, oil spikes, and investors flee equities.",
    "shock": {"SPY": -0.15, "EFA": -0.18, "EEM": -0.22, "AGG": -0.04, "GLD": 0.05, "XLE": 0.10, "XLK": -0.20},
}


def historical_stress(weights, r):
    s = portfolio_simple_returns(weights, r)
    rows = {}
    for name, (a, b) in HISTORICAL.items():
        w = s.loc[a:b]
        path = (1 + w).cumprod()
        total = path.iloc[-1] - 1
        rows[name] = {"total_return_%": total * 100,
                      "worst_day_%": w.min() * 100,
                      "max_drawdown_in_window_%": (path / path.cummax() - 1).min() * 100,
                      "loss_USD": -total * PORTFOLIO_USD}
    return pd.DataFrame(rows).T


def hypothetical_stress(weights, scenario=HYPOTHETICAL):
    w = pd.Series(weights)
    shock = pd.Series(scenario["shock"]).reindex(w.index).fillna(0.0)
    impact = float((w * shock).sum())
    by_fund = (w * shock * 100).rename("contribution_%")
    return impact, by_fund


if __name__ == "__main__":
    r = load_returns()
    pd.set_option("display.width", 200)
    for name, w in CANDIDATES.items():
        print(f"\n=== {name} ===")
        print(historical_stress(w, r).round(2))
        imp, by_fund = hypothetical_stress(w)
        print(f"Hypothetical '{HYPOTHETICAL['name']}': {imp*100:.2f}%  (USD {-imp*PORTFOLIO_USD:,.0f} loss)")