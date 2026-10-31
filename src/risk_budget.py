"""Stage 7a: risk budgeting - which fund contributes how much of the portfolio's tail risk?

Component ES of fund i = the average, over the portfolio's worst days (days with loss >= VaR),
of that fund's weighted loss (weight_i * -return_i). The pieces add up to the portfolio ES.
Convention: a LOSS is a positive number, 1-day horizon.
"""
import numpy as np
import pandas as pd

from src.data import load_returns
from src.portfolio import CANDIDATES


def component_es(weights, r, conf=0.99):
    w = pd.Series(weights).reindex(r.columns).fillna(0.0)
    simple = np.exp(r) - 1
    contrib_daily = -(simple * w)                 # each fund's weighted loss, every day
    port_loss = contrib_daily.sum(axis=1)
    var = np.quantile(port_loss, conf)
    tail = port_loss >= var
    comp = contrib_daily[tail].mean()             # average contribution on the worst days
    out = pd.DataFrame({"weight_%": w * 100,
                        "ES_contribution_%": comp * 100,
                        "share_of_ES_%": comp / comp.sum() * 100})
    out["risk_to_weight_ratio"] = out["share_of_ES_%"] / out["weight_%"].replace(0, np.nan)
    return out, float(comp.sum())


if __name__ == "__main__":
    r = load_returns()
    pd.set_option("display.width", 200)
    for name, w in CANDIDATES.items():
        table, es = component_es(w, r)
        print(f"\n=== {name}: 99% ES = {es*100:.2f}% of the portfolio (USD {es*10_000_000:,.0f} on 10m) ===")
        print(table.round(2))