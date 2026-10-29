"""Stage 1: how each fund has actually behaved."""
import numpy as np
import pandas as pd
from src.data import load_returns

r = load_returns()

# 1. Whole-period facts for each fund
cum = np.exp(r.cumsum())                       # growth of one dollar invested
facts = pd.DataFrame({
    "annual_return": np.exp(r.mean() * 252) - 1,
    "annual_volatility": r.std() * np.sqrt(252),
    "max_drawdown": (cum / cum.cummax() - 1).min(),
}).round(3)
print("FUND FACTS\n", facts, "\n")

# 2. The ten biggest daily moves, now WITH their sign (+ rise, - fall)
s = r.stack()
top = s.abs().sort_values(ascending=False).head(10).index
print("BIGGEST MOVES (signed)\n", s.loc[top].round(3), "\n")

# 3. How each fund did in the three crises the brief names
def period_return(a, b):
    return (np.exp(r.loc[a:b].sum()) - 1).round(3)

crises = pd.DataFrame({
    "GFC (Oct 2007 - Mar 2009)": period_return("2007-10-01", "2009-03-09"),
    "COVID (19 Feb - 23 Mar 2020)": period_return("2020-02-19", "2020-03-23"),
    "2022 (calendar year)": period_return("2022-01-01", "2022-12-31"),
})
print("CRISIS RETURNS\n", crises, "\n")

# 4. Do the funds move together? Normal times vs the 2008 crash
print("CORRELATION, WHOLE PERIOD\n", r.corr().round(2), "\n")
print("CORRELATION, SEP 2008 - MAR 2009\n", r.loc["2008-09-01":"2009-03-09"].corr().round(2))