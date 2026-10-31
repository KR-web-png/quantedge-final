"""Data pipeline: download, clean and return log returns for the 7-fund universe."""
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore", module="yfinance")

TICKERS = ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]
START, END = "2005-01-01", "2026-10-24"
CACHE = Path(__file__).resolve().parent.parent / "data" / "prices_raw.csv"


def download_prices(refresh=False):
    """Download adjusted closing prices once, then reuse the saved file."""
    CACHE.parent.mkdir(exist_ok=True)
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    px = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    px = px[TICKERS]
    px.to_csv(CACHE)
    return px


def clean_report(px):
    """Print things a human should look at. Nothing is deleted silently."""
    print("First valid date per fund:\n", px.apply(lambda s: s.first_valid_index()))
    print("Missing values per fund:\n", px.isna().sum())
    print("Share of days with unchanged price:\n", (px.diff() == 0).mean().round(3))


def load_returns(refresh=False):
    """Daily log returns, on days when all seven funds traded."""
    px = download_prices(refresh).dropna()
    return np.log(px).diff().dropna()


if __name__ == "__main__":
    px = download_prices()
    clean_report(px)
    r = load_returns()
    print(r.shape)
    print(r.abs().stack().sort_values(ascending=False).head(10))
    from statsmodels.tsa.stattools import adfuller
    for c in r.columns:
        print(c, "ADF p-value:", round(adfuller(r[c], result_object=False)[1], 4))