"""SAIFA Quant Edge 1.0 - Final Challenge. Run with:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import TICKERS
from src import dashboard as dash
from src import optimise
from src.backtest import run_backtest
from src.data import load_returns
from src.kpis import kpi_table, risk_free_rate
from src.portfolio import CANDIDATES, portfolio_simple_returns
from src.risk_budget import component_es
from src.stress import HISTORICAL, HYPOTHETICAL, hypothetical_stress, historical_stress

st.set_page_config(page_title="Endowment risk dashboard", layout="wide")


@st.cache_data(show_spinner="Loading prices...")
def get_returns():
    return load_returns()


@st.cache_data(show_spinner="Fetching risk-free rate...")
def get_rf():
    return risk_free_rate()


@st.cache_data(show_spinner="Running rolling backtest (about a minute)...")
def get_backtest(weights_key, conf, method):
    r = get_returns()
    loss = -portfolio_simple_returns(dict(weights_key), r).values
    res, var, hits = run_backtest(loss, r.index, conf, method)
    return res, var, loss


r = get_returns()
rf, rf_note = get_rf()
spy = np.exp(r["SPY"]) - 1

# ---------------- Sidebar: choose / edit the portfolio ----------------
st.sidebar.title("Portfolio")
choice = st.sidebar.selectbox("Start from", list(CANDIDATES) + ["Custom"], index=1)
base = CANDIDATES.get(choice, CANDIDATES["B Conservative"])
st.sidebar.caption("Weights in %. They are normalised to 100%.")
raw = {t: st.sidebar.number_input(t, 0.0, 100.0, float(round(base.get(t, 0) * 100, 2)), 1.0) for t in TICKERS}
if sum(raw.values()) <= 0:
    st.error("Give at least one fund a positive weight.")
    st.stop()
weights = dash.clean_weights(raw)
st.sidebar.write(f"Total entered: {sum(raw.values()):.1f}%")

pages = ["1 Overview", "2 Performance & KPIs", "3 VaR & ES", "4 Backtesting", "5 Stress tests",
         "6 Risk budget & Optimiser", "Assumptions & Limitations"]
page = st.sidebar.radio("Page", pages)
st.title("USD 10m conservative endowment - risk dashboard")

# ---------------- 1 Overview ----------------
if page == pages[0]:
    c1, c2 = st.columns(2)
    w = pd.Series(weights)
    c1.plotly_chart(px.pie(values=w.values, names=w.index, title="Allocation"), use_container_width=True)
    c2.dataframe((w * dash.PORTFOLIO_USD).rename("USD").round(0).to_frame())
    wealth, dd = dash.wealth_and_drawdown(weights, r)
    fig = go.Figure(go.Scatter(x=wealth.index, y=wealth, name="Portfolio"))
    fig.add_trace(go.Scatter(x=wealth.index, y=(1 + spy).cumprod(), name="SPY"))
    fig.update_layout(title="Growth of 1 USD (daily rebalanced to the weights)")
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.area(dd * 100, title="Drawdown (%)"), use_container_width=True)

# ---------------- 2 KPIs ----------------
elif page == pages[1]:
    st.caption(f"Risk-free rate {rf:.2%} - {rf_note}")
    ret = portfolio_simple_returns(weights, r)
    k = kpi_table(ret, spy, rf)
    st.dataframe(k.round(3))
    st.caption("Treynor, Jensen alpha and beta are measured against SPY as the market proxy.")
    roll = ret.rolling(252).std() * np.sqrt(252) * 100
    st.plotly_chart(px.line(roll, title="Rolling 1-year volatility (%)"), use_container_width=True)

# ---------------- 3 VaR / ES ----------------
elif page == pages[2]:
    conf = st.radio("Confidence", [0.95, 0.99], horizontal=True, index=1)
    t = dash.risk_table(weights, r)
    t = t[t.confidence == conf].drop(columns="confidence").set_index("method")
    st.dataframe(t.round(2))
    st.plotly_chart(px.bar(t["ES_USD"].reset_index(), x="method", y="ES_USD",
                           title=f"Expected shortfall at {conf:.0%} (USD, 1 day)"), use_container_width=True)
    loss = -portfolio_simple_returns(weights, r)
    fig = px.histogram(loss * 100, nbins=120, title="Distribution of daily losses (%)")
    fig.add_vline(x=t.loc["Historical", "VaR_%"], line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Normal (parametric) VaR understates the extreme tail; EVT models the tail directly.")

# ---------------- 4 Backtesting ----------------
elif page == pages[3]:
    method = st.selectbox("Method", ["historical", "parametric", "evt", "filtered"], index=3)
    conf = st.radio("Confidence", [0.95, 0.99], horizontal=True, index=1)
    if st.button("Run backtest"):
        res, var, loss = get_backtest(tuple(sorted(weights.items())), conf, method)
        st.dataframe(pd.Series(res).to_frame("result"))
        df = pd.DataFrame({"loss": loss * 100, "VaR": var.values * 100}, index=r.index).dropna()
        fig = go.Figure(go.Scatter(x=df.index, y=df.loss, name="Daily loss %", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df.VaR, name="VaR forecast %"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Kupiec: right number of violations? Christoffersen: do they cluster? "
                   "Small p-value (<0.05) = model rejected. Basel light uses the last 250 days.")
    else:
        st.info("Press 'Run backtest'. Each forecast uses only the previous 1000 days (no peeking).")

# ---------------- 5 Stress ----------------
elif page == pages[4]:
    st.subheader("Historical crises")
    st.dataframe(historical_stress(weights, r).round(2))
    st.subheader("Hypothetical scenario (editable)")
    st.write(HYPOTHETICAL["name"])
    st.caption(HYPOTHETICAL["story"])
    shock = {t: st.number_input(f"{t} shock %", -100.0, 100.0, float(HYPOTHETICAL["shock"].get(t, 0) * 100), 1.0,
                                key="sh" + t) / 100 for t in TICKERS}
    out = dash.custom_scenario(weights, shock)
    st.metric("Portfolio return", f"{out['return_%']:.2f}%", f"USD {-out['loss_USD']:,.0f}")

# ---------------- 6 Risk budget & optimiser ----------------
elif page == pages[5]:
    tab1, tab2 = st.tabs(["Risk budget", "Optimiser"])
    with tab1:
        table, es = component_es(weights, r)
        st.write(f"99% expected shortfall: {es*100:.2f}% of the portfolio (USD {es*dash.PORTFOLIO_USD:,.0f})")
        st.dataframe(table.round(2))
        st.plotly_chart(px.bar(table[["weight_%", "share_of_ES_%"]], barmode="group",
                               title="Share of capital vs share of tail risk"), use_container_width=True)
        st.caption("Risk-to-weight above 1 = that fund contributes more tail risk than its share of money.")
    with tab2:
        max_w = st.slider("Max weight per fund", 0.2, 1.0, 0.5, 0.05)
        mu, cov = optimise.inputs(r)
        fr = optimise.efficient_frontier(mu, cov, max_w)
        mv, ms = optimise.min_variance(mu, cov, max_w), optimise.max_sharpe(mu, cov, rf, max_w)
        fig = go.Figure(go.Scatter(x=fr.vol * 100, y=fr["return"] * 100, mode="lines", name="Efficient frontier"))
        for n, ww in {**CANDIDATES, "Your portfolio": weights}.items():
            s = optimise.stats_of(ww, mu, cov, rf)
            fig.add_trace(go.Scatter(x=[s["vol"] * 100], y=[s["return"] * 100], mode="markers+text",
                                     text=[n], name=n, textposition="top center"))
        fig.update_layout(xaxis_title="Volatility % (annual)", yaxis_title="Return % (annual)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({"Min variance": mv, "Max Sharpe": ms}).mul(100).round(1))
        st.caption("Optimiser inputs are historical averages: noisy. Treat as a reference, not the answer.")

# ---------------- Assumptions ----------------
else:
    st.header("Assumptions & Limitations")
    st.markdown(f"""
- **Data:** daily adjusted closes from yfinance (approved source) for SPY, EFA, EEM, AGG, GLD, XLE, XLK; common
  calendar only (days where any fund is missing are dropped). Funds trade in USD.
- **Risk-free rate:** {rf_note}.
- **Returns:** statistics on log returns; portfolio aggregation on simple returns, weights reset daily (no drift).
- **VaR/ES:** one-day horizon, loss quoted as a positive number, USD 10m portfolio. Methods: historical,
  normal (parametric), EVT (peaks-over-threshold, GPD, threshold = 90th percentile of losses), and
  volatility-filtered historical simulation (EWMA lambda = 0.94).
- **Backtest:** rolling 1000-day window, no look-ahead; EVT refitted every 21 days.
- **Past is not the future:** the sample includes 2008 and 2020 but not every possible crisis; correlations rise in crashes.
- **Not modelled:** transaction costs, taxes, liquidity, fund tracking error, USD/LKR exchange-rate risk,
  Sri Lankan sovereign or inflation risk. A Sri Lankan investor would hold these funds in rupees, so the
  rupee value moves with USD/LKR on top of the numbers shown.
- **Optimiser:** historical mean returns are very noisy; weights are capped and shown only as a reference.
- **AI use:** see the AI-use disclosure table in the report/README.
""")