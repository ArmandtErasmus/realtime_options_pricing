import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timezone
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from scipy.interpolate import griddata

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def fetch_intraday_data(ticker, interval="1m"):
    tick = yf.Ticker(ticker)
    df = tick.history(interval=interval, period="1d")

    # Keep only relevant columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = df.index.tz_localize(None)
    df.index = df.index.to_pydatetime()
    df["timestamp"] = df.index.astype(str)
    df["ticker"] = ticker
    df.reset_index(drop=True, inplace=True)
    return df

def fetch_options_data(ticker):
    tick = yf.Ticker(ticker)
    expirations = tick.options
    if not expirations:
        return None, None
    nearest_exp = expirations[0]
    opt_chain = tick.option_chain(nearest_exp)
    calls = opt_chain.calls.copy()
    puts = opt_chain.puts.copy()
    calls["expiration"] = nearest_exp
    puts["expiration"] = nearest_exp
    return calls, puts

def add_bs_prices(options_df, S, r=0.05):
    today = datetime.now(timezone.utc)
    df = options_df.copy()
    df["bs_price"] = None
    for i, row in df.iterrows():
        K = row["strike"]
        exp_date = datetime.strptime(row["expiration"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        T = (exp_date - today).total_seconds() / (365.25 * 24 * 3600)
        sigma = row.get("impliedVolatility")
        if not sigma or sigma <= 0:
            continue
        option_type = "call" if "C" in row["contractSymbol"] else "put"
        df.at[i, "bs_price"] = black_scholes(S, K, T, r, sigma, option_type)
    return df

conn = sqlite3.connect("intraday_stock_prices.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS intraday_prices (
        ticker TEXT,
        timestamp TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (ticker, timestamp)
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS options_prices (
        ticker TEXT,
        contractSymbol TEXT PRIMARY KEY,
        expiration TEXT,
        strike REAL,
        type TEXT,
        lastPrice REAL,
        impliedVol REAL,
        bs_price REAL,
        updated_at TEXT
    )
""")

conn.commit()

def get_latest_timestamp(ticker):
    cursor.execute("SELECT MAX(timestamp) FROM intraday_prices WHERE ticker = ?", (ticker,))
    return cursor.fetchone()[0]

def store_data_to_db(data, latest_timestamp):
    if latest_timestamp:
        data = data[data["timestamp"] > latest_timestamp]
    if not data.empty:
        data.to_sql("intraday_prices", conn, if_exists="append", index=False)

def store_options_to_db(df, ticker):
    if df.empty:
        return
    df = df.copy()
    df["ticker"] = ticker
    df["updated_at"] = datetime.now(timezone.utc).isoformat()
    keep_cols = [
        "ticker", "contractSymbol", "expiration", "strike",
        "type", "lastPrice", "impliedVolatility", "bs_price", "updated_at"
    ]
    df = df[keep_cols]
    df.rename(columns={"impliedVolatility": "impliedVol"}, inplace=True)
    df.to_sql("options_prices", conn, if_exists="replace", index=False)

def get_latest_close_price(ticker):
    df = pd.read_sql(
        "SELECT close FROM intraday_prices WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
        conn, params=(ticker,)
    )
    return df["close"].iloc[0] if not df.empty else None

st.set_page_config(layout="wide", page_title="Live Options Dashboard")
st.markdown("<meta http-equiv='refresh' content='120'>", unsafe_allow_html=True)
st.title("📈 Real-Time Derivative Pricing Dashboard")

ticker = st.text_input("Enter Ticker Symbol", value="AAPL")

st.markdown("""
<style>
.card-left { display: flex; border: 1px solid #ccc; border-radius: 10px; overflow: hidden; box-shadow: 1px 1px 6px rgba(0,0,0,0.05); margin-bottom: 10px; }
.bar-green { width: 8px; background-color: #28a745; }
.bar-red { width: 8px; background-color: #dc3545; }
.content { padding: 15px; background-color: #ffffff; width: 100%; }
.label { font-size: 16px; font-weight: bold; margin-bottom: 4px; }
.label-green { color: #28a745; }
.label-red { color: #dc3545; }
.price { font-size: 28px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

if ticker:
    price = get_latest_close_price(ticker)
    if price is None:
        df = fetch_intraday_data(ticker)
        store_data_to_db(df, None)
        price = df["close"].iloc[-1]

    calls, puts = fetch_options_data(ticker)
    if calls is None:
        st.error("No options found for this ticker.")
    else:
        calls_bs = add_bs_prices(calls, price)
        puts_bs = add_bs_prices(puts, price)
        calls_bs["type"] = "call"
        puts_bs["type"] = "put"
        store_options_to_db(calls_bs, ticker)
        store_options_to_db(puts_bs, ticker)

        atm_call = calls_bs.iloc[(calls_bs["strike"]).abs().argsort()[0]]
        atm_put  = puts_bs.iloc[(puts_bs["strike"]).abs().argsort()[0]]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="card-left">
                <div class="bar-green"></div>
                <div class="content">
                    <div class="label label-green">📈 CALL Option ({atm_call['contractSymbol']})</div>
                    <div class="price">${atm_call['bs_price']:.2f}</div>
                    <div>Market: ${atm_call['lastPrice']:.2f}</div>
                    <div>Strike: ${atm_call['strike']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="card-left">
                <div class="bar-red"></div>
                <div class="content">
                    <div class="label label-red">📉 PUT Option ({atm_put['contractSymbol']})</div>
                    <div class="price">${atm_put['bs_price']:.2f}</div>
                    <div>Market: ${atm_put['lastPrice']:.2f}</div>
                    <div>Strike: ${atm_put['strike']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    tick = yf.Ticker(ticker)
    expirations = tick.options

    if expirations:
        all_calls, all_puts = [], []
        for exp in expirations:
            chain = tick.option_chain(exp)
            calls_exp = chain.calls.copy()
            puts_exp = chain.puts.copy()
            calls_exp["expiration"] = exp
            puts_exp["expiration"] = exp
            all_calls.append(calls_exp)
            all_puts.append(puts_exp)

        df_iv = pd.concat(all_calls + all_puts, ignore_index=True)
        df_iv["expiration_dt"] = pd.to_datetime(df_iv["expiration"])
        today = pd.Timestamp.now()
        df_iv["days_to_expiry"] = (df_iv["expiration_dt"] - today).dt.days

        df_iv = df_iv[(df_iv["days_to_expiry"] > 0) & (df_iv["impliedVolatility"].notna())]

        df_iv = df_iv[df_iv["impliedVolatility"] < 5]

        if not df_iv.empty:
            from scipy.interpolate import Rbf

            strikes = df_iv["strike"].values
            days = df_iv["days_to_expiry"].values
            iv = df_iv["impliedVolatility"].values

            strike_grid = np.linspace(strikes.min(), strikes.max(), 50)
            days_grid = np.linspace(days.min(), days.max(), 50)
            X, Y = np.meshgrid(days_grid, strike_grid)

            rbf = Rbf(days, strikes, iv, function='multiquadric', smooth=0.1)
            Z = rbf(X, Y)

            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
            fig.update_layout(
                title=f"Smoothed Implied Volatility Surface for {ticker}",
                scene=dict(
                    xaxis_title='Days to Expiry',
                    yaxis_title='Strike',
                    zaxis_title='Implied Volatility'
                ),
                autosize=True,
                margin=dict(l=0, r=0, b=0, t=50)
            )
            st.subheader("📈 Smoothed Implied Volatility Surface")
            st.plotly_chart(fig, use_container_width=True)
