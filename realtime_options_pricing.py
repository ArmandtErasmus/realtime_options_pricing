# import libraries
import pandas as pd
import sqlite3
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timezone
import streamlit as st

# get a list of available option maturity dates
def get_option_maturity_dates(symbol):

    asset = yf.Ticker(symbol)
    maturity_dates = asset.options

    return list(maturity_dates)

# get a dataframe containing a call or put option chain for a given maturity date, or the current asset price
def get_option_data(symbol, maturity_date = None, derivative_type = "asset"):

    asset = yf.Ticker(symbol)
    symbol = symbol.upper()

    if maturity_date is None:
        maturity_date = asset.options[0]

    if derivative_type == "asset":

        asset_price = asset.info['regularMarketPrice']

        return asset_price
    
    elif derivative_type == "call":

        option_chain = asset.option_chain(maturity_date)

        calls = pd.DataFrame()

        calls['contract_symbol'] = option_chain.calls['contractSymbol']
        calls['strike_price'] = option_chain.calls['strike']
        calls['bid'] = option_chain.calls['bid']
        calls['ask'] = option_chain.calls['ask']
        calls['implied_volatility'] = option_chain.calls['impliedVolatility']

        return calls
    elif derivative_type == "put":

        option_chain = asset.option_chain(maturity_date)

        puts = pd.DataFrame()

        puts['contract_symbol'] = option_chain.puts['contractSymbol']
        puts['strike_price'] = option_chain.puts['strike']
        puts['bid'] = option_chain.puts['bid']
        puts['ask'] = option_chain.puts['ask']
        puts['implied_volatility'] = option_chain.puts['impliedVolatility']

        return puts
    else:
        return pd.DataFrame()

def get_user_asset_info():

    symbol = st.text_input("Enter a stock symbol", placeholder="e.g. AAPL")

    return symbol.upper()

def get_asset_info(symbol):

    current_market_price = get_option_data(symbol, derivative_type = "asset")

    return current_market_price

def main():

    st.set_page_config(
        page_title = "Real-Time Derivatives Pricing Dashboard",
        initial_sidebar_state = "collapsed",
        layout = "wide"
    )

    st.title("Real-Time Derivatives Pricing Dashboard")

    # dashboard
    try:

        col11, col12 = st.columns(2, border = True)

        with col11:
            symbol = get_user_asset_info()

        with col12:
            if symbol:
                dates = get_option_maturity_dates(symbol)
                maturity_date = st.selectbox(
                    label="Select a maturity date",
                    options=dates,
                    placeholder="Select a maturity date"
                )
            else:
                st.selectbox(
                    label="Select a maturity date",
                    options=[], 
                    placeholder="Enter a symbol first"
                )


        current_market_price = get_asset_info(symbol)
        calls = get_option_data(symbol, maturity_date = maturity_date, derivative_type = "call")
        puts = get_option_data(symbol, maturity_date = maturity_date, derivative_type = "put")

        col21, col22 = st.columns(2, border = True)

        with col21:
            st.write(f"Asset: {symbol}")

        with col22:
            st.write(f"Current Market Price: ${current_market_price}")

        st.subheader(f"Available Option Contracts for {symbol}:")

        col31, col32 = st.columns(2, border = True)

        with col31:
            st.subheader("Calls")
            st.dataframe(calls)

        with col32:
            st.subheader("Puts")
            st.dataframe(puts)

        
    except:
        pass
    

if __name__ == "__main__":
    main()