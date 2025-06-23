import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Stock Price Trend Dashboard with Indicators")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS)", "AAPL")

if st.button("Show Analysis"):
    df = yf.download(ticker, start='2015-01-01', end='2024-12-31')
    
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    st.subheader("📊 Stock Price with Moving Averages")
    st.line_chart(df[['Close', 'MA50', 'MA200']])

    st.subheader("📉 RSI Indicator")
    st.line_chart(df['RSI'])
