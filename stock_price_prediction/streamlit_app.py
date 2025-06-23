import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Stock Price Dashboard", layout="wide")
st.title("📈 Stock Price Trend Dashboard with Indicators")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS)", "AAPL")

if st.button("Show Analysis"):
    # Download data
    df = yf.download(ticker, start='2015-01-01', end='2024-12-31', auto_adjust=True)

    if df.empty:
        st.error("❌ Could not fetch data. Please check the ticker symbol.")
    else:
        # Handle MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten the MultiIndex by taking the first level (the actual column names)
            df.columns = df.columns.get_level_values(0)
        
        # Check if 'Close' column exists
        if 'Close' not in df.columns:
            st.error("❌ 'Close' column not found in the data.")
        else:
            # Keep only 'Close' column
            df = df[['Close']].copy()

            # Calculate Moving Averages
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()

            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Drop NaN values
            df.dropna(inplace=True)

            # Plot Moving Averages
            st.subheader(f"📊 {ticker} Stock Price with Moving Averages")
            st.line_chart(df[['Close', 'MA50', 'MA200']])

            # Plot RSI
            st.subheader(f"📉 {ticker} RSI Indicator")
            st.line_chart(df['RSI'])

            st.success("✅ Analysis complete!")