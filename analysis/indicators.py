import pandas as pd
import ta
import streamlit as st

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to a price DataFrame.
    Requires columns: 'Open', 'High', 'Low', 'Close' (or 'Adj Close' or 'Price'), and 'Volume'.
    """
    df = df.copy()
    
    # Ensure we have a valid 'Close' column
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        elif 'Price' in df.columns:
            df['Close'] = df['Price']
        else:
            st.warning("No 'Close', 'Adj Close', or 'Price' column found.")
            return df

    # Define required numeric columns
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

    # Identify missing or all-NaN columns
    missing = []
    for col in numeric_columns:
        # 1️⃣ Missing column?
        if col not in df.columns:
            missing.append(col)
            continue

        # 2️⃣ All values NaN?
        if df[col].isna().all():
            missing.append(col)


    if missing:
        for col in missing:
            st.error(f"Column '{col}' is missing or contains only NaNs. Filling with 0.")
            df[col] = 0  # Optional fallback: fill with 0s
        # Optionally return df early if you don't want to compute indicators on fallback data
        # return df

    # Convert required columns to numeric (safeguard)
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute technical indicators
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["MACD"]   = ta.trend.macd(df["Close"])
    df["RSI"]    = ta.momentum.rsi(df["Close"], window=14)

    # Bollinger Bands
    boll = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_upper"] = boll.bollinger_hband()
    df["BB_lower"] = boll.bollinger_lband()

    # Trend and Volume indicators
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    df["CCI"] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=20)
    df["MFI"] = ta.volume.money_flow_index(df["High"], df["Low"], df["Close"], df["Volume"], window=14)

    return df

