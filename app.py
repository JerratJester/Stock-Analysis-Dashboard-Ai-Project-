import streamlit as st
import pandas as pd
import os
import yfinance as yf

from datetime import timedelta
from analysis.indicators import add_indicators
from models.lstm_model import forecast_next_days,train_lstm_model
from models.prophet_model import forecast_prophet,plot_prophet_forecast
from analysis.sentiment import prepare_sentiment_data_from_news, summarize_sentiment
from analysis.recommendation import explain_recommendation
@st.cache_data
def load_price_data(ticker):
    path = f"data/{ticker}.csv"

    if os.path.exists(path):
        try:
            # Your CSV has junk headers in the first two rows, skip them
            df = pd.read_csv(path, header=0, skiprows=[1, 2])
            # Rename 'Price' to 'Date'
            if "Price" in df.columns:
                df.rename(columns={"Price": "Date"}, inplace=True)
            else:
                st.warning("Expected 'Price' column for Date not found.")
                return pd.DataFrame()

            # Convert 'Date' column to datetime and set as index
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.dropna(subset=["Date"], inplace=True)
            df.set_index("Date", inplace=True)

            # Ensure numeric types for all required columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    st.warning(f"Missing column: {col}")
                    return pd.DataFrame()

        except Exception as e:
            st.error(f"❌ Failed to load or process CSV: {e}")
            return pd.DataFrame()

    else:
        try:
            df = yf.download(ticker, start="2020-01-01")
            if df.empty:
                st.error(f"❌ Failed to download data for {ticker}")
                return pd.DataFrame()

            os.makedirs("data", exist_ok=True)
            df.to_csv(path)
        except Exception as e:
            st.error(f"❌ Error downloading data for {ticker}: {e}")
            return pd.DataFrame()

    return df

# Streamlit app config
st.set_page_config(layout="wide")
st.title("📊 Stock & Crypto Dashboard with AI Recommendations")

# define separate ticker lists
stock_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "META", "NFLX", "INTC", "CSCO",
    "CRM", "ADBE", "ORCL", "IBM", "SAP"
]
crypto_tickers = [
    "BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD",
    "XRP-USD", "SOL-USD", "DOGE-USD", "DOT-USD",
    "LTC-USD", "AVAX-USD", "MATIC-USD", "LINK-USD"
]

tabs = st.tabs(["Stocks", "Crypto"])

for tab, market in zip(tabs, ["stocks", "crypto"]):
    with tab:
        # pick the right list
        options = stock_tickers if market == "stocks" else crypto_tickers

        # give each selectbox its own key so Streamlit doesn't mix them up
        ticker = st.selectbox(
            f"Select {market.title()} Ticker",
            options,
            key=f"{market}_ticker"
        )

        df = load_price_data(ticker)
        if df.empty:
            st.warning("No data available for this ticker.")
            continue

        df = add_indicators(df)

        # Verify necessary columns exist before plotting
        required_cols = ["Close", "SMA_20", "SMA_50"]
        if all(col in df.columns for col in required_cols):
            st.subheader("📈 Price Data with Indicators")
            st.line_chart(df[required_cols].dropna())
        else:
            st.warning(f"Missing one or more columns: {required_cols}")

        
        st.subheader("🔮 Forecasting")

        if 'Close' in df.columns:
            price_series = df['Close'].dropna()
            MODEL_PATH = "models/lstm_model.h5"

            # Train model only if not saved yet
            if not os.path.exists(MODEL_PATH):
                st.info("Training LSTM model...")
                train_lstm_model(price_series, model_path=MODEL_PATH)

            try:
                lstm_forecast = forecast_next_days(price_series, model_path=MODEL_PATH)

                # Plot with future dates
                last_date = price_series.index[-1]
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(lstm_forecast), freq='D')
                forecast_df = pd.DataFrame({'LSTM Forecast': lstm_forecast}, index=forecast_index)

                st.line_chart(forecast_df, use_container_width=True)
            except Exception as e:
                st.error(f"LSTM Forecast Error: {e}")

            else:
                try:
                    # get the next 30 days of forecast
                    prophet_forecast = forecast_prophet(price_series, days=30, model_path="models/prophet_model.pkl")

                    # plot using your helper (this draws onto plt)
                    plot_prophet_forecast(price_series, prophet_forecast)

                    # then tell Streamlit to render the current figure
                    st.pyplot()
                except Exception as e:
                    st.error(f"Prophet Forecast Error: {e}")
        else:
            st.warning("No 'Close' column found in data.")
        


        # Sentiment
        st.subheader("📰 News Sentiment")
        try:
            sentiment_score, sentiment_summary = summarize_sentiment(ticker)
            st.metric("Sentiment Score", f"{sentiment_score:.2f}")
            st.json(sentiment_summary)
        except Exception as e:
            st.error(f"Sentiment Analysis Error: {e}")

        # Recommendation
        st.subheader("🤖 AI Recommendation")
        try:
            recommendation = explain_recommendation(df, sentiment_score)
            st.success(f"Recommendation: {recommendation}")
        except Exception as e:
            st.error(f"Recommendation Error: {e}")
