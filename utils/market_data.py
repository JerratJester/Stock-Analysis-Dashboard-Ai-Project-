import os
import requests
import yfinance as yf
import pandas as pd
from data.env_loader import load_env_keys

load_env_keys()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# --- YFinance: Stocks & Crypto ---
def fetch_yfinance_data(symbol, period="1y", interval="1d"):
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError(f"No data from yfinance for {symbol}")
        return data
    except Exception as e:
        print(f"[YFinance ERROR] {symbol} - {e}")
        return pd.DataFrame()

# --- Alpha Vantage: Crypto ---
def fetch_alpha_vantage_crypto(symbol="BTC", market="USD"):
    try:
        url = (
            f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY"
            f"&symbol={symbol}&market={market}&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        response = requests.get(url)
        data = response.json()
        key = "Time Series (Digital Currency Daily)"
        if key not in data:
            raise ValueError(f"Invalid data for {symbol}: {data}")
        df = pd.DataFrame.from_dict(data[key], orient="index").sort_index()
        df = df.rename(columns={
            "1a. open (USD)": "Open",
            "2a. high (USD)": "High",
            "3a. low (USD)": "Low",
            "4a. close (USD)": "Close",
            "5. volume": "Volume"
        }).astype(float)
        return df
    except Exception as e:
        print(f"[AlphaVantage ERROR] {symbol} - {e}")
        return pd.DataFrame()

# --- Polygon.io: Crypto ---
def fetch_polygon_crypto(symbol="X:BTCUSD", timespan="day", limit=365):
    try:
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/2023-01-01/2023-12-31"
            f"?adjusted=true&sort=desc&limit={limit}&apiKey={POLYGON_API_KEY}"
        )
        response = requests.get(url)
        data = response.json().get("results", [])
        if not data:
            raise ValueError("No results returned from Polygon API")
        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("t", inplace=True)
        return df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    except Exception as e:
        print(f"[Polygon ERROR] {symbol} - {e}")
        return pd.DataFrame()

# --- Unified Dispatcher ---
def fetch_market_data(symbol, source="yfinance", type="stock", **kwargs):
    symbol = symbol.strip().upper()

    # Smart detection for crypto ticker cleanup
    crypto_symbols_map = {
        "ETH-USD": "ETH",
        "BTC-USD": "BTC",
        "DOGE-USD": "DOGE",
        "ADA-USD": "ADA",
        "SOL-USD": "SOL",
        "XRP-USD": "XRP",
        "BNB-USD": "BNB",
    }

    if type == "stock":
        if source == "yfinance":
            return fetch_yfinance_data(symbol, **kwargs)
        # Placeholder: Add other stock sources here later

    elif type == "crypto":
        if source == "yfinance":
            return fetch_yfinance_data(symbol, **kwargs)
        elif source == "alphavantage":
            base_symbol = crypto_symbols_map.get(symbol, symbol.split("-")[0])
            return fetch_alpha_vantage_crypto(base_symbol, market="USD")
        elif source == "polygon":
            polygon_symbol = f"X:{symbol.replace('-USD', 'USD')}"
            return fetch_polygon_crypto(symbol=polygon_symbol, timespan=kwargs.get("timespan", "day"))

    raise ValueError(f"[Market Data ERROR] Unsupported combination: {type}/{source}")
