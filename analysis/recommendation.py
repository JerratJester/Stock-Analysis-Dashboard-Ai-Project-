import numpy as np
from typing import List, Dict, Union
import pandas as pd


def interpret_sentiment(score: float) -> str:
    """Classify compound sentiment score."""
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def interpret_rsi(rsi: float) -> str:
    """Classify RSI value into oversold/overbought/neutral."""
    if rsi < 30:
        return "oversold"
    elif rsi > 70:
        return "overbought"
    return "neutral"


def forecast_trend(forecast: List[float]) -> str:
    """Determine trend direction from forecast list."""
    if len(forecast) < 2:
        return "neutral"
    slope = forecast[-1] - forecast[0]
    if slope > 0:
        return "up"
    elif slope < 0:
        return "down"
    return "neutral"


def recommend_stock_action(
    forecasted_prices: List[float],
    sentiment_score: float,
    rsi_value: float
) -> str:
    """
    Combine forecast trend, sentiment, and RSI to generate a Buy/Hold/Sell recommendation.
    Rules:
      - Buy:    trend up, sentiment positive, RSI oversold
      - Sell:   trend down, sentiment negative, RSI overbought
      - Hold:   otherwise
    """
    trend = forecast_trend(forecasted_prices)
    sentiment = interpret_sentiment(sentiment_score)
    rsi_status = interpret_rsi(rsi_value)

    if trend == "up" and sentiment == "positive" and rsi_status == "oversold":
        return "Buy"
    if trend == "down" and sentiment == "negative" and rsi_status == "overbought":
        return "Sell"
    return "Hold"


def explain_recommendation(
    df: Union[pd.DataFrame, List[float]],
    sentiment_score: float
) -> Dict[str, Union[str, float]]:
    """
    Extracts forecast and RSI from `df` (DataFrame or a list of prices),
    then returns a breakdown dict with keys:
      - Forecast Trend
      - Sentiment
      - RSI
      - Recommendation

    Usage in Streamlit:
        rec = explain_recommendation(df, sentiment_score)
        st.success(f"Recommendation: {rec['Recommendation']}")
    """
    # Determine forecasted prices list
    if isinstance(df, pd.DataFrame):
        if "Forecast" in df.columns:
            forecasted_prices = df["Forecast"].dropna().tolist()
        else:
            # fallback: use last 30 close prices
            forecasted_prices = df.get("Close", df).tolist()[-30:]
        # extract latest RSI if available
        rsi_value = df["RSI"].dropna().iloc[-1] if "RSI" in df.columns else 50.0
    elif isinstance(df, list):
        forecasted_prices = df
        rsi_value = 50.0
    else:
        raise TypeError("Unsupported type for df: expected DataFrame or list of floats.")

    trend = forecast_trend(forecasted_prices)
    sentiment = interpret_sentiment(sentiment_score)
    rsi_status = interpret_rsi(rsi_value)
    action = recommend_stock_action(forecasted_prices, sentiment_score, rsi_value)

    return {
        "Forecast Trend": trend,
        "Sentiment": sentiment,
        "RSI": rsi_status,
        "Recommendation": action
    }

