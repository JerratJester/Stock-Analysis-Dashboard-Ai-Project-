import os
import csv
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.news_api import fetch_news

analyzer = SentimentIntensityAnalyzer()

def prepare_sentiment_data_from_news(ticker: str, limit: int = 5) -> List[Dict]:
    """
    Fetches news articles for the given ticker using utils.news_api.fetch_news,
    analyzes headline sentiment, and returns a list of dicts with keys:
    'title', 'url', 'publishedAt', 'sentiment'.
    """
    raw = fetch_news(ticker, limit=limit)
    if not isinstance(raw, list):
        raise TypeError(f"fetch_news returned {type(raw)}, expected List[Dict].")

    results: List[Dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        title = entry.get("title") or entry.get("description") or ""
        score = analyzer.polarity_scores(title).get("compound", 0.0)
        results.append({
            "title": title,
            "url": entry.get("url", ""),
            "publishedAt": entry.get("publishedAt", ""),
            "sentiment": score
        })
    return results


def summarize_sentiment(ticker: str, limit: int = 5) -> Tuple[float, Dict[str, float]]:
    """
    Fetches sentiment data for a ticker and returns a tuple:
    (average_sentiment_score, summary_dict).

    summary_dict has keys: average, positive, neutral, negative, total.
    """
    data = prepare_sentiment_data_from_news(ticker, limit=limit)
    if not data:
        summary = {"average": 0.0, "positive": 0, "neutral": 0, "negative": 0, "total": 0}
        return 0.0, summary

    total = len(data)
    avg = sum(item["sentiment"] for item in data) / total
    pos = sum(1 for item in data if item["sentiment"] >= 0.05)
    neg = sum(1 for item in data if item["sentiment"] <= -0.05)
    neu = total - pos - neg

    summary = {
        "average": round(avg, 4),
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "total": total
    }
    return summary["average"], summary


def summarize_combined_sentiment(news_data: List[Dict], tweet_data: List[Dict]) -> Dict[str, float]:
    """
    Combines two sentiment lists (news and tweets) and summarizes overall sentiment.
    """
    combined = news_data + tweet_data
    if not isinstance(combined, list):
        raise TypeError("Expected lists of dicts for sentiment data.")
    # Reuse summary logic on combined list
    total = len(combined)
    if total == 0:
        return {"average": 0.0, "positive": 0, "neutral": 0, "negative": 0, "total": 0}
    avg = sum(item.get("sentiment", 0.0) for item in combined) / total
    pos = sum(1 for item in combined if item.get("sentiment", 0.0) >= 0.05)
    neg = sum(1 for item in combined if item.get("sentiment", 0.0) <= -0.05)
    neu = total - pos - neg
    return {
        "average": round(avg, 4),
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "total": total
    }


def save_sentiment_summary(ticker: str, summary: Dict[str, float], filepath: str = "sentiment_log.csv") -> None:
    """
    Appends a summary row to a CSV log with timestamp and counts.
    """
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "ticker", "average", "positive", "neutral", "negative", "total"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            ticker,
            summary.get("average", 0.0),
            summary.get("positive", 0),
            summary.get("neutral", 0),
            summary.get("negative", 0),
            summary.get("total", 0)
        ])


def plot_sentiment_trend(filepath: str = "sentiment_log.csv", ticker: str = None):
    """
    Reads the sentiment log CSV and plots average sentiment and volume over time.
    If ticker is provided, filters log for that ticker.
    """
    if not os.path.isfile(filepath):
        print(f"[INFO] No sentiment log found at {filepath}.")
        return None

    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    if ticker:
        df = df[df["ticker"].str.upper() == ticker.upper()]

    if df.empty:
        print(f"[INFO] No sentiment data available for {ticker}.")
        return None

    df = df.sort_values("timestamp")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["timestamp"], df["average"], marker="o", label="Avg Sentiment")
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Average Sentiment")
    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["total"], linestyle="--", marker="x", label="Volume")
    ax2.set_ylabel("Volume")
    fig.tight_layout()
    plt.title(f"Sentiment & Volume Trend{' for ' + ticker if ticker else ''}")
    plt.grid(True)
    return fig
