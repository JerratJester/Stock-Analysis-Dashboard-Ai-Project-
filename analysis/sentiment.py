from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def analyze_text(text: str) -> float:
    return analyzer.polarity_scores(text)['compound']

def summarize_sentiment(sentiment_data: List[dict]) -> dict:
    if not sentiment_data:
        return {
            "average": 0.0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "total": 0
        }

    total = len(sentiment_data)
    avg = sum(item['sentiment'] for item in sentiment_data) / total
    classified = [classify_sentiment(item['sentiment']) for item in sentiment_data]

    return {
        "average": round(avg, 4),
        "positive": classified.count("positive"),
        "neutral": classified.count("neutral"),
        "negative": classified.count("negative"),
        "total": total
    }

def prepare_sentiment_data_from_news(articles: List[dict]) -> List[dict]:
    return [
        {"text": article["title"], "sentiment": analyze_text(article["title"])}
        for article in articles if "title" in article
    ]

def prepare_sentiment_data_from_tweets(tweets: List[dict]) -> List[dict]:
    return [
        {"text": tweet["text"], "sentiment": analyze_text(tweet["text"])}
        for tweet in tweets if "text" in tweet
    ]

def summarize_combined_sentiment(news_data: List[dict], tweet_data: List[dict]) -> dict:
    combined = news_data + tweet_data
    return summarize_sentiment(combined)

def save_sentiment_summary(ticker: str, summary: dict, filepath: str = "sentiment_log.csv") -> None:
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
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
    if not os.path.isfile(filepath):
        print(f"[INFO] No sentiment file found at {filepath}.")
        return

    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    if ticker:
        df = df[df["ticker"] == ticker.upper()]

    if df.empty:
        print(f"[INFO] No sentiment data available for {ticker}.")
        return

    df = df.sort_values("timestamp")
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df["timestamp"], df["average"], color="tab:blue", marker="o", label="Avg Sentiment")
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Average Sentiment", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["total"], color="tab:orange", linestyle="--", marker="x", label="Volume")
    ax2.set_ylabel("Volume (News + Tweets)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(f"Sentiment & Volume Trend for {ticker}")
    fig.tight_layout()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()
