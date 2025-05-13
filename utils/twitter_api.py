import os
import tweepy
from data.env_loader import load_env_keys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_env_keys()

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# --- Tweepy client setup ---
try:
    twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
except Exception as e:
    print(f"[Twitter AUTH ERROR] {e}")
    twitter_client = None

# --- Sentiment Analyzer ---
sentiment_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = sentiment_analyzer.polarity_scores(text)
    return score['compound']

# --- Tweet Search ---
def fetch_recent_tweets(query, max_results=5):
    if not twitter_client:
        return []
    try:
        response = twitter_client.search_recent_tweets(
            query=query,
            tweet_fields=["created_at", "text"],
            max_results=max_results
        )
        tweets = response.data or []
        return [
            {
                "text": tweet.text,
                "created_at": tweet.created_at.isoformat() if tweet.created_at else "",
                "sentiment": analyze_sentiment(tweet.text)
            }
            for tweet in tweets
        ]
    except Exception as e:
        print(f"[Twitter API ERROR] {query} - {e}")
        return []

# --- User Tweet Fetcher ---
def fetch_user_tweets(username, max_results=5):
    if not twitter_client:
        return []
    try:
        user_response = twitter_client.get_user(username=username)
        if not user_response.data:
            raise ValueError("User not found")

        user_id = user_response.data.id
        response = twitter_client.get_users_tweets(
            id=user_id,
            tweet_fields=["created_at", "text"],
            max_results=max_results
        )
        tweets = response.data or []
        return [
            {
                "text": tweet.text,
                "created_at": tweet.created_at.isoformat() if tweet.created_at else "",
                "sentiment": analyze_sentiment(tweet.text)
            }
            for tweet in tweets
        ]
    except Exception as e:
        print(f"[Twitter USER ERROR] {username} - {e}")
        return []