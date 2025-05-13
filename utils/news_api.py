import os
import requests
from data.env_loader import load_env_keys

load_env_keys()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# --- NewsAPI ---
def fetch_newsapi_articles(query, language="en", page_size=5):
    try:
        url = (
            f"https://newsapi.org/v2/everything?q={query}&language={language}"
            f"&sortBy=publishedAt&pageSize={page_size}&apiKey={NEWS_API_KEY}"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": article["title"],
                "url": article["url"],
                "publishedAt": article["publishedAt"]
            }
            for article in data.get("articles", [])
        ]
    except Exception as e:
        print(f"[NewsAPI ERROR] {query} - {e}")
        return []

# --- GNews (Fallback) ---
def fetch_gnews_articles(query, lang="en", max_articles=5):
    try:
        url = (
            f"https://gnews.io/api/v4/search?q={query}&lang={lang}"
            f"&max={max_articles}&token={GNEWS_API_KEY}"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": article["title"],
                "url": article["url"],
                "publishedAt": article["publishedAt"]
            }
            for article in data.get("articles", [])
        ]
    except Exception as e:
        print(f"[GNews ERROR] {query} - {e}")
        return []

# --- Combined News Fetcher ---
def fetch_news(query, language="en", limit=5):
    articles = fetch_newsapi_articles(query, language=language, page_size=limit)
    if not articles:
        articles = fetch_gnews_articles(query, lang=language, max_articles=limit)
    return articles
