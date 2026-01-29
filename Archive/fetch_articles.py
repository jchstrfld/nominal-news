# fetch_articles.py — pulls from NewsAPI + reputable RSS feeds

import json
import os
import feedparser
import requests
from dotenv import load_dotenv

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWS_API_KEY")

NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.dw.com/rdf/rss-en-all",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://feeds.npr.org/1001/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://feeds.foxnews.com/foxnews/latest",
    "https://www.abc.net.au/news/feed/51120/rss.xml",
    "https://www.independent.co.uk/news/world/rss",
    "https://www.latimes.com/world-nation/rss2.0.xml",
    "https://www.euronews.com/rss?level=theme&name=news"
]

OUTPUT_FILE = "articles_raw.json"


def fetch_newsapi():
    print("🔍 Fetching articles from NewsAPI...")
    params = {
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "pageSize": 100,
        "page": 1
    }
    try:
        res = requests.get(NEWSAPI_URL, params=params)
        data = res.json()
        if data.get("status") != "ok":
            print(f"❌ NewsAPI error: {data.get('code')} — {data.get('message')}")
            return []
        articles = [
            {
                "title": a["title"],
                "description": a.get("description", ""),
                "url": a["url"],
                "source": a["source"]["name"]
            } for a in data["articles"]
        ]
        print(f"✅ {len(articles)} valid articles fetched from NewsAPI")
        return articles
    except Exception as e:
        print(f"❌ Failed to fetch NewsAPI: {e}")
        return []


def fetch_rss():
    print("🔍 Fetching articles from RSS feeds...")
    articles = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:25]
            print(f"📡 {feed.feed.get('title', url)} — {len(entries)} entries")
            for e in entries:
                articles.append({
                    "title": e.get("title", ""),
                    "description": e.get("summary", ""),
                    "url": e.get("link", ""),
                    "source": feed.feed.get("title", url)
                })
        except Exception as e:
            print(f"⚠️ Error fetching {url}: {e}")
    print(f"✅ {len(articles)} valid articles from RSS")
    return articles


def main():
    newsapi = fetch_newsapi()
    rss = fetch_rss()
    all_articles = newsapi + rss
    print(f"📊 Total valid articles collected: {len(all_articles)}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2)
    print(f"📦 Articles saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
