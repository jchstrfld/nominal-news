# fetch_articles_by_date.py — fetches NewsAPI + robust RSS feeds for a specific date with fallback support

import os
import json
import feedparser
import requests
from datetime import datetime
from dotenv import load_dotenv
import argparse
import re
from datetime import timezone

try:
    from langdetect import detect
except Exception:
    detect = None

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

RSS_FEEDS = [
    # Academic / university-affiliated
    "https://hub.jhu.edu/rss/home-feed/",
    "https://news.berkeley.edu/feed/",
    "https://news.harvard.edu/gazette/feed/",
    "https://news.psu.edu/rss/rss.xml",
    "https://news.umich.edu/feed/",
    "https://news.uchicago.edu/rss.xml",
    "https://news.utexas.edu/feed/",
    "https://theconversation.com/rss.xml",
    "https://uncnews.unc.edu/feed/",
    "https://www.buffalo.edu/news/rss.xml",
    "https://www.princeton.edu/news/rss.xml",
    "https://www.science.org/rss/news_current.xml",
    "https://www.sciencenews.org/feed/",
    "https://www.washington.edu/news/feed/",

    # Global/international
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://allafrica.com/tools/headlines/rdf/all.rdf",
    "https://foreignpolicy.com/feed/",
    "https://globalvoices.org/feed",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://feeds.foxnews.com/foxnews/latest",
    "https://feeds.npr.org/1001/rss.xml",
    "https://rss.dw.com/rdf/rss-en-all",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.abc.net.au/news/feed/51120/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.euronews.com/rss?level=theme&name=news",
    "https://www.theguardian.com/world/rss",
    "https://www.independent.co.uk/news/world/rss",
    "https://www.latimes.com/world-nation/rss2.0.xml",

    # Investigative / independent / culture
    "https://feeds.edweek.org/edweek/topstories",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://fortune.com/feed/",
    "https://nautil.us/feed",
    "https://pitchfork.com/feed/feed-news/rss",
    "https://thebaffler.com/feed",
    "https://theintercept.com/rss.xml",
    "https://www.newyorker.com/feed/everything",
    "https://www.propublica.org/feeds/articles",
    "https://www.rollingstone.com/feed/",
    "https://www.smithsonianmag.com/rss/",
    "https://www.space.com/feeds/all",
    "https://www.statnews.com/feed/",
    "https://www.techradar.com/rss",
    "https://www.theatlantic.com/feed/all/",
    "https://www.vanityfair.com/vf-rss.xml",
    "https://www.vogue.com/rss",
    "https://www.wired.com/feed/rss",

    # Local / regional news — major city representation
    "https://billypenn.com/feed",
    "https://chicago.suntimes.com/rss/",
    "https://chicagoreader.com/feed",
    "https://citylimits.org/feed",
    "https://kdvr.com/feed",
    "https://nbcwashington.com/?rss=y",
    "https://news10.com/feed/",
    "https://www.ajc.com/arcio/rss/category/news/",
    "https://www.chron.com/rss/",
    "https://www.inquirer.com/news/nation-world/index.rss",
    "https://www.kabc.com/feed/",
    "https://www.khou.com/rss",
    "https://www.kiro7.com/news/?outputType=rss",
    "https://www.newsday.com/rss",
    "https://www.sfchronicle.com/default/feed/news",
    "https://www.startribune.com/rss/",
    "https://www.wcvb.com/topstories-rss",
    "https://www.wweek.com/feed/",
    "https://www.westword.com/denver/Rss.xml",
    "https://www.wusa9.com/rss",
    "https://wgntv.com/feed/",
    "https://www.wxyz.com/news/rss.xml",

    # National / mainstream US
    "https://abcnews.go.com/abcnews/topstories",
    "https://feeds.axios.com/axios/newsroom",
    "https://feeds.feedburner.com/yahoonewsroom",
    "https://rssfeeds.usatoday.com/usatoday-NewsTopStories",
    "https://rssfeeds.usnews.com/usnews/top-news",
    "https://thehill.com/feed/",
    "https://www.breitbart.com/feed/",
    "https://www.cbsnews.com/latest/rss/main",
    "https://www.commentary.org/feed/",
    "https://www.nationalreview.com/feed/",
    "https://www.nbcnews.com/id/3032091/device/rss/rss.xml",
    "https://www.oann.com/feed/",
    "https://www.realclearpolitics.com/index.xml",
    "https://thedispatch.com/feed/",
    "https://www.thebulwark.com/feed/",
    "https://www.theepochtimes.com/feed",
    "https://www.thegatewaypundit.com/feed/",
    "https://www.washingtonexaminer.com/feed",
    "https://www.washingtontimes.com/rss/headlines/news/politics/",

    # Nonprofit / think tank / policy
    "https://www.aei.org/feed/",
    "https://www.brookings.edu/feed/",
    "https://www.city-journal.org/rss.xml",
    "https://www.heritage.org/rss/all",
    "https://www.kff.org/feed/",
    "https://www.pewresearch.org/feed/",
    "https://www.rand.org/rss.xml",
    "https://www.urban.org/rss.xml"
]

FALLBACK_FEEDS = {
    "https://www.breitbart.com/feed/": "https://feeds.feedburner.com/breitbart",
    "https://www.oann.com/feed/": "https://rss.feedspot.com/oann_rss_feeds/",
    "https://www.theepochtimes.com/feed": "https://www.theepochtimes.com/focus/rss"
}

def iso_from_struct_time(st):
    """Convert feedparser time struct to ISO string (UTC)."""
    try:
        dt = datetime(*st[:6], tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return ""

def date_from_iso(iso: str) -> str:
    """Extract YYYY-MM-DD from an ISO string."""
    if not iso:
        return ""
    return iso[:10]

def fetch_newsapi(date_str):
    print(f"🔍 Fetching NewsAPI articles for {date_str}...")
    params = {
        "q": "politics OR war OR elections OR protest OR conflict OR economy",
        "language": "en",
        "pageSize": 100,
        "sortBy": "publishedAt",
        "from": date_str,
        "to": date_str,
        "apiKey": NEWS_API_KEY,
    }
    try:
        res = requests.get(NEWSAPI_URL, params=params)
        data = res.json()
        if data.get("status") != "ok":
            print(f"❌ NewsAPI error: {data.get('code')} — {data.get('message')}")
            return []
        return [
            {
                "title": a["title"],
                "description": a.get("description", ""),
                "url": a["url"],
                "source": a["source"]["name"],
                "published_at": a.get("publishedAt", ""),
                "published_date": (a.get("publishedAt", "") or "")[:10],
            } for a in data["articles"]
        ]
    except Exception as e:
        print(f"❌ NewsAPI fetch error: {e}")
        return []

def is_english_text(text: str) -> bool:
    """
    Ingestion-time language gate (token-free).
    - Uses langdetect if installed.
    - Fail-open for short/empty text to avoid dropping real English headlines.
    """
    if not text:
        return True
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 25:
        return True
    if detect is None:
        return True
    try:
        return detect(t) == "en"
    except Exception:
        return True

def fetch_rss():
    print("🔍 Fetching RSS articles...")
    articles = []

    for url in RSS_FEEDS:
        success = False
        for attempt in range(2):
            try:
                feed = feedparser.parse(url)
                if feed.entries:
                    print(f"📡 {feed.feed.get('title', url)} — {len(feed.entries[:25])} entries")
                    for e in feed.entries[:25]:
                        title = e.get("title", "") or ""
                        desc = e.get("summary", "") or ""
                        text_for_lang = f"{title}. {desc}"[:800]

                        if not is_english_text(text_for_lang):
                            continue
                        
                        published_at = ""
                        if e.get("published_parsed"):
                            published_at = iso_from_struct_time(e.get("published_parsed"))
                        elif e.get("updated_parsed"):
                            published_at = iso_from_struct_time(e.get("updated_parsed"))

                        published_date = date_from_iso(published_at)

                        articles.append({
                            "title": title,
                            "description": desc,
                            "url": e.get("link", ""),
                            "source": feed.feed.get("title", url),
                            "published_at": published_at,
                            "published_date": published_date,
                        })
                    success = True
                    break
                else:
                    print(f"⚠️ Empty feed attempt {attempt + 1}: {url}")
            except Exception as e:
                print(f"⚠️ Error fetching {url} (attempt {attempt + 1}): {e}")

        if not success and url in FALLBACK_FEEDS:
            fallback_url = FALLBACK_FEEDS[url]
            print(f"↩️ Attempting fallback: {fallback_url}")
            try:
                fallback_feed = feedparser.parse(fallback_url)
                if fallback_feed.entries:
                    print(f"📡 {fallback_feed.feed.get('title', fallback_url)} — {len(fallback_feed.entries[:25])} entries (fallback)")
                    for e in fallback_feed.entries[:25]:
                        title = e.get("title", "") or ""
                        desc = e.get("summary", "") or ""
                        text_for_lang = f"{title}. {desc}"[:800]

                        if not is_english_text(text_for_lang):
                            continue

                        published_at = ""
                        if e.get("published_parsed"):
                            published_at = iso_from_struct_time(e.get("published_parsed"))
                        elif e.get("updated_parsed"):
                            published_at = iso_from_struct_time(e.get("updated_parsed"))

                        published_date = date_from_iso(published_at)

                        articles.append({
                            "title": title,
                            "description": desc,
                            "url": e.get("link", ""),
                            "source": fallback_feed.feed.get("title", fallback_url),
                            "published_at": published_at,
                            "published_date": published_date,
                        })
                else:
                    print(f"⚠️ Fallback feed also empty: {fallback_url}")
            except Exception as e:
                print(f"❌ Fallback failed for {url}: {e}")

    print(f"✅ {len(articles)} total RSS articles collected")
    return articles

def main(date):
    iso_date = date.strftime("%Y-%m-%d")
    output_file = f"articles_raw_{iso_date}.json"

    newsapi = fetch_newsapi(iso_date)
    rss = fetch_rss()
    all_articles = newsapi + rss
    print(f"📦 Collected {len(all_articles)} articles on {iso_date}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2)
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    args = parser.parse_args()

    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    except Exception:
        print("❌ Invalid date. Use format YYYY-MM-DD")
        exit(1)

    main(target_date)
