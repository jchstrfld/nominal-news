import os
import requests
from dotenv import load_dotenv
import openai

# Load your API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
openai.api_key = OPENAI_API_KEY

# Fetch the top 5 headlines
def fetch_top_headlines():
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "language": "en",
        "pageSize": 5,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    articles = response.json().get("articles", [])
    return articles

# Use GPT to summarize each article
def summarize_article(title, content, url):
    prompt = f"""
You are a news summarizer.

Summarize the following news article in 1–3 sentences and provide up to 3 bullet-point takeaways.

Title: {title}

Content: {content}

Format your response like this:

Summary: [Your summary]

Takeaways:
• [Bullet 1]
• [Bullet 2]
• [Bullet 3]

Include this source at the bottom:
Source: {url}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ Failed to summarize article: {e}"

# Orchestrates the news fetching and summarizing
def run_news_summary():
    print("🗞️ Fetching today's top 5 news stories...\n")
    articles = fetch_top_headlines()

    for i, article in enumerate(articles):
        title = article.get("title", "")
        content = article.get("description", "") or article.get("content", "")
        url = article.get("url", "")

        if not content:
            print(f"⚠️ Story {i+1}: Skipped due to missing content — {title}")
            continue

        print(f"\n📌 Story {i+1}: {title}")
        summary = summarize_article(title, content, url)
        print(summary)
        print("-" * 60)

# Entry point
if __name__ == "__main__":
    run_news_summary()
