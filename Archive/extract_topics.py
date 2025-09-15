# extract_topics.py — use GPT-4 to label and group article topics with caching
import json
import time
import os
import openai
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_PATH = "articles_with_bias.json"
OUTPUT_PATH = "grouped_articles.json"
CACHE_PATH = "topic_cache.json"
MODEL = "gpt-4"

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def extract_topic_label(title, description, cache):
    key = title.strip()
    if key in cache:
        return cache[key]

    prompt = (
        "You are an expert news analyst. "
        "Given the following title and description of a news article, generate a short 3–6 word topic label that clearly summarizes the main story or subject. "
        "Avoid generic or vague phrases."
        f"\n\nTitle: {title}\nDescription: {description}\n\nTopic Label:"
    )

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        label = response.choices[0].message["content"].strip()
        cache[key] = label
        save_cache(cache)
        return label
    except Exception as e:
        print(f"⚠️ Error from OpenAI: {e}")
        return "Uncategorized"

def group_articles_by_topic(articles, cache):
    grouped = defaultdict(list)

    for i, article in enumerate(articles):
        title = article.get("title", "")
        description = article.get("description", "")

        if not title and not description:
            continue

        topic = extract_topic_label(title, description, cache)
        print(f"[{i+1}/{len(articles)}] → Topic: {topic}")
        grouped[topic].append(article)
        time.sleep(1.2)

    return grouped

def save_grouped_articles(grouped):
    grouped_list = [
        {"topic": topic, "articles": articles}
        for topic, articles in grouped.items()
    ]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(grouped_list, f, indent=2)
    print(f"\n✅ Saved {len(grouped_list)} topic groups → {OUTPUT_PATH}")

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    cache = load_cache()
    grouped = group_articles_by_topic(articles, cache)
    save_grouped_articles(grouped)

if __name__ == "__main__":
    main()
