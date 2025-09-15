# summarize_clusters.py (Final Version - Organic Volume Model)
import json
import os
from dotenv import load_dotenv
import openai
from collections import Counter

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

credibility_scores = {
    "Associated Press": 1.0,
    "AP": 1.0,
    "Reuters": 1.0,
    "AFP": 1.0,
    "NPR": 1.0,
    "BBC News": 0.9,
    "Deutsche Welle": 0.9,
    "The Wall Street Journal": 0.9,
    "The New York Times": 0.9,
    "Bloomberg": 0.9,
    "Al Jazeera": 0.9,
    "NHK WORLD-JAPAN": 0.9,
    "ABC News (Australia)": 0.9,
    "CBC": 0.9,
    "The Guardian": 0.8,
    "CNN": 0.8,
    "Politico": 0.8,
    "The Hill": 0.8,
    "Yahoo News": 0.8,
    "Fox News": 0.7,
    "MSNBC": 0.7,
    "The Telegraph": 0.7,
    "India Today": 0.7,
    "Daily Wire": 0.6,
    "The Blaze": 0.6,
    "Jacobin": 0.6,
    "Democracy Now": 0.6,
    "Breitbart": 0.5,
    "Epoch Times": 0.5,
    "OANN": 0.4,
    "The Gateway Pundit": 0.3,
    "unknown": 0.0
}

bias_weights = {
    "center": 1.0,
    "left": 0.8,
    "right": 0.8,
    "far-left": 0.5,
    "far-right": 0.5,
    "unknown": 0.6
}

def load_clusters(path="clustered_articles.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_opinion(article):
    title = (article.get("title") or "").lower()
    url = (article.get("url") or "").lower()
    return any(k in title for k in ["opinion", "editorial", "column", "viewpoint"]) or \
           any(k in url for k in ["opinion", "editorial", "op-ed", "column"])

def should_filter_article(article):
    if len(article.get("description") or "") < 30:
        return True
    if article.get("bias") in ["far-left", "far-right"] and is_opinion(article):
        return True
    return False

def summarize_topic(articles):
    weighted_snippets = []
    for a in articles:
        if should_filter_article(a):
            continue
        weight = bias_weights.get(a.get("bias", "unknown"), 0.6)
        if is_opinion(a):
            weight *= 0.6
        credibility = credibility_scores.get(a.get("source", "unknown"), 0.0)
        weight *= credibility
        snippet = f"Title: {(a.get('title') or '')}\nDescription: {(a.get('description') or '')}"
        weighted_snippets.extend([snippet] * max(1, int(weight * 5)))

    if not weighted_snippets:
        return "⚠️ Not enough quality data to summarize this topic."

    contents = "\n\n".join(weighted_snippets)
    prompt = f"""
You are a politically neutral news summarizer.
You will be given excerpts from multiple news articles about the same topic. Your task is to:
1. Write a short 1–3 sentence summary of the overall topic.
2. Provide 1–3 concise bullet-point takeaways.
3. Avoid speculation, blame, emotional language, or political framing.
4. Focus only on widely corroborated facts from credible sources.

Articles:
{contents}

Respond with:
Summary: [your summary here]
Takeaways:
- Point 1
- Point 2
- Point 3
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Failed to summarize: {e}"

def generate_topic_title(articles):
    titles = [a.get("title") for a in articles if a.get("title")]
    prompt = """
Here are several article headlines about the same topic. Create a short, clear title (3–6 words) that summarizes the common story.

Headlines:
""" + "\n".join(titles[:5]) + "\n\nTitle:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "Untitled Topic"

def calculate_bias_distribution(articles):
    if not articles:
        return {"unknown": "100%"}, "unknown"
    bias_counts = Counter(a.get("bias", "unknown") for a in articles)
    total = sum(bias_counts.values())
    distribution = {
        label: f"{round((count / total) * 100)}%"
        for label, count in bias_counts.items()
    }
    majority_bias = max(bias_counts, key=bias_counts.get)
    return distribution, majority_bias

def generate_topic_summaries(input_path="clustered_articles.json", output_path="topic_summaries.json"):
    clusters = load_clusters(input_path)
    summaries = []

    sorted_clusters = sorted(clusters, key=lambda c: len(c["articles"]), reverse=True)

    for cluster in sorted_clusters[:10]:
        topic_id = cluster["topic_id"]
        articles = cluster["articles"]

        summary = summarize_topic(articles)
        topic_title = generate_topic_title(articles)
        bias_breakdown, majority_bias = calculate_bias_distribution(articles)
        sources = list({a.get("url") for a in articles if a.get("url")})

        summaries.append({
            "topic_id": topic_id,
            "topic_title": topic_title,
            "bias_majority": majority_bias,
            "summary": summary,
            "bias_distribution": bias_breakdown,
            "sources": sources
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"✅ Saved {len(summaries)} topic summaries to {output_path}")

if __name__ == "__main__":
    generate_topic_summaries()
