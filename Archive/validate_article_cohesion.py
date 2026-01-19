# validate_article_cohesion.py — Final cleanup of topic clusters by matching sources to summary

import openai
import json
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_ARTICLES = "grouped_articles_filtered.json"
INPUT_SUMMARIES = "topic_summaries.json"
OUTPUT_CLEANED = "grouped_articles_final.json"

with open(INPUT_ARTICLES, "r", encoding="utf-8") as f:
    article_clusters = json.load(f)

with open(INPUT_SUMMARIES, "r", encoding="utf-8") as f:
    summaries = json.load(f)

final_clusters = []

for i, (cluster, summary_data) in enumerate(zip(article_clusters, summaries)):
    articles = cluster.get("articles", [])
    summary_text = summary_data.get("summary", "").strip()
    topic_title = summary_data.get("topic_title", "").strip()

    if len(articles) < 2 or not summary_text:
        print(f"⚠️ Skipping topic {i} — missing data")
        continue

    titles = [a.get("title", "") for a in articles if a.get("title")]
    title_list = "\n".join([f"- {t}" for t in titles])

    prompt = f"""
You are a news consistency checker. Below is a topic title, a topic summary, and a list of article titles that contributed to this topic.

Your job is to identify which article titles do NOT clearly support or relate to the core topic described in the summary.

Respond with ONLY a list of off-topic article titles. If all titles are clearly related to the topic, say "All titles match."

Topic title:
{topic_title}

Summary:
{summary_text}

Article titles:
{title_list}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message["content"].strip()

        if reply.lower().startswith("all titles match"):
            final_clusters.append(cluster)
            print(f"✅ Topic {i} accepted — all sources aligned")
        else:
            bad_titles = [line.strip("- ").strip() for line in reply.splitlines() if line.strip().startswith("-")]
            if not bad_titles:
                print(f"⚠️ Topic {i} flagged but no titles parsed — keeping all")
                final_clusters.append(cluster)
                continue

            cleaned_articles = [a for a in articles if a.get("title", "").strip() not in bad_titles]
            if len(cleaned_articles) >= 2:
                final_clusters.append({"articles": cleaned_articles})
                print(f"🧹 Topic {i} cleaned — removed {len(articles) - len(cleaned_articles)} articles")
            else:
                print(f"❌ Topic {i} dropped — too few articles after cleanup")

    except Exception as e:
        print(f"⚠️ GPT error on topic {i}: {e}")

# Save final cleaned clusters
with open(OUTPUT_CLEANED, "w", encoding="utf-8") as f:
    json.dump(final_clusters, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(final_clusters)} fully validated clusters to {OUTPUT_CLEANED}")
