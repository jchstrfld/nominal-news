# filter_loose_clusters.py — Strict YES/NO GPT cohesion with logging and redundancy

import openai
import json
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_FILE = "grouped_articles.json"
OUTPUT_FILE = "grouped_articles_filtered.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    clusters = json.load(f)

filtered_clusters = []

for i, cluster in enumerate(clusters):
    articles = cluster.get("articles", [])
    if len(articles) < 2:
        print(f"⚠️ Skipping cluster {i} — too few articles")
        continue

    titles = [a.get("title", "") for a in articles if a.get("title")]
    if not titles:
        print(f"⚠️ Skipping cluster {i} — no usable titles")
        continue

    joined_titles = "\n".join(titles)
    prompt = f"""
You are a news clustering assistant. Below is a list of article titles that were grouped together.

Your task is to determine whether these article titles all refer to the same specific real-world news story.

Only answer YES if all the titles clearly refer to the same event, incident, or news development.
If the titles refer to a mix of unrelated events (even if they share a theme or topic), answer NO.

After your YES or NO, explain your reasoning briefly (1–2 sentences).

Titles:
{joined_titles}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0].message["content"].strip()
        decision_line = result.splitlines()[0].strip().upper()
        explanation = " ".join(result.splitlines()[1:]).strip()

        if decision_line.startswith("YES"):
            filtered_clusters.append(cluster)
            print(f"✅ Cluster {i} accepted — GPT: YES — {explanation}")
        else:
            print(f"❌ Cluster {i} removed — GPT: NO — {explanation}")

    except Exception as e:
        print(f"⚠️ GPT error on cluster {i}: {e}")

# Save filtered clusters
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered_clusters, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(filtered_clusters)} cohesive clusters to {OUTPUT_FILE}")
