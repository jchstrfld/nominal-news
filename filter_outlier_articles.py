# filter_outlier_articles.py — Refines clusters by removing off-topic articles (now supports --date)

import openai
import json
import os
from dotenv import load_dotenv
from pathlib import Path
import sys
from datetime import datetime

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Handle --date argument
args = sys.argv
if "--date" in args:
    date_idx = args.index("--date") + 1
    if date_idx < len(args):
        date_str = args[date_idx]
    else:
        print("❌ No date provided after --date")
        sys.exit(1)
else:
    date_str = datetime.today().strftime("%Y-%m-%d")

INPUT_FILE = f"grouped_articles_{date_str}.json"
OUTPUT_FILE = f"grouped_articles_filtered_{date_str}.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    clusters = json.load(f)

filtered_clusters = []

for i, cluster in enumerate(clusters):
    articles = cluster.get("articles", [])
    if len(articles) < 3:
        print(f"⚠️ Skipping cluster {i} — too few articles")
        continue

    titles = [a.get("title", "") for a in articles if a.get("title")]
    if not titles:
        print(f"⚠️ Skipping cluster {i} — no usable titles")
        continue

    print(f"🔍 Evaluating cluster {i} with {len(titles)} titles")
    remaining_articles = []

    for j, article in enumerate(articles):
        current_title = article.get("title", "").strip()
        other_titles = [t for idx, t in enumerate(titles) if idx != j and t.strip()]
        context = "\n".join(other_titles)

        prompt = f"""
You are a news clustering assistant. One article title is shown below along with other titles grouped in the same topic cluster.
Determine whether the first title is about the same real-world news story as the rest.

Only answer YES if it clearly refers to the same event or development. Otherwise, answer NO.

Article title:
{current_title}

Other titles:
{context}
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message["content"].strip().upper()
            decision = result.splitlines()[0]

            if decision.startswith("YES"):
                remaining_articles.append(article)
                print(f"  ✅ Article {j} kept — GPT: YES")
            else:
                print(f"  ❌ Article {j} removed — GPT: NO")

        except Exception as e:
            print(f"⚠️ GPT error on article {j} in cluster {i}: {e}")

    if len(remaining_articles) >= 2:
        filtered_clusters.append({"articles": remaining_articles})
        print(f"✅ Cluster {i} retained with {len(remaining_articles)} articles after filtering\n")
    else:
        print(f"❌ Cluster {i} discarded after filtering — too few cohesive articles\n")

# Save filtered clusters
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered_clusters, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(filtered_clusters)} refined clusters to {OUTPUT_FILE}")
