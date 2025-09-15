# final_cohesion_check.py — supports --date, adds source_diversity & bias_distribution, saves grouped_articles_final_{date}.json

import json
import os
import openai
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import sys
from datetime import datetime
from collections import Counter
from math import log2
from urllib.parse import urlparse

# Load API key
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Parse optional --date argument
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

# Load NLP + embedding model
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

INPUT_FILE = f"grouped_articles_filtered_{date_str}.json"
OUTPUT_FILE = f"grouped_articles_final_{date_str}.json"

# ----------------------------
# Helpers (ADDITIVE)
# ----------------------------

def domain_from_url(url: str) -> str:
    try:
        return urlparse(url or "").netloc.replace("www.", "").lower()
    except Exception:
        return ""

def source_diversity(articles):
    """
    Compute simple diversity metrics over source domains for a cluster.
    Returns:
      {
        "unique_domains": int,
        "total_with_domain": int,
        "top_domain": [domain, share_float],
        "entropy": float
      }
    """
    domains = [domain_from_url(a.get("url", "")) for a in articles if a.get("url")]
    domains = [d for d in domains if d]
    total = len(domains)
    uniq = len(set(domains))
    if total == 0:
        return {
            "unique_domains": 0,
            "total_with_domain": 0,
            "top_domain": ["", 0.0],
            "entropy": 0.0
        }
    counts = Counter(domains)
    top_dom, top_cnt = counts.most_common(1)[0]
    probs = [c / total for c in counts.values()]
    H = -sum(p * log2(p) for p in probs)
    return {
        "unique_domains": uniq,
        "total_with_domain": total,
        "top_domain": [top_dom, round(top_cnt / total, 3)],
        "entropy": round(H, 3)
    }

BIAS_ORDER = ["Far Left", "Left", "Center", "Right", "Far Right", "Unknown"]

def aggregate_bias_distribution(articles):
    """
    Build a percentage distribution over canonical bias labels for a list of articles.
    Ensures integer percentages that sum to 100 using largest-remainder rounding.
    """
    # Count raw occurrences by canonical label
    raw_counts = Counter()
    for a in articles:
        label = (a.get("bias") or "Center").strip()
        # Normalize some common variants just in case
        norm = label.title().replace("-", " ")
        if norm not in BIAS_ORDER:
            norm = "Unknown" if norm.lower() == "unknown" else ("Center" if norm not in BIAS_ORDER else norm)
        raw_counts[norm] += 1

    total = sum(raw_counts.values())
    if total == 0:
        return {}

    # Compute exact percentages
    exact = {k: (raw_counts.get(k, 0) * 100.0 / total) for k in BIAS_ORDER}
    # Floor to ints and track remainders
    floored = {k: int(exact[k]) for k in BIAS_ORDER}
    remainders = {k: exact[k] - floored[k] for k in BIAS_ORDER}
    # Distribute leftover to highest remainders
    leftover = 100 - sum(floored.values())
    for k, _ in sorted(remainders.items(), key=lambda x: x[1], reverse=True):
        if leftover <= 0:
            break
        floored[k] += 1
        leftover -= 1
    # Remove zeros to keep the chart clean, but keep order when present
    return {k: floored[k] for k in BIAS_ORDER if floored[k] > 0}

# ----------------------------
# Main
# ----------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    clusters = json.load(f)

final_clusters = []

for i, cluster in enumerate(clusters):
    articles = cluster.get("articles", [])
    if len(articles) < 3:
        print(f"⚠️ Skipping cluster {i} — too few articles")
        continue

    titles = [a.get("title", "") for a in articles if a.get("title")]
    if not titles:
        print(f"⚠️ Skipping cluster {i} — no titles")
        continue

    embeddings = embedder.encode(titles)

    avg_vector = np.mean(embeddings, axis=0)
    sim_scores = cosine_similarity([avg_vector], embeddings)[0]
    sim_threshold = 0.6

    kept_articles = [article for score, article in zip(sim_scores, articles) if score >= sim_threshold]
    dropped_articles = [article for score, article in zip(sim_scores, articles) if score < sim_threshold]

    print(f"🔍 Cluster {i} pre-filtered — kept {len(kept_articles)}, dropped {len(dropped_articles)} (by cosine similarity)\n")

    # Named-entity overlap heuristic
    doc_entities = [set(ent.text.lower() for ent in nlp(a.get("title", "")).ents) for a in kept_articles]
    entity_overlap_counts = []

    for idx, entities in enumerate(doc_entities):
        overlap = sum(len(entities & other) for j, other in enumerate(doc_entities) if j != idx)
        entity_overlap_counts.append(overlap)

    min_entity_threshold = 1
    cohesive_articles = [a for a, score in zip(kept_articles, entity_overlap_counts) if score >= min_entity_threshold]

    print(f"🧠 Entity filter — {len(cohesive_articles)} articles pass named entity overlap check")

    if len(cohesive_articles) < 2:
        print(f"❌ Cluster {i} dropped — not enough cohesive articles after heuristic checks\n")
        continue

    titles_cleaned = [a.get("title", "") for a in cohesive_articles]
    joined_titles = "\n".join(titles_cleaned)

    prompt = f"""
You are a news cluster validator. Below are article titles that may describe related events.

Your task is to determine whether these titles refer to the same specific real-world story.

If they clearly describe one story, say: Type: Specific Event
If they describe multiple unrelated or only loosely related stories, say: Type: Mixed
If they are just thematically similar (e.g. multiple protests or crimes), say: Type: Thematic Similarity

Then explain your reasoning briefly.

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
        label_line = result.splitlines()[0].strip()
        explanation = " ".join(result.splitlines()[1:]).strip()

        print(f"🧾 GPT cluster analysis: {label_line} — {explanation}")

        if label_line.lower().startswith("type: specific"):
            # ADDITIVE: compute diversity & bias distribution and attach
            diversity = source_diversity(cohesive_articles)

            # Prefer upstream bias_distribution if present, else compute from articles
            existing_bias_dist = cluster.get("bias_distribution") or {}
            bias_dist = existing_bias_dist if existing_bias_dist else aggregate_bias_distribution(cohesive_articles)

            final_clusters.append({
                "articles": cohesive_articles,
                "source_diversity": diversity,
                "bias_distribution": bias_dist
            })
            print(f"✅ Cluster {i} accepted — passed all filters\n")
        elif label_line.lower().startswith("type: mixed"):
            print(f"🔄 Cluster {i} flagged as Mixed — consider re-clustering manually\n")
        else:
            print(f"❌ Cluster {i} removed — too thematically vague\n")

    except Exception as e:
        print(f"⚠️ GPT error in cluster {i}: {e}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_clusters, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(final_clusters)} high-precision clusters to {OUTPUT_FILE}")
