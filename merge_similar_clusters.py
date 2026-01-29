# merge_similar_clusters.py — now supports --date and preserves core merging logic

import json
import openai
import os
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import sys
import tiktoken  # ✅ NEW
import time      # ✅ NEW
import random    # ✅ NEW

load_dotenv()
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

INPUT_FILE = f"clustered_articles_{date_str}.json"
OUTPUT_FILE = f"grouped_articles_{date_str}.json"
EMBED_MODEL = "text-embedding-3-small"
SIM_THRESHOLD = 0.85

# ✅ Token safety constants
MAX_TOKENS = 7000

# tiktoken may not recognize newer models if the package is old; fallback safely.
try:
    ENCODING = tiktoken.encoding_for_model(EMBED_MODEL)
except KeyError:
    ENCODING = tiktoken.get_encoding("cl100k_base")

def truncate_to_token_limit(text, max_tokens):
    tokens = ENCODING.encode(text)
    return ENCODING.decode(tokens[:max_tokens])

# ✅ Retry-safe embedding wrapper
def get_embedding_with_retry(text, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(
                input=text,
                model=EMBED_MODEL
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            print(f"⚠️ Embedding failed (attempt {attempt + 1}): {e}")
            wait = random.uniform(1.5, 3.5) * (attempt + 1)
            print(f"⏳ Waiting {wait:.1f} seconds before retrying...")
            time.sleep(wait)
    print("❌ Failed to get embedding after retries.")
    return None

def get_cluster_embedding(cluster):
    texts = [f"{a.get('title') or ''}. {a.get('description') or ''}" for a in cluster["articles"]]
    combined = "\n".join(texts[:10])  # Only use first 10 articles
    truncated = truncate_to_token_limit(combined, MAX_TOKENS)  # ✅ now token-safe
    return get_embedding_with_retry(truncated)  # ✅ now retry-safe

def merge_clusters(clusters, embeddings):
    num = len(embeddings)
    used = set()
    merged = []

    for i in range(num):
        if i in used:
            continue

        group = clusters[i]["articles"][:]

        for j in range(i + 1, num):
            if j in used:
                continue
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim >= SIM_THRESHOLD:
                group.extend(clusters[j]["articles"])
                used.add(j)

        # Compute bias distribution
        bias_counts = {}
        total = 0
        for article in group:
            bias_raw = (article.get("bias") or "Unknown").strip()
            bias = bias_raw.title().replace("-", " ")
            if bias != "Unknown":
                bias_counts[bias] = bias_counts.get(bias, 0) + 1
                total += 1

        bias_distribution = {}
        if total > 0:
            for k, v in bias_counts.items():
                bias_distribution[k] = round((v / total) * 100)

        merged.append({
            "topic": f"Topic {i}",
            "articles": group,
            "bias_distribution": bias_distribution
        })

    print(f"✅ Merged {num} clusters into {len(merged)} topic groups")
    return merged

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"🔍 Loaded {len(raw)} topic clusters")
    embeddings = []

    for i, cluster in enumerate(raw):
        print(f"🔄 Processing cluster {i + 1}/{len(raw)}")
        emb = get_cluster_embedding(cluster)
        if emb:
            embeddings.append(emb)
        else:
            embeddings.append(np.zeros(1536, dtype=np.float32))

    grouped = merge_clusters(raw, embeddings)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2)

    print(f"📦 Saved merged topics → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
