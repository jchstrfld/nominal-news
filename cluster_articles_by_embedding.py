# cluster_articles_by_embedding.py — Adds --date support, truncation, retry & delay logic + caching (token cap stays at 7000)

import openai
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from cache import load_cache, save_cache, key_for, get_cached_vector, put_cached_vector
import sys
import tiktoken
import time
import random

# Load API key from .env
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
MODEL_NAME = "text-embedding-ada-002"
EMBED_DIM = 1536
MAX_TOKENS = 7000  # ✅ Safe limit under 8192 (token-based, not chars)
ENCODING = tiktoken.encoding_for_model(MODEL_NAME)

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

INPUT_FILE = f"articles_with_bias_{date_str}.json"
OUTPUT_FILE = f"clustered_articles_{date_str}.json"

# --- Token truncation ---
def truncate_to_token_limit(text, max_tokens):
    tokens = ENCODING.encode(text or "")
    if len(tokens) <= max_tokens:
        return text or ""
    return ENCODING.decode(tokens[:max_tokens])

# --- Embedding with retry logic ---
def get_embedding_with_retry(text, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(
                input=text,
                model=MODEL_NAME
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            print(f"⚠️ Embedding failed (attempt {attempt + 1}): {e}")
            wait = random.uniform(1.5, 3.5) * (attempt + 1)
            print(f"⏳ Waiting {wait:.1f} seconds before retrying...")
            time.sleep(wait)
    print("❌ Failed to get embedding after retries.")
    return None

# --- Load articles ---
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

texts = [truncate_to_token_limit(f"{a.get('title', '')} {a.get('description', '')}", MAX_TOKENS) for a in articles]

print(f"📊 Embedding {len(texts)} articles...")

# --- Load cache once ---
cache = load_cache()
cache_dirty = False

embeddings = []
for idx, text in enumerate(texts):
    # 🔑 Cache key is based on text+model
    k = key_for(text, MODEL_NAME)
    cached = get_cached_vector(cache, k)
    if cached is not None:
        embeddings.append(cached)
    else:
        emb = get_embedding_with_retry(text)
        if emb is None:
            emb = list(np.zeros(EMBED_DIM))  # fallback to zero vector
        else:
            # store in cache
            put_cached_vector(cache, k, emb, MODEL_NAME)
            cache_dirty = True
        embeddings.append(emb)

    # optional small pacing to be gentle
    time.sleep(0.25)

    # periodically flush cache so you keep progress on long runs
    if cache_dirty and (idx % 50 == 0):
        save_cache(cache)
        cache_dirty = False

# final cache flush
if cache_dirty:
    save_cache(cache)

# --- KMeans clustering ---
num_clusters = max(5, len(articles) // 10)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# ensure embeddings is a numeric array for sklearn
labels = kmeans.fit_predict(np.array(embeddings, dtype=np.float32))

clusters = [{} for _ in range(num_clusters)]
for label, article in zip(labels, articles):
    if "articles" not in clusters[label]:
        clusters[label]["articles"] = []
    clusters[label]["articles"].append(article)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(clusters, f, indent=2, ensure_ascii=False)

print(f"✅ Saved clustered output to {OUTPUT_FILE}")
