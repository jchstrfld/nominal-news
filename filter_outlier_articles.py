# filter_outlier_articles.py — Math-only cluster refinement (NO GPT)
# - Reads grouped_articles_{date}.json
# - Optionally trims low-cohesion tail articles from large, tail-contaminated clusters
# - Writes grouped_articles_filtered_{date}.json
#
# Goal: Preserve coverage while improving cluster purity.
# No OpenAI usage. No keyword lists. All math-based (local embeddings + centroid similarity).

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# ----------------------------
# CLI date handling
# ----------------------------
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


# ----------------------------
# Local embedder (token-free)
# ----------------------------
_EMBEDDER = None

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        if SentenceTransformer is None:
            return None
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _article_text(a: dict) -> str:
    # Math-only: title + short desc (no keyword lists)
    t = _clean(a.get("title") or "")
    d = _clean(a.get("description") or "")
    if d:
        d = d[:400]
    if t and d:
        return f"{t}. {d}"
    return t or d or _clean(a.get("url") or "")


def _cohesion_stats(vecs: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """
    vecs: (n, d) L2-normalized embeddings
    returns: mean_sim, p10_sim, std_sim, sims_to_centroid
    """
    n = int(vecs.shape[0])
    if n < 2:
        return (0.0, 0.0, 0.0, np.zeros((n,), dtype=np.float32))

    centroid = vecs.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid = centroid / np.maximum(denom, 1e-12)

    sims = (vecs @ centroid.T).reshape(-1)
    mean_sim = float(np.mean(sims))
    p10_sim = float(np.percentile(sims, 10))
    std_sim = float(np.std(sims))
    return mean_sim, p10_sim, std_sim, sims


def trim_cluster_tail_math_only(
    articles: list[dict],
    *,
    min_cluster_size: int = 6,     # only trim big clusters (coverage-first)
    min_keep: int = 6,              # never fall below this
    max_remove_cap: int = 8,        # never remove more than this
    max_remove_frac: float = 0.20,  # or more than this fraction
    p10_flag: float = 0.55,         # tail suspicious below this
    std_flag: float = 0.13,         # spread suspicious above this
    p10_improve: float = 0.04,      # meaningful improvement
    std_improve: float = 0.03,      # meaningful improvement
) -> list[dict]:
    """
    Conservative trimming:
      - Only considers trimming large clusters
      - Only trims if cluster is flagged (low p10 or high std)
      - Only accepts trimming if cohesion improves meaningfully
      - Returns articles in original order
    """
    if not isinstance(articles, list) or len(articles) < min_cluster_size:
        return articles

    embedder = get_embedder()
    if embedder is None:
        return articles

    texts = [_article_text(a) for a in articles]
    if sum(1 for x in texts if x) < 2:
        return articles

    try:
        vecs = embedder.encode(texts, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)
    except Exception:
        return articles

    mean0, p10_0, std0, sims0 = _cohesion_stats(vecs)

    # Extra conservatism for small clusters:
    # only trim small clusters if they are severely incoherent.
    if len(articles) < 10 and not (p10_0 < 0.40 and std0 > 0.18):
        return articles

    # Only act if flagged
    if not (p10_0 < p10_flag or std0 > std_flag):
        return articles

    n = len(articles)
    max_remove = int(min(max_remove_cap, np.ceil(n * max_remove_frac)))
    max_remove = min(max_remove, n - min_keep)
    if max_remove <= 0:
        return articles

    # Worst-first indices (lowest similarity to centroid)
    order = np.argsort(sims0)

    best_keep_idx = None
    best_score = None

    # Try removing 1..max_remove tail items
    for k in range(1, max_remove + 1):
        remove_set = set(order[:k].tolist())
        keep_idx = [i for i in range(n) if i not in remove_set]
        if len(keep_idx) < min_keep:
            break

        vecs_k = vecs[keep_idx, :]
        mean_k, p10_k, std_k, _ = _cohesion_stats(vecs_k)

        improved = (p10_k - p10_0 >= p10_improve) or (std0 - std_k >= std_improve)
        if not improved:
            continue

        # Composite score: higher p10 + mean, lower std
        score = (0.75 * p10_k + 0.25 * mean_k) - (0.50 * std_k)

        if best_score is None or score > best_score:
            best_score = score
            best_keep_idx = keep_idx

    if best_keep_idx is None:
        return articles

    keep_set = set(best_keep_idx)
    return [a for i, a in enumerate(articles) if i in keep_set]


# ----------------------------
# Main
# ----------------------------
def main():
    if not Path(INPUT_FILE).exists():
        print(f"❌ Missing {INPUT_FILE}. Run merge_similar_clusters.py first.")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    out = []
    for c in clusters:
        # Preserve full cluster dict, only modify articles
        arts = c.get("articles", []) if isinstance(c, dict) else []
        if not isinstance(arts, list) or len(arts) < 2:
            continue

        trimmed = trim_cluster_tail_math_only(arts)

        c2 = dict(c)
        c2["articles"] = trimmed
        out.append(c2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(out)} clusters → {OUTPUT_FILE} (math-only, no GPT)")

if __name__ == "__main__":
    main()
