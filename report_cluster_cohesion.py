# report_cluster_cohesion.py
# Diagnostic-only cohesion report (math-only, token-free).
# Reads grouped_articles_filtered_{date}.json and prints a ranked report of least-coherent clusters.

import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    p.add_argument("--top", type=int, default=10, help="How many worst clusters to print (default 10)")
    p.add_argument("--min-articles", type=int, default=4, help="Skip clusters smaller than this (default 4)")
    p.add_argument("--max-per-cluster", type=int, default=18, help="Max articles sampled per cluster (default 18)")
    p.add_argument("--input", type=str, default="", help="Override input file path (optional)")
    p.add_argument("--json-out", type=str, default="", help="Optional: write report JSON to this path")
    return p.parse_args()


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _article_text(a: dict) -> str:
    # Keep it stable + token-free: title + short description
    t = _clean(a.get("title") or "")
    d = _clean(a.get("description") or "")
    if d:
        d = d[:400]
    if t and d:
        return f"{t}. {d}"
    return t or d or _clean(a.get("url") or "")


def _cluster_vecs(model: SentenceTransformer, articles: list[dict], max_per_cluster: int) -> np.ndarray:
    texts = []
    for a in articles[:max_per_cluster]:
        tx = _article_text(a)
        if tx:
            texts.append(tx)

    if len(texts) < 2:
        return np.zeros((0, 384), dtype=np.float32)

    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def _cohesion_stats(vecs: np.ndarray) -> dict:
    # vecs: (n, d) normalized
    n = int(vecs.shape[0])
    if n < 2:
        return {"n": n, "mean_sim": None, "p10_sim": None, "std_sim": None}

    centroid = vecs.mean(axis=0, keepdims=True)
    # normalize centroid
    denom = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid = centroid / np.maximum(denom, 1e-12)

    sims = (vecs @ centroid.T).reshape(-1)
    mean_sim = float(np.mean(sims))
    p10_sim = float(np.percentile(sims, 10))
    std_sim = float(np.std(sims))

    return {"n": n, "mean_sim": mean_sim, "p10_sim": p10_sim, "std_sim": std_sim}


def _score_for_sort(stats: dict) -> float:
    """
    Lower score = worse cohesion.
    Heavily penalize low p10 (tail incoherence), then std.
    """
    if stats["n"] is None or stats["mean_sim"] is None:
        return 999.0
    return (stats["p10_sim"] * 0.75 + stats["mean_sim"] * 0.25) - (stats["std_sim"] * 0.50)


def main():
    args = _parse_args()
    date_str = args.date or datetime.today().strftime("%Y-%m-%d")

    input_file = args.input.strip() or f"grouped_articles_filtered_{date_str}.json"
    if not os.path.exists(input_file):
        print(f"❌ Missing {input_file}. Run upstream steps first.")
        raise SystemExit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        grouped = json.load(f)

    clusters = grouped["clusters"] if isinstance(grouped, dict) and "clusters" in grouped else grouped
    if not isinstance(clusters, list):
        print("❌ Unexpected JSON shape: expected a list of clusters or {'clusters': [...]} ")
        raise SystemExit(1)

    # Token-free local embeddings (fast)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    report_rows = []
    for idx, c in enumerate(clusters):
        arts = c.get("articles", []) if isinstance(c, dict) else []
        if len(arts) < args.min_articles:
            continue

        vecs = _cluster_vecs(model, arts, args.max_per_cluster)
        stats = _cohesion_stats(vecs)

        # A short “label” for printing: first article title
        label = ""
        if arts and isinstance(arts[0], dict):
            label = _clean(arts[0].get("title") or "")
        label = label[:120]

        report_rows.append({
            "cluster_index": idx,
            "articles_in_cluster": len(arts),
            "sampled_articles": stats["n"],
            "mean_sim": None if stats["mean_sim"] is None else round(stats["mean_sim"], 3),
            "p10_sim": None if stats["p10_sim"] is None else round(stats["p10_sim"], 3),
            "std_sim": None if stats["std_sim"] is None else round(stats["std_sim"], 3),
            "sort_score": round(_score_for_sort(stats), 3),
            "label": label,
        })

    if not report_rows:
        print("ℹ️ No clusters met minimum size for cohesion reporting.")
        return

    # Worst-first
    report_rows.sort(key=lambda r: r["sort_score"])

    # Print concise report
    print("\n🧪 Cohesion diagnostic (worst clusters first)")
    print(f"   Input: {input_file}")
    print(f"   Showing worst {min(args.top, len(report_rows))} clusters (min_articles={args.min_articles}, max_per_cluster={args.max_per_cluster})\n")

    for r in report_rows[: args.top]:
        print(
            f"- idx={r['cluster_index']:>3}  size={r['articles_in_cluster']:>2}  "
            f"p10={r['p10_sim']}  mean={r['mean_sim']}  std={r['std_sim']}  score={r['sort_score']}  "
            f"| {r['label']}"
        )

    # Optional JSON output
    if args.json_out.strip():
        out_path = args.json_out.strip()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "date": date_str,
                    "input_file": input_file,
                    "min_articles": args.min_articles,
                    "max_per_cluster": args.max_per_cluster,
                    "rows": report_rows,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\n✅ Wrote cohesion report JSON → {out_path}")


if __name__ == "__main__":
    main()
