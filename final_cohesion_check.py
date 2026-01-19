# final_cohesion_check.py — high-precision, token-efficient final pass
# - Loads grouped_articles_filtered_{date}.json
# - Runs exact + semantic de-duplication (token-free)
# - Optionally validates clusters with GPT (can be skipped via --no-openai)
# - Adds source diversity & bias distribution
# - Caps to top-K clusters
# - Writes grouped_articles_final_{date}.json
#
# Usage examples:
#   python final_cohesion_check.py --date 2025-09-24 --no-openai --top-k 10 --print-report
#   python final_cohesion_check.py --date 2025-09-24
#
# Notes:
# - No OpenAI tokens are used if you pass --no-openai (or no key is present).
# - To see changes on your webpage (index.html) you typically need to re-run summarization,
#   because the page reads topic_summaries_{date}.json. See run notes at the end of file.

from __future__ import annotations

import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
from math import log2
from urllib.parse import urlparse

# Optional deps
try:
    import spacy
except Exception:
    spacy = None

try:
    import openai
except Exception:
    openai = None

from dotenv import load_dotenv

# Lightweight ML
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# ----------------------------
# CLI args
# ----------------------------
def _parse_args():
    args = sys.argv[1:]
    date_str = None
    no_openai = False
    top_k = 10
    print_report = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--date" and i + 1 < len(args):
            date_str = args[i + 1]
            i += 2
        elif a == "--no-openai":
            no_openai = True
            i += 1
        elif a == "--top-k" and i + 1 < len(args):
            try:
                top_k = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif a == "--print-report":
            print_report = True
            i += 1
        else:
            i += 1

    if not date_str:
        date_str = datetime.today().strftime("%Y-%m-%d")

    return date_str, no_openai, top_k, print_report


# ----------------------------
# Utilities
# ----------------------------
_TRACKING_PARAMS = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "gclid","fbclid","mc_cid","mc_eid","igshid","si","s","ref","ref_src"
}

def canonicalize_url(u: str) -> str:
    """
    Make URLs comparable by removing scheme, www, trailing slash,
    and common tracking query params. Keeps domain + path + non-tracking query keys.
    """
    try:
        pu = urlparse(u or "")
        netloc = pu.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = re.sub(r"/+$", "", pu.path or "")
        kept = []
        if pu.query:
            for kv in pu.query.split("&"):
                if not kv:
                    continue
                k = kv.split("=", 1)[0].lower()
                if k not in _TRACKING_PARAMS:
                    kept.append(kv)
        q = "&".join(sorted(kept))  # stable signature
        return f"{netloc}{path}?{q}" if q else f"{netloc}{path}"
    except Exception:
        return (u or "").strip().lower()


def norm_url_hostpath(u: str) -> str:
    """Simpler normalization for set membership (host + path only)."""
    try:
        p = urlparse(u or "")
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return (host + (p.path or "")).rstrip("/")
    except Exception:
        return (u or "").lower().strip()


def domain_from_url(u: str) -> str:
    try:
        host = urlparse(u or "").netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def source_diversity(articles: list[dict]) -> dict:
    """
    Diversity metrics across source domains for a cluster.
    """
    domains = [domain_from_url(a.get("url", "")) for a in articles if a.get("url")]
    domains = [d for d in domains if d]
    total = len(domains)
    uniq = len(set(domains))
    if total == 0:
        return {"unique_domains": 0, "total_with_domain": 0, "top_domain": ["", 0.0], "entropy": 0.0}
    counts = Counter(domains)
    top_dom, top_cnt = counts.most_common(1)[0]
    probs = [c / total for c in counts.values()]
    H = -sum(p * log2(p) for p in probs)
    return {
        "unique_domains": uniq,
        "total_with_domain": total,
        "top_domain": [top_dom, round(top_cnt / total, 3)],
        "entropy": round(H, 3),
    }


BIAS_ORDER = ["Far Left", "Left", "Center", "Right", "Far Right", "Unknown"]

def aggregate_bias_distribution(articles: list[dict]) -> dict:
    """
    Build integer percentages over canonical labels that sum to 100 (largest-remainder rounding).
    """
    raw_counts = Counter()
    for a in articles:
        label = (a.get("bias") or "Center").strip()
        # normalize variants
        label = label.title().replace("-", " ")
        if label not in BIAS_ORDER:
            label = "Unknown" if label.lower() == "unknown" else ("Center" if label not in BIAS_ORDER else label)
        raw_counts[label] += 1

    total = sum(raw_counts.values())
    if total == 0:
        return {}

    exact = {k: (raw_counts.get(k, 0) * 100.0 / total) for k in BIAS_ORDER}
    floored = {k: int(exact[k]) for k in BIAS_ORDER}
    remainders = {k: exact[k] - floored[k] for k in BIAS_ORDER}
    leftover = 100 - sum(floored.values())
    for k, _ in sorted(remainders.items(), key=lambda x: x[1], reverse=True):
        if leftover <= 0:
            break
        floored[k] += 1
        leftover -= 1
    # drop zeros for a cleaner chart
    return {k: v for k, v in floored.items() if v > 0}


def matter_score(cluster: dict) -> float:
    """
    Simple ranking to keep the best topics: larger clusters, more diverse, balanced bias.
    """
    arts = cluster.get("articles", [])
    size = len(arts)
    div = (cluster.get("source_diversity") or {}).get("entropy", 0.0)
    # count non-empty bias labels
    bias_labels = { (a.get("bias") or "").strip() for a in arts if a.get("bias") }
    bias_uniq = len(bias_labels)
    # weights chosen to prioritize size, then diversity, then some bias spread
    return size * 1.0 + div * 0.6 + bias_uniq * 0.2


# ----------------------------
# Exact duplicate collapse
# ----------------------------
def collapse_identical_clusters(clusters: list[dict]) -> list[dict]:
    """
    Keep one cluster per unique set of canonical URLs.
    Prefer more articles; tie-break with higher source diversity entropy.
    """
    def sig(c: dict) -> frozenset[str]:
        return frozenset(canonicalize_url(a.get("url","")) for a in c.get("articles", []) if a.get("url"))

    def score(c: dict) -> tuple[float, float]:
        size = len(c.get("articles", []))
        ent = (c.get("source_diversity") or {}).get("entropy", 0.0)
        return (size, ent)

    best_by_sig = {}
    for c in clusters:
        s = sig(c)
        prev = best_by_sig.get(s)
        if prev is None or score(c) > score(prev):
            best_by_sig[s] = c

    return list(best_by_sig.values())

# ----------------------------
# Semantic near-duplicate merge (token-free)
# ----------------------------

def _cluster_text(c: dict) -> str:
    # Titles-only on purpose: descriptions frequently contain boilerplate that causes false TF-IDF matches.
    parts = []
    for a in c.get("articles", []):
        t = (a.get("title") or "").strip()
        if t:
            parts.append(t)
    if not parts:
        parts = [a.get("source","") for a in c.get("articles", []) if a.get("source")]
    return " ".join(parts)

def dedupe_topics(
    clusters: list[dict],
    url_overlap: float = 0.50,
    cos_thresh: float = 0.78,
    nlp=None,
) -> list[dict]:
    """
    Merge near-identical topics using:
      - URL Jaccard (host+path) >= url_overlap OR
      - (Cosine(TF-IDF over TITLES) >= threshold AND entity overlap if spaCy is available)

    This prevents late-stage false merges that create mixed-topic clusters.
    """
    n = len(clusters)
    if n <= 1:
        return clusters[:]

    texts = [_cluster_text(c) for c in clusters]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)
    XX = X @ X.T
    sim = XX.toarray() if hasattr(XX, "toarray") else np.asarray(XX)

    url_sets = []
    for c in clusters:
        urls = set()
        for a in c.get("articles", []):
            u = a.get("url_normalized") or a.get("url")
            key = norm_url_hostpath(u or "")
            if key:
                urls.add(key)
        url_sets.append(urls)

    # Precompute entity sets per cluster (titles-only) if spaCy is available
    ent_sets = None
    if nlp:
        ent_sets = []
        for c in clusters:
            t = _cluster_text(c)
            ent_sets.append(_entities(nlp, t))

    def domain_entropy(c: dict) -> float:
        doms = []
        for a in c.get("articles", []):
            u = a.get("url") or a.get("url_normalized") or ""
            d = urlparse(u).netloc.lower() if u else ""
            if d.startswith("www."):
                d = d[4:]
            if d:
                doms.append(d)
        if not doms:
            return 0.0
        cnt = Counter(doms)
        total = sum(cnt.values())
        ent = 0.0
        for v in cnt.values():
            p = v / total
            ent -= p * np.log2(p)
        return float(ent)

    keep = [True] * n

    def winner(i, j):
        ni, nj = len(clusters[i].get("articles", [])), len(clusters[j].get("articles", []))
        if ni != nj:
            return i if ni > nj else j
        ei, ej = domain_entropy(clusters[i]), domain_entropy(clusters[j])
        if abs(ei - ej) > 1e-6:
            return i if ei > ej else j
        return i if i < j else j

    # If no NER available, require a much higher cosine for TF-IDF merges
    cos_thresh_no_ner = max(cos_thresh, 0.88)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue

            a, b = url_sets[i], url_sets[j]
            jacc = (len(a & b) / len(a | b)) if (a or b) else 0.0

            tfidf_ok = False
            if jacc >= url_overlap:
                tfidf_ok = True
            else:
                if nlp and ent_sets is not None:
                    # require at least one shared entity token between clusters
                    shared = bool(ent_sets[i] & ent_sets[j])
                    tfidf_ok = (sim[i, j] >= cos_thresh) and shared
                else:
                    tfidf_ok = (sim[i, j] >= cos_thresh_no_ner)

            if tfidf_ok:
                w = winner(i, j)
                l = j if w == i else i

                # merge articles by canonical key
                by_key = {}
                for art in clusters[w].get("articles", []):
                    key = (art.get("url_normalized") or art.get("url") or "").split("?")[0].rstrip("/") \
                        or (art.get("title", "") + art.get("source", "")).lower()
                    if key:
                        by_key[key] = art
                for art in clusters[l].get("articles", []):
                    key = (art.get("url_normalized") or art.get("url") or "").split("?")[0].rstrip("/") \
                        or (art.get("title", "") + art.get("source", "")).lower()
                    if key:
                        by_key[key] = art

                clusters[w]["articles"] = list(by_key.values())
                keep[l] = False

    return [clusters[i] for i in range(n) if keep[i]]

# ----------------------------
# Heuristics + optional GPT validation
# ----------------------------

def _load_spacy():
    if spacy is None:
        return None
    try:
        # minimal pipeline
        return spacy.load("en_core_web_sm", disable=["tagger","parser","lemmatizer","textcat"])
    except Exception:
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            return None


def _entities(nlp, text: str) -> set[str]:
    if not nlp or not text:
        return set()
    doc = nlp(text)
    return { (ent.text or "").strip().lower() for ent in doc.ents if (ent.text or "").strip() }


def validate_cluster_with_gpt(titles: list[str]) -> tuple[bool, str]:
    """
    Returns (is_specific_event, explanation).
    Skips if no API/key; caller should gate on --no-openai or missing key.
    """
    if openai is None or not os.getenv("OPENAI_API_KEY"):
        return True, "OpenAI disabled — accepting by heuristics"

    prompt = f"""
You are a news cluster validator. Below are article titles that may describe related events.

Your task is to determine whether these titles refer to the same specific real-world story.

If they clearly describe one story, say: Type: Specific Event
If they describe multiple unrelated or only loosely related stories, say: Type: Mixed
If they are just thematically similar (e.g. multiple protests or crimes), say: Type: Thematic Similarity

Then explain your reasoning briefly.

Titles:
{chr(10).join(titles)}
""".strip()

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # uses very few tokens for short titles
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":prompt}
            ],
            temperature=0
        )
        content = resp.choices[0].message["content"].strip()
        first = content.splitlines()[0].strip().lower()
        is_specific = first.startswith("type: specific")
        explanation = " ".join(content.splitlines()[1:]).strip()
        return is_specific, explanation
    except Exception as e:
        return True, f"OpenAI error skipped: {e}"


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    date_str, no_openai, top_k, print_report = _parse_args()

    # Env & OpenAI
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    if openai is not None:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    input_file = f"grouped_articles_filtered_{date_str}.json"
    output_file = f"grouped_articles_final_{date_str}.json"

    if not Path(input_file).exists():
        print(f"❌ Missing {input_file}. Run the upstream grouping step first.")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        grouped = json.load(f)

    # grouped may be a list of clusters or an object with {"clusters":[...]} — normalize
    if isinstance(grouped, dict) and "clusters" in grouped:
        clusters_in = grouped["clusters"]
    else:
        clusters_in = grouped

    # 1) light cleanup: drop empty articles and compute diversity upfront
    cleaned = []
    for c in clusters_in:
        arts = [a for a in c.get("articles", []) if a.get("title") or a.get("url")]
        if len(arts) < 2:
            continue
        c2 = dict(c)
        c2["articles"] = arts
        c2["source_diversity"] = source_diversity(arts)
        cleaned.append(c2)

    # 2) collapse exact duplicates by canonical URL sets
    collapsed = collapse_identical_clusters(cleaned)

    # 3) heuristics + optional GPT validation
    nlp = _load_spacy()
    final_candidates = []

    for i, cluster in enumerate(collapsed, start=1):
        arts = cluster.get("articles", [])
        # basic heuristic: require at least 2 articles with some shared named entities across titles
        titles = [a.get("title","") for a in arts if a.get("title")]
        if len(titles) < 2:
            continue

        # Named-entity overlap count
        if nlp:
            doc_ents = [ _entities(nlp, t) for t in titles ]
            overlap_counts = []
            for idx, ents in enumerate(doc_ents):
                overlap = sum(1 for j, other in enumerate(doc_ents) if j != idx and (ents & other))
                overlap_counts.append(overlap)
            # keep titles that overlap with at least one other
            kept_idx = [k for k, v in enumerate(overlap_counts) if v >= 1]
            if len(kept_idx) >= 2:
                titles_kept = [titles[k] for k in kept_idx]
                arts_kept = [arts[k] for k in kept_idx]
            else:
                # fall back to all titles if NER too sparse
                titles_kept, arts_kept = titles, arts
        else:
            titles_kept, arts_kept = titles, arts

        if len(arts_kept) < 2:
            continue

        # Optional GPT validation
        accept = True
        explanation = "Accepted by heuristics (no-openai)."
        if not no_openai and os.getenv("OPENAI_API_KEY"):
            is_specific, explanation = validate_cluster_with_gpt(titles_kept)
            accept = is_specific

        if not accept:
            continue

        # Attach diversity & bias
        diversity = source_diversity(arts_kept)
        bias_dist = cluster.get("bias_distribution") or aggregate_bias_distribution(arts_kept)

        final_candidates.append({
            "topic": cluster.get("topic", cluster.get("topic_title", "Merged Topic")),
            "articles": arts_kept,
            "source_diversity": diversity,
            "bias_distribution": bias_dist
        })

    # Semantic near-duplicate merge (token-free)
    deduped = dedupe_topics(final_candidates, url_overlap=0.50, cos_thresh=0.78, nlp=nlp)

    # Rank and cap top-K
    ranked = sorted(deduped, key=matter_score, reverse=True)
    capped = ranked[: max(1, top_k)]

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(capped, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(capped)} high-precision clusters to {output_file}")

    # Optional console report (token-free)
    if print_report:
        print("\n=== Final clusters (console report) ===")
        for idx, c in enumerate(capped, start=1):
            arts = c.get("articles", [])
            print(f"\n[{idx}] size={len(arts)} entropy={(c.get('source_diversity') or {}).get('entropy',0)}")
            for a in arts[:5]:
                print("   -", (a.get("title") or a.get("url") or "").strip())
        print("\n(Only first 5 article titles per cluster shown.)")


if __name__ == "__main__":
    main()
