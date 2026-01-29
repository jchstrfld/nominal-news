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
from datetime import datetime, timedelta
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

# Optional: local embeddings for tail trimming
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

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

# Cluster-level GPT validation (eventness) — capped + cached

EVENT_MODEL = os.getenv("NN_EVENT_MODEL", "gpt-4o-mini")
EVENT_MAX_CALLS = int(os.getenv("NN_EVENT_MAX_CALLS", "24"))  # hard cap per run
EVENT_CACHE_FILE = os.getenv("NN_EVENT_CACHE_FILE", "eventness_cache.json")

def _load_event_cache() -> dict:
    try:
        with open(EVENT_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_event_cache(cache: dict) -> None:
    try:
        with open(EVENT_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _cluster_sig_urls(cluster: dict) -> str:
    # stable signature: sorted canonical URLs
    urls = sorted(canonicalize_url(a.get("url","")) for a in cluster.get("articles", []) if a.get("url"))
    return "|".join(urls)[:8000]  # safety cap

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

def compute_source_concentration(articles: list[dict]) -> float:
    """
    Fraction of articles coming from the single most-common 'source' label.
    1.0 means one publisher dominates; lower is better.
    """
    srcs = [(a.get("source") or "").strip() for a in articles]
    srcs = [s for s in srcs if s]
    if not srcs:
        return 0.0
    top = Counter(srcs).most_common(1)[0][1]
    return round(top / len(srcs), 3)

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

def _article_date(a: dict) -> str:
    """
    Return YYYY-MM-DD if present, else "".
    """
    d = (a.get("published_date") or "").strip()
    if d and len(d) >= 10:
        return d[:10]
    # fallback if only published_at exists
    pa = (a.get("published_at") or "").strip()
    return pa[:10] if pa and len(pa) >= 10 else ""

def cluster_today_ratio(articles: list[dict], date_str: str) -> float:
    """
    Fraction of dated articles in cluster that fall within date_str ± 1 day.
    This handles RSS timezone drift and late-night publishing.
    Fail-open (1.0) if dates are missing/unparseable.
    """
    if not articles:
        return 0.0

    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
        lo = target - timedelta(days=1)
        hi = target + timedelta(days=1)
    except Exception:
        lo = hi = None

    dated = 0
    in_window = 0

    for a in articles:
        d = _article_date(a)
        if not d:
            continue
        try:
            ad = datetime.strptime(d[:10], "%Y-%m-%d").date()
        except Exception:
            continue

        dated += 1
        if lo and hi and (lo <= ad <= hi):
            in_window += 1
        elif (not lo) and (d[:10] == date_str):
            in_window += 1

    if dated == 0:
        return 1.0

    return in_window / dated

def matter_score(cluster: dict) -> float:
    """
    Ranking for top topics: prioritize size, then cross-outlet diversity,
    and (new) time-density to favor "what's happening today" over evergreen buckets.
    Does not remove any sources; only affects ordering.
    """
    arts = cluster.get("articles", [])
    size = len(arts)

    div = (cluster.get("source_diversity") or {}).get("entropy", 0.0)
    uniq = (cluster.get("source_diversity") or {}).get("unique_domains", 0)

    bias_labels = { (a.get("bias") or "").strip() for a in arts if a.get("bias") }
    bias_uniq = len(bias_labels)

    base = (
        size * 1.0
        + div * 0.8
        + uniq * 0.12
        + bias_uniq * 0.15
    )

    # Time-density: favor clusters that spike on the target date
    tr = float(cluster.get("today_ratio", 1.0) or 1.0)

    src_conc = float(cluster.get("source_concentration", 0.0) or 0.0)

    # "Today-ness" matters most for larger clusters
    base += (tr - 0.5) * min(12.0, size * 0.4)

    # Penalty: evergreen buckets (big but low today_ratio)
    if size >= 12 and tr < 0.25:
        base -= 3.0

    # Publisher concentration penalty (ranking-only)
    if size >= 12 and src_conc >= 0.45:
        base -= 2.5

    # Tie-breaker: downrank clusters GPT labeled as MIXED
    if cluster.get("eventness_label") == "MIXED":
        base -= 1.0

    return base

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

_SEM_EMBEDDER = None

def _get_sem_embedder():
    global _SEM_EMBEDDER
    if _SEM_EMBEDDER is None:
        if SentenceTransformer is None:
            return None
        _SEM_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEM_EMBEDDER

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
    # Token-free semantic vectors for cluster-level near-duplicate detection
    sem = _get_sem_embedder() if "_get_sem_embedder" in globals() else None
    sem_vecs = None
    if sem is not None:
        try:
            sem_vecs = sem.encode(texts, normalize_embeddings=True)
            sem_vecs = np.asarray(sem_vecs, dtype=np.float32)
        except Exception:
            sem_vecs = None

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

            merge_ok = False
            if jacc >= url_overlap:
                merge_ok = True
            else:
                if nlp and ent_sets is not None:
                    shared = bool(ent_sets[i] & ent_sets[j])

                    # Original TF-IDF rule (kept)
                    if (sim[i, j] >= cos_thresh) and shared:
                        merge_ok = True

                    # NEW: semantic rule (token-free, math-only)
                    if (not merge_ok) and sem_vecs is not None and shared:
                        sem_sim = float(sem_vecs[i] @ sem_vecs[j].T)
                        if sem_sim >= 0.70:
                            merge_ok = True
                else:
                    # No NER available: keep conservative TF-IDF only
                    merge_ok = (sim[i, j] >= cos_thresh_no_ner)

            if merge_ok:
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

def validate_cluster_eventness_with_gpt(titles: list[str]) -> tuple[str, str]:
    """
    Returns (label, explanation)
    label ∈ {"SINGLE_EVENT","MIXED","THEMATIC_BUCKET"}
    Uses very small prompt (titles only).
    """
    if openai is None or not os.getenv("OPENAI_API_KEY"):
        return "SINGLE_EVENT", "OpenAI disabled — default keep"

    # Keep it short to control cost
    titles = [t.strip() for t in titles if t and t.strip()][:12]
    if len(titles) < 4:
        return "SINGLE_EVENT", "Too few titles — keep"

    prompt = (
        "You are a news clustering judge.\n"
        "Given these article titles, decide whether they describe:\n"
        "A) ONE specific real-world news event/development (SINGLE_EVENT) — same incident, same outcome, same protagonists.\n"
        "B) multiple different incidents (MIXED) — e.g., multiple separate accidents/crimes in different places.\n"
        "C) a broad theme/roundup/opinion pile (THEMATIC_BUCKET) — 'several things about X' with no single incident.\n\n"
        "Be strict: if there are clearly multiple distinct incidents, label MIXED.\n\n"
        "Return exactly two lines:\n"
        "Label: <SINGLE_EVENT|MIXED|THEMATIC_BUCKET>\n"
        "Why: <one short sentence>\n\n"
        "Titles:\n- " + "\n- ".join(titles)
    )

    # Try preferred model; fall back to gpt-3.5-turbo if needed
    for model in [EVENT_MODEL, "gpt-3.5-turbo"]:
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Be strict and concise."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0
            )
            content = (resp.choices[0].message["content"] or "").strip()
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            lab = ""
            why = ""
            for ln in lines:
                if ln.lower().startswith("label:"):
                    lab = ln.split(":", 1)[1].strip().upper()
                if ln.lower().startswith("why:"):
                    why = ln.split(":", 1)[1].strip()
            if lab not in {"SINGLE_EVENT", "MIXED", "THEMATIC_BUCKET"}:
                # fallback parse: first token on first line
                first = lines[0].upper() if lines else ""
                if "THEMATIC" in first:
                    lab = "THEMATIC_BUCKET"
                elif "MIXED" in first:
                    lab = "MIXED"
                else:
                    lab = "SINGLE_EVENT"
            return lab, why or "OK"
        except Exception:
            continue

    return "SINGLE_EVENT", "OpenAI error — default keep"

# ----------------------------
# Tail trimming
# ----------------------------

_EMBEDDER = None

def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        if SentenceTransformer is None:
            return None
        # Small + fast; same model used in your cohesion report script.
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER

def _trim_text(a: dict) -> str:
    t = (a.get("title") or "").strip()
    d = (a.get("description") or "").strip()
    if d:
        d = re.sub(r"\s+", " ", d)[:400]
    if t and d:
        return f"{t}. {d}"
    return t or d or (a.get("url") or "")

def _cohesion_stats(vecs: np.ndarray) -> tuple[float, float, float]:
    """
    vecs: (n, d) L2-normalized embeddings
    returns: (mean_sim, p10_sim, std_sim) to centroid
    """
    n = int(vecs.shape[0])
    if n < 2:
        return (0.0, 0.0, 0.0)

    centroid = vecs.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid = centroid / np.maximum(denom, 1e-12)

    sims = (vecs @ centroid.T).reshape(-1)
    mean_sim = float(np.mean(sims))
    p10_sim = float(np.percentile(sims, 10))
    std_sim = float(np.std(sims))
    return (mean_sim, p10_sim, std_sim)

def trim_cluster_tail_math_only(
    articles: list[dict],
    *,
    min_cluster_size: int = 10,
    min_keep: int = 6,
    max_remove_cap: int = 8,
    max_remove_frac: float = 0.20,
    p10_flag: float = 0.55,
    std_flag: float = 0.13,
    p10_improve: float = 0.04,
    std_improve: float = 0.03,
) -> list[dict]:
    """
    Removes a small number of lowest-similarity articles from a cluster
    only when:
      - cluster is large enough
      - cluster looks 'tail-contaminated' (low p10 or high std)
      - trimming yields meaningful cohesion improvement
    Always keeps at least min_keep articles.
    Token-free; uses local sentence-transformers embeddings if available.
    """
    if not isinstance(articles, list) or len(articles) < min_cluster_size:
        return articles

    embedder = _get_embedder()
    if embedder is None:
        # If sentence_transformers isn't installed, do nothing (don't break pipeline)
        return articles

    texts = [_trim_text(a) for a in articles]
    # Guard: if too many empty texts, skip
    if sum(1 for x in texts if x.strip()) < 2:
        return articles

    # Embed and normalize
    try:
        vecs = embedder.encode(texts, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)
    except Exception:
        return articles

    mean0, p10_0, std0 = _cohesion_stats(vecs)

    # Only act if flagged (tail looks off)
    if not (p10_0 < p10_flag or std0 > std_flag):
        return articles

    n = len(articles)
    max_remove = int(min(max_remove_cap, np.ceil(n * max_remove_frac)))
    max_remove = min(max_remove, n - min_keep)
    if max_remove <= 0:
        return articles

    # similarity to centroid to identify tail
    centroid = vecs.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid = centroid / np.maximum(denom, 1e-12)
    sims = (vecs @ centroid.T).reshape(-1)

    # indices sorted by similarity ascending (worst first)
    order = np.argsort(sims)

    best_keep_idx = None
    best_score = None

    # Try removing 1..max_remove worst items; pick first k that meaningfully improves
    # and has the best resulting (p10, std) tradeoff.
    for k in range(1, max_remove + 1):
        remove_set = set(order[:k].tolist())
        keep_idx = [i for i in range(n) if i not in remove_set]
        if len(keep_idx) < min_keep:
            break

        vecs_k = vecs[keep_idx, :]
        mean_k, p10_k, std_k = _cohesion_stats(vecs_k)

        improved = (p10_k - p10_0 >= p10_improve) or (std0 - std_k >= std_improve)
        if not improved:
            continue

        # score: favor higher p10 and lower std
        score = (0.75 * p10_k + 0.25 * mean_k) - (0.50 * std_k)

        if best_score is None or score > best_score:
            best_score = score
            best_keep_idx = keep_idx

    if best_keep_idx is None:
        return articles

    # Return trimmed articles in original order
    keep_set = set(best_keep_idx)
    return [a for i, a in enumerate(articles) if i in keep_set]

def _should_drop_cluster_math_only(articles: list[dict]) -> bool:
    """
    Conservative: drop only small clusters that are severely incoherent.
    Token-free local embeddings. No keyword lists.
    """
    if not articles or len(articles) < 4:
        return True  # too small to be meaningful coverage
    if len(articles) > 12:
        return False  # never auto-drop large clusters

    sem = _get_sem_embedder() if "_get_sem_embedder" in globals() else None
    if sem is None:
        return False

    texts = []
    for a in articles[:18]:
        t = (a.get("title") or "").strip()
        d = (a.get("description") or "").strip()
        if d:
            d = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", d))[:300]
        txt = (t + ". " + d).strip() if t else d
        if txt:
            texts.append(txt)

    if len(texts) < 4:
        return False

    try:
        vecs = sem.encode(texts, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)
    except Exception:
        return False

    # cohesion to centroid
    centroid = vecs.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid = centroid / np.maximum(denom, 1e-12)
    sims = (vecs @ centroid.T).reshape(-1)

    p10 = float(np.percentile(sims, 10))
    std = float(np.std(sims))

    # Severe incoherence gate (tuned to only catch the worst tails you showed)
    return (p10 < 0.40 and std > 0.18)

def cluster_cohesion_fast(articles: list[dict]) -> tuple[float, float]:
    """
    Returns (p10_sim, std_sim) for a cluster using token-free local embeddings.
    p10 is tail similarity to centroid; std is dispersion. Higher p10, lower std is better.
    """
    sem = _get_sem_embedder() if "_get_sem_embedder" in globals() else None
    if sem is None or not articles:
        return (1.0, 0.0)

    texts = []
    for a in articles[:18]:
        t = (a.get("title") or "").strip()
        d = (a.get("description") or "").strip()
        if d:
            d = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", d))[:250]
        txt = (t + ". " + d).strip() if t else d
        if txt:
            texts.append(txt)

    if len(texts) < 4:
        return (1.0, 0.0)

    try:
        vecs = sem.encode(texts, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)
        centroid = vecs.mean(axis=0, keepdims=True)
        denom = np.linalg.norm(centroid, axis=1, keepdims=True)
        centroid = centroid / np.maximum(denom, 1e-12)
        sims = (vecs @ centroid.T).reshape(-1)

        p10 = float(np.percentile(sims, 10))
        std = float(np.std(sims))
        return (p10, std)
    except Exception:
        return (1.0, 0.0)

def cluster_nn_tightness(articles: list[dict]) -> tuple[float, float]:
    """
    Token-free. Returns (nn_mean, nn_p10) where each article contributes its max cosine
    similarity to any other article in the cluster (nearest neighbor tightness).
    """
    sem = _get_sem_embedder() if "_get_sem_embedder" in globals() else None
    if sem is None or not articles:
        return (1.0, 1.0)

    texts = []
    for a in articles[:18]:
        t = (a.get("title") or "").strip()
        d = (a.get("description") or "").strip()
        if d:
            d = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", d))[:200]
        txt = (t + ". " + d).strip() if t else d
        if txt:
            texts.append(txt)

    if len(texts) < 4:
        return (1.0, 1.0)

    try:
        X = sem.encode(texts, normalize_embeddings=True)
        X = np.asarray(X, dtype=np.float32)
    except Exception:
        return (1.0, 1.0)

    S = X @ X.T
    np.fill_diagonal(S, -1.0)
    nn = S.max(axis=1)
    return (float(nn.mean()), float(np.percentile(nn, 10)))

def _cluster_is_thematic_multi_lump(articles: list[dict]) -> bool:
    """
    Math-only detection of 'thematic buckets':
    If a k=2 split significantly improves cohesion, cluster likely contains multiple topics.
    Token-free local embeddings (MiniLM). No keyword/domain lists.
    """
    sem = _get_sem_embedder() if "_get_sem_embedder" in globals() else None
    if sem is None or not articles or len(articles) < 12:
        return False  # only apply to bigger clusters where this becomes a problem

    texts = []
    for a in articles[:18]:
        t = (a.get("title") or "").strip()
        d = (a.get("description") or "").strip()
        if d:
            d = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", d))[:250]
        txt = (t + ". " + d).strip() if t else d
        if txt:
            texts.append(txt)

    if len(texts) < 10:
        return False

    try:
        X = sem.encode(texts, normalize_embeddings=True)
        X = np.asarray(X, dtype=np.float32)
    except Exception:
        return False

    # Baseline cohesion (p10, std)
    centroid = X.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid = centroid / np.maximum(denom, 1e-12)
    sims = (X @ centroid.T).reshape(-1)
    p10_before = float(np.percentile(sims, 10))
    std_before = float(np.std(sims))

    # Farthest-pair init for k=2 (approx, fast)
    # pick farthest from a random point, then farthest from that
    a0 = 0
    d1 = (X @ X[a0:a0+1].T).reshape(-1)
    a = int(np.argmin(d1))
    d2 = (X @ X[a:a+1].T).reshape(-1)
    b = int(np.argmin(d2))
    c1 = X[a].copy()
    c2 = X[b].copy()

    # 5 iterations of 2-means refinement (cosine space)
    for _ in range(5):
        s1 = (X @ c1.reshape(-1, 1)).reshape(-1)
        s2 = (X @ c2.reshape(-1, 1)).reshape(-1)
        lab = (s2 > s1).astype(np.int32)
        if lab.sum() == 0 or lab.sum() == len(lab):
            break
        c1 = X[lab == 0].mean(axis=0)
        c2 = X[lab == 1].mean(axis=0)
        c1 = c1 / max(np.linalg.norm(c1), 1e-12)
        c2 = c2 / max(np.linalg.norm(c2), 1e-12)

    # Cohesion after split: weighted p10/std
    def stats(subX):
        if len(subX) < 4:
            return (0.0, 0.0)
        cent = subX.mean(axis=0, keepdims=True)
        cent = cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)
        ss = (subX @ cent.T).reshape(-1)
        return (float(np.percentile(ss, 10)), float(np.std(ss)))

    X1 = X[lab == 0]
    X2 = X[lab == 1]
    if len(X1) < 4 or len(X2) < 4:
        return False

    p10_1, std_1 = stats(X1)
    p10_2, std_2 = stats(X2)

    p10_after = (len(X1) * p10_1 + len(X2) * p10_2) / (len(X1) + len(X2))
    std_after = (len(X1) * std_1 + len(X2) * std_2) / (len(X1) + len(X2))

    # If splitting improves tail cohesion a lot, it's a multi-topic thematic bucket.
    return (p10_after - p10_before) >= 0.10 and (std_before - std_after) >= 0.03

def cluster_entity_cohesion(nlp, titles: list[str]) -> float:
    """
    Token-free event-specificity proxy.
    Returns the fraction of title-pairs that share at least one named entity.
    Higher = more likely to be a single real-world story.
    """
    if not nlp or not titles or len(titles) < 4:
        return 1.0  # fail-open when NER unavailable or too small

    ents = []
    for t in titles[:18]:
        s = (t or "").strip()
        if not s:
            continue
        ents.append(_entities(nlp, s))

    if len(ents) < 4:
        return 1.0

    shared_pairs = 0
    total_pairs = 0
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            total_pairs += 1
            if ents[i] and ents[j] and (ents[i] & ents[j]):
                shared_pairs += 1

    if total_pairs == 0:
        return 1.0
    return shared_pairs / total_pairs

def merge_into_dominant_clusters(clusters: list[dict], date_str: str) -> list[dict]:
    """
    Directed merge: merge small, time-dense clusters into the top few dominant event clusters
    when they are highly similar (token-free, MiniLM over titles-only).
    """
    sem = _get_sem_embedder()
    if sem is None or len(clusters) < 3:
        return clusters

    texts = [_cluster_text(c) for c in clusters]
    try:
        V = sem.encode(texts, normalize_embeddings=True)
        V = np.asarray(V, dtype=np.float32)
    except Exception:
        return clusters

    ranked_idx = sorted(range(len(clusters)), key=lambda i: matter_score(clusters[i]), reverse=True)
    dom_idx = ranked_idx[:3]

    # Merge helper
    def merge_articles(into: dict, src: dict):
        by_url = {}
        for art in into.get("articles", []):
            u = canonicalize_url(art.get("url",""))
            if u:
                by_url[u] = art
        for art in src.get("articles", []):
            u = canonicalize_url(art.get("url",""))
            if u:
                by_url[u] = art
        into["articles"] = list(by_url.values())

        # refresh metrics
        into["source_diversity"] = source_diversity(into["articles"])
        into["source_concentration"] = compute_source_concentration(into["articles"])
        into["bias_distribution"] = aggregate_bias_distribution(into["articles"])
        into["today_ratio"] = round(cluster_today_ratio(into["articles"], date_str), 3)

    absorbed = set()

    for j in ranked_idx:
        if j in dom_idx or j in absorbed:
            continue

        cj = clusters[j]
        sj = len(cj.get("articles", []))
        trj = float(cj.get("today_ratio", 1.0) or 1.0)

        if sj > 25 or trj < 0.60:
            continue

        best_i = None
        best_sim = -1.0
        for di in dom_idx:
            ci = clusters[di]
            tri = float(ci.get("today_ratio", 1.0) or 1.0)
            if tri < 0.60:
                continue
            sim = float(V[di] @ V[j].T)
            if sim > best_sim:
                best_sim = sim
                best_i = di

        if best_i is not None and best_sim >= 0.86:
            merge_articles(clusters[best_i], cj)
            absorbed.add(j)

    return [clusters[i] for i in range(len(clusters)) if i not in absorbed]

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

        # Coverage-first: do NOT hard-drop articles based on NER overlap.
        # NER overlap is useful for diagnostics, but it's too aggressive as a filter and
        # can collapse large, valid clusters into tiny ones.
        titles_kept, arts_kept = titles, arts

        # Tail trimming to reduce "one story + a few off-topic articles" contamination.
        # Only acts on larger clusters and only if cohesion improves.
        arts_kept = trim_cluster_tail_math_only(arts_kept)

        # Refresh titles after trimming (keeps downstream logic consistent)
        titles_kept = [a.get("title","") for a in arts_kept if a.get("title")]
        if len(titles_kept) < 2:
            continue

        # Drop only small clusters that are severely incoherent (math-only).
        if _should_drop_cluster_math_only(arts_kept):
            continue

        # Do NOT run the early GPT filter — it shrinks the candidate pool too aggressively.
        # Eventness is handled later with a capped/cached GPT pass.
        accept = True

        # Attach diversity & bias
        diversity = source_diversity(arts_kept)
        bias_dist = cluster.get("bias_distribution") or aggregate_bias_distribution(arts_kept)

        final_candidates.append({
            "topic": cluster.get("topic", cluster.get("topic_title", "Merged Topic")),
            "articles": arts_kept,
            "source_diversity": diversity,
            "source_concentration": compute_source_concentration(arts_kept),
            "bias_distribution": bias_dist
        })

    # Attach time-density to each cluster (used for ranking and bucket control)
    for c in final_candidates:
        c["today_ratio"] = round(cluster_today_ratio(c.get("articles", []), date_str), 3)

    # Semantic near-duplicate merge (token-free), run until stable (max 2 passes)
    deduped = final_candidates
    for _ in range(2):
        merged = dedupe_topics(deduped, url_overlap=0.50, cos_thresh=0.78, nlp=nlp)
        if len(merged) == len(deduped):
            break
        deduped = merged

    deduped = merge_into_dominant_clusters(deduped, date_str)

    # Final, low-cost GPT pass: only on suspicious big buckets
    event_cache = _load_event_cache()
    event_calls = 0

    # Pre-rank to decide which ones are worth validating (top 15 by current score)
    pre_ranked = sorted(deduped, key=matter_score, reverse=True)
    kept = []

    for rank_pos, c in enumerate(pre_ranked):
        FORCE_POOL = max(top_k + 30, 60)
        force_review = rank_pos < FORCE_POOL

        arts = c.get("articles", [])
        titles = [a.get("title","") for a in arts if a.get("title")]

        # Only consider GPT for larger clusters that still look "bucket-ish"
        # (math-based signals; no lists)
        p10, std = cluster_cohesion_fast(arts)
        ent_coh = cluster_entity_cohesion(nlp, titles) if "cluster_entity_cohesion" in globals() else 1.0

        tr = float(c.get("today_ratio", 1.0) or 1.0)

        suspicious = (
            (len(arts) >= 12 and (
                (ent_coh < 0.10) or
                (p10 < 0.42) or
                (p10 < 0.50 and std > 0.14) or
                (len(arts) >= 25 and ent_coh < 0.16) or
                (len(arts) >= 15 and tr < 0.25)
            ))
        )

        if (force_review or suspicious):
            sig = _cluster_sig_urls(c)
            cache_key = f"{EVENT_MODEL}::{sig}"
            cached = event_cache.get(cache_key)

            if cached:
                lab = cached.get("label", "SINGLE_EVENT")
            else:
                if event_calls >= EVENT_MAX_CALLS:
                    # If we can't afford to vet a forced candidate, we should not allow it to pass.
                    if force_review:
                        continue
                    # Non-forced and no budget: let it pass through unlabelled
                    kept.append(c)
                    continue

                lab, why = validate_cluster_eventness_with_gpt(titles)
                event_cache[cache_key] = {"label": lab, "why": why}
                event_calls += 1

            c["eventness_label"] = lab

            if lab in {"THEMATIC_BUCKET", "MIXED"}:
                continue

        kept.append(c)

    # Save cache if we made any calls
    if event_calls > 0:
        _save_event_cache(event_cache)

    deduped = kept

    # Rank
    ranked = sorted(deduped, key=matter_score, reverse=True)

    # Ensure top candidates are event-vetted when OpenAI is enabled
    if not no_openai and os.getenv("OPENAI_API_KEY"):
        ranked = [c for c in ranked if c.get("eventness_label") == "SINGLE_EVENT"]

    # Cap top-K AFTER filtering
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
