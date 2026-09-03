# expand_cluster_coverage_gdelt.py
# Phase 1: Expand VERIFIED clusters with additional coverage from GDELT
# Runs AFTER final_cohesion_check.py and BEFORE summarize_grouped_topics.py
# Free / rate-limited only. No OpenAI calls.

import argparse
import json
import os
import time
import requests
import re
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from bias_labeler import lookup_bias_by_domain, canonicalize
from sentence_transformers import SentenceTransformer
UNMAPPED_BIAS_FILE = "unmapped_bias.json"


# ----------------------------
# Config (safe defaults)
# ----------------------------
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

# Local recovery: free, deterministic, and runs before GDELT.
LOCAL_CORPUS_PREFIX = "articles_raw_normalized_"
MIN_LOCAL_CORE_SIM = 0.53
MIN_LOCAL_PEER_SIM = 0.50
MIN_LOCAL_PEER_SUPPORT = 1
MIN_LOCAL_BEST_MARGIN = 0.05
MIN_EVENT_PEG_TERM_SUPPORT = 2
MIN_EVENT_PEG_SHARED_TERMS = 2
MAX_LOCAL_ADDITIONS_PER_CLUSTER = 40

# GDELT: optional best-effort global discovery.
MAX_QUERIES_PER_RUN = 12
REQUEST_SLEEP_SECONDS = 8.0
MAX_RECORDS_PER_QUERY = 75
MAX_ARTICLES_PER_CLUSTER = 120
MAX_PER_DOMAIN = 3
MAX_QUERY_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = (15.0, 45.0)
RATE_LIMIT_COOLDOWN_SECONDS = 90.0
TARGET_GDELT_NEW_ARTICLES_PER_CLUSTER = 8

SYNDICATION_TITLE_SIM = 0.92
MAX_GDELT_CONTENT_FAMILIES_PER_CLUSTER = 20

# Local semantic validation only — no OpenAI calls.
SEMANTIC_MODEL = "all-MiniLM-L6-v2"
MIN_CORE_SIM = 0.58
MIN_PEER_SIM = 0.54
MIN_PEER_SUPPORT = 2
QUERY_CORE_SIZE = 4

# ----------------------------
# Helpers
# ----------------------------
GDELT_CACHE_FILE = "gdelt_doc_cache.json"

def load_cache():
    if os.path.exists(GDELT_CACHE_FILE):
        with open(GDELT_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(GDELT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def extract_entities(title: str):
    """
    Lightweight title-only entity candidates.
    Avoids treating headline grammar such as "The American Nationalization"
    as a useful GDELT entity anchor.
    """
    ents = re.findall(r"\b([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,2})\b", title)
    bad_leads = {"The", "A", "An", "What", "Why", "How", "When", "Where", "After", "Before"}
    out = []
    for ent in ents:
        words = ent.split()
        while words and words[0] in bad_leads:
            words = words[1:]
        if not words:
            continue
        ent = " ".join(words)
        if len(ent) < 3:
            continue
        out.append(ent)
    return out[:6]

def gdelt_safe_query(title: str) -> str:
    import re
    t = title.lower()
    t = re.sub(r"\s[-–|].*$", "", t)       # remove source suffix
    t = re.sub(r"[^a-z0-9\s]", " ", t)     # strip punctuation
    words = [w for w in t.split() if len(w) >= 3]  # <-- key fix
    return " ".join(words[:12])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    p.add_argument(
        "--skip-gdelt",
        action="store_true",
        help="Run only local corpus recovery; useful for diagnostics.",
    )
    p.add_argument(
        "--diagnose-local",
        action="store_true",
        help="Print strongest rejected local candidates without accepting them.",
    )
    return p.parse_args()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_date_window(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    start = (d - timedelta(days=1)).strftime("%Y%m%d000000")
    end = (d + timedelta(days=1)).strftime("%Y%m%d235959")
    return start, end

def domain_from_url(url):
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""

def article_text(a):
    title = (a.get("title") or "").strip()
    desc = (a.get("description") or "").strip()
    desc = re.sub(r"<[^>]+>", " ", desc)
    desc = re.sub(r"\s+", " ", desc)[:240]
    return f"{title}. {desc}".strip()


def build_event_core(cluster, model):
    """
    Select the most mutually central verified articles. These are used both to
    construct precise GDELT queries and to validate returned coverage.
    """
    articles = [a for a in cluster.get("articles", []) if (a.get("title") or "").strip()]
    if not articles:
        return [], None, None

    texts = [article_text(a) for a in articles]
    X = np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    if len(X) == 1:
        return articles, X, X[0]

    S = X @ X.T
    np.fill_diagonal(S, np.nan)
    centrality = np.nanmean(S, axis=1)

    k = min(QUERY_CORE_SIZE, len(articles))
    core_idx = np.argsort(centrality)[-k:][::-1]
    core_articles = [articles[i] for i in core_idx]

    core_vec = X[core_idx].mean(axis=0)
    core_vec = core_vec / max(np.linalg.norm(core_vec), 1e-12)

    return core_articles, X, core_vec


def build_queries_from_cluster(cluster, model):
    """
    Build one compact event signature plus at most one fallback.
    """
    core_articles, _, _ = build_event_core(cluster, model)
    titles = [(a.get("title") or "").strip() for a in core_articles]
    titles = [t for t in titles if t]
    if not titles:
        return []

    entity_counts = {}
    entity_display = {}
    for title in titles:
        seen_here = set()
        for ent in extract_entities(title):
            key = ent.lower()
            if key in seen_here:
                continue
            seen_here.add(key)
            entity_counts[key] = entity_counts.get(key, 0) + 1
            entity_display.setdefault(key, ent)

    repeated_entities = [k for k, n in entity_counts.items() if n >= 2]
    repeated_entities.sort(
        key=lambda k: (entity_counts[k], len(k.split()), len(k)),
        reverse=True,
    )

    stop = {
        "the","and","for","with","from","that","this","after","before","into","over",
        "under","says","say","new","latest","live","amid","about","more","will","has",
        "have","had","was","were","are","its","their","his","her","what","when","where",
        "news","report","reports","update","updates",
    }

    token_counts = {}
    for title in titles:
        seen_here = set()
        for w in gdelt_safe_query(title).split():
            if len(w) < 4 or w in stop or w in seen_here:
                continue
            seen_here.add(w)
            token_counts[w] = token_counts.get(w, 0) + 1

    repeated_terms = [w for w, n in token_counts.items() if n >= 2]
    repeated_terms.sort(key=lambda w: (token_counts[w], len(w)), reverse=True)

    queries = []

    anchor_parts = [entity_display[k] for k in repeated_entities[:2]]
    terms = []
    anchor_words = {w.lower() for part in anchor_parts for w in part.split()}

    for term in repeated_terms:
        if term not in anchor_words:
            terms.append(term)
        if len(terms) >= 4:
            break

    if anchor_parts or terms:
        pieces = []
        for part in anchor_parts:
            pieces.append(f'"{part}"' if " " in part else part)
        pieces.extend(terms)
        q = " ".join(pieces[:6]).strip()
        if q:
            queries.append(q)

    fallback = " ".join(gdelt_safe_query(titles[0]).split()[:7]).strip()
    if fallback and fallback not in queries:
        queries.append(fallback)

    return queries[:2]


def prepare_article_bias(art):
    out = dict(art)
    if not out.get("bias"):
        bias = lookup_bias_by_domain(out.get("url", ""))
        out["bias"] = canonicalize(bias) if bias else "Unknown"
    return out



def event_peg_terms(core_articles):
    """
    Derive event-defining terms from the verified core itself.

    A term must appear in at least MIN_EVENT_PEG_TERM_SUPPORT core titles.
    This avoids hardcoded event vocab while filtering broad same-topic matches.
    """
    stop = {
        "the","and","for","with","from","that","this","after","before","into","over",
        "under","says","say","new","latest","live","amid","about","more","will","has",
        "have","had","was","were","are","its","their","his","her","what","when","where",
        "news","report","reports","update","updates","video","watch","why","how",
        "people","years","year","days","day","month","months",
    }

    counts = {}
    display = {}

    for art in core_articles:
        title = (art.get("title") or "").strip()
        if not title:
            continue

        seen = set()

        # Include lightweight named-entity phrases.
        for ent in extract_entities(title):
            key = ent.lower().strip()
            if key and key not in seen:
                seen.add(key)
                counts[key] = counts.get(key, 0) + 1
                display.setdefault(key, ent)

        # Include normalized lexical terms.
        for token in gdelt_safe_query(title).split():
            token = token.lower().strip()
            if len(token) < 4 or token in stop or token in seen:
                continue
            seen.add(token)
            counts[token] = counts.get(token, 0) + 1
            display.setdefault(token, token)

    peg = {
        key for key, n in counts.items()
        if n >= MIN_EVENT_PEG_TERM_SUPPORT
    }

    return peg


def article_event_peg_overlap(article, peg_terms):
    """
    Count how many event-peg terms appear in the candidate title/description.
    Longer entity phrases count as one peg term.
    """
    if not peg_terms:
        return 0, []

    haystack = article_text(article).lower()
    matched = []

    for term in peg_terms:
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", haystack):
            matched.append(term)

    return len(matched), matched


def recover_from_local_corpus(clusters, corpus, model, diagnose=False):
    """
    Re-examine the full normalized daily corpus after event verification.

    Each unused article can be assigned to at most one final event. In
    diagnostic mode, nothing is added; instead the strongest rejected
    candidates are printed so thresholds can be calibrated safely.
    """
    all_core_urls = {
        a.get("url")
        for cluster in clusters
        for a in cluster.get("articles", [])
        if a.get("url")
    }

    candidates = [
        a for a in corpus
        if a.get("url")
        and a.get("url") not in all_core_urls
        and (a.get("title") or "").strip()
    ]

    if not candidates or not clusters:
        return [0] * len(clusters)

    candidate_X = np.asarray(
        model.encode([article_text(a) for a in candidates], normalize_embeddings=True),
        dtype=np.float32,
    )

    cluster_cores = []
    for cluster in clusters:
        core_articles, _, _ = build_event_core(cluster, model)
        core_texts = [article_text(a) for a in core_articles]

        if not core_texts:
            cluster_cores.append(([], None, None, set()))
            continue

        core_X = np.asarray(
            model.encode(core_texts, normalize_embeddings=True),
            dtype=np.float32,
        )
        core_vec = core_X.mean(axis=0)
        core_vec = core_vec / max(np.linalg.norm(core_vec), 1e-12)
        peg_terms = event_peg_terms(core_articles)
        cluster_cores.append((core_articles, core_X, core_vec, peg_terms))

    assignments = [[] for _ in clusters]
    diagnostic_rows = [[] for _ in clusters]

    for art, vec in zip(candidates, candidate_X):
        scored = []

        for idx, (core_articles, core_X, core_vec, peg_terms) in enumerate(cluster_cores):
            if core_vec is None:
                continue

            core_sim = float(vec @ core_vec)
            peer_sims = core_X @ vec
            peer_support = int(np.sum(peer_sims >= MIN_LOCAL_PEER_SIM))
            needed = min(MIN_LOCAL_PEER_SUPPORT, len(core_articles))

            peg_overlap, peg_matches = article_event_peg_overlap(art, peg_terms)
            scored.append((core_sim, peer_support, needed, peg_overlap, peg_matches, idx))

        if not scored:
            continue

        scored.sort(reverse=True)
        best_sim, best_support, needed, best_peg_overlap, best_peg_matches, best_idx = scored[0]
        second_sim = scored[1][0] if len(scored) > 1 else -1.0
        margin = best_sim - second_sim if len(scored) > 1 else 1.0

        passes_sim = best_sim >= MIN_LOCAL_CORE_SIM
        passes_peer = best_support >= needed
        passes_margin = len(scored) == 1 or margin >= MIN_LOCAL_BEST_MARGIN
        passes_peg = best_peg_overlap >= MIN_EVENT_PEG_SHARED_TERMS

        # Calibrated recovery rule with event-peg verification:
        # semantic similarity identifies the topic,
        # cluster margin identifies the best event,
        # peg overlap verifies the same specific development.
        high_confidence = (
            best_sim >= 0.60
            and best_support >= 1
            and (len(scored) == 1 or margin >= 0.10)
            and passes_peg
        )

        accepted = (
            (passes_sim and passes_peer and passes_margin and passes_peg)
            or high_confidence
        )

        diagnostic_rows[best_idx].append({
            "title": (art.get("title") or "").strip(),
            "core_sim": best_sim,
            "peer_support": best_support,
            "peer_needed": needed,
            "margin": margin,
            "passes_sim": passes_sim,
            "passes_peer": passes_peer,
            "passes_margin": passes_margin,
            "peg_overlap": best_peg_overlap,
            "peg_matches": best_peg_matches,
            "passes_peg": passes_peg,
            "high_confidence": high_confidence,
            "accepted": accepted,
        })

        if diagnose:
            continue

        if not accepted:
            continue

        recovered = prepare_article_bias(art)
        recovered["local_recovered"] = True
        recovered["local_core_sim"] = round(best_sim, 3)
        recovered["local_peer_support"] = best_support
        recovered["local_best_margin"] = round(margin, 3)
        recovered["local_event_peg_overlap"] = best_peg_overlap
        recovered["local_event_peg_matches"] = sorted(best_peg_matches)
        assignments[best_idx].append(recovered)

    if diagnose:
        print("\n🔬 Local recovery diagnostic — strongest non-core candidates")
        for idx, rows in enumerate(diagnostic_rows):
            rows.sort(key=lambda r: (r["core_sim"], r["peer_support"], r["margin"]), reverse=True)
            print(f"\n  Cluster {idx}:")
            for row in rows[:8]:
                flags = []
                if not row["passes_sim"]:
                    flags.append("sim")
                if not row["passes_peer"]:
                    flags.append("peer")
                if not row["passes_margin"]:
                    flags.append("margin")
                if not row["passes_peg"]:
                    flags.append("peg")
                if row.get("accepted"):
                    reason = "PASS-HIGH" if row.get("high_confidence") else "PASS"
                else:
                    reason = "reject:" + ",".join(flags)
                print(
                    f"    sim={row['core_sim']:.3f} "
                    f"peers={row['peer_support']}/{row['peer_needed']} "
                    f"margin={row['margin']:.3f} "
                    f"peg={row['peg_overlap']} "
                    f"[{reason}] {row['title'][:115]}"
                )
        return [0] * len(clusters)

    added_counts = [0] * len(clusters)

    for idx, cluster in enumerate(clusters):
        existing_urls = {
            a.get("url")
            for a in cluster.get("coverage_articles", [])
            if a.get("url")
        }

        domain_counts = {}
        for a in cluster.get("coverage_articles", []):
            d = domain_from_url(a.get("url", ""))
            domain_counts[d] = domain_counts.get(d, 0) + 1

        assignments[idx].sort(
            key=lambda a: (
                a.get("local_core_sim", 0),
                a.get("local_peer_support", 0),
                a.get("local_best_margin", 0),
            ),
            reverse=True,
        )

        for art in assignments[idx]:
            if added_counts[idx] >= MAX_LOCAL_ADDITIONS_PER_CLUSTER:
                break
            if len(cluster["coverage_articles"]) >= MAX_ARTICLES_PER_CLUSTER:
                break

            url = art.get("url")
            domain = domain_from_url(url or "")

            if not url or url in existing_urls:
                continue
            if domain_counts.get(domain, 0) >= MAX_PER_DOMAIN:
                continue

            cluster["coverage_articles"].append(art)
            existing_urls.add(url)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            added_counts[idx] += 1

    return added_counts


def validate_gdelt_candidates(core_articles, candidates, model):
    """
    Local/free article-to-event validation.

    A GDELT candidate must be sufficiently similar to the verified event core
    and have support from multiple verified articles. This prevents a broad
    query from turning coverage_articles back into a thematic bucket.
    """
    if not candidates or not core_articles:
        return [], len(candidates)

    core_texts = [article_text(a) for a in core_articles]
    cand_texts = [article_text(a) for a in candidates]

    core_X = np.asarray(model.encode(core_texts, normalize_embeddings=True), dtype=np.float32)
    cand_X = np.asarray(model.encode(cand_texts, normalize_embeddings=True), dtype=np.float32)

    core_vec = core_X.mean(axis=0)
    core_vec = core_vec / max(np.linalg.norm(core_vec), 1e-12)

    accepted = []
    rejected = 0

    for art, vec in zip(candidates, cand_X):
        core_sim = float(vec @ core_vec)
        peer_sims = core_X @ vec
        peer_support = int(np.sum(peer_sims >= MIN_PEER_SIM))

        if core_sim >= MIN_CORE_SIM and peer_support >= min(MIN_PEER_SUPPORT, len(core_articles)):
            art["gdelt_core_sim"] = round(core_sim, 3)
            art["gdelt_peer_support"] = peer_support
            accepted.append(art)
        else:
            rejected += 1

    return accepted, rejected


def gdelt_query(query, start, end):
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": MAX_RECORDS_PER_QUERY,
        "startdatetime": start,
        "enddatetime": end,
        "sort": "HybridRel",
    }

    last_error = None

    for attempt in range(MAX_QUERY_ATTEMPTS):
        try:
            r = requests.get(GDELT_ENDPOINT, params=params, timeout=(20, 35))

            if r.status_code == 429:
                raise RuntimeError("GDELT_429")
            if r.status_code != 200:
                raise RuntimeError(f"GDELT HTTP {r.status_code}")

            body = (r.text or "").strip()
            if not body.startswith("{"):
                return {"articles": []}

            try:
                return r.json()
            except Exception:
                raise RuntimeError("GDELT_INVALID_JSON")

        except Exception as exc:
            last_error = exc

            if "GDELT_429" in str(exc):
                raise

            if attempt >= MAX_QUERY_ATTEMPTS - 1:
                break

            delay = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
            print(
                f"      ↻ retry {attempt + 1}/{MAX_QUERY_ATTEMPTS - 1} "
                f"after {type(exc).__name__}: {exc} ({delay:.0f}s)"
            )
            time.sleep(delay)

    raise RuntimeError(str(last_error) if last_error else "GDELT request failed")



def normalize_title_for_syndication(title):
    title = (title or "").lower()
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def title_similarity(a, b):
    from difflib import SequenceMatcher
    a = normalize_title_for_syndication(a)
    b = normalize_title_for_syndication(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def unique_domain(url):
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url or "")
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def add_source_to_family(family, art):
    family.setdefault("syndicated_sources", [])
    entry = {
        "domain": unique_domain(art.get("url", "")),
        "url": art.get("url"),
        "source": art.get("source"),
        "bias": art.get("bias", "Unknown"),
    }
    existing = {(x.get("domain"), x.get("url")) for x in family["syndicated_sources"]}
    if (entry["domain"], entry["url"]) not in existing:
        family["syndicated_sources"].append(entry)



def source_receipt_from_article(art, origin, syndicated=False):
    return {
        "domain": unique_domain(art.get("url", "")),
        "url": art.get("url"),
        "source": art.get("source"),
        "bias": art.get("bias", "Unknown"),
        "origin": origin,
        "syndicated": bool(syndicated),
    }


def dedupe_source_receipts(receipts):
    """
    Keep one receipt per distinct article URL. If a URL is missing, fall back to
    domain+source so we still preserve the outlet record without duplicating it.
    """
    out = []
    seen = set()

    for r in receipts:
        domain = (r.get("domain") or "").lower().strip()
        url = (r.get("url") or "").strip()
        source = (r.get("source") or "").strip()

        key = ("url", url) if url else ("fallback", domain, source)
        if key in seen:
            continue

        seen.add(key)
        rr = dict(r)
        rr["domain"] = domain
        out.append(rr)

    return out


def build_cluster_coverage_sources(cluster):
    """
    Build the complete source-receipt list for the cluster.

    Includes:
      - verified core articles
      - local recovered articles
      - retained GDELT representative families
      - every syndicated GDELT source attached to those families
      - previously stored cluster-level GDELT receipts
    """
    receipts = []

    core_urls = {
        a.get("url")
        for a in cluster.get("articles", [])
        if a.get("url")
    }

    for art in cluster.get("coverage_articles", []):
        url = art.get("url")

        if art.get("gdelt"):
            origin = "gdelt"
        elif art.get("local_recovered"):
            origin = "local"
        elif url in core_urls:
            origin = "core"
        else:
            origin = "coverage"

        receipts.append(
            source_receipt_from_article(
                art,
                origin=origin,
                syndicated=False,
            )
        )

        for src in art.get("syndicated_sources", []):
            receipts.append({
                "domain": unique_domain(src.get("url", "")) or (src.get("domain") or "").lower().strip(),
                "url": src.get("url"),
                "source": src.get("source"),
                "bias": src.get("bias", "Unknown"),
                "origin": "gdelt",
                "syndicated": True,
            })

    # Preserve any validated GDELT receipts that were not retained as content
    # families because of the downstream family cap.
    for src in cluster.get("_validated_gdelt_sources", []):
        receipts.append({
            "domain": unique_domain(src.get("url", "")) or (src.get("domain") or "").lower().strip(),
            "url": src.get("url"),
            "source": src.get("source"),
            "bias": src.get("bias", "Unknown"),
            "origin": "gdelt",
            "syndicated": bool(src.get("syndicated", False)),
        })

    return dedupe_source_receipts(receipts)


def refresh_cluster_coverage_metadata(cluster):
    sources = build_cluster_coverage_sources(cluster)
    cluster["coverage_sources"] = sources

    domains = sorted({
        s.get("domain")
        for s in sources
        if s.get("domain")
    })

    cluster["coverage_domains"] = domains
    cluster["coverage_outlet_count"] = len(domains)
    cluster["independent_report_count"] = len(cluster.get("coverage_articles", []))


def build_gdelt_syndication_families(articles):
    families = []
    articles = sorted(
        articles,
        key=lambda a: (a.get("gdelt_core_sim", 0), a.get("gdelt_peer_support", 0)),
        reverse=True,
    )
    for art in articles:
        matched = None
        for family in families:
            if title_similarity(art.get("title"), family.get("title")) >= SYNDICATION_TITLE_SIM:
                matched = family
                break
        if matched is None:
            family = dict(art)
            family["syndication_family"] = True
            family["syndicated_sources"] = []
            add_source_to_family(family, art)
            families.append(family)
        else:
            add_source_to_family(matched, art)
    return families


def collect_cluster_coverage_domains(cluster):
    domains = set()
    for art in cluster.get("coverage_articles", []):
        d = unique_domain(art.get("url", ""))
        if d:
            domains.add(d)
        for src in art.get("syndicated_sources", []):
            d2 = (src.get("domain") or "").lower().strip()
            if d2:
                domains.add(d2)
    for d in cluster.get("coverage_domains", []):
        if d:
            domains.add(d.lower().strip())
    return sorted(domains)


def normalize_gdelt_article(a):
    url = a.get("url")
    if not url:
        return None

    bias = lookup_bias_by_domain(url)
    bias = canonicalize(bias) if bias else "Unknown"

    return {
        "title": a.get("title"),
        "url": url,
        "description": a.get("seendate"),
        "published_at": a.get("seendate"),
        "source": domain_from_url(url),
        "bias": bias,
        "gdelt": True
    }

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    date_str = args.date or datetime.today().strftime("%Y-%m-%d")

    input_file = f"grouped_articles_final_{date_str}.json"
    local_corpus_file = f"{LOCAL_CORPUS_PREFIX}{date_str}.json"
    output_file = f"grouped_articles_final_expanded_{date_str}.json"

    if not os.path.exists(input_file):
        print(f"❌ Missing {input_file}. Run final_cohesion_check.py first.")
        return

    grouped = load_json(input_file)
    clusters = grouped["clusters"] if isinstance(grouped, dict) else grouped

    start, end = get_date_window(date_str)
    cache = load_cache()

    print(f"🧠 Loading local semantic validator: {SEMANTIC_MODEL}")
    semantic_model = SentenceTransformer(SEMANTIC_MODEL)
    cache_dirty = False

    for cluster in clusters:
        cluster["coverage_articles"] = list(cluster.get("articles", []))

    print(f"♻️ Recovering coverage from local corpus ({date_str})")
    if os.path.exists(local_corpus_file):
        corpus = load_json(local_corpus_file)
        local_added = recover_from_local_corpus(clusters, corpus, semantic_model, diagnose=args.diagnose_local)

        for idx, n in enumerate(local_added):
            print(f"  • Cluster {idx}: +{n} local articles")

        print(f"  ✓ Local recovery total: +{sum(local_added)}")
    else:
        print(f"⚠️ Missing {local_corpus_file}; skipping local recovery.")

    if args.skip_gdelt or args.diagnose_local:
        save_json(output_file, grouped)
        verified_total = sum(len(c.get("articles", [])) for c in clusters)
        coverage_total = sum(len(c.get("coverage_articles", [])) for c in clusters)

        print("⏭️ Skipping GDELT during local-only/diagnostic run.")
        print(f"📊 Verified core articles: {verified_total}")
        print(f"📊 Final coverage articles: {coverage_total}")
        print(f"📊 Net expansion: +{coverage_total - verified_total}")
        print(f"✅ Wrote expanded clusters → {output_file}")
        return

    print(f"🌍 Expanding clusters via GDELT ({date_str})")

    queries_used = 0
    unmapped = []
    gdelt_added_total = 0

    for idx, cluster in enumerate(clusters):
        existing_urls = {
            a.get("url")
            for a in cluster.get("coverage_articles", [])
            if a.get("url")
        }

        domain_counts = {}
        for a in cluster.get("coverage_articles", []):
            d = domain_from_url(a.get("url", ""))
            domain_counts[d] = domain_counts.get(d, 0) + 1

        queries = build_queries_from_cluster(cluster, semantic_model)
        core_articles_for_validation, _, _ = build_event_core(cluster, semantic_model)

        added = 0
        primary_succeeded = False

        for q_idx, q in enumerate(queries):
            if queries_used >= MAX_QUERIES_PER_RUN:
                print("🛑 Reached MAX_QUERIES_PER_RUN; stopping GDELT requests.")
                break

            if q_idx > 0 and not primary_succeeded:
                break
            if q_idx > 0 and added >= TARGET_GDELT_NEW_ARTICLES_PER_CLUSTER:
                break

            try:
                ck = f"{start}|{end}|{q}"

                if ck in cache:
                    data = cache[ck]
                    print(f"    cached query='{q}' → {len(data.get('articles', []))} hits")
                else:
                    data = gdelt_query(q, start, end)
                    cache[ck] = data
                    cache_dirty = True
                    queries_used += 1
                    print(f"    query='{q}' → {len(data.get('articles', []))} hits")
                    time.sleep(REQUEST_SLEEP_SECONDS)

                if q_idx == 0:
                    primary_succeeded = True

            except Exception as exc:
                msg = str(exc)

                if "GDELT_429" in msg:
                    print(
                        f"⚠️ GDELT rate-limited. Cooling down the entire stage "
                        f"for {RATE_LIMIT_COOLDOWN_SECONDS:.0f}s."
                    )
                    time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
                else:
                    print(f"⚠️ GDELT query failed: {exc}")

                continue

            normalized_candidates = []
            for raw in data.get("articles", []):
                art = normalize_gdelt_article(raw)
                if not art:
                    continue

                url = art["url"]
                domain = domain_from_url(url)

                if url in existing_urls:
                    continue
                if domain_counts.get(domain, 0) >= MAX_PER_DOMAIN:
                    continue

                normalized_candidates.append(art)

            accepted_candidates, rejected_semantic = validate_gdelt_candidates(
                core_articles_for_validation,
                normalized_candidates,
                semantic_model,
            )

            validated_domains_this_query = set()
            validated_source_receipts = []

            for art in accepted_candidates:
                d = unique_domain(art.get("url", ""))
                if d:
                    validated_domains_this_query.add(d)

                validated_source_receipts.append({
                    "domain": d,
                    "url": art.get("url"),
                    "source": art.get("source"),
                    "bias": art.get("bias", "Unknown"),
                    "origin": "gdelt",
                    "syndicated": False,
                })

                if art.get("bias") == "Unknown":
                    unmapped.append({
                        "url": art.get("url"),
                        "domain": domain_from_url(art.get("url", "")),
                        "title": art.get("title"),
                    })

            cluster.setdefault("_validated_gdelt_sources", [])
            cluster["_validated_gdelt_sources"].extend(validated_source_receipts)

            families = build_gdelt_syndication_families(accepted_candidates)
            accepted_this_query = 0

            for family in families:
                if accepted_this_query >= MAX_GDELT_CONTENT_FAMILIES_PER_CLUSTER:
                    break
                if added >= TARGET_GDELT_NEW_ARTICLES_PER_CLUSTER:
                    break
                if len(cluster["coverage_articles"]) >= MAX_ARTICLES_PER_CLUSTER:
                    break

                rep_url = family.get("url")
                rep_domain = domain_from_url(rep_url or "")

                if not rep_url or rep_url in existing_urls:
                    continue
                if domain_counts.get(rep_domain, 0) >= MAX_PER_DOMAIN:
                    continue

                cluster["coverage_articles"].append(family)
                existing_urls.add(rep_url)
                domain_counts[rep_domain] = domain_counts.get(rep_domain, 0) + 1

                accepted_this_query += 1
                added += 1
                gdelt_added_total += 1

            refresh_cluster_coverage_metadata(cluster)

            print(
                f"      candidates={len(normalized_candidates)} "
                f"validated={len(accepted_candidates)} "
                f"families_added={accepted_this_query} "
                f"validated_domains={len(validated_domains_this_query)} "
                f"semantic_rejected={rejected_semantic}"
            )

            if len(cluster["coverage_articles"]) >= MAX_ARTICLES_PER_CLUSTER:
                break
            if added >= TARGET_GDELT_NEW_ARTICLES_PER_CLUSTER:
                print(f"      ✓ GDELT expansion target reached (+{added})")
                break

        refresh_cluster_coverage_metadata(cluster)
        print(f"  • Cluster {idx}: +{added} GDELT content families")

    if unmapped:
        try:
            if os.path.exists(UNMAPPED_BIAS_FILE):
                with open(UNMAPPED_BIAS_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.extend(unmapped)

            seen = set()
            deduped = []
            for x in existing:
                d = x.get("domain")
                if d and d not in seen:
                    seen.add(d)
                    deduped.append(x)

            with open(UNMAPPED_BIAS_FILE, "w", encoding="utf-8") as f:
                json.dump(deduped, f, indent=2, ensure_ascii=False)

            print(f"🧭 Exported {len(deduped)} unmapped domains → {UNMAPPED_BIAS_FILE}")
        except Exception as exc:
            print(f"⚠️ Failed to write {UNMAPPED_BIAS_FILE}: {exc}")

    if cache_dirty:
        save_cache(cache)
        print(f"💾 Saved GDELT cache → {GDELT_CACHE_FILE}")

    for cluster in clusters:
        refresh_cluster_coverage_metadata(cluster)
        cluster.pop("_validated_gdelt_sources", None)

    save_json(output_file, grouped)

    verified_total = sum(len(c.get("articles", [])) for c in clusters)
    coverage_total = sum(len(c.get("coverage_articles", [])) for c in clusters)

    print(f"📊 Verified core articles: {verified_total}")
    print(f"📊 Final coverage articles: {coverage_total}")
    print(f"📊 Net expansion: +{coverage_total - verified_total}")
    total_outlets = sum(c.get("coverage_outlet_count", 0) for c in clusters)
    print(f"📊 GDELT content families added this run: +{gdelt_added_total}")
    print(f"📊 Total validated outlet-domain mentions across clusters: {total_outlets}")
    print(f"✅ Wrote expanded clusters → {output_file}")


if __name__ == "__main__":
    main()
