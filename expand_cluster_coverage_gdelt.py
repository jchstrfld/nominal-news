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
BIAS_OVERRIDES_FILE = "bias_overrides.json"


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
# Local recovery must contain a specific multiword event anchor.
# Semantic similarity alone is not enough for same-person/same-institution stories.
MIN_LOCAL_EVENT_TERM_HITS = 2
MIN_LOCAL_HIGH_CONF_SIM = 0.68
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
MAX_RATE_LIMITS_PER_RUN = 2
TARGET_GDELT_NEW_ARTICLES_PER_CLUSTER = 8

# Exact-development safeguards for post-selection GDELT receipts.
# GDELT should expand the selected development, not a later outcome or merely
# another article about the same person/case/country.
MIN_GDELT_EXACT_TITLE_SIM = 0.42
MIN_GDELT_IDENTITY_TITLE_SIM = 0.52
MIN_GDELT_HIGH_TITLE_SIM = 0.72
MIN_GDELT_EVENT_TERM_HITS = 3
MIN_GDELT_EVENT_TERM_COVERAGE = 0.50
MIN_GDELT_FUZZY_PHRASE_COVERAGE = 0.75
MIN_GDELT_NEXT_DAY_TITLE_SIM = 0.64

SYNDICATION_TITLE_SIM = 0.92
MAX_GDELT_CONTENT_FAMILIES_PER_CLUSTER = 20

# Conservative cross-origin report-family detection.
# These thresholds group likely wire/syndicated variants without collapsing
# independently written coverage merely because it discusses the same event.
REPORT_FAMILY_TITLE_SIM = 0.90
REPORT_FAMILY_DESC_SIM = 0.94
REPORT_FAMILY_TOKEN_JACCARD = 0.72
REPORT_FAMILY_SEM_SIM = 0.94
REPORT_FAMILY_SEM_MIN_JACCARD = 0.50

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
    p.add_argument(
        "--refresh-bias-only",
        action="store_true",
        help=(
            "Refresh saved outlet bias labels from bias_overrides.json without "
            "running local recovery or making GDELT requests."
        ),
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
    """Apply the current saved domain override, including manual edits."""
    out = dict(art)
    url = out.get("url", "")
    looked_up = lookup_bias_by_domain(url) if url else None

    if looked_up:
        out["bias"] = canonicalize(looked_up)
    else:
        out["bias"] = canonicalize(out.get("bias") or "Unknown")

    return out


def _refresh_source_bias(source):
    out = dict(source)
    url = out.get("url") or ""
    domain = (out.get("domain") or "").strip().lower()
    lookup_url = url or (f"https://{domain}/" if domain else "")
    looked_up = lookup_bias_by_domain(lookup_url) if lookup_url else None
    out["bias"] = canonicalize(looked_up or out.get("bias") or "Unknown")
    return out


def refresh_cluster_biases(cluster):
    """Refresh all persisted article/source receipts after manual override edits."""
    for key in ("articles", "related_articles", "coverage_articles"):
        refreshed = []
        for art in cluster.get(key, []) or []:
            item = prepare_article_bias(art)
            if item.get("syndicated_sources"):
                item["syndicated_sources"] = [
                    _refresh_source_bias(src)
                    for src in item.get("syndicated_sources", [])
                ]
            refreshed.append(item)
        if key in cluster or refreshed:
            cluster[key] = refreshed

    if cluster.get("coverage_sources"):
        cluster["coverage_sources"] = [
            _refresh_source_bias(src)
            for src in cluster.get("coverage_sources", [])
        ]

    return cluster


def prune_unmapped_bias_file(path=UNMAPPED_BIAS_FILE):
    """Remove domains from the review queue once a saved override categorizes them."""
    if not os.path.exists(path):
        return 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return 0

    if not isinstance(rows, list):
        return 0

    remaining = []
    removed = 0
    for row in rows:
        domain = (row.get("domain") or "").strip().lower()
        lookup_url = row.get("url") or (f"https://{domain}/" if domain else "")
        bias = lookup_bias_by_domain(lookup_url) if lookup_url else None
        if bias and canonicalize(bias) != "Unknown":
            removed += 1
        else:
            remaining.append(row)

    if removed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(remaining, f, indent=2, ensure_ascii=False)

    return removed




def sync_unmapped_domains_to_overrides(
    unmapped_path=UNMAPPED_BIAS_FILE,
    overrides_path=BIAS_OVERRIDES_FILE,
):
    """
    Ensure every domain in the GDELT review queue exists in the permanent
    bias registry. Existing manual entries are never overwritten.

    This turns unmapped_bias.json into a review queue only; the user edits
    bias_overrides.json as the single source of truth.
    """
    if not os.path.exists(unmapped_path):
        return 0

    try:
        with open(unmapped_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return 0

    if not isinstance(rows, list):
        return 0

    try:
        if os.path.exists(overrides_path):
            with open(overrides_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            if not isinstance(overrides, dict):
                overrides = {}
        else:
            overrides = {}
    except Exception:
        return 0

    added = 0
    for row in rows:
        domain = (row.get("domain") or "").strip().lower()
        if not domain or domain in overrides:
            continue
        overrides[domain] = {
            "bias": "Unknown",
            "sources": ["gdelt-auto-seen"],
            "notes": "",
        }
        added += 1

    if added:
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2, ensure_ascii=False)

    return added


def _parse_article_date(value):
    """Return YYYY-MM-DD from common GDELT/ISO date strings, else ''."""
    raw = str(value or "").strip()
    if not raw:
        return ""

    iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})", raw)
    if iso:
        return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"

    compact = re.match(r"^(\d{4})(\d{2})(\d{2})", raw)
    if compact:
        return f"{compact.group(1)}-{compact.group(2)}-{compact.group(3)}"

    return ""


def _event_phrase_coverage(title, event_phrases):
    """
    Best order-independent token coverage of a derived event phrase.
    Allows modest headline paraphrasing without accepting a different outcome.
    """
    candidate_tokens = set(_anchor_tokens(title))
    if not candidate_tokens:
        return 0.0, []

    best = 0.0
    hits = []
    for phrase in event_phrases or []:
        phrase_tokens = set(_anchor_tokens(phrase))
        if len(phrase_tokens) < 2:
            continue
        coverage = len(candidate_tokens & phrase_tokens) / len(phrase_tokens)
        if coverage > best:
            best = coverage
        if coverage >= MIN_GDELT_FUZZY_PHRASE_COVERAGE:
            hits.append(phrase)

    return best, hits


def _exact_development_match(cluster, core_articles, article):
    """
    Free, conservative same-development check for a GDELT title.

    Semantic similarity identifies the subject. This check additionally
    requires the title to preserve the selected event's action/state. It is
    deliberately stricter for articles first seen after the target date, where
    a later outcome can otherwise be mistaken for the earlier development.
    """
    candidate_title = (article.get("title") or "").strip()
    if not candidate_title:
        return False, {}

    core_titles = [
        (a.get("title") or "").strip()
        for a in core_articles
        if (a.get("title") or "").strip()
    ]
    canonical_event = (cluster.get("canonical_event") or "").strip()
    reference_titles = ([canonical_event] if canonical_event else []) + core_titles

    profile = build_event_anchor_profile(cluster, core_articles)
    anchor = article_event_anchor_overlap(article, profile)

    title_sim = 0.0
    for reference in reference_titles:
        title_sim = max(title_sim, title_similarity(candidate_title, reference))

    identity_tokens = {
        token
        for phrase in profile.get("identity_phrases", [])
        for token in _anchor_tokens(phrase)
    }
    event_vocab = set()
    for reference in reference_titles:
        event_vocab.update(_anchor_tokens(reference))
    event_vocab -= identity_tokens

    candidate_tokens = set(_anchor_tokens(candidate_title)) - identity_tokens
    event_term_hits = sorted(candidate_tokens & event_vocab)
    term_coverage = (
        len(event_term_hits) / len(candidate_tokens)
        if candidate_tokens else 0.0
    )

    fuzzy_phrase_coverage, fuzzy_phrase_hits = _event_phrase_coverage(
        candidate_title,
        profile.get("event_phrases", []),
    )

    exact_phrase = bool(anchor.get("event_phrase_hits"))
    fuzzy_phrase = bool(fuzzy_phrase_hits)
    identity_hit = bool(anchor.get("identity_hits"))

    same_development = (
        ((exact_phrase or fuzzy_phrase) and title_sim >= MIN_GDELT_EXACT_TITLE_SIM)
        or (
            identity_hit
            and len(event_term_hits) >= MIN_GDELT_EVENT_TERM_HITS
            and term_coverage >= MIN_GDELT_EVENT_TERM_COVERAGE
            and title_sim >= MIN_GDELT_IDENTITY_TITLE_SIM
        )
        or (identity_hit and title_sim >= MIN_GDELT_HIGH_TITLE_SIM)
    )

    target_date = (cluster.get("_expansion_target_date") or "").strip()
    candidate_date = _parse_article_date(
        article.get("published_at") or article.get("description")
    )

    # Keep the ±1-day retrieval window, but demand stronger textual evidence
    # before attaching a next-day article to the prior day's development.
    if same_development and target_date and candidate_date and candidate_date > target_date:
        same_development = (
            exact_phrase
            or title_sim >= MIN_GDELT_NEXT_DAY_TITLE_SIM
            or (
                fuzzy_phrase_coverage >= MIN_GDELT_FUZZY_PHRASE_COVERAGE
                and term_coverage >= 0.55
            )
            or (
                identity_hit
                and title_sim >= MIN_GDELT_IDENTITY_TITLE_SIM
                and term_coverage >= MIN_GDELT_EVENT_TERM_COVERAGE
            )
        )

    diag = {
        "title_sim": round(float(title_sim), 3),
        "identity_hits": anchor.get("identity_hits", []),
        "event_phrase_hits": anchor.get("event_phrase_hits", []),
        "fuzzy_event_phrase_hits": fuzzy_phrase_hits,
        "fuzzy_event_phrase_coverage": round(float(fuzzy_phrase_coverage), 3),
        "event_term_hits": event_term_hits,
        "event_term_coverage": round(float(term_coverage), 3),
        "candidate_date": candidate_date,
        "target_date": target_date,
    }
    return bool(same_development), diag


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



_ANCHOR_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "after", "before",
    "into", "over", "under", "says", "say", "said", "new", "latest", "live",
    "amid", "about", "more", "will", "has", "have", "had", "was", "were",
    "are", "its", "their", "his", "her", "what", "when", "where", "news",
    "report", "reports", "update", "updates", "video", "watch", "why", "how",
    "people", "years", "year", "days", "day", "month", "months", "could",
    "would", "should", "may", "might", "still", "now", "again", "top",
    "asks", "asked", "calls", "called", "tells", "told", "announces",
    "announced", "reports", "reported",
}


def _anchor_tokens(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    return [
        t.strip("'-")
        for t in re.sub(r"\s+", " ", text).split()
        if len(t.strip("'-")) >= 3 and t.strip("'-") not in _ANCHOR_STOP
    ]


def _phrase_present(phrase, haystack):
    phrase = re.sub(r"\s+", " ", (phrase or "").strip().lower())
    if not phrase:
        return False
    return bool(re.search(r"\b" + re.escape(phrase) + r"\b", haystack))


def build_event_anchor_profile(cluster, core_articles):
    """
    Build event-specific anchors from the verified core and canonical event.

    Acceptance requires either:
      - a specific multiword event phrase, or
      - a multiword identity phrase plus multiple event terms.

    No story-specific vocabulary is hardcoded.
    """
    canonical_event = (cluster.get("canonical_event") or "").strip()
    titles = [
        (a.get("title") or "").strip()
        for a in core_articles
        if (a.get("title") or "").strip()
    ]
    source_texts = ([canonical_event] if canonical_event else []) + titles

    # Multiword identity phrases: names, organizations, institutions, and places.
    # Only multiword identities are used as identity anchors; a single broad
    # actor/country name cannot satisfy the event-specific anchor by itself.
    identity_counts = {}
    for text in source_texts:
        seen = set()
        for phrase in extract_entities(text):
            p = re.sub(r"\s+", " ", phrase.lower()).strip()
            if len(p.split()) < 2 or p in seen:
                continue
            seen.add(p)
            identity_counts[p] = identity_counts.get(p, 0) + 1

    identity_phrases = {
        p for p, count in identity_counts.items()
        if count >= 2 or (canonical_event and _phrase_present(p, canonical_event.lower()))
    }
    identity_tokens = {
        token
        for phrase in identity_phrases
        for token in _anchor_tokens(phrase)
    }

    # Event terms come from the canonical event plus terms repeated across core titles.
    title_term_counts = {}
    for title in titles:
        for token in set(_anchor_tokens(title)):
            title_term_counts[token] = title_term_counts.get(token, 0) + 1

    canonical_terms = set(_anchor_tokens(canonical_event))
    repeated_terms = {
        token for token, count in title_term_counts.items()
        if count >= 2
    }
    event_terms = (canonical_terms | repeated_terms) - identity_tokens

    # Multiword event phrases from the canonical event and repeated core-title n-grams.
    phrase_counts = {}
    for source_index, source_text in enumerate(source_texts):
        tokens = _anchor_tokens(source_text)
        seen_here = set()
        for n in (2, 3, 4):
            for i in range(0, len(tokens) - n + 1):
                phrase_tokens = tokens[i:i+n]
                phrase = " ".join(phrase_tokens)
                if phrase in seen_here:
                    continue
                seen_here.add(phrase)
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    event_phrases = set()
    for phrase, count in phrase_counts.items():
        phrase_tokens = set(phrase.split())
        is_identity_only = bool(phrase_tokens) and phrase_tokens.issubset(identity_tokens)
        event_term_count = len(phrase_tokens & event_terms)
        in_canonical = canonical_event and _phrase_present(phrase, canonical_event.lower())

        # A phrase needs at least two event-defining terms. This prevents broad
        # actor/institution phrases from qualifying as the specific event peg.
        if is_identity_only or event_term_count < 2:
            continue
        if count >= 2 or in_canonical:
            event_phrases.add(phrase)

    # Keep the profile compact and deterministic.
    identity_phrases = sorted(identity_phrases, key=lambda p: (-len(p.split()), p))[:12]
    event_phrases = sorted(event_phrases, key=lambda p: (-len(p.split()), p))[:24]
    event_terms = sorted(event_terms)[:24]

    return {
        "identity_phrases": identity_phrases,
        "event_phrases": event_phrases,
        "event_terms": event_terms,
        "available": bool(identity_phrases or event_phrases),
    }


def article_event_anchor_overlap(article, profile):
    haystack = article_text(article).lower()
    identity_hits = [
        phrase for phrase in profile.get("identity_phrases", [])
        if _phrase_present(phrase, haystack)
    ]
    event_phrase_hits = [
        phrase for phrase in profile.get("event_phrases", [])
        if _phrase_present(phrase, haystack)
    ]
    event_token_hits = [
        token for token in profile.get("event_terms", [])
        if re.search(r"\b" + re.escape(token) + r"\b", haystack)
    ]

    strict_pass = bool(event_phrase_hits) or (
        bool(identity_hits) and len(set(event_token_hits)) >= MIN_LOCAL_EVENT_TERM_HITS
    )

    return {
        "available": bool(profile.get("available")),
        "strict_pass": strict_pass,
        "identity_hits": identity_hits,
        "event_phrase_hits": event_phrase_hits,
        "event_token_hits": sorted(set(event_token_hits)),
    }

def recover_from_local_corpus(clusters, corpus, model, diagnose=False):
    """
    Re-examine the full normalized daily corpus after event verification.

    Each unused article can be assigned to at most one final event. A candidate
    must pass the existing semantic/peer/margin checks AND an event-specific
    multiword-anchor check derived from the verified event itself.
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
            cluster_cores.append(None)
            continue

        core_X = np.asarray(
            model.encode(core_texts, normalize_embeddings=True),
            dtype=np.float32,
        )
        core_vec = core_X.mean(axis=0)
        core_vec = core_vec / max(np.linalg.norm(core_vec), 1e-12)
        peg_terms = event_peg_terms(core_articles)
        anchor_profile = build_event_anchor_profile(cluster, core_articles)

        cluster_cores.append({
            "core_articles": core_articles,
            "core_X": core_X,
            "core_vec": core_vec,
            "peg_terms": peg_terms,
            "anchor_profile": anchor_profile,
        })

    assignments = [[] for _ in clusters]
    diagnostic_rows = [[] for _ in clusters]

    for art, vec in zip(candidates, candidate_X):
        scored = []

        for idx, core in enumerate(cluster_cores):
            if not core:
                continue

            core_sim = float(vec @ core["core_vec"])
            peer_sims = core["core_X"] @ vec
            peer_support = int(np.sum(peer_sims >= MIN_LOCAL_PEER_SIM))
            needed = min(MIN_LOCAL_PEER_SUPPORT, len(core["core_articles"]))

            peg_overlap, peg_matches = article_event_peg_overlap(art, core["peg_terms"])
            anchor = article_event_anchor_overlap(art, core["anchor_profile"])

            scored.append({
                "idx": idx,
                "core_sim": core_sim,
                "peer_support": peer_support,
                "peer_needed": needed,
                "peg_overlap": peg_overlap,
                "peg_matches": peg_matches,
                "anchor": anchor,
            })

        if not scored:
            continue

        scored.sort(key=lambda row: row["core_sim"], reverse=True)
        best = scored[0]
        second_sim = scored[1]["core_sim"] if len(scored) > 1 else -1.0
        margin = best["core_sim"] - second_sim if len(scored) > 1 else 1.0

        passes_sim = best["core_sim"] >= MIN_LOCAL_CORE_SIM
        passes_peer = best["peer_support"] >= best["peer_needed"]
        passes_margin = len(scored) == 1 or margin >= MIN_LOCAL_BEST_MARGIN
        passes_peg = best["peg_overlap"] >= MIN_EVENT_PEG_SHARED_TERMS
        passes_anchor = best["anchor"]["available"] and best["anchor"]["strict_pass"]

        # High-confidence path still requires a specific anchor. One identity
        # phrase plus one event term is allowed only at substantially higher
        # semantic similarity and event margin.
        high_conf_anchor = (
            bool(best["anchor"]["identity_hits"])
            and len(best["anchor"]["event_token_hits"]) >= 1
        )
        high_confidence = (
            best["core_sim"] >= MIN_LOCAL_HIGH_CONF_SIM
            and best["peer_support"] >= 1
            and (len(scored) == 1 or margin >= 0.12)
            and high_conf_anchor
        )

        accepted = (
            passes_sim
            and passes_peer
            and passes_margin
            and passes_peg
            and passes_anchor
        ) or high_confidence

        row = {
            "title": (art.get("title") or "").strip(),
            "core_sim": best["core_sim"],
            "peer_support": best["peer_support"],
            "peer_needed": best["peer_needed"],
            "margin": margin,
            "passes_sim": passes_sim,
            "passes_peer": passes_peer,
            "passes_margin": passes_margin,
            "peg_overlap": best["peg_overlap"],
            "peg_matches": best["peg_matches"],
            "passes_peg": passes_peg,
            "passes_anchor": passes_anchor,
            "identity_hits": best["anchor"]["identity_hits"],
            "event_phrase_hits": best["anchor"]["event_phrase_hits"],
            "event_token_hits": best["anchor"]["event_token_hits"],
            "high_confidence": high_confidence,
            "accepted": accepted,
        }
        diagnostic_rows[best["idx"]].append(row)

        if diagnose or not accepted:
            continue

        recovered = prepare_article_bias(art)
        recovered["local_recovered"] = True
        recovered["local_core_sim"] = round(best["core_sim"], 3)
        recovered["local_peer_support"] = best["peer_support"]
        recovered["local_best_margin"] = round(margin, 3)
        recovered["local_event_peg_overlap"] = best["peg_overlap"]
        recovered["local_event_peg_matches"] = sorted(best["peg_matches"])
        recovered["local_identity_anchor_hits"] = sorted(best["anchor"]["identity_hits"])
        recovered["local_event_phrase_hits"] = sorted(best["anchor"]["event_phrase_hits"])
        recovered["local_event_token_hits"] = sorted(best["anchor"]["event_token_hits"])
        assignments[best["idx"]].append(recovered)

    if diagnose:
        print("\n🔬 Local recovery diagnostic — strongest non-core candidates")
        for idx, rows in enumerate(diagnostic_rows):
            rows.sort(
                key=lambda r: (r["core_sim"], r["peer_support"], r["margin"]),
                reverse=True,
            )
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
                if not row["passes_anchor"]:
                    flags.append("anchor")

                if row.get("accepted"):
                    reason = "PASS-HIGH" if row.get("high_confidence") else "PASS"
                else:
                    reason = "reject:" + ",".join(flags)

                print(
                    f"    sim={row['core_sim']:.3f} "
                    f"peers={row['peer_support']}/{row['peer_needed']} "
                    f"margin={row['margin']:.3f} "
                    f"peg={row['peg_overlap']} "
                    f"anchors={len(row['identity_hits'])}/"
                    f"{len(row['event_phrase_hits'])}/"
                    f"{len(row['event_token_hits'])} "
                    f"[{reason}] {row['title'][:105]}"
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


def validate_gdelt_candidates(cluster, core_articles, candidates, model):
    """
    Local/free article-to-event validation.

    Stage 1: semantic similarity + peer support identify the same subject.
    Stage 2: exact-development matching rejects later outcomes or adjacent
    developments involving the same person, case, country, or institution.
    """
    if not candidates or not core_articles:
        return [], len(candidates), 0

    core_texts = [article_text(a) for a in core_articles]
    cand_texts = [article_text(a) for a in candidates]

    core_X = np.asarray(model.encode(core_texts, normalize_embeddings=True), dtype=np.float32)
    cand_X = np.asarray(model.encode(cand_texts, normalize_embeddings=True), dtype=np.float32)

    core_vec = core_X.mean(axis=0)
    core_vec = core_vec / max(np.linalg.norm(core_vec), 1e-12)

    accepted = []
    rejected_semantic = 0
    rejected_development = 0

    for art, vec in zip(candidates, cand_X):
        core_sim = float(vec @ core_vec)
        peer_sims = core_X @ vec
        peer_support = int(np.sum(peer_sims >= MIN_PEER_SIM))

        semantic_pass = (
            core_sim >= MIN_CORE_SIM
            and peer_support >= min(MIN_PEER_SUPPORT, len(core_articles))
        )
        if not semantic_pass:
            rejected_semantic += 1
            continue

        development_pass, development_diag = _exact_development_match(
            cluster,
            core_articles,
            art,
        )
        if not development_pass:
            rejected_development += 1
            continue

        art["gdelt_core_sim"] = round(core_sim, 3)
        art["gdelt_peer_support"] = peer_support
        art["gdelt_exact_title_sim"] = development_diag.get("title_sim", 0.0)
        art["gdelt_event_term_coverage"] = development_diag.get("event_term_coverage", 0.0)
        art["gdelt_event_phrase_hits"] = development_diag.get("event_phrase_hits", [])
        art["gdelt_fuzzy_event_phrase_hits"] = development_diag.get("fuzzy_event_phrase_hits", [])
        accepted.append(art)

    return accepted, rejected_semantic, rejected_development

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
        "title": art.get("title"),
        "bias": prepare_article_bias(art).get("bias", "Unknown"),
    }
    existing = {(x.get("domain"), x.get("url")) for x in family["syndicated_sources"]}
    if (entry["domain"], entry["url"]) not in existing:
        family["syndicated_sources"].append(entry)



def source_receipt_from_article(art, origin, syndicated=False):
    refreshed = prepare_article_bias(art)
    return {
        "domain": unique_domain(refreshed.get("url", "")),
        "url": refreshed.get("url"),
        "source": refreshed.get("source"),
        "title": refreshed.get("title"),
        "bias": refreshed.get("bias", "Unknown"),
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
                "title": src.get("title") or art.get("title"),
                "bias": _refresh_source_bias(src).get("bias", "Unknown"),
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
            "title": src.get("title"),
            "bias": _refresh_source_bias(src).get("bias", "Unknown"),
            "origin": "gdelt",
            "syndicated": bool(src.get("syndicated", False)),
        })

    return dedupe_source_receipts(receipts)


_REPORT_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "after", "before",
    "into", "over", "under", "says", "said", "new", "latest", "live", "amid",
    "about", "more", "will", "has", "have", "had", "was", "were", "are",
    "its", "their", "news", "report", "reports", "update", "updates",
}


def _report_tokens(text):
    text = normalize_title_for_syndication(text)
    return {
        token for token in text.split()
        if len(token) >= 3 and token not in _REPORT_STOP
    }


def _report_token_jaccard(a, b):
    A = _report_tokens(a)
    B = _report_tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _clean_report_description(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text[:500]


def _report_family_items(cluster):
    """Include retained coverage plus validated GDELT receipts beyond family caps."""
    items = []
    seen_urls = set()

    for art in cluster.get("coverage_articles", []) or []:
        url = (art.get("url") or "").strip()
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        items.append(art)

    for src in cluster.get("_validated_gdelt_sources", []) or []:
        url = (src.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        items.append({
            "title": src.get("title") or "",
            "description": src.get("description") or "",
            "url": url,
            "source": src.get("source"),
            "bias": src.get("bias", "Unknown"),
            "gdelt": True,
            "receipt_only": True,
        })

    return items


def build_cross_origin_report_families(cluster, model=None):
    """
    Conservatively group likely duplicate/syndicated report variants across
    core, local-recovery, and GDELT origins. This is for transparency counts;
    it does not remove outlet receipts or alter the verified event core.
    """
    items = _report_family_items(cluster)
    n = len(items)
    if n == 0:
        return []

    titles = [(item.get("title") or "").strip() for item in items]
    descriptions = [_clean_report_description(item.get("description") or "") for item in items]

    sem_vecs = None
    if model is not None and n > 1:
        try:
            sem_texts = [title or article_text(item) for title, item in zip(titles, items)]
            sem_vecs = np.asarray(
                model.encode(sem_texts, normalize_embeddings=True),
                dtype=np.float32,
            )
        except Exception:
            sem_vecs = None

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        if not titles[i]:
            continue
        for j in range(i + 1, n):
            if not titles[j]:
                continue

            title_sim = title_similarity(titles[i], titles[j])
            token_jacc = _report_token_jaccard(titles[i], titles[j])

            desc_sim = 0.0
            if len(descriptions[i]) >= 80 and len(descriptions[j]) >= 80:
                desc_sim = title_similarity(descriptions[i], descriptions[j])

            semantic_match = False
            if sem_vecs is not None:
                sem_sim = float(sem_vecs[i] @ sem_vecs[j])
                semantic_match = (
                    sem_sim >= REPORT_FAMILY_SEM_SIM
                    and token_jacc >= REPORT_FAMILY_SEM_MIN_JACCARD
                )

            likely_same_report = (
                title_sim >= REPORT_FAMILY_TITLE_SIM
                or desc_sim >= REPORT_FAMILY_DESC_SIM
                or token_jacc >= REPORT_FAMILY_TOKEN_JACCARD
                or semantic_match
            )

            if likely_same_report:
                union(i, j)

    grouped = {}
    for idx, item in enumerate(items):
        grouped.setdefault(find(idx), []).append(item)

    families = []
    for family_num, members in enumerate(grouped.values(), start=1):
        representative = max(
            members,
            key=lambda a: (
                bool((a.get("description") or "").strip()),
                len((a.get("title") or "").strip()),
            ),
        )
        domains = set()
        origins = set()

        for member in members:
            d = unique_domain(member.get("url", ""))
            if d:
                domains.add(d)
            if member.get("gdelt"):
                origins.add("gdelt")
            elif member.get("local_recovered"):
                origins.add("local")
            else:
                origins.add("core")

            for src in member.get("syndicated_sources", []) or []:
                d2 = unique_domain(src.get("url", "")) or (src.get("domain") or "").lower().strip()
                if d2:
                    domains.add(d2)

        family_id = f"report-{family_num}"
        for member in members:
            member["content_family_id"] = family_id

        families.append({
            "family_id": family_id,
            "representative_title": representative.get("title") or "",
            "article_record_count": len(members),
            "outlet_count": len(domains),
            "origins": sorted(origins),
        })

    return families


def refresh_cluster_coverage_metadata(cluster, model=None):
    sources = build_cluster_coverage_sources(cluster)
    cluster["coverage_sources"] = sources

    domains = sorted({
        s.get("domain")
        for s in sources
        if s.get("domain")
    })

    cluster["coverage_domains"] = domains
    cluster["coverage_outlet_count"] = len(domains)

    if model is not None:
        families = build_cross_origin_report_families(cluster, model=model)
        cluster["coverage_report_families"] = families
        cluster["distinct_report_count"] = len(families)
        # Backward-compatible field consumed by the current template.
        cluster["independent_report_count"] = len(families)
    elif "independent_report_count" not in cluster:
        fallback_count = len(cluster.get("coverage_articles", []))
        cluster["distinct_report_count"] = fallback_count
        cluster["independent_report_count"] = fallback_count


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


def process_gdelt_result_for_cluster(
    *,
    cluster,
    data,
    query,
    core_articles,
    semantic_model,
    existing_urls,
    domain_counts,
    unmapped,
    remaining_family_slots,
):
    """Validate one cached/live GDELT response and attach only exact-development coverage."""
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

    accepted_candidates, rejected_semantic, rejected_development = validate_gdelt_candidates(
        cluster,
        core_articles,
        normalized_candidates,
        semantic_model,
    )

    validated_domains = set()
    validated_source_receipts = []
    for art in accepted_candidates:
        d = unique_domain(art.get("url", ""))
        if d:
            validated_domains.add(d)

        validated_source_receipts.append({
            "domain": d,
            "url": art.get("url"),
            "source": art.get("source"),
            "title": art.get("title"),
            "description": art.get("description"),
            "bias": art.get("bias", "Unknown"),
            "origin": "gdelt",
            "syndicated": False,
        })

        if art.get("bias") == "Unknown":
            unmapped.append({
                "url": art.get("url"),
                "domain": d,
                "title": art.get("title"),
            })

    cluster.setdefault("_validated_gdelt_sources", [])
    cluster["_validated_gdelt_sources"].extend(validated_source_receipts)

    families = build_gdelt_syndication_families(accepted_candidates)
    added = 0

    for family in families:
        if added >= min(MAX_GDELT_CONTENT_FAMILIES_PER_CLUSTER, remaining_family_slots):
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
        added += 1

    refresh_cluster_coverage_metadata(cluster, model=semantic_model)

    print(
        f"      candidates={len(normalized_candidates)} "
        f"validated={len(accepted_candidates)} "
        f"families_added={added} "
        f"validated_domains={len(validated_domains)} "
        f"semantic_rejected={rejected_semantic} "
        f"development_rejected={rejected_development}"
    )

    return added


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

    if args.refresh_bias_only:
        synced = sync_unmapped_domains_to_overrides()
        if synced:
            print(
                f"🧩 Added {synced} missing review domains to "
                f"{BIAS_OVERRIDES_FILE} as Unknown"
            )

        source_file = output_file if os.path.exists(output_file) else input_file
        grouped = load_json(source_file)
        clusters = grouped["clusters"] if isinstance(grouped, dict) else grouped
        for cluster in clusters:
            refresh_cluster_biases(cluster)
        save_json(output_file, grouped)
        pruned = prune_unmapped_bias_file()
        print(f"✅ Refreshed saved bias labels from bias_overrides.json → {output_file}")
        if pruned:
            print(f"🧹 Removed {pruned} categorized entries from {UNMAPPED_BIAS_FILE}")
        print("⏭️ No local recovery or GDELT requests were made.")
        return

    grouped = load_json(input_file)
    clusters = grouped["clusters"] if isinstance(grouped, dict) else grouped

    start, end = get_date_window(date_str)
    cache = load_cache()

    print(f"🧠 Loading local semantic validator: {SEMANTIC_MODEL}")
    semantic_model = SentenceTransformer(SEMANTIC_MODEL)
    cache_dirty = False

    for cluster in clusters:
        refresh_cluster_biases(cluster)
        cluster["_expansion_target_date"] = date_str
        cluster["coverage_articles"] = [
            prepare_article_bias(a) for a in cluster.get("articles", [])
        ]

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
        for cluster in clusters:
            refresh_cluster_coverage_metadata(cluster, model=semantic_model)
            cluster.pop("_expansion_target_date", None)
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
    rate_limit_count = 0
    live_requests_disabled = False

    states = []
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
        core_articles, _, _ = build_event_core(cluster, semantic_model)
        states.append({
            "idx": idx,
            "cluster": cluster,
            "queries": queries,
            "core_articles": core_articles,
            "existing_urls": existing_urls,
            "domain_counts": domain_counts,
            "processed": set(),
            "added": 0,
        })

    # Pass 1: consume every useful cached result before any live request.
    for state in states:
        for q in state["queries"]:
            if state["added"] >= TARGET_GDELT_NEW_ARTICLES_PER_CLUSTER:
                break
            ck = f"{start}|{end}|{q}"
            if ck not in cache:
                continue

            data = cache[ck]
            state["processed"].add(ck)
            print(f"    cached query='{q}' → {len(data.get('articles', []))} hits")

            remaining = TARGET_GDELT_NEW_ARTICLES_PER_CLUSTER - state["added"]
            added_now = process_gdelt_result_for_cluster(
                cluster=state["cluster"],
                data=data,
                query=q,
                core_articles=state["core_articles"],
                semantic_model=semantic_model,
                existing_urls=state["existing_urls"],
                domain_counts=state["domain_counts"],
                unmapped=unmapped,
                remaining_family_slots=remaining,
            )
            state["added"] += added_now
            gdelt_added_total += added_now

    # Pass 2: make only the live requests still needed. Two 429 responses open
    # a circuit breaker for the remainder of this run; cached data above is kept.
    for state in states:
        if state["added"] >= TARGET_GDELT_NEW_ARTICLES_PER_CLUSTER:
            print(
                f"  • Cluster {state['idx']}: +{state['added']} "
                f"GDELT content families"
            )
            continue

        for q in state["queries"]:
            if state["added"] >= TARGET_GDELT_NEW_ARTICLES_PER_CLUSTER:
                break

            ck = f"{start}|{end}|{q}"
            if ck in state["processed"] or ck in cache:
                continue

            if live_requests_disabled:
                continue
            if queries_used >= MAX_QUERIES_PER_RUN:
                print("🛑 Reached MAX_QUERIES_PER_RUN; stopping new GDELT requests.")
                live_requests_disabled = True
                break

            try:
                data = gdelt_query(q, start, end)
                cache[ck] = data
                cache_dirty = True
                queries_used += 1
                state["processed"].add(ck)
                print(f"    query='{q}' → {len(data.get('articles', []))} hits")
                time.sleep(REQUEST_SLEEP_SECONDS)

            except Exception as exc:
                if "GDELT_429" in str(exc):
                    rate_limit_count += 1
                    if rate_limit_count >= MAX_RATE_LIMITS_PER_RUN:
                        live_requests_disabled = True
                        print(
                            "⚠️ GDELT rate-limited twice; disabling further live "
                            "requests for this run. Cached results remain usable."
                        )
                    else:
                        print(
                            f"⚠️ GDELT rate-limited. Cooling down once for "
                            f"{RATE_LIMIT_COOLDOWN_SECONDS:.0f}s."
                        )
                        time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
                else:
                    print(f"⚠️ GDELT query failed: {exc}")
                continue

            remaining = TARGET_GDELT_NEW_ARTICLES_PER_CLUSTER - state["added"]
            added_now = process_gdelt_result_for_cluster(
                cluster=state["cluster"],
                data=data,
                query=q,
                core_articles=state["core_articles"],
                semantic_model=semantic_model,
                existing_urls=state["existing_urls"],
                domain_counts=state["domain_counts"],
                unmapped=unmapped,
                remaining_family_slots=remaining,
            )
            state["added"] += added_now
            gdelt_added_total += added_now

        refresh_cluster_coverage_metadata(state["cluster"], model=semantic_model)
        print(
            f"  • Cluster {state['idx']}: +{state['added']} "
            f"GDELT content families"
        )

    if live_requests_disabled and rate_limit_count >= MAX_RATE_LIMITS_PER_RUN:
        print(
            f"🛡️ GDELT circuit breaker opened after {rate_limit_count} rate limits; "
            f"{queries_used} live queries were made."
        )

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
            synced = sync_unmapped_domains_to_overrides()
            if synced:
                print(
                    f"🧩 Added {synced} missing review domains to "
                    f"{BIAS_OVERRIDES_FILE} as Unknown"
                )
        except Exception as exc:
            print(f"⚠️ Failed to write {UNMAPPED_BIAS_FILE}: {exc}")

    if cache_dirty:
        save_cache(cache)
        print(f"💾 Saved GDELT cache → {GDELT_CACHE_FILE}")

    for cluster in clusters:
        refresh_cluster_coverage_metadata(cluster, model=semantic_model)
        cluster.pop("_validated_gdelt_sources", None)
        cluster.pop("_expansion_target_date", None)

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
