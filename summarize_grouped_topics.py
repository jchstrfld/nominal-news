# summarize_grouped_topics.py — summarize top clusters with restored bias & source info + summaries cache

import json
import openai
import os
from dotenv import load_dotenv
from datetime import datetime
import argparse
import tiktoken
import difflib
import re
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np
from urllib.parse import urlparse, parse_qsl, urlencode

# Summaries cache helpers (add summaries_cache.py next to this file)
from summaries_cache import (
    load_summ_cache, save_summ_cache,
    make_summ_key, get_cached_summary, put_cached_summary
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()
IMAGES_OFF = os.getenv("NN_IMAGES_OFF", "0") == "1"

# Local semantic model for relevance checks (no OpenAI tokens)
# MiniLM is small + fast and good enough for relevance gating.
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

parser = argparse.ArgumentParser()
parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
parser.add_argument(
    "--input-file",
    type=str,
    default="",
    help=(
        "Optional cluster input path. When omitted, the existing freshness "
        "check chooses the normal expanded or final date-based file."
    ),
)
parser.add_argument(
    "--output-file",
    type=str,
    default="",
    help=(
        "Optional summaries output path. Defaults to "
        "topic_summaries_{date}.json."
    ),
)
args = parser.parse_args()

date_str = args.date or datetime.today().strftime("%Y-%m-%d")
print(f"📅 Using input date: {date_str}")

if args.input_file:
    # Explicit shadow/custom input always wins. The default production path
    # retains the existing freshness protection below.
    INPUT_FILE = args.input_file
else:
    expanded = f"grouped_articles_final_expanded_{date_str}.json"
    final_file = f"grouped_articles_final_{date_str}.json"

    # Use expanded only if it is at least as fresh as final.
    # Prevents stale expanded files from causing 1-topic summaries.
    if os.path.exists(expanded) and (
        not os.path.exists(final_file)
        or os.path.getmtime(expanded) >= os.path.getmtime(final_file)
    ):
        INPUT_FILE = expanded
    else:
        INPUT_FILE = final_file

print(f"📄 Summarizer input: {INPUT_FILE}")
OUTPUT_FILE = args.output_file or f"topic_summaries_{date_str}.json"

MIN_ARTICLES = 4
MAX_ARTICLES_PER_CLUSTER = 10
MAX_CLUSTERS = 10

MAX_TOKENS = 7000
ENCODING = tiktoken.encoding_for_model("gpt-4")

# Cache config (bump PROMPT_VERSION when you change the prompt format)
PROMPT_VERSION = "v1.1-evidence-only-2026-09-05"
SUMM_MODEL = "gpt-4"


# ----------------------------
# Cluster Topic Categories
# ----------------------------

CATEGORY_DEFS = [
    {"slug": "politics-government", "name": "Politics & Government"},
    {"slug": "global-affairs", "name": "Global Affairs"},
    {"slug": "economy-markets", "name": "Economy & Markets"},
    {"slug": "business", "name": "Business"},
    {"slug": "technology", "name": "Technology"},
    {"slug": "science-health", "name": "Science & Health"},
    {"slug": "climate-environment", "name": "Climate & Environment"},
    {"slug": "culture-society", "name": "Culture & Society"},
]

# Zero-shot classifier (local; no OpenAI tokens)
_ZS_MODEL = None
_ZS = None


def get_zero_shot():
    global _ZS, _ZS_MODEL
    if _ZS is None:
        _ZS_MODEL = os.getenv("NN_ZS_MODEL", "facebook/bart-large-mnli")
        print(f"🧠 Zero-shot model: {_ZS_MODEL}")
        _ZS = pipeline("zero-shot-classification", model=_ZS_MODEL)
    return _ZS

def select_central_articles(articles: list[dict], k: int) -> list[dict]:
    """
    Pick the k most central articles in a cluster using local embeddings (token-free).
    This improves summary quality by focusing on the semantic core of the cluster.
    """
    if not articles or k <= 0:
        return []

    if len(articles) <= k:
        return articles[:]

    texts = []
    for a in articles:
        t = (a.get("title") or "").strip()
        d = (a.get("description") or "").strip()
        d = re.sub(r"<[^>]+>", " ", d)
        d = re.sub(r"\s+", " ", d).strip()
        if d:
            d = d[:300]
        txt = (t + ". " + d).strip() if t else d
        texts.append(txt or (a.get("url") or ""))

    try:
        vecs = EMBEDDER.encode(texts, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)

        centroid = vecs.mean(axis=0, keepdims=True)
        denom = np.linalg.norm(centroid, axis=1, keepdims=True)
        centroid = centroid / np.maximum(denom, 1e-12)

        sims = (vecs @ centroid.T).reshape(-1)
        top_idx = np.argsort(sims)[::-1][:k].tolist()

        # Return in the order of "most central first" (best for summarization)
        return [articles[i] for i in top_idx]
    except Exception:
        # Fail-open: preserve current behavior
        return articles[:k]

def classify_topic_category(topic_title: str, summary: str, source_domains: list[str] | None = None) -> dict:
    """
    Token-free category classification using local zero-shot MNLI.
    Returns best + runner-up + scores; may return category=None if ambiguous.
    """
    parts = [topic_title or "", summary or ""]
    text = " ".join([p.strip() for p in parts if p and p.strip()])
    text = text[:1200]
    if source_domains:
        text = text + " Sources: " + ", ".join(source_domains[:8])

    if not text:
        return {
            "category": None,
            "category_slug": None,
            "category_score": None,
            "category_runner_up": None,
            "category_runner_up_score": None,
        }

    labels = [
        "politics and government",
        "world and international affairs",
        "economy and financial markets",
        "business and companies",
        "technology and computing",
        "science and health",
        "climate and environment",
        "culture and society",
    ]

    label_to_category = {
        "politics and government": ("Politics & Government", "politics-government"),
        "world and international affairs": ("Global Affairs", "global-affairs"),
        "economy and financial markets": ("Economy & Markets", "economy-markets"),
        "business and companies": ("Business", "business"),
        "technology and computing": ("Technology", "technology"),
        "science and health": ("Science & Health", "science-health"),
        "climate and environment": ("Climate & Environment", "climate-environment"),
        "culture and society": ("Culture & Society", "culture-society"),
    }

    zs = get_zero_shot()
    out = zs(
        text,
        labels,
        multi_label=False,
        hypothesis_template="This news topic is mainly about {}."
    )

    ranked_labels = out["labels"]
    ranked_scores = [float(s) for s in out["scores"]]

    best_label = ranked_labels[0]
    best_score = ranked_scores[0]
    runner_label = ranked_labels[1] if len(ranked_labels) > 1 else None
    runner_score = ranked_scores[1] if len(ranked_scores) > 1 else 0.0
    margin = best_score - runner_score

    # --- Top-3 runoff when the top-2 are close ---
    RUNOFF_MARGIN = 0.04
    if len(ranked_labels) >= 3 and margin < RUNOFF_MARGIN:
        top3 = ranked_labels[:3]
        out3 = zs(
            text,
            top3,
            multi_label=False,
            hypothesis_template="This news topic is mainly about {}."
        )
        ranked_labels = out3["labels"]
        ranked_scores = [float(s) for s in out3["scores"]]

        best_label = ranked_labels[0]
        best_score = ranked_scores[0]
        runner_label = ranked_labels[1] if len(ranked_labels) > 1 else runner_label
        runner_score = ranked_scores[1] if len(ranked_scores) > 1 else runner_score
        margin = best_score - runner_score

    # --- ALWAYS do a final 2-label runoff to get meaningful probabilities ---
    if runner_label:
        out2 = zs(
            text,
            [best_label, runner_label],
            multi_label=False,
            hypothesis_template="This news topic is mainly about {}."
        )
        best_label = out2["labels"][0]
        best_score = float(out2["scores"][0])
        runner_label = out2["labels"][1] if len(out2["labels"]) > 1 else runner_label
        runner_score = float(out2["scores"][1]) if len(out2["scores"]) > 1 else runner_score
        margin = best_score - runner_score

    best_name, best_slug = label_to_category[best_label]
    runner_name = label_to_category[runner_label][0] if runner_label else None

    MIN_SCORE = 0.42
    MIN_MARGIN = 0.12

    if best_score < MIN_SCORE or margin < MIN_MARGIN:
        return {
            "category": None,
            "category_slug": None,
            "category_score": round(best_score, 3),
            "category_runner_up": runner_name,
            "category_runner_up_score": round(runner_score, 3),
        }

    return {
        "category": best_name,
        "category_slug": best_slug,
        "category_score": round(best_score, 3),
        "category_runner_up": runner_name,
        "category_runner_up_score": round(runner_score, 3),
    }


# ----------------------------
# Build image queries
# ----------------------------

def build_image_query(topic_title: str, summary_text: str | None = None, max_words: int = 12) -> str:
    """
    Build a short, search-friendly image query from the topic title + summary.
    - Prioritizes the topic_title (headline).
    - Optionally adds 3–5 informative words from summary (no stopwords, no duplicates).
    - Truncates to max_words.
    """
    title = (topic_title or "").strip()
    summary = (summary_text or "").strip()

    if not title and summary:
        first_sent = re.split(r"[.!?]", summary)[0]
        title = first_sent.strip()
    if not title:
        return ""

    title_clean = re.sub(r"\s+", " ", title).strip(" .,:;–-")

    extra_words = []
    if summary:
        text = re.sub(r"[^a-z0-9\s]", " ", summary.lower())
        tokens = [t for t in text.split() if len(t) > 3]
        stopwords = {
            "this", "that", "with", "from", "about", "after", "before", "through",
            "into", "which", "their", "there", "where", "while", "being", "have",
            "has", "had", "will", "would", "should", "could", "might", "also",
            "over", "under", "between", "among", "other", "more", "most",
            "very", "just", "like", "than", "such", "many", "some", "only"
        }
        seen = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", title_clean))
        for tok in tokens:
            if tok in stopwords or tok in seen:
                continue
            extra_words.append(tok)
            if len(extra_words) >= 5:
                break

    combined = (title_clean.split() + extra_words)[:max_words]
    return " ".join(combined).strip()


def build_short_image_queries(topic_title: str, headline: str) -> list[str]:
    """
    Accuracy-first query variants.
    We DO NOT add generic coverage queries here.
    """
    def clean(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    q1 = clean(topic_title)

    h = (headline or "")
    h = re.sub(r"[\(\)\[\]\{\}]", " ", h)
    h = re.sub(r"\b\d+%?\b", " ", h)
    h = re.sub(r"[^A-Za-z0-9\s\-]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    q2 = " ".join(h.split()[:8]).strip()

    out = []
    for q in [q1, q2]:
        if q and q not in out:
            out.append(q)
    return out


def build_wikimedia_query_from_headline(headline: str, max_terms: int = 4) -> str:
    """
    Wikimedia works best with entity names (people/places/orgs).
    Pulls capitalized tokens first; falls back to first few non-stopwords.
    """
    words = re.findall(r"[A-Za-z0-9']+", (headline or ""))
    stop = {"the", "and", "or", "to", "of", "in", "on", "at", "amid", "after", "before", "with", "from", "a", "an"}
    entities = [w.strip("'") for w in words if w[:1].isupper() and w.lower() not in stop]

    if len(entities) >= 2:
        return " ".join(entities[:max_terms])

    tokens = [w.strip("'") for w in words if w.lower() not in stop]
    return " ".join(tokens[:max_terms])

def most_central_title(articles: list[dict]) -> str:
    """
    Math-only: pick the most central article title in the cluster using local embeddings.
    Returns "" if not enough signal.
    """
    titles = []
    for a in (articles or []):
        t = (a.get("title") or "").strip()
        if t:
            titles.append(t)

    if len(titles) < 2:
        return titles[0] if titles else ""

    try:
        vecs = EMBEDDER.encode(titles)
        centroid = np.mean(vecs, axis=0, keepdims=True)
        sims = cosine_similarity(vecs, centroid).reshape(-1)
        i = int(np.argmax(sims))
        return titles[i]
    except Exception:
        return titles[0] if titles else ""

def shorten_query(q: str, max_words: int = 6) -> str:
    q = (q or "").strip()
    q = re.sub(r"\b\d+%?\b", " ", q)
    q = re.sub(r"[^A-Za-z0-9\s\-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return " ".join(q.split()[:max_words]).strip()

# ----------------------------
# Relevance gating (semantic + overlap)
# ----------------------------

_STOP = {
    "the", "and", "for", "with", "from", "into", "amid", "after", "before", "over", "under", "about",
    "this", "that", "these", "those", "are", "was", "were", "been", "being", "has", "have", "had",
    "will", "would", "should", "could", "might", "also", "says", "said", "its", "their", "them",
    "a", "an", "of", "in", "on", "at", "to"
}


def keyword_overlap_ok(topic_text: str, image_text: str, min_hits: int = 2) -> bool:
    """
    Token overlap gate (#2). Computed automatically.
    Require at least min_hits overlapping meaningful tokens.
    """
    def toks(s: str) -> set:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\s]+", " ", s)
        return {t for t in s.split() if len(t) > 2 and t not in _STOP}

    T = toks(topic_text)
    I = toks(image_text)
    if not T or not I:
        return False
    return len(T & I) >= min_hits


def semantic_sim_ok(topic_text: str, image_text: str, thresh: float = 0.32) -> bool:
    """
    Semantic similarity gate (#1) using local embeddings.
    """
    if not topic_text or not image_text:
        return False
    try:
        v1 = EMBEDDER.encode([topic_text])[0]
        v2 = EMBEDDER.encode([image_text])[0]
        sim = float(cosine_similarity(np.array(v1).reshape(1, -1), np.array(v2).reshape(1, -1))[0][0])
        return sim >= thresh
    except Exception:
        return False


def image_relevance_ok(topic_text: str, image_text: str, *, sim_thresh: float = 0.32, min_kw_hits: int = 2) -> bool:
    """
    Combined relevance gate: semantic AND keyword overlap.
    """
    return semantic_sim_ok(topic_text, image_text, thresh=sim_thresh) and keyword_overlap_ok(topic_text, image_text, min_hits=min_kw_hits)


# ----------------------------
# Image fetching: Wikimedia → Unsplash → none
# ----------------------------

def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> set:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    toks = [t for t in text.split() if len(t) > 2]
    stop = {
        "the", "and", "for", "with", "from", "into", "amid", "after", "before", "over", "under", "about",
        "this", "that", "these", "those", "are", "was", "were", "been", "being", "has", "have", "had",
        "will", "would", "should", "could", "might", "also", "says", "said"
    }
    return {t for t in toks if t not in stop}


_BAD_WIKI_WORDS = {
    "book", "cover", "report", "document", "scan", "scanned", "page", "pages",
    "volume", "issue", "journal", "catalog", "brochure", "pamphlet", "poster",
    "manuscript", "handbook", "proceedings", "thesis", "pdf", "title page"
}
_BAD_WIKI_EXT = (".pdf", ".djvu", ".tif", ".tiff", ".svg")


def _is_bad_wiki_candidate(title: str, desc: str, url: str) -> bool:
    hay = f"{title} {desc}".lower()
    u = (url or "").lower()
    if any(u.endswith(ext) or f"{ext}?" in u for ext in _BAD_WIKI_EXT):
        return True
    if any(w in hay for w in _BAD_WIKI_WORDS):
        return True
    if any(title.lower().endswith(ext) for ext in _BAD_WIKI_EXT):
        return True
    return False


def _relevance_score(topic_query: str, title: str, desc: str) -> int:
    q = _tokenize(topic_query)
    t = _tokenize(title)
    d = _tokenize(desc)
    return len(q & (t | d))


def fetch_wikimedia_image(image_query: str, topic_text: str = ""):
    """
    Conservative Wikimedia fetch:
      - File namespace only (images)
      - Reject scans/docs/covers/PDF/SVG
      - Require license metadata
      - Require relevance score >= 3
      - Reject illustration/painting/engraving
      - Relevance gate uses topic_text (headline + summary)
    """
    if not image_query or IMAGES_OFF:
        return None

    headers = {
        "User-Agent": "NominalNewsBot/0.1 (contact: 221876385+jchstrfld@users.noreply.github.com)"
    }

    try:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": image_query,
            "gsrnamespace": 6,
            "gsrlimit": 10,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 600,
            "format": "json",
        }
        r = requests.get("https://commons.wikimedia.org/w/api.php", params=params, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"ℹ️ Wikimedia fetch failed: {e}")
        return None

    pages = (data.get("query") or {}).get("pages") or {}
    if not pages:
        return None

    best = None
    best_score = -1

    for page in pages.values():
        imageinfo = page.get("imageinfo") or []
        if not imageinfo:
            continue

        info = imageinfo[0]
        url = info.get("thumburl")
        if not url:
            continue

        ext = info.get("extmetadata") or {}
        desc = _clean_html((ext.get("ImageDescription") or {}).get("value") or "")
        artist = _clean_html((ext.get("Artist") or {}).get("value") or "")
        license_short = _clean_html((ext.get("LicenseShortName") or {}).get("value") or "")
        license_url = _clean_html((ext.get("LicenseUrl") or {}).get("value") or "")
        title = page.get("title") or ""

        if not (license_short or license_url):
            continue

        if _is_bad_wiki_candidate(title, desc, url):
            continue

        hay = f"{title} {desc}".lower()
        if "illustration" in hay or "painting" in hay or "engraving" in hay:
            continue

        score = _relevance_score(image_query, title, desc)
        if score < 3:
            continue

        source_url = f"https://commons.wikimedia.org/?curid={page.get('pageid')}"

        alt = (desc or image_query).strip()
        alt = re.sub(r"\s+", " ", alt)

        credit_bits = []
        if artist:
            credit_bits.append(artist)
        credit_bits.append("Wikimedia Commons")
        if license_short:
            credit_bits.append(license_short)
        credit = "Photo: " + " / ".join(credit_bits)

        candidate = {
            "url": url,
            "alt": alt,
            "description": desc or alt,
            "credit": credit,
            "source": "wikimedia",
            "source_url": source_url,
            "license": license_short or "Wikimedia license"
        }

        topic_text_clean = (topic_text or image_query).strip()
        image_text = f"{title} {desc}".strip()
        if not image_relevance_ok(topic_text_clean, image_text, sim_thresh=0.30, min_kw_hits=1):
            continue

        if score > best_score:
            best = candidate
            best_score = score

    return best


def build_unsplash_query_variants(image_query: str, topic_text: str, headline: str = "", max_words: int = 5) -> list[str]:
    """
    Build 3 query variants:
      1) original image_query
      2) token-trimmed headline tokens (automatic)
      3) compressed frequent tokens from topic_text
    """
    def toks(s: str) -> list[str]:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\s]+", " ", s)
        return [w for w in s.split() if len(w) > 2 and w not in _STOP]

    q1 = (image_query or "").strip()

    h_words = toks(headline)
    q2 = " ".join(h_words[:max_words]).strip()

    t_words = toks(topic_text)
    freq = {}
    for w in t_words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    q3 = " ".join([w for w, _ in top[:max_words]]).strip()

    out = []
    for q in [q1, q2, q3]:
        q = (q or "").strip()
        if q and q not in out:
            out.append(q)
    return out


def fetch_unsplash_image(image_query: str, topic_text: str = "", headline: str = "", core_text: str = ""):
    """
    Unsplash fallback (accuracy-first):
      - Try query variants
      - Score candidate image_text against:
          sim_topic = cos(emb(topic_text), emb(image_text))
          sim_query = cos(emb(query),      emb(image_text))
      - Accept only if both pass thresholds.
      - If ambiguous AND not strong, drop (prefer no image to wrong image).
    """
    if IMAGES_OFF:
        return None
    if not UNSPLASH_ACCESS_KEY:
        return None

    topic_text_clean = (topic_text or image_query or "").strip()
    if not topic_text_clean:
        return None

    queries = build_unsplash_query_variants(
        image_query=image_query,
        topic_text=topic_text_clean,
        headline=headline,
        max_words=6
    )
    if not queries:
        return None

    try:
        topic_vec = EMBEDDER.encode([topic_text_clean])[0]
    except Exception:
        return None
    
    core_vec = None
    core_text_clean = (core_text or "").strip()
    if core_text_clean:
        try:
            core_vec = EMBEDDER.encode([core_text_clean])[0]
        except Exception:
            core_vec = None

    def _cos(a, b) -> float:
        return float(cosine_similarity(np.array(a).reshape(1, -1), np.array(b).reshape(1, -1))[0][0])

    # Tuned to keep Unsplash present while filtering obvious mismatches
    SIM_TOPIC = 0.31
    SIM_QUERY = 0.30
    MARGIN = 0.020
    SIM_TOPIC_RELAX = 0.29
    SIM_QUERY_RELAX = 0.27

    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}

    best = None
    best_score = -1.0
    second_best_score = -1.0
    
    found_any = False

    for q in queries:
        params = {"query": q, "per_page": 8, "orientation": "landscape"}
        try:
            r = requests.get("https://api.unsplash.com/search/photos", headers=headers, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        results = data.get("results") or []
        if not results:
            continue

        try:
            q_vec = EMBEDDER.encode([q])[0]
        except Exception:
            q_vec = None

        for photo in results:
            urls = photo.get("urls") or {}
            raw = urls.get("raw")
            if raw:
                url = f"{raw}&w=900&fit=max&q=80"
            else:
                url = urls.get("regular") or urls.get("small")
            if not url:
                continue

            alt = (photo.get("alt_description") or photo.get("description") or q).strip()
            alt = re.sub(r"\s+", " ", alt)
            desc = (photo.get("description") or photo.get("alt_description") or alt).strip()
            desc = re.sub(r"\s+", " ", desc)

            image_text = f"{alt} {desc}".strip()

            try:
                img_vec = EMBEDDER.encode([image_text])[0]
            except Exception:
                continue

            sim_topic = _cos(topic_vec, img_vec)
            sim_query = _cos(q_vec, img_vec) if q_vec is not None else sim_topic

            # Compute core similarity BEFORE using it
            sim_core = sim_topic
            if core_vec is not None:
                sim_core = _cos(core_vec, img_vec)

            strict_ok = (sim_topic >= SIM_TOPIC and sim_query >= SIM_QUERY)
            relax_ok = (sim_core >= 0.36 and sim_topic >= SIM_TOPIC_RELAX and sim_query >= SIM_QUERY_RELAX)

            if not (strict_ok or relax_ok):
                continue

            # Allow a slightly weaker query match when the cluster's semantic core is strong
            if sim_query < SIM_QUERY and sim_core >= 0.36:
                sim_query = SIM_QUERY

            # Prefer matching the cluster's semantic core (core title) more than headline framing.
            score = (0.50 * sim_core) + (0.35 * sim_topic) + (0.15 * sim_query)

            if score > best_score:
                second_best_score = best_score
                best_score = score

                user = photo.get("user") or {}
                photographer = (user.get("name") or "").strip()
                link_html = ((photo.get("links") or {}).get("html") or "").strip()

                credit = "Photo: "
                credit += f"{photographer} / Unsplash" if photographer else "Unsplash"

                best = {
                    "url": url,
                    "alt": alt,
                    "description": desc,
                    "credit": credit,
                    "source": "unsplash",
                    "source_url": link_html,
                    "license": "Unsplash License"
                }
            elif score > second_best_score:
                second_best_score = score

    # Drop only if ambiguous AND not strong
    if best is not None:
        if (best_score - second_best_score) < MARGIN and best_score < (SIM_TOPIC + 0.05):
            return None

    return best


# ----------------------------
# Everything below here is your original non-image logic
# ----------------------------

BIAS_LABELS = ["Far Left", "Left", "Center", "Right", "Far Right", "Unknown"]


def normalize_bias_label(raw):
    value = (raw or "Unknown").strip().lower().replace("_", "-")
    aliases = {
        "far-left": "Far Left",
        "far left": "Far Left",
        "left": "Left",
        "center-left": "Left",
        "lean-left": "Left",
        "lean left": "Left",
        "center": "Center",
        "right": "Right",
        "center-right": "Right",
        "lean-right": "Right",
        "lean right": "Right",
        "far-right": "Far Right",
        "far right": "Far Right",
        "unknown": "Unknown",
        "uncategorized": "Unknown",
    }
    return aliases.get(value, "Unknown")


def compute_bias_counts(articles):
    """Exact outlet counts, including uncategorized/Unknown domains."""
    counts = {label: 0 for label in BIAS_LABELS}
    for article in articles:
        counts[normalize_bias_label(article.get("bias"))] += 1
    return {label: count for label, count in counts.items() if count > 0}


def compute_bias_distribution(articles):
    """Largest-remainder percentages over every displayed outlet; sums to 100."""
    counts = compute_bias_counts(articles)
    total = sum(counts.values())
    if total <= 0:
        return {}

    exact = {label: counts.get(label, 0) * 100.0 / total for label in BIAS_LABELS}
    whole = {label: int(exact[label]) for label in BIAS_LABELS}
    leftover = 100 - sum(whole.values())

    ranked_remainders = sorted(
        BIAS_LABELS,
        key=lambda label: (exact[label] - whole[label], -BIAS_LABELS.index(label)),
        reverse=True,
    )
    for label in ranked_remainders[:leftover]:
        whole[label] += 1

    return {label: whole[label] for label in BIAS_LABELS if whole[label] > 0}


def format_article(a):
    title = (a.get("title") or "")[:200].strip()
    desc = (a.get("description") or "")[:300].strip()
    return f"{title}. {desc}"


def truncate_prompt(prompt, max_tokens):
    tokens = ENCODING.encode(prompt or "")
    if len(tokens) <= max_tokens:
        return prompt or ""
    return ENCODING.decode(tokens[:max_tokens])


def build_display_sources(cluster, coverage_articles):
    """
    Build one display/source-balance receipt per unique outlet domain.

    Prefer the new cluster-level coverage_sources receipts when available so
    syndicated GDELT outlets remain visible even when their duplicate content
    is collapsed out of coverage_articles. Fall back to coverage_articles for
    older expanded files.
    """
    receipts = cluster.get("coverage_sources") or []

    if not receipts:
        receipts = []
        for a in coverage_articles:
            url = a.get("url")
            if not url:
                continue
            from urllib.parse import urlparse
            domain = (urlparse(url).hostname or "").lower()
            if domain.startswith("www."):
                domain = domain[4:]
            receipts.append({
                "domain": domain,
                "url": url,
                "source": a.get("source") or a.get("source_name") or domain,
                "bias": a.get("bias", "Unknown"),
                "origin": "coverage",
                "syndicated": False,
            })

    origin_rank = {"core": 0, "local": 1, "gdelt": 2, "coverage": 3}

    by_domain = {}
    for r in receipts:
        domain = (r.get("domain") or "").lower().strip()
        if not domain:
            continue

        candidate = dict(r)
        candidate["source_name"] = (
            r.get("source")
            or r.get("source_name")
            or domain
        )

        current = by_domain.get(domain)
        if current is None:
            by_domain[domain] = candidate
            continue

        candidate_key = (
            origin_rank.get(candidate.get("origin"), 9),
            bool(candidate.get("syndicated")),
            candidate.get("bias") in (None, "", "Unknown"),
        )
        current_key = (
            origin_rank.get(current.get("origin"), 9),
            bool(current.get("syndicated")),
            current.get("bias") in (None, "", "Unknown"),
        )
        if candidate_key < current_key:
            by_domain[domain] = candidate

    return list(by_domain.values())



def _coverage_domain(url):
    try:
        host = (urlparse(url or "").hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _coverage_canonical_url(url):
    try:
        parsed = urlparse((url or "").strip())
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return ""
        path = re.sub(r"/+$", "", parsed.path or "")
        tracking = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term",
            "utm_content", "gclid", "fbclid", "mc_cid", "mc_eid",
            "igshid", "ref", "ref_src",
        }
        kept = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in tracking
        ]
        query = urlencode(sorted(kept))
        base = f"{host}{path}"
        return f"{base}?{query}" if query else base
    except Exception:
        return (url or "").strip().lower()


def _coverage_normalize_title(title):
    text = re.sub(r"<[^>]+>", " ", title or "")
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _coverage_title_score(left, right):
    a = _coverage_normalize_title(left)
    b = _coverage_normalize_title(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    aset = set(a.split())
    bset = set(b.split())
    jaccard = len(aset & bset) / max(1, len(aset | bset))
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    return max(jaccard, seq)


def _coverage_title_match(left, right):
    return _coverage_title_score(left, right) >= 0.94 or (
        len(set(_coverage_normalize_title(left).split()) | set(_coverage_normalize_title(right).split())) > 0
        and (
            len(set(_coverage_normalize_title(left).split()) & set(_coverage_normalize_title(right).split()))
            / max(1, len(set(_coverage_normalize_title(left).split()) | set(_coverage_normalize_title(right).split())))
        ) >= 0.90
    )


def _coverage_receipt(record, default_origin="coverage"):
    url = (record.get("url") or "").strip()
    if not url:
        return None
    domain = (record.get("domain") or _coverage_domain(url)).lower().strip()
    if not domain:
        return None
    source_name = record.get("source") or record.get("source_name") or domain
    try:
        declared = int(
            record.get("writeup_family_outlet_count")
            or record.get("gdelt_writeup_family_outlet_count")
            or 0
        )
    except Exception:
        declared = 0
    return {
        "domain": domain,
        "url": url,
        "canonical_url": _coverage_canonical_url(url),
        "source_name": str(source_name).strip() or domain,
        "title": (record.get("title") or "").strip(),
        "bias": normalize_bias_label(record.get("bias")),
        "origin": (record.get("origin") or default_origin).strip(),
        "syndicated": bool(record.get("syndicated")),
        "day_basis": (record.get("day_basis") or "").strip(),
        "displayable_english": record.get("displayable_english", True) is not False,
        "writeup_family_id": (
            record.get("writeup_family_id")
            or record.get("gdelt_writeup_family_id")
            or ""
        ),
        "declared_family_outlet_count": declared,
    }


def _coverage_receipt_preference(receipt):
    origin_rank = {
        "core": 0,
        "local": 1,
        "gdelt_discovery": 2,
        "coverage": 3,
        "gdelt": 4,
        "gdelt_gsg": 5,
    }
    return (
        origin_rank.get(receipt.get("origin"), 9),
        0 if receipt.get("day_basis") == "TARGET_EXPLICIT" else 1,
        1 if receipt.get("syndicated") else 0,
        1 if receipt.get("bias") == "Unknown" else 0,
        1 if receipt.get("source_name") == receipt.get("domain") else 0,
        -len(receipt.get("title") or ""),
    )


def _coverage_bias_slug(label):
    return {
        "Far Left": "far-left",
        "Left": "left",
        "Center": "center",
        "Right": "right",
        "Far Right": "far-right",
        "Unknown": "unknown",
    }.get(normalize_bias_label(label), "unknown")


def build_coverage_writeups(cluster, coverage_articles):
    """
    Build one primary row per distinct article/write-up and nest every known
    outlet copy beneath that row. Political-bias metrics remain outlet-based
    elsewhere; this structure is for transparent article browsing only.
    """
    family_meta = {
        str(row.get("family_id")): row
        for row in (cluster.get("coverage_report_families") or [])
        if row.get("family_id")
    }
    groups = {}
    url_to_group = {}

    def ensure_group(key, *, local_family_id="", global_family_id="", title=""):
        group = groups.setdefault(key, {
            "key": key,
            "local_family_ids": set(),
            "global_family_ids": set(),
            "titles": [],
            "origins": set(),
            "receipts_by_domain": {},
            "declared_outlet_count": 0,
        })
        if local_family_id:
            group["local_family_ids"].add(str(local_family_id))
        if global_family_id:
            group["global_family_ids"].add(str(global_family_id))
        if title and title not in group["titles"]:
            group["titles"].append(title)
        return group

    def add_receipt(key, receipt):
        if receipt is None or not receipt.get("displayable_english", True):
            return
        group = groups[key]
        domain = receipt["domain"]
        current = group["receipts_by_domain"].get(domain)
        if current is None or _coverage_receipt_preference(receipt) < _coverage_receipt_preference(current):
            group["receipts_by_domain"][domain] = receipt
        if receipt.get("canonical_url"):
            url_to_group[receipt["canonical_url"]] = key
        title = receipt.get("title") or ""
        if title and title not in group["titles"]:
            group["titles"].append(title)
        group["origins"].add(receipt.get("origin") or "coverage")
        group["declared_outlet_count"] = max(
            group["declared_outlet_count"],
            int(receipt.get("declared_family_outlet_count") or 0),
        )

    # Local/cross-origin families already computed by the expansion stage.
    for idx, article in enumerate(coverage_articles or []):
        receipt = _coverage_receipt(article, "coverage")
        if receipt is None:
            continue
        local_id = (
            article.get("content_family_id")
            or article.get("report_family_id")
            or f"url-{receipt['canonical_url'] or idx}"
        )
        key = f"local:{local_id}"
        meta = family_meta.get(str(local_id)) or {}
        group = ensure_group(
            key,
            local_family_id=str(local_id),
            title=(meta.get("representative_title") or receipt.get("title") or ""),
        )
        for origin in meta.get("origins") or []:
            group["origins"].add(origin)
        add_receipt(key, receipt)

    def best_local_title_group(title):
        best_key = None
        best_score = 0.0
        for key, group in groups.items():
            # Never collapse one GSG family into another merely because their
            # headlines resemble each other. Title matching is only a bridge
            # from a GSG family to an already-computed local family.
            if not group.get("local_family_ids"):
                continue
            for existing in group.get("titles") or []:
                score = _coverage_title_score(title, existing)
                if _coverage_title_match(title, existing) and score > best_score:
                    best_key = key
                    best_score = score
        return best_key

    attached_global_families = list(
        cluster.get("gdelt_global_writeup_families") or []
    )

    if attached_global_families:
        for family in attached_global_families:
            family_id = str(family.get("family_id") or "").strip()
            if not family_id:
                continue
            receipts = [
                _coverage_receipt(row, "gdelt_gsg")
                for row in (family.get("receipts") or [])
            ]
            receipts = [row for row in receipts if row is not None]

            rep_url = (family.get("representative_url") or "").strip()
            rep_title = (family.get("representative_title") or "").strip()
            if rep_url and not any(
                row.get("canonical_url") == _coverage_canonical_url(rep_url)
                for row in receipts
            ):
                synthetic = _coverage_receipt({
                    "url": rep_url,
                    "title": rep_title,
                    "source": _coverage_domain(rep_url),
                    "bias": "Unknown",
                    "origin": "gdelt_gsg",
                    "writeup_family_id": family_id,
                    "writeup_family_outlet_count": family.get("outlet_count"),
                    "displayable_english": family.get("displayable_english", True),
                }, "gdelt_gsg")
                if synthetic:
                    receipts.append(synthetic)

            key = None
            for receipt in receipts:
                key = url_to_group.get(receipt.get("canonical_url"))
                if key:
                    break
            if key is None and rep_title:
                key = best_local_title_group(rep_title)
            if key is None:
                key = f"gsg:{family_id}"
                ensure_group(
                    key,
                    global_family_id=family_id,
                    title=rep_title,
                )
            else:
                groups[key]["global_family_ids"].add(family_id)
                if rep_title and rep_title not in groups[key]["titles"]:
                    groups[key]["titles"].append(rep_title)

            groups[key]["declared_outlet_count"] = max(
                groups[key]["declared_outlet_count"],
                int(family.get("outlet_count", 0) or 0),
            )
            for receipt in receipts:
                add_receipt(key, receipt)
    else:
        # Backward-compatible fallback for receipt files created before the
        # family-level catalog was attached.
        rows_by_family = {}
        for raw in cluster.get("gdelt_global_source_receipts") or []:
            receipt = _coverage_receipt(raw, "gdelt_gsg")
            if receipt is None or not receipt.get("writeup_family_id"):
                continue
            rows_by_family.setdefault(
                str(receipt["writeup_family_id"]), []
            ).append(receipt)

        for family_id, receipts in rows_by_family.items():
            key = None
            for receipt in receipts:
                key = url_to_group.get(receipt.get("canonical_url"))
                if key:
                    break
            if key is None:
                key = best_local_title_group(receipts[0].get("title") or "")
            if key is None:
                key = f"gsg:{family_id}"
                ensure_group(
                    key,
                    global_family_id=family_id,
                    title=receipts[0].get("title") or "",
                )
            else:
                groups[key]["global_family_ids"].add(family_id)
            for receipt in receipts:
                add_receipt(key, receipt)

    # Add local/DOC outlet receipts. Global GSG receipts were handled above.
    for raw in cluster.get("coverage_sources") or []:
        if raw.get("gdelt_gsg_receipt") and attached_global_families:
            continue
        receipt = _coverage_receipt(raw, "coverage")
        if receipt is None:
            continue
        key = url_to_group.get(receipt.get("canonical_url"))
        if key is None:
            # Map syndicated copies to a known family conservatively by title.
            best_key = None
            best_score = 0.0
            for candidate_key, group in groups.items():
                for existing in group.get("titles") or []:
                    score = _coverage_title_score(receipt.get("title") or "", existing)
                    if _coverage_title_match(receipt.get("title") or "", existing) and score > best_score:
                        best_key = candidate_key
                        best_score = score
            key = best_key
        if key is None:
            key = f"source:{receipt.get('canonical_url') or len(groups)}"
            ensure_group(key, title=receipt.get("title") or "")
        add_receipt(key, receipt)

    writeups = []
    for key, group in groups.items():
        receipts = list(group["receipts_by_domain"].values())
        if not receipts:
            continue
        receipts.sort(key=lambda row: (
            _coverage_receipt_preference(row),
            (row.get("source_name") or "").lower(),
            row.get("domain") or "",
        ))
        representative = receipts[0]
        other_outlets = sorted(
            receipts[1:],
            key=lambda row: (
                BIAS_LABELS.index(normalize_bias_label(row.get("bias"))),
                (row.get("source_name") or "").lower(),
                row.get("domain") or "",
            ),
        )
        title = (
            representative.get("title")
            or next((t for t in group.get("titles") or [] if t), "")
            or representative.get("source_name")
        )
        origins = sorted(origin for origin in group["origins"] if origin)
        global_ids = sorted(group["global_family_ids"])
        local_ids = sorted(group["local_family_ids"])
        family_id = global_ids[0] if global_ids else (
            local_ids[0] if local_ids else key
        )
        bias = normalize_bias_label(representative.get("bias"))
        writeups.append({
            "family_id": family_id,
            "local_family_ids": local_ids,
            "global_family_ids": global_ids,
            "title": title,
            "url": representative.get("url"),
            "source_name": representative.get("source_name"),
            "domain": representative.get("domain"),
            "bias": bias,
            "bias_slug": _coverage_bias_slug(bias),
            "origin": representative.get("origin"),
            "origins": origins,
            "global_family": bool(global_ids),
            "republisher_count": len(receipts),
            "additional_outlet_count": max(0, len(receipts) - 1),
            "declared_outlet_count": max(
                len(receipts), int(group.get("declared_outlet_count") or 0)
            ),
            "other_outlets": [
                {
                    "source_name": row.get("source_name"),
                    "domain": row.get("domain"),
                    "url": row.get("url"),
                    "bias": normalize_bias_label(row.get("bias")),
                    "bias_slug": _coverage_bias_slug(row.get("bias")),
                }
                for row in other_outlets
            ],
        })

    writeups.sort(key=lambda row: (
        -int(row.get("republisher_count") or 0),
        0 if "core" in (row.get("origins") or []) else 1,
        0 if "local" in (row.get("origins") or []) else 1,
        (row.get("title") or "").lower(),
    ))
    return writeups


def make_html_chips(articles):
    color_map = {
        "Far Left": "#0B36B8",
        "Left": "#275BF5",
        "Center": "#894AB3",
        "Right": "#EB4040",
        "Far Right": "#C43D31",
        "Unknown": "#C6C6C6"
    }
    bias_order = ["Far Left", "Left", "Center", "Right", "Far Right", "Unknown"]

    def bias_sort_key(article):
        bias = normalize_bias_label(article.get("bias"))
        return (bias_order.index(bias), (article.get("source_name") or "").lower())

    sorted_articles = sorted(articles, key=bias_sort_key)

    chips = []
    for a in sorted_articles:
        url = a.get("url", "#")
        source = a.get("source_name") or url.split("//")[-1].split("/")[0]
        bias = normalize_bias_label(a.get("bias"))
        bias_color = color_map[bias]
        bias_color_20 = bias_color + "33"
        chips.append(
            f'<a href="{url}" target="_blank" class="chip" '
            f'style="background-color: {bias_color_20}; border-color: {bias_color};">{source}</a>'
        )
    return " ".join(chips)


def summarize_cluster(articles):
    text_block = "\n\n".join([format_article(a) for a in articles])
    prompt = f"""
You are a neutral, evidence-bound news editor.

Use only facts explicitly stated in the supplied articles. Do not add background
knowledge, inferred motives, unsupported causes, assumed consequences, or facts
from memory. Do not claim a condition is ongoing unless an article explicitly
says so. If the articles disagree or remain uncertain, state that uncertainty.

All output must describe the single event shared by these articles. Provide:
1. A short, factual headline
2. A factual summary in 3-4 sentences

Articles:
{text_block}

Respond in this exact format:
Headline: <headline>
Summary: <summary>
"""
    prompt = truncate_prompt(prompt.strip(), MAX_TOKENS)

    try:
        response = openai.ChatCompletion.create(
            model=SUMM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write neutral news copy using only the evidence supplied "
                        "by the user. Never introduce outside facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"⚠️ Failed to summarize cluster: {e}")
        return None


def extract_headline(summary_text):
    for line in (summary_text or "").splitlines():
        if line.lower().startswith("headline:"):
            return line.replace("Headline:", "").strip()
    return "Untitled"


def extract_body(summary_text):
    text = (summary_text or "").strip()
    if not text:
        return ""

    lines = text.splitlines()
    body_lines = []
    capturing = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("summary:"):
            capturing = True
            remainder = stripped.split(":", 1)[1].strip()
            if remainder:
                body_lines.append(remainder)
            continue
        if capturing:
            body_lines.append(stripped)

    if body_lines:
        return " ".join(line for line in body_lines if line).strip()

    # Fail-open for an unexpected model format: remove a leading headline label
    # but keep the remaining evidence-bound copy.
    return re.sub(r"(?is)^\s*headline\s*:\s*[^\n]+\n?", "", text).strip()


# --- Safety valve: collapse near-duplicate topics after summarization (no tokens) ---
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "in", "on", "at", "over", "under", "after", "before",
    "with", "without", "by", "from", "as", "about", "into", "during", "including", "until", "against", "among",
    "between", "through", "because", "so", "since", "due", "has", "have", "had", "is", "was", "are", "were",
    "be", "been", "being", "will", "would", "should", "may", "might", "can", "could"
}


def _norm_title_tokens(text: str):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    toks = [w for w in text.split() if len(w) > 2 and w not in _STOPWORDS]
    return toks


def _jaccard(a, b):
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _overlap_ratio(urls_a, urls_b):
    s1 = set(urls_a or [])
    s2 = set(urls_b or [])
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / max(1, min(len(s1), len(s2)))


def _seq_ratio(a: str, b: str):
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def _looks_like_duplicate(x: dict, y: dict, url_thresh: float, title_thresh: float, body_thresh: float):
    title_x = x.get("topic_title") or ""
    title_y = y.get("topic_title") or ""
    body_x = (x.get("summary") or "")[:600]
    body_y = (y.get("summary") or "")[:600]
    urls_x = x.get("sources") or []
    urls_y = y.get("sources") or []

    url_ov = _overlap_ratio(urls_x, urls_y)
    t_ratio = _seq_ratio(title_x.lower(), title_y.lower())
    t_jacc = _jaccard(_norm_title_tokens(title_x), _norm_title_tokens(title_y))
    b_ratio = _seq_ratio(body_x.lower(), body_y.lower())

    return (url_ov >= url_thresh) and ((t_ratio >= title_thresh) or (t_jacc >= 0.70) or (b_ratio >= body_thresh))


def dedupe_topic_summaries(items, url_overlap=0.50, title_sim=0.86, body_sim=0.88):
    result = []
    for cand in items:
        merged = False
        for kept in result:
            if _looks_like_duplicate(kept, cand, url_overlap, title_sim, body_sim):
                merged_sources = list({*(kept.get("sources") or []), *(cand.get("sources") or [])})
                kept["sources"] = merged_sources
                kept["independent_report_count"] = max(
                    int(kept.get("independent_report_count") or 0),
                    int(cand.get("independent_report_count") or 0),
                )
                kept["distinct_report_count"] = max(
                    int(kept.get("distinct_report_count") or 0),
                    int(cand.get("distinct_report_count") or 0),
                )
                # Display chips/counts remain those of the higher-ranked kept card.
                merged = True
                break
        if not merged:
            result.append(cand)
    return result[:MAX_CLUSTERS]


# ----------------------------
# Load clusters and run
# ----------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    clusters = json.load(f)

valid_clusters = [c for c in clusters if len(c["articles"]) >= MIN_ARTICLES]

# IMPORTANT:
# final_cohesion_check.py already writes clusters in final importance order.
# Preserve that order here. Re-sorting by verified-core article count would
# overwrite the ranking decision and make the webpage order misleading.
top_clusters = valid_clusters[:MAX_CLUSTERS]

# load summaries cache once
summ_cache = load_summ_cache()
summ_cache_dirty = False

summaries = []

for idx, cluster in enumerate(top_clusters):
    print(f"🧠 Summarizing topic {idx + 1}/{len(top_clusters)} with {len(cluster['articles'])} articles")

    # Verified core drives the summary.
    core_articles = cluster["articles"]

    # Expanded content stays separate from the complete outlet receipt list.
    coverage_articles = cluster.get("coverage_articles") or core_articles
    display_sources = build_display_sources(cluster, coverage_articles)
    coverage_writeups = build_coverage_writeups(cluster, coverage_articles)

    selected_articles = select_central_articles(
        core_articles,
        MAX_ARTICLES_PER_CLUSTER
    )

    # Cache key should be stable for the cluster (use ALL URLs, not just the selected subset)
    all_urls = sorted([a.get("url") for a in core_articles if a.get("url")])
    cache_key = make_summ_key(all_urls, SUMM_MODEL, PROMPT_VERSION, MAX_ARTICLES_PER_CLUSTER)

    cached = get_cached_summary(summ_cache, cache_key)
    if cached:
        print(f"💾 Cache hit for topic {idx + 1} — reused headline/summary")
        headline = cached["headline"]
        body = cached["summary"]
    else:
        summary_text = summarize_cluster(selected_articles)
        if not summary_text:
            continue
        headline = extract_headline(summary_text)
        body = extract_body(summary_text)
        # summaries_cache.py keeps a legacy takeaways slot for compatibility.
        # Store an empty list; no takeaways are generated or consumed.
        put_cached_summary(summ_cache, cache_key, headline, body, [])
        summ_cache_dirty = True

    all_articles = core_articles

    # Build image queries
    image_query = build_image_query(headline, body)
    image_query_wiki = build_wikimedia_query_from_headline(headline)
    topic_text = f"{headline}. {body}".strip()

    # Add a math-only representative title query from the cluster itself
    central_title = most_central_title(all_articles if "all_articles" in locals() else cluster.get("articles", []))

    seed_queries = [
        image_query_wiki,
        *build_short_image_queries(topic_title=headline, headline=headline),
        image_query,
        central_title,
        shorten_query(central_title, max_words=6),
    ]

    query_candidates = []
    for q in seed_queries:
        q = (q or "").strip()
        if q and q not in query_candidates:
            query_candidates.append(q)

    # Wikimedia first
    img = None
    for q in query_candidates:
        img = fetch_wikimedia_image(q, topic_text=topic_text)
        if img:
            print(f"🖼️ Topic {idx+1}: Wikimedia hit (query='{q}')")
            break
    if not img:
        print(f"🖼️ Topic {idx+1}: Wikimedia miss (tried {len(query_candidates)} queries)")

    # Unsplash fallback
    if not img:
        for q in query_candidates:
            img = fetch_unsplash_image(q, topic_text=topic_text, headline=headline, core_text=central_title)
            if img:
                print(f"🖼️ Topic {idx+1}: Unsplash hit (query='{q}')")
                break
        if not img:
            print(f"🖼️ Topic {idx+1}: Unsplash miss (tried {len(query_candidates)} queries)")

    image_url = img["url"] if img else ""
    image_alt = img["alt"] if img else ""
    image_description = img["description"] if img else ""
    image_credit = img["credit"] if img else ""
    image_source = img["source"] if img else ""
    image_source_url = img["source_url"] if img else ""
    image_license = img["license"] if img else ""

    # Bias/source breadth is outlet-based: one receipt per unique domain.
    bias_counts = compute_bias_counts(display_sources)
    bias_dist = compute_bias_distribution(display_sources)
    if not bias_dist:
        print(f"⚠️ Cluster {idx} has no bias_distribution field")

    source_domains = sorted({
        s.get("domain")
        for s in display_sources
        if s.get("domain")
    })

    cat = classify_topic_category(headline, body, source_domains)

    # Local report families and GDELT's global coverage measurements are
    # intentionally separate. The source list may include both local receipts
    # and GSG receipts, so its linked-source count can exceed the bounded GSG
    # global-outlet count.
    try:
        independent_report_count = int(
            cluster.get("distinct_report_count")
            or cluster.get("independent_report_count")
            or len(coverage_articles)
        )
    except Exception:
        independent_report_count = len(coverage_articles)

    linked_source_count = len(display_sources)
    coverage_writeup_count = len(coverage_writeups)
    coverage_writeup_link_count = sum(
        int(row.get("republisher_count") or 0)
        for row in coverage_writeups
    )
    global_linked_writeup_count = sum(
        1 for row in coverage_writeups if row.get("global_family")
    )
    global_coverage = cluster.get("gdelt_global_coverage") or {}
    bridge = cluster.get("gdelt_receipt_bridge") or {}

    def _nonnegative_int(value):
        try:
            return max(0, int(value or 0))
        except Exception:
            return 0

    global_outlet_count = _nonnegative_int(
        global_coverage.get("global_outlet_count")
        or cluster.get("global_coverage_outlet_count")
    )
    global_unique_writeup_count = _nonnegative_int(
        global_coverage.get("global_unique_writeup_count")
        or cluster.get("global_unique_writeup_count")
    )
    global_english_source_count = _nonnegative_int(
        global_coverage.get("english_source_receipt_count")
        or cluster.get("global_english_source_count")
    )
    global_non_english_source_count = _nonnegative_int(
        global_coverage.get("non_english_source_receipt_count")
        or cluster.get("global_non_english_source_count")
    )
    global_coverage_available = bool(
        bridge.get("status") == "ATTACHED"
        and global_coverage.get("receipt_catalog_complete") is True
        and global_outlet_count > 0
        and global_unique_writeup_count > 0
    )

    summaries.append({
        "topic_title": headline,
        "summary": body,
        "bias_distribution": bias_dist,
        "bias_counts": bias_counts,
        # Keep article URLs for the existing topic-dedupe safety valve, but
        # display/count the complete unique outlet set from coverage_sources.
        "sources": [a.get("url") for a in coverage_articles if a.get("url")],
        "num_sources": linked_source_count,
        "html_chips": make_html_chips(display_sources),
        "coverage_outlet_count": linked_source_count,
        "linked_source_count": linked_source_count,
        "bias_outlet_count": linked_source_count,
        # Structured article-level transparency: one row per distinct write-up,
        # with syndicated outlet copies nested beneath it. Bias remains counted
        # once per outlet domain through display_sources above.
        "coverage_writeups": coverage_writeups,
        "coverage_writeup_count": coverage_writeup_count,
        "coverage_writeup_link_count": coverage_writeup_link_count,
        "global_linked_writeup_count": global_linked_writeup_count,
        "independent_report_count": independent_report_count,
        "distinct_report_count": independent_report_count,
        "global_coverage_available": global_coverage_available,
        "global_coverage_outlet_count": global_outlet_count,
        "global_unique_writeup_count": global_unique_writeup_count,
        "global_english_source_count": global_english_source_count,
        "global_non_english_source_count": global_non_english_source_count,
        "global_coverage_candidate_id": global_coverage.get("candidate_id", ""),
        "coverage_count_scope": (
            "GDELT_GLOBAL_SAMPLE" if global_coverage_available else "COLLECTED_SOURCES"
        ),

        "image_query": image_query,

        "image_url": image_url,
        "image_alt": image_alt,
        "image_description": image_description,
        "image_credit": image_credit,
        "image_credit_short": (image_credit[:80] + "…") if image_credit and len(image_credit) > 80 else image_credit,
        "image_source": image_source,
        "image_source_url": image_source_url,
        "image_license": image_license,
        "image_width": 600,
        "image_max_width": 600,

        "category": cat["category"],
        "category_slug": cat["category_slug"],
        "category_score": cat["category_score"],
        "category_runner_up": cat["category_runner_up"],
        "category_runner_up_score": cat["category_runner_up_score"],
    })

# Safety valve — de-duplicate near-identical topic cards before writing JSON
summaries = dedupe_topic_summaries(
    summaries,
    url_overlap=0.50,
    title_sim=0.86,
    body_sim=0.88
)

# Save cache if we wrote anything
if summ_cache_dirty:
    save_summ_cache(summ_cache)

tmp_file = OUTPUT_FILE + ".tmp"
with open(tmp_file, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)

os.replace(tmp_file, OUTPUT_FILE)
print(f"✅ Saved top summaries to {OUTPUT_FILE}")
