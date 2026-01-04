# summarize_grouped_topics.py — summarize top clusters with restored bias & source info + summaries cache

import json
import openai
import os
from dotenv import load_dotenv
from datetime import datetime
import argparse
import tiktoken
import difflib, re
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
args = parser.parse_args()

date_str = args.date or datetime.today().strftime("%Y-%m-%d")
print(f"📅 Using input date: {date_str}")

INPUT_FILE = f"grouped_articles_final_{date_str}.json"
OUTPUT_FILE = f"topic_summaries_{date_str}.json"

MIN_ARTICLES = 6
MAX_ARTICLES_PER_CLUSTER = 10
MAX_CLUSTERS = 10
MAX_TOKENS = 7000
ENCODING = tiktoken.encoding_for_model("gpt-4")

# Cache config (bump PROMPT_VERSION when you change the prompt format)
PROMPT_VERSION = "v1.0-2025-08-12"
SUMM_MODEL = "gpt-4"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    clusters = json.load(f)

valid_clusters = [c for c in clusters if len(c["articles"]) >= MIN_ARTICLES]
ranked = sorted(valid_clusters, key=lambda c: len(c["articles"]), reverse=True)
top_clusters = ranked[:MAX_CLUSTERS]

summaries = []


# ----------------------------
# Step 2: Build image queries
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

# ----------------------------
# Relevance gating (1 + 2)
# ----------------------------

_STOP = {
    "the","and","for","with","from","into","amid","after","before","over","under","about",
    "this","that","these","those","are","was","were","been","being","has","have","had",
    "will","would","should","could","might","also","says","said","its","their","them",
    "a","an","of","in","on","at","to"
}

def keyword_overlap_ok(topic_text: str, image_text: str, min_hits: int = 2) -> bool:
    """
    Token overlap gate (#2). Computed automatically; no hand-built lists.
    Require at least min_hits overlapping meaningful tokens.
    """
    def toks(s: str) -> set:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\s]+", " ", s)
        out = {t for t in s.split() if len(t) > 2 and t not in _STOP}
        return out

    T = toks(topic_text)
    I = toks(image_text)
    if not T or not I:
        return False
    hits = len(T & I)
    return hits >= min_hits

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
        "the","and","for","with","from","into","amid","after","before","over","under","about",
        "this","that","these","those","are","was","were","been","being","has","have","had",
        "will","would","should","could","might","also","says","said"
    }
    return {t for t in toks if t not in stop}


def _extract_strong_entity_token(query: str) -> str:
    """
    Conservative: pick a strong entity token (usually surname/org keyword) if query looks like a name/entity.
    If query is vague (e.g., "Controversy Surrounds Trump Administration"), return "" so we don't force Wikimedia.
    """
    words = re.findall(r"[A-Za-z0-9']+", query or "")
    stop = {"the","and","or","to","of","in","on","at","amid","after","before","with","from","a","an"}
    candidates = [w.strip("'") for w in words if w and w.lower() not in stop]
    if len(candidates) < 2:
        return ""
    token = candidates[-1].lower()
    # prevent generic political tokens from acting as "entity"
    if token in {"administration","president","government","claims","controversy","pressure","conflict","tariffs"}:
        return ""
    return token


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
      - Require either:
          * strong entity token match in metadata, OR
          * relevance score >= 3
      - Reject illustration/painting/engraving
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
            "gsrnamespace": 6,   # File namespace only
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

    strong_token = _extract_strong_entity_token(image_query)

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

        # Must have license metadata to be email-safe
        if not (license_short or license_url):
            continue

        if _is_bad_wiki_candidate(title, desc, url):
            continue

        hay = f"{title} {desc}".lower()
        if "illustration" in hay or "painting" in hay or "engraving" in hay:
            continue

        # Entity token gating: if we found a strong token, it MUST appear
        if strong_token and strong_token not in hay:
            continue

        score = _relevance_score(image_query, title, desc)

        # If there is no strong token, require higher relevance
        if not strong_token and score < 3:
            continue

        # If there is a strong token, still require at least minimal relevance
        if strong_token and score < 1:
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

        # Relevance gating: compare against topic meaning (headline + summary), not just the query
        topic_text_clean = (topic_text or image_query).strip()
        image_text = f"{title} {desc}".strip()
        if not image_relevance_ok(topic_text_clean, image_text, sim_thresh=0.30, min_kw_hits=1):
            continue

        if score > best_score:
            best = candidate
            best_score = score

    return best

def extract_person_name_from_headline(headline: str):
    """
    Return (first, last) only when we see a plausible person name.

    This avoids false positives like "Amidst", "Escalate", "Pharmaceuticals".
    """
    words = re.findall(r"[A-Za-z']+", headline or "")
    if len(words) < 2:
        return ("", "")

    # headline connector/verb words that are NOT names (small + stable)
    not_name_words = {
        "Amid", "Amidst", "After", "Before", "Following", "Returns", "Return", "Imposes",
        "Approves", "Escalate", "Escalates", "Surrounds", "Surround", "Announces", "Calls",
        "Says", "Said", "Dies", "Killed", "Kills", "Convicted", "Claims", "Deal", "Tariffs",
        "Imported", "Pharmaceuticals", "Pressure", "International", "Violence", "Protests",
        "Conflict", "Administration", "Liberation"
    }
    stop = {"The","A","An","And","Or","To","Of","In","On","At","With","From"}

    # scan consecutive pairs
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]

        if w1 in stop or w2 in stop:
            continue
        if not (w1[:1].isupper() and w2[:1].isupper()):
            continue
        if w1 in not_name_words or w2 in not_name_words:
            continue
        # reject acronyms
        if w1.isupper() or w2.isupper():
            continue
        # surname-ish length
        if len(w2) < 3:
            continue

        return (w1, w2)

    return ("", "")

def is_person_specific(headline: str, summary: str) -> tuple[bool, str]:
    """
    Returns (is_person_specific, last_name).
    Person-specific means a likely First Last exists AND the last name appears >=2 times
    in headline+summary (strong anchor to the person).
    """
    first, last = extract_person_name_from_headline(headline)
    if not last:
        return (False, "")
    hay = f"{headline} {summary}".lower()
    return (hay.count(last.lower()) >= 2, last)

def build_unsplash_query_variants(image_query: str, topic_text: str, headline: str = "", max_words: int = 5) -> list[str]:
    """
    Build 3 query variants (B):
      1) original image_query (your existing output)
      2) entity-ish query from headline (shorter)
      3) compressed keyword query from topic_text (headline+summary), top frequent tokens

    All computed automatically; no hardcoded topic categories.
    """
    def toks(s: str) -> list[str]:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\s]+", " ", s)
        words = [w for w in s.split() if len(w) > 2 and w not in _STOP]
        return words

    # 1) Original (as-is)
    q1 = (image_query or "").strip()

    # 2) Headline/entity-short: take first max_words meaningful tokens from headline
    h_words = toks(headline)
    q2 = " ".join(h_words[:max_words]).strip()

    # 3) Compressed: top frequent tokens from topic_text
    t_words = toks(topic_text)
    freq = {}
    for w in t_words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    q3 = " ".join([w for w, _ in top[:max_words]]).strip()

    # Deduplicate, keep non-empty, keep order
    out = []
    for q in [q1, q2, q3]:
        q = (q or "").strip()
        if q and q not in out:
            out.append(q)
    return out

def fetch_unsplash_image(image_query: str, topic_text: str = "", headline: str = ""):
    """
    Unsplash fallback (B): try 3 query variants, score all candidates, pick best.

    - Pull top N results per query variant (6 each)
    - Score each candidate against topic_text using semantic similarity + keyword overlap
    - Accept best candidate if it clears thresholds; else None

    This increases hit rate without lowering quality.
    """
    if IMAGES_OFF:
        return None
    if not UNSPLASH_ACCESS_KEY:
        return None

    topic_text_clean = (topic_text or image_query or "").strip()
    if not topic_text_clean:
        return None

    # Build 3 query variants
    queries = build_unsplash_query_variants(image_query=image_query, topic_text=topic_text_clean, headline=headline, max_words=6)
    if not queries:
        return None

    # Embed topic once
    try:
        topic_vec = EMBEDDER.encode([topic_text_clean])[0]
    except Exception:
        topic_vec = None

    def candidate_score(image_text: str) -> tuple[float, int]:
        """
        Returns (semantic_similarity, keyword_hits).
        Higher is better.
        """
        # semantic similarity
        sim = 0.0
        if topic_vec is not None:
            try:
                img_vec = EMBEDDER.encode([image_text])[0]
                sim = float(cosine_similarity(np.array(topic_vec).reshape(1, -1),
                                              np.array(img_vec).reshape(1, -1))[0][0])
            except Exception:
                sim = 0.0

        # keyword overlap hits
        def toks(s: str) -> set:
            s = (s or "").lower()
            s = re.sub(r"[^a-z0-9\s]+", " ", s)
            return {t for t in s.split() if len(t) > 2 and t not in _STOP}

        T = toks(topic_text_clean)
        I = toks(image_text)
        hits = len(T & I) if T and I else 0
        return sim, hits

    # Acceptance thresholds (tuneable)
    SIM_THRESH = 0.30
    MIN_HITS = 2

    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    best = None
    best_sim = -1.0
    best_hits = -1

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

        for photo in results:
            urls = photo.get("urls") or {}
            # Prefer high-quality source and scale DOWN for email/web
            raw = urls.get("raw")
            if raw:
                # Request a server-resized image ~900px wide for crisp 600px display
                url = f"{raw}&w=900&fit=max&q=80"
            else:
                # Fallbacks (still fine if raw isn't available)
                url = urls.get("regular") or urls.get("small")

            if not url:
                continue

            alt = (photo.get("alt_description") or photo.get("description") or q).strip()
            alt = re.sub(r"\s+", " ", alt)

            desc = (photo.get("description") or photo.get("alt_description") or alt).strip()
            desc = re.sub(r"\s+", " ", desc)

            image_text = f"{alt} {desc}".strip()

            sim, hits = candidate_score(image_text)

            # Extra guards for person-specific topics & inverted/protest images
            first, last = extract_person_name_from_headline(headline)
            topic_lower = topic_text_clean.lower()

            person_specific = False
            if last:
                # If last name appears 2+ times in topic text, it’s likely truly about that person
                person_specific = topic_lower.count(last.lower()) >= 2

            # If person-specific: require either (a) last name appears in image text OR (b) very high semantic match
            if person_specific:
                hay = image_text.lower()
                if (last.lower() not in hay) and (sim < 0.42):
                    continue

                # Reject protest-sign style inversions for person-specific conviction/death topics
                # (e.g., “convict killer cops” sign for “cop killer dies” story)
                if any(w in topic_lower for w in ["dies", "died", "death", "killed", "convicted", "sentenced"]):
                    if any(w in hay for w in ["protest", "rally", "march", "demonstration", "sign", "placard"]):
                        # Inversion cue: "killer cops" or "kill cops" is likely protest context, not the biography event
                        if ("killer" in hay and "cops" in hay) or ("kill" in hay and "cops" in hay):
                            continue

            # Reject weak matches
            if sim < SIM_THRESH and hits < MIN_HITS:
                continue

            # Keep best by semantic similarity, then keyword overlap
            if (sim > best_sim) or (sim == best_sim and hits > best_hits):
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
                best_sim = sim
                best_hits = hits

    return best

# ----------------------------
# Everything below here is your original non-image logic
# (unchanged except using image results)
# ----------------------------
def compute_bias_distribution(articles):
    counts = {}
    total = 0
    for a in articles:
        bias_raw = a.get("bias") or "Unknown"
        bias = bias_raw.title().replace("-", " ")
        if bias != "Unknown":
            counts[bias] = counts.get(bias, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {k: round(v / total * 100) for k, v in counts.items()}


def format_article(a):
    title = (a.get("title") or "")[:200].strip()
    desc = (a.get("description") or "")[:300].strip()
    return f"{title}. {desc}"


def truncate_prompt(prompt, max_tokens):
    tokens = ENCODING.encode(prompt or "")
    if len(tokens) <= max_tokens:
        return prompt or ""
    return ENCODING.decode(tokens[:max_tokens])


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
        bias = (article.get("bias") or "Unknown").title().replace("-", " ")
        if bias not in bias_order:
            bias = "Unknown"
        return (bias_order.index(bias), (article.get("source_name") or "").lower())

    sorted_articles = sorted(articles, key=bias_sort_key)

    chips = []
    for a in sorted_articles:
        url = a.get("url", "#")
        source = a.get("source_name") or url.split("//")[-1].split("/")[0]
        bias = (a.get("bias") or "Unknown").title().replace("-", " ")
        if bias not in color_map:
            bias = "Unknown"
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
You are a neutral news assistant. Summarize the key theme across the following news articles. Provide:
1. A short, clear headline summarizing the topic
2. A factual summary in 3-4 sentences
3. Three bullet-point takeaways explaining why it matters (political, economic, legal, civil, etc)

Articles:
{text_block}

Respond in this format:
Headline: <headline>
Summary: <summary>
Takeaways:
- <point 1>
- <point 2>
- <point 3>
"""
    prompt = truncate_prompt(prompt.strip(), MAX_TOKENS)

    try:
        response = openai.ChatCompletion.create(
            model=SUMM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
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


def extract_body_and_takeaways(summary_text):
    parts = (summary_text or "").split("Takeaways:")
    summary = ""
    takeaways = []
    if len(parts) == 2:
        summary = parts[0].split("Summary:")[-1].strip().rstrip("Key")
        takeaways = [line.strip("- ").strip() for line in parts[1].strip().splitlines() if line.strip()]
    else:
        summary = summary_text or ""
    return summary.strip(), takeaways


# --- Safety valve: collapse near-duplicate topics after summarization (no tokens) ---
_STOPWORDS = {
    "a","an","the","and","or","but","to","of","for","in","on","at","over","under","after","before",
    "with","without","by","from","as","about","into","during","including","until","against","among",
    "between","through","because","so","since","due","has","have","had","is","was","are","were",
    "be","been","being","will","would","should","may","might","can","could"
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
    body_x  = (x.get("summary") or "")[:600]
    body_y  = (y.get("summary") or "")[:600]
    urls_x  = x.get("sources") or []
    urls_y  = y.get("sources") or []

    url_ov = _overlap_ratio(urls_x, urls_y)
    t_ratio = _seq_ratio(title_x.lower(), title_y.lower())
    t_jacc  = _jaccard(_norm_title_tokens(title_x), _norm_title_tokens(title_y))
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
                kept["num_sources"] = len(merged_sources)
                merged = True
                break
        if not merged:
            result.append(cand)
    return result[:MAX_CLUSTERS]


# load summaries cache once
summ_cache = load_summ_cache()
summ_cache_dirty = False

for idx, cluster in enumerate(top_clusters):
    print(f"🧠 Summarizing topic {idx + 1}/{len(top_clusters)} with {len(cluster['articles'])} articles")

    selected_articles = cluster["articles"][:MAX_ARTICLES_PER_CLUSTER]
    selected_urls = sorted([a.get("url") for a in selected_articles if a.get("url")])
    cache_key = make_summ_key(selected_urls, SUMM_MODEL, PROMPT_VERSION, MAX_ARTICLES_PER_CLUSTER)

    cached = get_cached_summary(summ_cache, cache_key)
    if cached:
        print(f"💾 Cache hit for topic {idx + 1} — reused summary")
        headline = cached["headline"]
        body = cached["summary"]
        takeaways = cached["takeaways"]
    else:
        summary_text = summarize_cluster(selected_articles)
        if not summary_text:
            continue
        headline = extract_headline(summary_text)
        body, takeaways = extract_body_and_takeaways(summary_text)
        put_cached_summary(summ_cache, cache_key, headline, body, takeaways)
        summ_cache_dirty = True

    # Build image queries
    image_query = build_image_query(headline, body)
    image_query_wiki = build_wikimedia_query_from_headline(headline)

    # Try Wikimedia short query → Wikimedia full query → Unsplash
    topic_text = f"{headline}. {body}".strip()

    img = fetch_wikimedia_image(image_query_wiki, topic_text=topic_text)
    if img:
        print(f"🖼️ Topic {idx+1}: Wikimedia hit (query='{image_query_wiki}')")
    else:
        print(f"🖼️ Topic {idx+1}: Wikimedia miss (query='{image_query_wiki}')")

    if not img:
        img = fetch_wikimedia_image(image_query, topic_text=topic_text)
        if img:
            print(f"🖼️ Topic {idx+1}: Wikimedia hit (fallback query='{image_query}')")
        else:
            print(f"🖼️ Topic {idx+1}: Wikimedia miss (fallback query='{image_query}')")

    if not img:
        topic_text = f"{headline}. {body}".strip()
        img = fetch_unsplash_image(image_query, topic_text=topic_text, headline=headline)
    if img:
        print(f"🖼️ Topic {idx+1}: Unsplash hit (query='{image_query}')")
    else:
        print(f"🖼️ Topic {idx+1}: Unsplash miss (query='{image_query}')")

    image_url = img["url"] if img else ""
    image_alt = img["alt"] if img else ""
    image_description = img["description"] if img else ""
    image_credit = img["credit"] if img else ""
    image_source = img["source"] if img else ""
    image_source_url = img["source_url"] if img else ""
    image_license = img["license"] if img else ""

    all_articles = cluster["articles"]
    bias_dist = cluster.get("bias_distribution") or compute_bias_distribution(all_articles)
    if not bias_dist:
        print(f"⚠️ Cluster {idx} has no bias_distribution field")

    summaries.append({
        "topic_title": headline,
        "summary": body,
        "takeaways": takeaways,
        "bias_distribution": bias_dist,
        "sources": [a.get("url") for a in all_articles if a.get("url")],
        "num_sources": len(all_articles),
        "html_chips": make_html_chips(all_articles),

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

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)

print(f"✅ Saved top summaries to {OUTPUT_FILE}")