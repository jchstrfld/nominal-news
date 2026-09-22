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
import numpy as np
from urllib.parse import urlparse, parse_qsl, urlencode

# Summaries cache helpers (add summaries_cache.py next to this file)
from summaries_cache import (
    load_summ_cache, save_summ_cache,
    make_summ_key, get_cached_summary, put_cached_summary
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
IMAGES_OFF = os.getenv("NN_IMAGES_OFF", "0") == "1"
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()

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

try:
    STORY_DATE = datetime.strptime(date_str, "%Y-%m-%d")
except ValueError:
    STORY_DATE = datetime.today()
STORY_YEAR = STORY_DATE.year

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


# ----------------------------
# Context-image selection
# ----------------------------

# Images in Nominal News are contextual illustrations, not claims that the
# photograph depicts the exact breaking-news event. Search queries therefore
# prioritize named people, places, organizations, and institutions that are
# actually present in the verified core headlines. Generic event concepts
# ("protest", "attack", "shooting", etc.) are never used as standalone fallbacks.

_STOP = {
    "the", "and", "for", "with", "from", "into", "amid", "after", "before", "over", "under", "about",
    "this", "that", "these", "those", "are", "was", "were", "been", "being", "has", "have", "had",
    "will", "would", "should", "could", "might", "also", "says", "said", "its", "their", "them",
    "a", "an", "of", "in", "on", "at", "to"
}

_IMAGE_CONNECTORS = {"of", "the", "and", "&"}

# Do not maintain news-topic vocabulary here. Proper-name candidates are
# extracted structurally from capitalization/acronym patterns, then a local
# semantic classifier decides whether the phrase is a specific named subject
# or merely a generic headline/action phrase. This keeps image anchoring
# reusable as daily news vocabulary changes.


def _clean_name_token(token: str) -> str:
    token = (token or "").strip(" \t\r\n,;:!?()[]{}\"“”")
    token = token.replace("’", "'")
    if token.lower().endswith("'s") and len(token) > 2:
        token = token[:-2]
    token = token.rstrip(".'") if token.endswith(".'") else token
    return token


def _is_proper_name_token(token: str) -> bool:
    raw = _clean_name_token(token)
    if not raw:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", raw)
    if not compact:
        return False
    # Acronyms (CNN, ICE, DC) and normal proper-name tokens (Greenland, Sheeran).
    if compact.isupper() and len(compact) >= 2:
        return True
    return raw[:1].isupper()


def _extract_proper_phrases(text: str) -> set[str]:
    """Extract short name-like phrases without using a fixed entity taxonomy."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9.'’&-]*|&", text or "")
    runs: list[list[str]] = []
    run: list[str] = []

    for tok in tokens:
        low = tok.lower().strip(".'’")
        if _is_proper_name_token(tok):
            run.append(_clean_name_token(tok))
        elif run and low in _IMAGE_CONNECTORS:
            run.append(low)
        else:
            while run and run[-1].lower() in _IMAGE_CONNECTORS:
                run.pop()
            if run:
                runs.append(run)
            run = []

    while run and run[-1].lower() in _IMAGE_CONNECTORS:
        run.pop()
    if run:
        runs.append(run)

    phrases: set[str] = set()
    for run in runs:
        # Include the useful short subphrases rather than an entire Title Case headline.
        n = len(run)
        for size in range(1, min(4, n) + 1):
            for i in range(0, n - size + 1):
                part = run[i:i + size]
                while part and part[0].lower() in _IMAGE_CONNECTORS:
                    part = part[1:]
                while part and part[-1].lower() in _IMAGE_CONNECTORS:
                    part = part[:-1]
                meaningful = [w for w in part if w.lower() not in _IMAGE_CONNECTORS]
                if not meaningful:
                    continue
                if len(meaningful) == 1:
                    token = re.sub(r"[^A-Za-z0-9]", "", meaningful[0])
                    if len(token) < 3:
                        continue
                phrase = " ".join(part).strip()
                if phrase:
                    phrases.add(phrase)
    return phrases


def _anchor_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def build_context_image_queries(
    headline: str,
    articles: list[dict],
    central_title: str = "",
    max_queries: int = 6,
) -> list[str]:
    """
    Build entity-first image queries from verified core headlines.

    Proper-name candidates come from structure, while generic headline phrases
    are rejected by a local semantic prototype classifier. Repeated named
    subjects win. Multiword entities receive a modest bonus,
    which favors anchors such as "White House", "Ed Sheeran", "Saudi Arabia",
    "Kennedy Center", and "Munich Oktoberfest" over generic action words.
    """
    source_titles = []
    for article in articles or []:
        title = (article.get("title") or "").strip()
        if title:
            source_titles.append(title)

    if central_title and central_title not in source_titles:
        source_titles.append(central_title)
    if not source_titles and headline:
        source_titles.append(headline)

    counts: dict[str, float] = {}
    source_support: dict[str, int] = {}
    display: dict[str, str] = {}

    for title in source_titles:
        seen = set()
        for phrase in _extract_proper_phrases(title):
            key = _anchor_key(phrase)
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0.0) + 1.0
            source_support[key] = source_support.get(key, 0) + 1
            display.setdefault(key, phrase)

    # The generated Nominal News headline may strengthen an anchor that was
    # already observed in purifier-approved source headlines, but it must not
    # manufacture a new entity. Generated headlines often use Title Case, which
    # can make ordinary headline wording look like a proper name.
    for phrase in _extract_proper_phrases(headline):
        key = _anchor_key(phrase)
        if key and key in counts:
            counts[key] += 1.25

    scored = []
    for key, count in counts.items():
        phrase = display[key]
        meaningful = [t for t in key.split() if t not in _IMAGE_CONNECTORS]
        if not meaningful:
            continue

        # Generic actions, roles, and descriptive headline fragments are
        # rejected semantically rather than through a maintained word list.
        # The remaining labels are durable subject shapes used elsewhere by
        # the image selector (person/place/institution/event/other).
        anchor_type = classify_context_anchor(phrase)
        if anchor_type == "generic":
            continue

        # Generated headlines may title-case ordinary nouns and make them look
        # like named entities (for example, a generic structure type). For
        # institution/other anchors, require the proper-name form to be
        # independently corroborated by at least two purifier-approved core
        # headlines. Named events join person/geographic anchors as eligible
        # with one source-title occurrence because event names are themselves
        # the visual subject we want to search. This is a structural evidence
        # rule, not a maintained news-vocabulary list.
        support = int(source_support.get(key, 0))
        if anchor_type in {"institution", "other"} and len(source_titles) >= 2 and support < 2:
            continue

        multiword_bonus = 12.0 if len(meaningful) >= 2 else 0.0
        acronym_penalty = 0.0
        compact_words = [re.sub(r"[^A-Za-z0-9]", "", w) for w in phrase.split()]
        if compact_words and all(w.isupper() and len(w) <= 4 for w in compact_words if w):
            acronym_penalty = 5.0
        score = (count * 10.0) + multiword_bonus + min(len(key), 20) * 0.05 - acronym_penalty
        scored.append((score, count, len(meaningful), phrase, key))

    scored.sort(key=lambda row: (-row[0], -row[1], -row[2], row[4]))

    out: list[str] = []
    seen_keys = set()
    for _, _, _, phrase, key in scored:
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(phrase)
        if len(out) >= max_queries:
            break
    return out



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


# ----------------------------
# Image relevance gates
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
    return {t for t in toks if t not in _STOP}


def _normalized_words(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP]


def _normalized_phrase(text: str) -> str:
    return " ".join(_normalized_words(text))


def anchor_match_score(anchor: str, image_text: str) -> int:
    """
    Score how explicitly image metadata names the searched subject.

    Multiword anchors are intentionally strict: the normalized phrase itself
    must occur. This prevents near-name collisions such as one institution
    being mistaken for another with a shared surname or generic noun.
    """
    anchor_words = _normalized_words(anchor)
    image_words = _normalized_words(image_text)
    if not anchor_words or not image_words:
        return 0

    anchor_norm = " ".join(anchor_words)
    image_norm = " ".join(image_words)

    if len(anchor_words) >= 2:
        return 4 if anchor_norm in image_norm else 0

    return 3 if anchor_words[0] in set(image_words) else 0


def semantic_similarity(topic_text: str, image_text: str) -> float:
    if not topic_text or not image_text:
        return 0.0
    try:
        v1 = EMBEDDER.encode([topic_text])[0]
        v2 = EMBEDDER.encode([image_text])[0]
        return float(cosine_similarity(np.array(v1).reshape(1, -1), np.array(v2).reshape(1, -1))[0][0])
    except Exception:
        return 0.0


_ANCHOR_TYPE_CACHE: dict[str, str] = {}
_SCENE_TYPE_CACHE: dict[str, str] = {}
_VISUAL_MEDIUM_CACHE: dict[str, str] = {}
_PNG_PHOTO_EVIDENCE_CACHE: dict[str, bool] = {}

_ANCHOR_TYPE_PROTOTYPES = {
    "person": (
        "a named human person",
        "a specific individual person",
        "a celebrity politician musician athlete or other person",
    ),
    "geographic": (
        "a city country region island geographic location",
        "a named place on earth",
        "a geographic area landscape or territory",
    ),
    "institution": (
        "a named building venue institution organization or facility",
        "an airport museum university government building or center",
        "a physical institution campus venue or architectural site",
    ),
    "event": (
        "the proper name of a specific recurring event festival conference ceremony or competition",
        "a specifically named organized public event",
        "the name of a particular festival tournament conference parade or annual event",
    ),
    "other": (
        "the proper name of a specific object product law program work vehicle or other named subject",
        "a uniquely named thing that is not a person place institution or event",
    ),
    "generic": (
        "a generic action incident conflict policy issue or common noun rather than a proper name",
        "a generic job title role group label or broad category rather than a specific named entity",
        "a descriptive news headline phrase saying what happened rather than naming a specific subject",
        "a generic object type building type or event concept that could describe many unrelated stories",
    ),
}


_VISUAL_MEDIUM_PROTOTYPES = {
    "photo": (
        "a camera photograph of a real-world scene, person, building, landscape, crowd, vehicle, or physical environment",
        "a photographic image showing real people, places, architecture, objects, or events as they appeared in the physical world",
        "a real-world photograph captured by a camera rather than a designed graphic or document reproduction",
    ),
    "graphic": (
        "a map, flag map, diagram, chart, logo, emblem, illustration, icon, symbol, or designed informational graphic",
        "a typographic image whose main content is words, lettering, an alphabet, a script, or written text rather than a photographed scene",
        "a screenshot, drawing, infographic, cartographic image, or other non-photographic visual design",
    ),
    "artifact": (
        "a scan or reproduction of a banknote, coin, currency, stamp, passport, ticket, certificate, document, poster, sign, or printed page",
        "a document or printed artifact presented as the image itself rather than a photograph of a broader real-world scene",
        "a standalone archival paper, card, label, book page, currency note, or manufactured ephemera",
    ),
}

_SCENE_TYPE_PROTOTYPES = {
    "person": (
        "a photograph mainly depicting a human person or portrait",
        "a person is the main subject of the photograph",
        "a portrait or photograph of an individual",
    ),
    "geographic": (
        "an establishing photograph of a city country region landscape or skyline",
        "a geographic aerial satellite landscape or city view",
        "a broad photograph of a place rather than a specific event",
    ),
    "institution": (
        "an architectural photograph of a building venue institution airport or facility",
        "a building exterior interior campus or physical institution is the main subject",
        "a photograph mainly depicting a named venue or structure",
    ),
    "event": (
        "a photograph of people participating in a public event performance festival ceremony protest competition or gathering",
        "an action scene from an organized event",
        "a crowd performance competition ceremony rally or festival scene",
    ),
    "graphic": (
        "a map diagram chart logo flag emblem illustration or informational graphic",
        "a non-photographic map or graphic",
        "a diagrammatic cartographic typographic or text-only image",
        "an image whose main content is a written word phrase name alphabet script or lettering rather than a photographed scene",
        "a screenshot icon symbol label or text graphic rather than a real-world photograph",
    ),
    "artifact": (
        "a photograph of a banknote coin currency stamp passport ticket document or printed paper",
        "a photograph of a poster sign banner book page certificate card label or archival document",
        "a standalone printed or manufactured artifact photographed as an object rather than a place or event",
        "a photograph centered on a physical sign document printed object or piece of ephemera rather than the surrounding scene",
    ),
    "other": (
        "a photograph of an object vehicle product animal or other subject",
        "a photograph that is not mainly a person place building event graphic or document artifact",
    ),
}


def _semantic_prototype_classify(text: str, prototypes: dict[str, tuple[str, ...]], cache: dict[str, str]) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return "other"

    cache_key = cleaned[:700].lower()
    if cache_key in cache:
        return cache[cache_key]

    try:
        labels = list(prototypes.keys())
        proto_texts = [p for label in labels for p in prototypes[label]]
        vecs = EMBEDDER.encode([cleaned] + proto_texts)
        subject_vec = np.array(vecs[0]).reshape(1, -1)

        offset = 1
        scores = []
        for label in labels:
            sims = []
            for _ in prototypes[label]:
                proto_vec = np.array(vecs[offset]).reshape(1, -1)
                sims.append(float(cosine_similarity(subject_vec, proto_vec)[0][0]))
                offset += 1
            # Mean prototype similarity is more stable than a single phrase.
            scores.append((float(sum(sims) / max(1, len(sims))), label))

        scores.sort(reverse=True)
        result = scores[0][1] if scores else "other"
    except Exception:
        result = "other"

    cache[cache_key] = result
    return result


def classify_context_anchor(anchor: str) -> str:
    """
    Classify the searched entity into a durable semantic shape.
    No story/entity dictionary is maintained.
    """
    return _semantic_prototype_classify(
        anchor,
        _ANCHOR_TYPE_PROTOTYPES,
        _ANCHOR_TYPE_CACHE,
    )


def classify_candidate_scene(title: str, desc: str) -> str:
    """
    Classify what the candidate image metadata says the image actually depicts.
    This lets a geographic story reject an unrelated event photo without a
    maintained blacklist of event words.
    """
    candidate_text = f"{title}. {desc}".strip()
    return _semantic_prototype_classify(
        candidate_text,
        _SCENE_TYPE_PROTOTYPES,
        _SCENE_TYPE_CACHE,
    )


def classify_candidate_medium(title: str, desc: str) -> str:
    """Classify the visual medium before classifying the depicted subject.

    This prevents a map, text graphic, or document about a place from being
    accepted merely because its *subject* is geographic. The ontology is about
    durable media shapes, not story-specific vocabulary.
    """
    candidate_text = f"{title}. {desc}".strip()
    return _semantic_prototype_classify(
        candidate_text,
        _VISUAL_MEDIUM_PROTOTYPES,
        _VISUAL_MEDIUM_CACHE,
    )


def _wikimedia_medium_context(desc: str, extmetadata: dict) -> str:
    """Build medium-classification evidence from structural Commons metadata.

    Long free-form ImageDescription values can contain quotations, speeches,
    archival prose, or catalog text that describe the *story* rather than the
    visual medium. Those strings can make a real photograph look text-like to
    an embedding classifier. Prefer stable file/category metadata, and only add
    the description when it is short enough to behave like a visual caption.
    """
    extmetadata = extmetadata or {}

    categories = _clean_html(
        ((extmetadata.get("Categories") or {}).get("value") or "")
    )
    object_name = _clean_html(
        ((extmetadata.get("ObjectName") or {}).get("value") or "")
    )
    short_desc = re.sub(r"\s+", " ", _clean_html(desc or "")).strip()

    parts = []
    for value in (object_name, categories):
        value = re.sub(r"\s+", " ", value or "").strip()
        if value and value not in parts:
            parts.append(value)

    # Short captions often say exactly what the file is (for example, a map or
    # banknote). Long narrative/quoted descriptions are poor medium evidence.
    if short_desc and len(short_desc) <= 220 and short_desc not in parts:
        parts.append(short_desc)

    return ". ".join(parts)


# Automated production accepts only simple, well-understood reuse terms:
# Public Domain / CC0 or Creative Commons Attribution (CC BY).
# Licenses with ShareAlike, NonCommercial, NoDerivatives, GFDL, unclear,
# or permission-based restrictions are rejected. Attribution-required CC BY
# is allowed because Nominal News displays a source/credit line under images.

def _commons_license_is_approved(license_short: str, license_url: str, extmetadata: dict) -> bool:
    short = _clean_html(license_short or "").strip().lower()
    url = _clean_html(license_url or "").strip().lower()
    combined = f"{short} {url}"

    restrictions = _clean_html(
        ((extmetadata or {}).get("Restrictions") or {}).get("value") or ""
    ).strip().lower()
    if restrictions not in {"", "none", "false", "0"}:
        return False

    # Reject anything that carries obligations beyond straightforward credit,
    # or that is not suitable for a potentially commercial newsletter.
    blocked = (
        "by-sa", "/by-sa/", "sharealike", "share alike",
        "by-nc", "/by-nc/", "noncommercial", "non-commercial",
        "by-nd", "/by-nd/", "no derivatives", "noderivatives",
        "gfdl", "gnu free documentation", "all rights reserved",
        "permission", "fair use", "copyrighted free use",
    )
    if any(token in combined for token in blocked):
        return False

    # Public-domain/CC0 material is preferred and always acceptable.
    if "cc0" in short or "cc zero" in short or "creative commons zero" in short:
        return True
    if "public domain" in short or short.startswith("pd-") or short.startswith("pdm"):
        return True
    if "creativecommons.org/publicdomain/zero" in url:
        return True
    if "creativecommons.org/publicdomain/mark" in url:
        return True

    # Straight CC BY is acceptable: commercial reuse is allowed with attribution.
    if "creativecommons.org/licenses/by/" in url:
        return True
    if re.search(r"\bcc\s*by(?:\s|$|[0-9.])", short) and not any(
        token in short for token in ("sa", "nc", "nd")
    ):
        return True

    return False


def _commons_license_preference(license_short: str, license_url: str) -> int:
    """Small ranking bonus for lower-friction licenses; relevance still dominates."""
    short = _clean_html(license_short or "").strip().lower()
    url = _clean_html(license_url or "").strip().lower()
    if (
        "cc0" in short
        or "cc zero" in short
        or "public domain" in short
        or short.startswith("pd-")
        or short.startswith("pdm")
        or "creativecommons.org/publicdomain/" in url
    ):
        return 2
    if "creativecommons.org/licenses/by/" in url or re.search(r"\bcc\s*by(?:\s|$|[0-9.])", short):
        return 1
    return 0


def _title_anchor_focus(anchor: str, title: str) -> float:
    """Share of meaningful title tokens occupied by the anchor."""
    anchor_words = _normalized_words(anchor)
    title_words = _normalized_words(title)
    if not anchor_words or not title_words:
        return 0.0

    remaining = list(title_words)
    matched = 0
    for word in anchor_words:
        if word in remaining:
            remaining.remove(word)
            matched += 1
    return matched / max(1, len(title_words))


def _specific_proper_phrases(text: str) -> list[tuple[str, set[str]]]:
    """
    Keep only the most specific proper-name phrases found in text.

    `_extract_proper_phrases` deliberately emits useful subphrases. For context
    contamination checks we collapse nested phrases so one named entity is not
    counted several times.
    """
    rows = []
    for phrase in _extract_proper_phrases(text or ""):
        # Keep contamination checks focused on named subjects. Generic
        # title-case/action phrases are filtered by the same local semantic
        # classifier used for query construction, not by story-word lists.
        if classify_context_anchor(phrase) == "generic":
            continue
        words = set(_normalized_words(phrase))
        if not words:
            continue
        rows.append((phrase, words))

    rows.sort(key=lambda row: (-len(row[1]), -len(row[0]), row[0].lower()))

    kept: list[tuple[str, set[str]]] = []
    for phrase, words in rows:
        if any(words < existing_words for _, existing_words in kept):
            continue
        kept.append((phrase, words))
    return kept


def _proper_context_counts(anchor: str, topic_text: str, text: str) -> tuple[int, int, int]:
    """
    Return:
      (specific proper phrases, unsupported proper phrases,
       phrases that extend the anchor with unsupported proper-name material)

    This is story-relative. Nothing here depends on an event/category blacklist.
    """
    anchor_words = set(_normalized_words(anchor))
    topic_words = set(_normalized_words(topic_text))
    rows = _specific_proper_phrases(text)

    unsupported = 0
    anchor_extensions = 0

    for _, words in rows:
        if not words:
            continue

        # A phrase is supported when nearly all of its content is already
        # present in the verified story text.
        story_overlap = len(words & topic_words) / max(1, len(words))
        supported = story_overlap >= 0.80

        if not supported:
            unsupported += 1

        if anchor_words and anchor_words <= words:
            extras = words - anchor_words
            unsupported_extras = extras - topic_words
            if unsupported_extras:
                anchor_extensions += 1

    return len(rows), unsupported, anchor_extensions


def _candidate_context_metrics(anchor: str, topic_text: str, title: str, desc: str) -> dict:
    candidate_text = f"{title} {desc}".strip()
    anchor_score = anchor_match_score(anchor, candidate_text)
    topic_semantic = semantic_similarity(topic_text or anchor, candidate_text)
    anchor_semantic = semantic_similarity(anchor, candidate_text)
    title_focus = _title_anchor_focus(anchor, title)
    anchor_type = classify_context_anchor(anchor)
    scene_type = classify_candidate_scene(title, desc)
    title_anchor_match = anchor_match_score(anchor, title) > 0

    title_total, title_foreign, title_extensions = _proper_context_counts(
        anchor, topic_text, title
    )
    desc_total, desc_foreign, desc_extensions = _proper_context_counts(
        anchor, topic_text, desc
    )

    foreign_ratio = (
        (title_foreign + 0.35 * desc_foreign)
        / max(1.0, title_total + 0.35 * desc_total)
    )

    return {
        "anchor_score": anchor_score,
        "topic_semantic": topic_semantic,
        "anchor_semantic": anchor_semantic,
        "title_focus": title_focus,
        "anchor_type": anchor_type,
        "scene_type": scene_type,
        "title_anchor_match": title_anchor_match,
        "title_foreign": title_foreign,
        "title_extensions": title_extensions,
        "desc_foreign": desc_foreign,
        "desc_extensions": desc_extensions,
        "foreign_ratio": foreign_ratio,
    }


def _context_candidate_is_safe(metrics: dict) -> bool:
    """
    Fail closed using structure rather than maintained content blacklists.

    Person anchors are held to the strictest rule because a photo of the right
    person at the wrong event can look like event photography. Places/venues
    can safely use broader establishing context. Named-event searches get one
    additional structural signal: provider retrieval for the exact event name.
    """
    anchor_type = metrics.get("anchor_type") or "other"
    scene_type = metrics.get("scene_type") or "other"
    anchor_score = int(metrics.get("anchor_score", 0))
    topic_semantic = float(metrics.get("topic_semantic", 0.0))
    anchor_semantic = float(metrics.get("anchor_semantic", 0.0))
    title_focus = float(metrics.get("title_focus", 0.0))
    title_anchor_match = bool(metrics.get("title_anchor_match"))
    title_foreign = int(metrics.get("title_foreign", 0))
    title_extensions = int(metrics.get("title_extensions", 0))
    foreign_ratio = float(metrics.get("foreign_ratio", 0.0))

    # Provider search is a legitimate relevance signal for a *named event*.
    # Provider captions are often sparse, so allow missing literal event-name
    # metadata only when the returned scene is itself an event and remains
    # semantically aligned with the verified story. All other anchor types
    # still require an explicit metadata match.
    event_retrieval_support = (
        anchor_type == "event"
        and scene_type == "event"
        and anchor_score <= 0
        and topic_semantic >= 0.24
        and anchor_semantic >= 0.18
        and foreign_ratio < 0.60
    )

    # Unsplash full-photo metadata can provide an exact event-name match even
    # when the visible alt text is generic. Treat that as stronger evidence
    # than provider retrieval alone, but only for an event-looking scene that
    # is also strongly aligned with the verified story. This avoids forcing
    # sparse captions to literally repeat the event name without weakening the
    # rule for ordinary search-result metadata.
    event_metadata_support = (
        anchor_type == "event"
        and scene_type == "event"
        and bool(metrics.get("provider_metadata_anchor_match"))
        and topic_semantic >= 0.42
        and anchor_semantic >= 0.30
        and title_foreign == 0
        and title_extensions == 0
    )

    if anchor_score <= 0 and not event_retrieval_support:
        return False

    # A small, permanent compatibility matrix prevents category mistakes
    # such as a country selecting an air-show/event image or a place selecting
    # a map. These are semantic shapes, not news-specific terms.
    compatible_scene_types = {
        "person": {"person"},
        "geographic": {"geographic", "institution"},
        "institution": {"institution", "geographic"},
        "event": {"event"},
        # Object/document imagery can be correct when the verified anchor is
        # itself a named object, product, work, vehicle, law, or similar
        # non-place subject. It is never an acceptable substitute for a
        # person, place, venue, or named event.
        "other": {"person", "geographic", "institution", "event", "artifact", "other"},
        "generic": set(),
    }

    if scene_type == "graphic":
        return False
    if scene_type not in compatible_scene_types.get(anchor_type, compatible_scene_types["other"]):
        return False

    if topic_semantic < 0.14 and anchor_semantic < 0.45:
        return False

    if anchor_type == "person":
        # For people, the person must be the actual subject of the file title,
        # not merely named somewhere in a caption/description.
        if not title_anchor_match:
            return False
        if title_focus < 0.38:
            return False
        if title_foreign > 0 and title_focus < 0.55:
            return False
        if foreign_ratio >= 0.60 and title_focus < 0.65:
            return False
        return True

    if anchor_type == "geographic":
        # Establishing imagery is allowed for cities/countries/regions, even
        # when captions mention a local landmark. Reject only when a different
        # named institution dominates the *title/alt text* itself. This keeps
        # broad skyline/context views while rejecting a landmark photo chosen
        # merely because it is located in the right city.
        if scene_type == "institution" and title_foreign > 0 and title_focus < 0.28:
            return False
        if title_extensions > 0 and topic_semantic < 0.36:
            return False
        if title_foreign > 0 and title_focus < 0.30 and topic_semantic < 0.32:
            return False
        if foreign_ratio >= 0.70 and topic_semantic < 0.34:
            return False
        return True

    if anchor_type == "institution":
        # Named buildings/venues/institutions should be the actual subject,
        # not merely the location or namesake of an unrelated scene.
        if not title_anchor_match and anchor_semantic < 0.58:
            return False
        if title_extensions > 0 and topic_semantic < 0.38:
            return False
        if title_foreign > 0 and title_focus < 0.38 and topic_semantic < 0.36:
            return False
        if foreign_ratio >= 0.62 and topic_semantic < 0.40:
            return False
        return True

    if anchor_type == "event":
        # Exact full-provider metadata support is allowed to compensate for a
        # sparse visible caption. The event name is still explicitly present in
        # provider metadata, the scene must look like an event, and the candidate
        # must remain strongly aligned with the verified story.
        if event_metadata_support:
            return True

        # If exact named-event retrieval already passed the structural event
        # support gate above, do not require a second, stronger literal-name
        # match here. That would nullify the provider-retrieval signal we added
        # specifically for sparse event metadata.
        if event_retrieval_support:
            if title_extensions > 0 and topic_semantic < 0.30:
                return False
            return True

        # When provider retrieval support is absent, keep the stricter literal
        # evidence requirement for named events.
        if not title_anchor_match and anchor_semantic < 0.55:
            return False
        if title_extensions > 0 and topic_semantic < 0.30:
            return False
        return True

    # Unknown/other anchors remain conservative.
    if not title_anchor_match and title_focus < 0.35:
        return False
    if title_extensions > 0 and topic_semantic < 0.40:
        return False
    if foreign_ratio >= 0.60 and topic_semantic < 0.40:
        return False
    return True


def _context_quality_score(metrics: dict) -> float:
    """Continuous quality score used after the hard safety gate."""
    topic_semantic = float(metrics.get("topic_semantic", 0.0))
    anchor_semantic = float(metrics.get("anchor_semantic", 0.0))
    title_focus = float(metrics.get("title_focus", 0.0))
    foreign_ratio = float(metrics.get("foreign_ratio", 0.0))
    title_extensions = int(metrics.get("title_extensions", 0))
    desc_extensions = int(metrics.get("desc_extensions", 0))

    anchor_type = metrics.get("anchor_type") or "other"
    person_focus_bonus = title_focus * 2.0 if anchor_type == "person" else 0.0

    score = (
        topic_semantic * 8.0
        + anchor_semantic * 2.5
        + title_focus * 5.0
        + person_focus_bonus
        - foreign_ratio * 3.5
        - title_extensions * 2.0
        - desc_extensions * 0.5
    )
    return max(-10.0, min(12.0, score))


def _selection_confidence(metrics: dict) -> float:
    anchor_score = float(metrics.get("anchor_score", 0.0))
    anchor_norm = min(1.0, anchor_score / 4.0)
    topic_semantic = max(0.0, min(1.0, float(metrics.get("topic_semantic", 0.0))))
    anchor_semantic = max(0.0, min(1.0, float(metrics.get("anchor_semantic", 0.0))))
    title_focus = max(0.0, min(1.0, float(metrics.get("title_focus", 0.0)) * 2.0))
    foreign_ratio = max(0.0, min(1.0, float(metrics.get("foreign_ratio", 0.0))))
    anchor_type = metrics.get("anchor_type") or "other"
    scene_type = metrics.get("scene_type") or "other"

    # Full provider metadata can explicitly name a named event even when the
    # user-facing alt text is generic. This path is only reachable after the
    # event-scene and strong semantic safety gates pass, so confidence should
    # reflect that exact metadata evidence rather than penalizing missing title
    # focus as if the event name were absent everywhere.
    if (
        anchor_type == "event"
        and scene_type == "event"
        and bool(metrics.get("provider_metadata_anchor_match"))
    ):
        confidence = (
            0.32
            + 0.35 * topic_semantic
            + 0.25 * anchor_semantic
            + 0.08 * (1.0 - foreign_ratio)
        )
        return max(0.0, min(1.0, confidence))

    # Named-event retrieval carries useful evidence even when a sparse caption
    # omits the event name. This path is only reachable after the event-scene
    # and semantic safety gates pass.
    if anchor_type == "event" and scene_type == "event" and anchor_score <= 0:
        confidence = (
            0.35
            + 0.35 * topic_semantic
            + 0.20 * anchor_semantic
            + 0.10 * (1.0 - foreign_ratio)
        )
        return max(0.0, min(1.0, confidence))

    confidence = (
        0.45 * anchor_norm
        + 0.35 * topic_semantic
        + 0.20 * title_focus
        - 0.08 * foreign_ratio
    )
    return max(0.0, min(1.0, confidence))


# Image age is intentionally handled mathematically rather than with year
# blacklists. Half-life controls how quickly older context loses ranking power;
# max age is only a fail-safe for obviously stale current-news context.
_TEMPORAL_HALF_LIFE_YEARS = {
    "person": 7.0,
    "geographic": 24.0,
    "institution": 9.0,
    "event": 6.0,
    "other": 12.0,
}

_TEMPORAL_MAX_AGE_YEARS = {
    "person": 30.0,
    "geographic": 80.0,
    "institution": 35.0,
    "event": 25.0,
    "other": 45.0,
}

# Minimum exponential freshness for dated material. These are durable visual
# relevance rules, not calendar-year blacklists. Undated images remain eligible.
_TEMPORAL_MIN_FRESHNESS = {
    "person": 0.30,
    "geographic": 0.05,
    "institution": 0.10,
    "event": 0.15,
    "other": 0.10,
}


def _plausible_years(text: str) -> list[int]:
    if not text:
        return []
    years = []
    for token in re.findall(r"(?<!\d)(18\d{2}|19\d{2}|20\d{2}|21\d{2})(?!\d)", str(text)):
        try:
            year = int(token)
        except Exception:
            continue
        if 1800 <= year <= STORY_YEAR + 1:
            years.append(year)
    return years


def _commons_image_year(extmetadata: dict, title: str, desc: str) -> int | None:
    """
    Estimate the year of the depicted context, not merely the year the Commons
    asset or derivative was created.

    A year embedded in the file title is the strongest conservative signal of
    depicted context (for example, a recently uploaded file whose title says
    1914). Wikimedia's DateTimeOriginal is next-best, followed by description
    text when neither stronger signal is available. Upload/digitization dates
    are intentionally ignored.
    """
    title_years = _plausible_years(title)
    if title_years:
        return max(title_years)

    raw_original = _clean_html(
        ((extmetadata or {}).get("DateTimeOriginal") or {}).get("value") or ""
    )
    structured = _plausible_years(raw_original)
    if structured:
        return max(structured)

    desc_years = _plausible_years(desc)
    if desc_years:
        # Descriptions can contain several historical references; the newest
        # plausible year is the safest approximation when no title or
        # structured creation/capture date is available.
        return max(desc_years)

    return None


def _unsplash_image_year(photo: dict) -> int | None:
    raw = str((photo or {}).get("created_at") or "").strip()
    years = _plausible_years(raw)
    return max(years) if years else None


def _temporal_context(anchor_type: str, image_year: int | None) -> dict:
    """
    Convert image age to a 0..1 freshness score with exponential decay.

    Unknown dates are allowed but receive a neutral-mid score, so a known
    recent image beats an equally relevant undated image without discarding the
    larger image pool.
    """
    anchor_type = anchor_type if anchor_type in _TEMPORAL_HALF_LIFE_YEARS else "other"

    if image_year is None:
        return {
            "image_year": None,
            "age_years": None,
            "freshness": 0.55,
            "allowed": True,
        }

    age = max(0.0, float(STORY_YEAR - image_year))
    half_life = float(_TEMPORAL_HALF_LIFE_YEARS[anchor_type])
    max_age = float(_TEMPORAL_MAX_AGE_YEARS[anchor_type])

    freshness = float(0.5 ** (age / max(half_life, 1e-6)))
    freshness = max(0.0, min(1.0, freshness))
    min_freshness = float(_TEMPORAL_MIN_FRESHNESS.get(anchor_type, 0.10))

    return {
        "image_year": int(image_year),
        "age_years": age,
        "freshness": freshness,
        "allowed": age <= max_age and freshness >= min_freshness,
    }


# ----------------------------
# Image fetching: Wikimedia Commons (PD/CC0/CC BY) → constrained Unsplash → none
# ----------------------------

def _wiki_media_is_photo_candidate(info: dict) -> bool:
    """
    Admit common still-image raster formats, then let semantic scene
    classification determine whether the candidate is actually photographic
    context or a graphic/artifact. File extension alone is not a reliable
    photo detector: legitimate photographs may be PNG. Animated/legacy GIF
    files remain excluded so motion never appears opportunistically in the
    calm news-card experience.
    """
    media_type = (info.get("mediatype") or "").strip().upper()
    mime = (info.get("mime") or "").strip().lower()

    if media_type and media_type != "BITMAP":
        return False

    # These are common still-image raster containers. PNG is intentionally
    # allowed because it may contain real photography; semantic graphic/artifact
    # gates downstream decide what the image depicts. GIF remains excluded here
    # because it may animate and is not needed for contextual news photography.
    approved_photo_mimes = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/tiff",
        "image/webp",
    }
    if not mime or mime not in approved_photo_mimes:
        return False

    return True


def _normalize_metadata_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _commons_png_has_photo_evidence(
    page_title: str,
    title: str,
    desc: str,
    extmetadata: dict,
    headers: dict,
) -> bool:
    """Require positive photographic evidence for ambiguous Commons PNGs.

    PNG is a container, not a visual medium: it can hold a real photograph,
    but Commons also contains many rasterized maps, diagrams, text graphics,
    flags, and other designed assets in PNG. Rather than banning PNG outright,
    keep it eligible when Commons supplies durable evidence of photographic
    origin.

    Evidence is either:
      1) explicit photo/photograph wording in concise Commons metadata, or
      2) camera/capture EXIF fields from a one-file detail lookup.

    The lookup is cached and is only reached after the normal relevance and
    semantic medium gates have already accepted the candidate, so this remains
    a narrow fail-closed safety check rather than a broad extra API stage.
    """
    cache_key = (page_title or title or "").strip().lower()
    if cache_key in _PNG_PHOTO_EVIDENCE_CACHE:
        return _PNG_PHOTO_EVIDENCE_CACHE[cache_key]

    extmetadata = extmetadata or {}
    categories = _clean_html(
        ((extmetadata.get("Categories") or {}).get("value") or "")
    )
    object_name = _clean_html(
        ((extmetadata.get("ObjectName") or {}).get("value") or "")
    )
    short_desc = re.sub(r"\s+", " ", _clean_html(desc or "")).strip()
    if len(short_desc) > 220:
        short_desc = ""

    text_evidence = " ".join(
        part for part in (title, object_name, categories, short_desc) if part
    )
    # Stable medium vocabulary only; no story/entity terms are encoded here.
    if re.search(r"\b(?:photo|photos|photograph|photographs|photographic)\b", text_evidence, re.I):
        _PNG_PHOTO_EVIDENCE_CACHE[cache_key] = True
        return True

    if not page_title:
        _PNG_PHOTO_EVIDENCE_CACHE[cache_key] = False
        return False

    try:
        params = {
            "action": "query",
            "prop": "imageinfo",
            "titles": page_title,
            "iiprop": "metadata|commonmetadata",
            "format": "json",
        }
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            headers=headers,
            timeout=6,
        )
        r.raise_for_status()
        payload = r.json()
        pages = (payload.get("query") or {}).get("pages") or {}

        metadata_names = set()
        for detail_page in pages.values():
            infos = detail_page.get("imageinfo") or []
            if not infos:
                continue
            detail = infos[0]
            for field in ("metadata", "commonmetadata"):
                for item in detail.get(field) or []:
                    if isinstance(item, dict):
                        name = _normalize_metadata_name(str(item.get("name") or ""))
                        if name:
                            metadata_names.add(name)

        camera_identity = {
            "make",
            "model",
            "lensmodel",
            "lens",
        }
        capture_settings = {
            "exposuretime",
            "fnumber",
            "focallength",
            "isospeedratings",
            "photographicsensitivity",
            "shutterspeedvalue",
            "aperturevalue",
        }
        result = bool(
            (metadata_names & camera_identity)
            and (metadata_names & capture_settings)
        )
    except Exception as e:
        # Ambiguous PNGs fail closed if their photographic origin cannot be
        # verified. JPEG/TIFF/WebP behavior is unaffected by this check.
        print(f"ℹ️ Wikimedia PNG photo-evidence check failed: {e}")
        result = False

    _PNG_PHOTO_EVIDENCE_CACHE[cache_key] = result
    return result


def _wikimedia_display_text(anchor: str, desc: str) -> str:
    """Return concise user-facing text without changing scoring metadata.

    Commons descriptions are valuable for relevance scoring, but some are
    archival catalog records, long quotations, or metadata dumps that make
    poor alt text. In those cases the already-validated anchor is the safest
    concise accessibility fallback.
    """
    text = re.sub(r"\s+", " ", _clean_html(desc or "")).strip()
    fallback = re.sub(r"\s+", " ", (anchor or "").strip())

    if not text:
        return fallback

    lowered = text.lower()
    metadata_markers = (
        "physical description:",
        "notes:",
        "credit line:",
        "forms part of",
        "gift and purchase",
        "digital image produced",
    )

    # Alt text should stay concise. Long catalog records and quotations are
    # useful internal evidence but not appropriate screen-reader output.
    if len(text) > 180:
        return fallback
    if any(marker in lowered for marker in metadata_markers):
        return fallback
    if text.count(";") >= 3:
        return fallback

    return text


def _relevance_score(topic_query: str, title: str, desc: str) -> int:
    q = _tokenize(topic_query)
    t = _tokenize(title)
    d = _tokenize(desc)
    return len(q & (t | d))


def fetch_wikimedia_image(image_query: str, topic_text: str = ""):
    """
    Return the strongest Wikimedia candidate for one entity anchor.

    Every result for this anchor is evaluated before a winner is chosen. The
    returned candidate includes internal selection metadata so winners from
    different anchors can then compete globally.
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
            "gsrlimit": 24,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|mediatype",
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
    best_score = -1e9

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
        page_title = (page.get("title") or "").strip()
        title = re.sub(r"^File:\s*", "", page_title, flags=re.I).strip()

        if not (license_short or license_url):
            continue
        if not _commons_license_is_approved(license_short, license_url, ext):
            continue
        if not _wiki_media_is_photo_candidate(info):
            continue

        # First decide whether Commons metadata describes actual photographic
        # context at all. Subject classification comes second. Without this
        # stage, a map or text graphic can look "geographic" simply because
        # it names a country.
        medium_context = _wikimedia_medium_context(desc, ext)
        visual_medium = classify_candidate_medium(title, medium_context)
        if visual_medium != "photo":
            continue

        metrics = _candidate_context_metrics(image_query, topic_text, title, desc)
        if not _context_candidate_is_safe(metrics):
            continue

        # PNG remains supported, but because it is commonly used for both
        # photographs and rasterized graphics, require positive photographic
        # evidence before a PNG can become a production image.
        if (info.get("mime") or "").strip().lower() == "image/png":
            if not _commons_png_has_photo_evidence(
                page_title, title, desc, ext, headers
            ):
                continue

        temporal = _temporal_context(
            metrics.get("anchor_type") or "other",
            _commons_image_year(ext, title, desc),
        )
        if not temporal["allowed"]:
            continue

        image_text = f"{title} {desc}".strip()
        anchor_score = int(metrics["anchor_score"])
        semantic = float(metrics["topic_semantic"])
        overlap = _relevance_score(topic_text or image_query, title, desc)
        quality = _context_quality_score(metrics)
        license_bonus = _commons_license_preference(license_short, license_url)

        source_url = f"https://commons.wikimedia.org/?curid={page.get('pageid')}"
        display_text = _wikimedia_display_text(image_query, desc)
        alt = display_text or image_query

        credit_bits = []
        if artist:
            credit_bits.append(artist)
        credit_bits.append("Wikimedia Commons")
        credit = "Photo: " + " / ".join(credit_bits)

        event_retrieval_bonus = 30.0 if (
            metrics.get("anchor_type") == "event"
            and metrics.get("scene_type") == "event"
            and anchor_score <= 0
        ) else 0.0

        score = (
            (anchor_score * 20.0)
            + event_retrieval_bonus
            + (semantic * 28.0)
            + (min(overlap, 6) * 2.0)
            + (quality * 3.0)
            + (license_bonus * 1.5)
            + (float(temporal["freshness"]) * 16.0)
        )

        confidence = _selection_confidence(metrics)

        candidate = {
            "url": url,
            "alt": alt,
            "description": display_text or alt,
            "credit": credit,
            "source": "wikimedia",
            "source_url": source_url,
            "license": license_short or "Wikimedia license",
            "license_url": license_url,
            "photographer_url": "",
            "provider_url": "",
            "query": image_query,
            "_selection_score": score,
            "_confidence": confidence,
            "_context_quality": quality,
            "_semantic": semantic,
            "_anchor_match": anchor_score,
            "_title_focus": metrics["title_focus"],
            "_foreign_ratio": metrics["foreign_ratio"],
            "_anchor_type": metrics["anchor_type"],
            "_scene_type": metrics["scene_type"],
            "_image_year": temporal["image_year"],
            "_image_age_years": temporal["age_years"],
            "_freshness": temporal["freshness"],
        }

        if score > best_score:
            best = candidate
            best_score = score

    return best



_UNSPLASH_DETAIL_CACHE: dict[str, dict] = {}


def _unsplash_referral_url(url: str) -> str:
    if not url:
        return ""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}utm_source=nominal_news&utm_medium=referral"


def _unsplash_full_photo(photo: dict, headers: dict) -> dict:
    """Fetch and cache full Unsplash metadata for one photo when needed."""
    photo_id = str((photo or {}).get("id") or "").strip()
    if not photo_id:
        return photo or {}

    if photo_id in _UNSPLASH_DETAIL_CACHE:
        detail = _UNSPLASH_DETAIL_CACHE[photo_id]
        return {**(photo or {}), **(detail or {})}

    try:
        r = requests.get(
            f"https://api.unsplash.com/photos/{photo_id}",
            headers=headers,
            timeout=6,
        )
        r.raise_for_status()
        detail = r.json() or {}
    except Exception:
        detail = {}

    _UNSPLASH_DETAIL_CACHE[photo_id] = detail
    return {**(photo or {}), **detail}


def _track_unsplash_download(candidate: dict) -> None:
    """Notify Unsplash when a selected API photo is actually used.

    This is a best-effort analytics/compliance call. Selection must never fail
    merely because Unsplash's tracking endpoint is temporarily unavailable.
    """
    if IMAGES_OFF or not UNSPLASH_ACCESS_KEY or not candidate:
        return

    download_location = str(candidate.get("_download_location") or "").strip()
    if not download_location:
        return

    try:
        requests.get(
            download_location,
            headers={
                "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}",
                "Accept-Version": "v1",
            },
            timeout=5,
        ).raise_for_status()
    except Exception as e:
        print(f"ℹ️ Unsplash download tracking failed: {e}")


def _unsplash_scoring_text(photo: dict) -> tuple[str, str, str]:
    """Return display alt/description plus richer metadata used only for scoring."""
    raw_alt = str((photo or {}).get("alt_description") or "").strip()
    raw_desc = str((photo or {}).get("description") or "").strip()

    alt = re.sub(r"\s+", " ", raw_alt or raw_desc).strip()
    desc = re.sub(r"\s+", " ", raw_desc or raw_alt).strip()

    tag_titles = []
    for tag in (photo or {}).get("tags") or []:
        if isinstance(tag, dict):
            value = str(tag.get("title") or "").strip()
        else:
            value = str(tag or "").strip()
        if value:
            tag_titles.append(value)

    location = (photo or {}).get("location") or {}
    location_parts = []
    if isinstance(location, dict):
        for key in ("name", "city", "country"):
            value = str(location.get(key) or "").strip()
            if value and value not in location_parts:
                location_parts.append(value)

    scoring_parts = [alt, desc]
    if tag_titles:
        scoring_parts.append("Tags: " + "; ".join(tag_titles[:12]))
    if location_parts:
        scoring_parts.append("Location: " + ", ".join(location_parts))

    scoring_text = re.sub(r"\s+", " ", " ".join(p for p in scoring_parts if p)).strip()
    return alt, desc, scoring_text


def _unsplash_display_alt(image_query: str, alt: str, desc: str) -> str:
    """Choose the more accessible provider caption without affecting scoring.

    Unsplash's ``alt_description`` can be visually literal but context-poor.
    When the provider's description has a stronger explicit match to the
    selected image anchor, prefer that description as the user-facing alt text.
    Relevance scoring continues to use the original provider fields unchanged.
    """
    clean_alt = re.sub(r"\s+", " ", alt or "").strip()
    clean_desc = re.sub(r"\s+", " ", desc or "").strip()
    if not clean_desc:
        return clean_alt
    if not clean_alt:
        return clean_desc

    alt_match = anchor_match_score(image_query, clean_alt)
    desc_match = anchor_match_score(image_query, clean_desc)
    return clean_desc if desc_match > alt_match else clean_alt


def fetch_unsplash_image(image_query: str, topic_text: str = "", headline: str = "", core_text: str = ""):
    """
    Return the strongest Unsplash candidate for one entity anchor.

    Unsplash remains a neutral-context fallback only. Generic protest, police,
    war, explosion, or other event-action imagery cannot stand in for the
    reported event.
    """
    if IMAGES_OFF or not UNSPLASH_ACCESS_KEY or not image_query:
        return None

    topic_text_clean = (topic_text or headline or core_text or image_query).strip()
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}",
        "Accept-Version": "v1",
    }
    query_anchor_type = classify_context_anchor(image_query)

    try:
        params = {"query": image_query, "per_page": 16, "orientation": "landscape"}
        r = requests.get("https://api.unsplash.com/search/photos", headers=headers, params=params, timeout=8)
        r.raise_for_status()
        results = (r.json() or {}).get("results") or []
    except Exception as e:
        print(f"ℹ️ Unsplash fetch failed: {e}")
        return None

    best = None
    best_score = -1e9

    for result_rank, photo in enumerate(results):
        scoring_photo = photo
        alt, desc, scoring_text = _unsplash_scoring_text(scoring_photo)
        if not scoring_text:
            continue

        metrics = _candidate_context_metrics(
            image_query, topic_text_clean, alt, scoring_text
        )
        metrics["provider_metadata_anchor_match"] = False

        # Search results are abbreviated objects. For the first few results of
        # a named-event query, fetch full metadata only when the sparse search
        # result would otherwise fail. Full photo objects add provider tags and
        # location, which are stronger evidence than weakening relevance gates.
        if (
            query_anchor_type == "event"
            and result_rank < 3
            and not _context_candidate_is_safe(metrics)
        ):
            scoring_photo = _unsplash_full_photo(photo, headers)
            alt, desc, scoring_text = _unsplash_scoring_text(scoring_photo)
            if not scoring_text:
                continue
            metrics = _candidate_context_metrics(
                image_query, topic_text_clean, alt, scoring_text
            )
            metrics["provider_metadata_anchor_match"] = (
                scoring_photo is not photo
                and anchor_match_score(image_query, scoring_text) > 0
            )

        # Unsplash remains a broad pool for places/venues/context. For people,
        # sparse captions make it too easy to select symbolic or unrelated
        # imagery, so fail closed to Wikimedia-or-none.
        if metrics.get("anchor_type") == "person":
            continue

        if not _context_candidate_is_safe(metrics):
            continue

        image_text = scoring_text

        temporal = _temporal_context(
            metrics.get("anchor_type") or "other",
            _unsplash_image_year(scoring_photo),
        )
        if not temporal["allowed"]:
            continue

        semantic = float(metrics["topic_semantic"])
        core_semantic = semantic_similarity(core_text, image_text) if core_text else semantic
        quality = _context_quality_score(metrics)

        if max(semantic, core_semantic) < 0.18 and quality < 2.0:
            continue

        urls = photo.get("urls") or {}
        raw = urls.get("raw")
        if raw:
            sep = "&" if "?" in raw else "?"
            url = f"{raw}{sep}w=900&fit=max&q=80"
        else:
            url = urls.get("regular") or urls.get("small")
        if not url:
            continue

        user = scoring_photo.get("user") or photo.get("user") or {}
        photographer = (user.get("name") or "").strip()
        photo_page = _unsplash_referral_url(
            ((scoring_photo.get("links") or photo.get("links") or {}).get("html") or "").strip()
        )
        photographer_page = _unsplash_referral_url(
            ((user.get("links") or {}).get("html") or "").strip()
        )

        credit = "Photo: " + (f"{photographer} / Unsplash" if photographer else "Unsplash")

        blended_semantic = max(semantic, core_semantic)
        event_retrieval_bonus = 30.0 if (
            metrics.get("anchor_type") == "event"
            and metrics.get("scene_type") == "event"
            and int(metrics.get("anchor_score", 0)) <= 0
        ) else 0.0

        score = (
            (float(metrics["anchor_score"]) * 20.0)
            + event_retrieval_bonus
            + (blended_semantic * 28.0)
            + (quality * 3.0)
            + (float(temporal["freshness"]) * 16.0)
        )

        metrics_for_confidence = dict(metrics)
        metrics_for_confidence["topic_semantic"] = blended_semantic
        confidence = _selection_confidence(metrics_for_confidence)


        display_alt = _unsplash_display_alt(image_query, alt, desc)

        candidate = {
            "url": url,
            "alt": display_alt,
            "description": desc,
            "credit": credit,
            "source": "unsplash",
            "source_url": photo_page,
            "license": "Unsplash License",
            "license_url": "https://unsplash.com/license",
            "photographer_url": photographer_page,
            "provider_url": "https://unsplash.com/?utm_source=nominal_news&utm_medium=referral",
            "query": image_query,
            "_download_location": (
                ((scoring_photo.get("links") or photo.get("links") or {}).get("download_location") or "").strip()
            ),
            "_selection_score": score,
            "_confidence": confidence,
            "_context_quality": quality,
            "_semantic": blended_semantic,
            "_anchor_match": metrics["anchor_score"],
            "_title_focus": metrics["title_focus"],
            "_foreign_ratio": metrics["foreign_ratio"],
            "_anchor_type": metrics["anchor_type"],
            "_scene_type": metrics["scene_type"],
            "_image_year": temporal["image_year"],
            "_image_age_years": temporal["age_years"],
            "_freshness": temporal["freshness"],
            "_metadata_enriched": scoring_photo is not photo,
        }

        if score > best_score:
            best = candidate
            best_score = score

    return best


def choose_best_context_candidate(candidates: list[dict], source: str) -> dict | None:
    """
    Compare the strongest candidate from every entity anchor instead of taking
    the first acceptable hit.

    Earlier/more-central anchors get only a small tie-break bonus. Relevance and
    contextual quality remain dominant. Borderline ambiguous candidates fail
    closed to no image.
    """
    if not candidates:
        return None

    ranked = []
    for candidate in candidates:
        rank = int(candidate.get("_anchor_rank", 0))
        anchor_words = _normalized_words(candidate.get("query") or "")

        rank_bonus = max(0.0, 4.0 - (rank * 2.0))
        specificity_bonus = 2.0 if len(anchor_words) >= 2 else 0.0

        final_score = (
            float(candidate.get("_selection_score", 0.0))
            + rank_bonus
            + specificity_bonus
        )
        candidate["_final_selection_score"] = final_score
        ranked.append(candidate)

    ranked.sort(
        key=lambda c: (
            -float(c.get("_final_selection_score", 0.0)),
            -float(c.get("_freshness", 0.55)),
            -float(c.get("_confidence", 0.0)),
            -float(c.get("_context_quality", 0.0)),
        )
    )

    best = ranked[0]
    confidence_floor = 0.48 if source == "wikimedia" else 0.50
    if float(best.get("_confidence", 0.0)) < confidence_floor:
        return None

    if len(ranked) > 1 and float(best.get("_confidence", 0.0)) < 0.58:
        margin = (
            float(best.get("_final_selection_score", 0.0))
            - float(ranked[1].get("_final_selection_score", 0.0))
        )
        if margin < 2.0:
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
used_context_image_urls: set[str] = set()

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

    # Entity-first context-image queries. These are derived only from the
    # purifier-approved core headlines; generic event concepts are not used as
    # standalone image fallbacks.
    topic_text = f"{headline}. {body}".strip()
    central_title = most_central_title(all_articles)
    query_candidates = build_context_image_queries(
        headline=headline,
        articles=all_articles,
        central_title=central_title,
        max_queries=3,
    )
    # Do not auto-expand anchors from prose. The original extracted entity
    # is safer than inventing a longer phrase such as "Greenland Prime".

    # Evaluate every anchor before choosing. Each fetcher already ranks all
    # provider results for that anchor; this second stage lets those anchor
    # winners compete with one another instead of accepting the first hit.
    wiki_candidates = []
    for anchor_rank, q in enumerate(query_candidates):
        candidate = fetch_wikimedia_image(q, topic_text=topic_text)
        if candidate and candidate.get("url") not in used_context_image_urls:
            candidate["_anchor_rank"] = anchor_rank
            wiki_candidates.append(candidate)

    img = choose_best_context_candidate(wiki_candidates, source="wikimedia")
    if img:
        print(
            f"🖼️ Topic {idx+1}: Wikimedia context hit "
            f"(query='{img.get('query', '')}', license='{img.get('license', '')}', "
            f"type='{img.get('_anchor_type', '')}', "
            f"scene='{img.get('_scene_type', '')}', "
            f"year={img.get('_image_year') if img.get('_image_year') is not None else '?'}, "
            f"freshness={img.get('_freshness', 0.55):.2f}, "
            f"confidence={img.get('_confidence', 0.0):.2f}, "
            f"focus={img.get('_title_focus', 0.0):.2f})"
        )
    else:
        print(
            f"🖼️ Topic {idx+1}: Wikimedia context miss/ambiguous "
            f"(tried {len(query_candidates)} anchors)"
        )

    # Unsplash remains a larger-pool fallback, but its strongest result from
    # every anchor must pass the same contextual confidence test.
    if not img:
        unsplash_candidates = []
        for anchor_rank, q in enumerate(query_candidates):
            candidate = fetch_unsplash_image(
                q,
                topic_text=topic_text,
                headline=headline,
                core_text=central_title,
            )
            if candidate and candidate.get("url") not in used_context_image_urls:
                candidate["_anchor_rank"] = anchor_rank
                unsplash_candidates.append(candidate)

        img = choose_best_context_candidate(unsplash_candidates, source="unsplash")
        if img:
            print(
                f"🖼️ Topic {idx+1}: Unsplash context hit "
                f"(query='{img.get('query', '')}', type='{img.get('_anchor_type', '')}', "
                f"scene='{img.get('_scene_type', '')}', "
                f"year={img.get('_image_year') if img.get('_image_year') is not None else '?'}, "
                f"freshness={img.get('_freshness', 0.55):.2f}, "
                f"confidence={img.get('_confidence', 0.0):.2f}, "
                f"focus={img.get('_title_focus', 0.0):.2f})"
            )
        else:
            print(
                f"🖼️ Topic {idx+1}: No defensible context image "
                f"(tried {len(query_candidates)} anchors)"
            )

    if img and img.get("url"):
        used_context_image_urls.add(img["url"])
        if img.get("source") == "unsplash":
            _track_unsplash_download(img)

    image_query = (img or {}).get("query") or (query_candidates[0] if query_candidates else "")
    image_url = img["url"] if img else ""
    image_alt = img["alt"] if img else ""
    image_description = img["description"] if img else ""
    image_credit = img["credit"] if img else ""
    image_source = img["source"] if img else ""
    image_source_url = img["source_url"] if img else ""
    image_license = img["license"] if img else ""
    image_license_url = img.get("license_url", "") if img else ""
    image_photographer_url = img.get("photographer_url", "") if img else ""
    image_provider_url = img.get("provider_url", "") if img else ""

    # Bias/source breadth is outlet-based: one receipt per unique domain.
    bias_counts = compute_bias_counts(display_sources)
    bias_dist = compute_bias_distribution(display_sources)
    if not bias_dist:
        print(f"⚠️ Cluster {idx} has no bias_distribution field")

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
        "image_license_url": image_license_url,
        "image_photographer_url": image_photographer_url,
        "image_provider_url": image_provider_url,
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

tmp_file = OUTPUT_FILE + ".tmp"
with open(tmp_file, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)

os.replace(tmp_file, OUTPUT_FILE)
print(f"✅ Saved top summaries to {OUTPUT_FILE}")
