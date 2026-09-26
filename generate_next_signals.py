# generate_next_signals.py — add optional Next Signal fields without touching story summaries

import argparse
import hashlib
import html as html_lib
import ipaddress
import json
import os
import re
from datetime import datetime, date, timedelta
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse, urljoin
from urllib.request import Request, urlopen

try:
    import openai
except ImportError:
    openai = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


load_dotenv()
if openai is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = os.getenv("NN_NEXT_SIGNAL_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
PROMPT_VERSION = "next-signal-v9-grounded-gate-2026-09-25"
TRIAGE_PROMPT_VERSION = "next-signal-triage-v2-grounded-2026-09-25"
FINAL_PROMPT_VERSION = "next-signal-final-v2-grounded-2026-09-25"
CACHE_PATH = "next_signal_cache.json"
SIDECAR_CACHE_PATH = "next_signal_sidecar_cache.json"
MAX_EVIDENCE_ARTICLES = 12
MAX_DESCRIPTION_CHARS = 420
PREFILTER_VERSION = "next-signal-prefilter-v6-grounded-2026-09-25"
MAX_SIDECAR_CANDIDATES = 10
MAX_SIDECAR_FETCHES = 6
MAX_SIDECAR_MATCHES = 8
MAX_SIDECAR_SNIPPETS_PER_SOURCE = 2
MAX_FETCH_BYTES = 1_500_000
MAX_PAGE_TEXT_CHARS = 40_000
MAX_SIDECAR_EXCERPT_CHARS = 700
SIDECAR_SIMILARITY_MIN = 0.22

# Generic final-stage evidence discovery. This is intentionally domain-agnostic and provider-resilient:
# it searches for coverage of the already-finalized story, then applies the same
# future-milestone extractor and strict model validator used everywhere else.
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MAX_EXTERNAL_SEARCH_RESULTS = 30
MAX_EXTERNAL_CANDIDATES = 10
MAX_EXTERNAL_FETCHES = 6
MAX_EXTERNAL_MATCHES = 6
MAX_MODEL_EVIDENCE = 8
EXTERNAL_SIMILARITY_MIN = 0.18
EXTERNAL_SEARCH_WINDOW_DAYS = 2

_EXTERNAL_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "over", "the", "their", "to",
    "with", "after", "amid", "against", "during", "off", "post", "plans", "plan", "says",
    "said", "holds", "hold", "addresses", "address", "discuss", "discusses", "signs",
    "sign", "launches", "launch", "rejects", "reject", "granted", "grant", "major",
}

# Cheap, local gate: require evidence of both a concrete future time/window and
# a plausible milestone before spending an API call. These are durable temporal
# and event-structure cues, not story/topic vocabulary. The GPT check remains
# the final authority and can still reject every prefilter candidate.
_MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_RE = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"(?:\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?)?(?:,?\s+(?P<year>20\d{2}))?\b",
    re.I,
)
_WEEKDAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_WEEKDAY_RE = re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
_NUMERIC_DATE_RE = re.compile(
    r"\b(?:(?P<ymd_year>20\d{2})[-/](?P<ymd_month>\d{1,2})[-/](?P<ymd_day>\d{1,2})|"
    r"(?P<md_month>\d{1,2})[-/](?P<md_day>\d{1,2})(?:[-/](?P<md_year>20\d{2}))?)\b"
)
_RELATIVE_TIME_RE = re.compile(
    r"\b(?:tomorrow|tonight|next\s+(?:weekend|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"later\s+(?:today|this\s+week|this\s+month|this\s+year)|"
    r"this\s+(?:weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"within\s+\d+\s+(?:hours?|days?|weeks?|months?)|in\s+\d+\s+(?:hours?|days?|weeks?|months?)|"
    r"by\s+the\s+end\s+of\s+(?:the\s+)?(?:day|week|month|year)|over\s+the\s+weekend)\b",
    re.I,
)
_FUTURE_FRAMING_RE = re.compile(
    r"\b(?:scheduled|slated|set\s+(?:to|for)|expected|forecast|planned|plans\s+to|"
    r"will|is\s+to|are\s+to|looms?|upcoming|on\s+track\s+to|due(?!\s+to\b))\b",
    re.I,
)
# A broad structural cue that something concrete is scheduled/expected to happen.
# The downstream model still decides whether it truly advances the exact story.
_MILESTONE_RE = re.compile(
    r"\b(?:vote|voting|hearing|trial|ruling|decision|verdict|sentenc(?:e|ing)|"
    r"meet(?:ing)?|talks?|negotiations?|summit|deadline|report|release|findings?|results?|"
    r"election|referendum|launch|landfall|arriv(?:e|al)|approach|testif(?:y|ies|ied)|"
    r"resume|reopen|begin|start|take\s+effect|expire|expiration|court\s+date|"
    r"scheduled|slated|set\s+(?:to|for)|due(?!\s+to\b)|expected|forecast)\b",
    re.I,
)

# Deterministic grounding rules used before and after GPT. These are generic
# event-structure rules, not topic/domain rules. Their purpose is to prevent a
# model call unless the same evidence passage contains a real future time anchor
# tied to a future action, and to reject any generated timing that the source did
# not actually state.
_REPORTING_VERB_RE = re.compile(
    r"\b(?:said|told|reported|wrote|posted|announced|stated|added|noted|according\s+to)\b",
    re.I,
)
_FINALITY_RE = re.compile(
    r"\b(?:final|last)\s+(?:chance|opportunit(?:y|ies)|votes?|hearings?|meetings?|sessions?|rounds?|appearances?)\b|"
    r"\bno\s+(?:further|more|additional|new)\b|"
    r"\bnot\s+(?:scheduled|planned|expected|set)\b|"
    r"\b(?:nothing|none)\s+(?:is\s+|are\s+)?(?:scheduled|planned|expected)\b|"
    r"\b(?:unlikely|not\s+likely)\b",
    re.I,
)
_ROUTINE_ACTIVITY_RE = re.compile(
    r"\b(?:tour|visit|arriv(?:e|al)|depart(?:ure)?|leave|return|travel|reception|ceremony|gala|"
    r"photo\s+op(?:portunity)?|holiday|vacation|appearance|attend|speech|address)\b",
    re.I,
)
_SUBSTANTIVE_ACTIVITY_RE = re.compile(
    r"\b(?:vote|hearing|trial|ruling|decision|verdict|sentenc(?:e|ing)|talks?|negotiations?|"
    r"summit|deadline|report|release|findings?|results?|election|referendum|launch|landfall|"
    r"approach|take\s+effect|expire|expiration|court\s+date|investigation|inquiry|forecast|"
    r"agreement|deal|announcement|approval|filing|appeal)\b",
    re.I,
)
_UNCERTAINTY_RE = re.compile(
    r"\b(?:forecast|expected|could|may|might|likely|possible|possibly|as\s+early\s+as|projected)\b",
    re.I,
)
_CONTINUATION_SIGNAL_RE = re.compile(
    r"\b(?:further|more|additional|another|next)\s+(?:vote|hearing|meeting|session|round|appearance)s?\b",
    re.I,
)
_UNSUPPORTED_VAGUE_TIME_RE = re.compile(
    r"\b(?:soon|shortly|eventually|in\s+the\s+coming\s+days|in\s+coming\s+days)\b",
    re.I,
)
_RULE_CONTEXT_RE = re.compile(
    r"\b(?:dictates|requires?|required\s+to|must\s+notify|caps|permits?|allows?)\b",
    re.I,
)
_PAST_EVENT_RE = re.compile(
    r"\b(?:grew|became|occurred|happened|met|voted|signed|approved|rejected|granted|sentenced|"
    r"arrived|left|spoke|addressed|announced|reported|said|told|opened|closed|ended|concluded)\b",
    re.I,
)


class _ArticleHTMLTextParser(HTMLParser):
    """Small stdlib-only article-text extractor for signal evidence."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._in_jsonld = False
        self._jsonld_parts = []
        self._jsonld_scripts = []
        self._capture_p = False
        self._p_parts = []
        self._paragraphs = []
        self._article_depth = 0
        self._main_depth = 0
        self._article_paragraphs = []
        self.meta_description = ""

    def handle_starttag(self, tag, attrs):
        tag = (tag or "").lower()
        attr_map = {str(k).lower(): str(v or "") for k, v in attrs}
        if tag == "script" and "ld+json" in attr_map.get("type", "").lower():
            self._in_jsonld = True
            self._jsonld_parts = []
        if tag == "article":
            self._article_depth += 1
        if tag == "main":
            self._main_depth += 1
        if tag == "p":
            self._capture_p = True
            self._p_parts = []
        if tag == "meta" and not self.meta_description:
            name = attr_map.get("name", "").lower()
            prop = attr_map.get("property", "").lower()
            if name == "description" or prop == "og:description":
                self.meta_description = _clean_text(attr_map.get("content", ""))

    def handle_endtag(self, tag):
        tag = (tag or "").lower()
        if tag == "script" and self._in_jsonld:
            raw = "".join(self._jsonld_parts).strip()
            if raw:
                self._jsonld_scripts.append(raw)
            self._jsonld_parts = []
            self._in_jsonld = False
        if tag == "p" and self._capture_p:
            text = _clean_text(" ".join(self._p_parts))
            if len(text) >= 35:
                self._paragraphs.append(text)
                if self._article_depth > 0 or self._main_depth > 0:
                    self._article_paragraphs.append(text)
            self._capture_p = False
            self._p_parts = []
        if tag == "article" and self._article_depth > 0:
            self._article_depth -= 1
        if tag == "main" and self._main_depth > 0:
            self._main_depth -= 1

    def handle_data(self, data):
        if self._in_jsonld:
            self._jsonld_parts.append(data)
        if self._capture_p:
            self._p_parts.append(data)

    @property
    def jsonld_texts(self):
        return list(self._jsonld_scripts)

    @property
    def paragraphs(self):
        return list(self._paragraphs)

    @property
    def article_paragraphs(self):
        return list(self._article_paragraphs)


def _walk_article_bodies(value):
    if isinstance(value, dict):
        body = value.get("articleBody")
        if isinstance(body, str) and body.strip():
            yield _clean_text(body)
        for child in value.values():
            yield from _walk_article_bodies(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_article_bodies(child)


def _extract_article_text(html_text: str) -> str:
    if not html_text:
        return ""
    parser = _ArticleHTMLTextParser()
    try:
        parser.feed(html_text)
    except Exception:
        pass

    # JSON-LD articleBody is the most precise source when publishers expose it.
    bodies = []
    for raw in parser.jsonld_texts:
        try:
            payload = json.loads(html_lib.unescape(raw.strip()))
        except Exception:
            continue
        bodies.extend(body for body in _walk_article_bodies(payload) if body)
    if bodies:
        return max(bodies, key=len)[:MAX_PAGE_TEXT_CHARS]

    paragraphs = parser.article_paragraphs or parser.paragraphs
    if paragraphs:
        return _clean_text(" ".join(paragraphs))[:MAX_PAGE_TEXT_CHARS]
    return parser.meta_description[:MAX_PAGE_TEXT_CHARS]


def _safe_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url or "")
        if parsed.scheme not in {"http", "https"}:
            return False
        host = (parsed.hostname or "").strip().lower()
        if not host or host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            return False
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        except ValueError:
            pass
        return True
    except Exception:
        return False


def _fetch_article_text(url: str, sidecar_cache: dict) -> tuple[str, dict, bool]:
    """Fetch article text for signal-only evidence. Returns text, metadata, cache_dirty."""
    canonical = _canonical_url(url)
    if not canonical or not _safe_public_url(canonical):
        return "", {"status": "rejected_url", "url": canonical or url}, False

    cached = sidecar_cache.get(canonical)
    if isinstance(cached, dict):
        return str(cached.get("text") or ""), dict(cached), False

    meta = {"status": "fetch_failed", "url": canonical, "final_url": canonical, "text": ""}
    try:
        request = Request(
            canonical,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
            },
        )
        with urlopen(request, timeout=8) as response:
            final_url = response.geturl() or canonical
            if not _safe_public_url(final_url):
                raise ValueError("redirected to a non-public URL")
            content_type = (response.headers.get("Content-Type") or "").lower()
            if content_type and "html" not in content_type and "text/" not in content_type:
                raise ValueError(f"unsupported content type: {content_type}")
            raw = response.read(MAX_FETCH_BYTES + 1)
            if len(raw) > MAX_FETCH_BYTES:
                raw = raw[:MAX_FETCH_BYTES]
            charset = response.headers.get_content_charset() or "utf-8"
            page = raw.decode(charset, errors="replace")
            text = _extract_article_text(page)
            meta = {
                "status": "ok" if text else "no_article_text",
                "url": canonical,
                "final_url": final_url,
                "chars": len(text),
                "text": text,
            }
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        meta = {
            "status": "fetch_failed",
            "url": canonical,
            "final_url": canonical,
            "error": _clean_text(str(exc))[:180],
            "text": "",
        }
    except Exception as exc:
        meta = {
            "status": "fetch_failed",
            "url": canonical,
            "final_url": canonical,
            "error": _clean_text(str(exc))[:180],
            "text": "",
        }

    sidecar_cache[canonical] = meta
    return str(meta.get("text") or ""), dict(meta), True


def _fetch_raw_document(url: str, sidecar_cache: dict, cache_prefix: str) -> tuple[str, dict, bool]:
    """Fetch raw text/HTML/JSON for signal-only enrichment without mutating frozen story data."""
    canonical = _canonical_url(url)
    if not canonical or not _safe_public_url(canonical):
        return "", {"status": "rejected_url", "url": canonical or url}, False

    key = f"{cache_prefix}::{canonical}"
    cached = sidecar_cache.get(key)
    if isinstance(cached, dict):
        return str(cached.get("raw") or ""), dict(cached), False

    meta = {"status": "fetch_failed", "url": canonical, "final_url": canonical, "raw": ""}
    try:
        request = Request(
            canonical,
            headers={
                "User-Agent": "NominalNews/1.0",
                "Accept": "text/html,application/xhtml+xml,text/plain,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
            },
        )
        with urlopen(request, timeout=8) as response:
            final_url = response.geturl() or canonical
            if not _safe_public_url(final_url):
                raise ValueError("redirected to a non-public URL")
            raw_bytes = response.read(MAX_FETCH_BYTES + 1)
            if len(raw_bytes) > MAX_FETCH_BYTES:
                raw_bytes = raw_bytes[:MAX_FETCH_BYTES]
            charset = response.headers.get_content_charset() or "utf-8"
            raw = raw_bytes.decode(charset, errors="replace")
            meta = {
                "status": "ok" if raw else "empty",
                "url": canonical,
                "final_url": final_url,
                "chars": len(raw),
                "raw": raw,
            }
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        meta = {
            "status": "fetch_failed",
            "url": canonical,
            "final_url": canonical,
            "error": _clean_text(str(exc))[:180],
            "raw": "",
        }
    except Exception as exc:
        meta = {
            "status": "fetch_failed",
            "url": canonical,
            "final_url": canonical,
            "error": _clean_text(str(exc))[:180],
            "raw": "",
        }

    sidecar_cache[key] = meta
    return str(meta.get("raw") or ""), dict(meta), True


def _sentence_windows(text: str) -> list[str]:
    clean = _clean_text(text)
    if not clean:
        return []
    parts = [
        _clean_text(part)
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])", clean)
        if _clean_text(part)
    ]
    windows = []
    for idx, sentence in enumerate(parts):
        windows.append(sentence)
        if idx > 0:
            windows.append(_clean_text(parts[idx - 1] + " " + sentence))
        if idx + 1 < len(parts):
            windows.append(_clean_text(sentence + " " + parts[idx + 1]))
    return windows


def _future_milestone_snippets(text: str, reference_date: date) -> list[dict]:
    clean = _clean_text(text)
    if not clean:
        return []

    parts = [
        _clean_text(part)
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])", clean)
        if _clean_text(part)
    ]
    if not parts:
        return []

    vague_subject_re = re.compile(
        r"^(?:it|they|he|she|this|that|the\s+[a-z][a-z-]{2,}(?:\s+[a-z][a-z-]{2,}){0,2})\b",
        re.I,
    )

    ranked = []
    seen = set()
    for idx, sentence in enumerate(parts):
        direct_cues = _future_time_cues(sentence, reference_date)
        direct_qualifies = bool(
            direct_cues
            and (_MILESTONE_RE.search(sentence) or _FUTURE_FRAMING_RE.search(sentence))
        )

        pair = ""
        pair_cues = []
        pair_qualifies = False
        if idx + 1 < len(parts):
            pair = _clean_text(sentence + " " + parts[idx + 1])
            pair_cues = _future_time_cues(pair, reference_date)
            pair_qualifies = bool(
                pair_cues
                and (_MILESTONE_RE.search(pair) or _FUTURE_FRAMING_RE.search(pair))
            )

        if not direct_qualifies and not pair_qualifies:
            continue

        if direct_qualifies:
            trigger = sentence
            cues = direct_cues
            # Preserve the named antecedent when the signal sentence starts with a
            # vague subject such as "the storm" or "it". This is critical when one
            # article discusses multiple events.
            if idx > 0 and vague_subject_re.search(sentence):
                excerpt = _clean_text(parts[idx - 1] + " " + sentence)
            else:
                excerpt = sentence
        else:
            trigger = pair
            cues = pair_cues
            excerpt = pair

        excerpt = excerpt[:MAX_SIDECAR_EXCERPT_CHARS]
        key = _normalize_title(excerpt[:350])
        if not key or key in seen:
            continue
        seen.add(key)

        score = len(cues)
        if _FUTURE_FRAMING_RE.search(trigger):
            score += 2
        score += min(3, len(_MILESTONE_RE.findall(trigger)))
        ranked.append({
            "text": excerpt,
            "time_cues": cues,
            "score": score,
        })

    ranked.sort(key=lambda row: (-row["score"], -len(row["text"])))
    final = []
    final_norms = []
    for row in ranked:
        norm = _normalize_title(row["text"])
        if any(norm in existing or existing in norm for existing in final_norms):
            continue
        final.append(row)
        final_norms.append(norm)
        if len(final) >= 3:
            break
    return final

def _default_article_pool_file(date_str: str) -> str:
    candidates = [
        f"articles_with_bias_{date_str}.json",
        f"articles_raw_normalized_{date_str}.json",
        f"articles_raw_{date_str}.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _candidate_from_row(row: dict, origin: str, priority: float = 0.0) -> dict | None:
    if not isinstance(row, dict):
        return None
    url = row.get("url") or row.get("canonical_url") or row.get("url_normalized") or ""
    title = _clean_text(row.get("title") or row.get("representative_title") or "")
    desc = _clean_text(row.get("description") or "")[:MAX_DESCRIPTION_CHARS]
    if not url or not title:
        return None
    return {
        "origin": origin,
        "priority": float(priority),
        "title": title,
        "description": desc,
        "url": url,
        "published_at": str(row.get("published_at") or row.get("published_date") or row.get("target_date") or ""),
        "source": row.get("source") or row.get("domain") or "",
        "family_id": row.get("writeup_family_id") or row.get("family_id") or "",
    }


def _story_profile_text(topic: dict, cluster: dict) -> str:
    parts = [
        topic.get("topic_title") or "",
        topic.get("summary") or "",
        _cluster_label(cluster),
    ]
    parts.extend(
        article.get("title") or ""
        for article in (cluster.get("articles") or [])[:6]
    )
    return _clean_text(" ".join(part for part in parts if part))


def _rank_sidecar_candidates(topic: dict, cluster: dict, evidence: list[dict], article_pool: list[dict]) -> list[dict]:
    candidates = []

    # First include the exact already-associated URLs so full text can reveal details
    # that were absent from RSS title/description metadata.
    for row in evidence:
        candidate = _candidate_from_row(row, "existing_story_source", priority=0.14)
        if candidate:
            candidates.append(candidate)

    # Daily article pool is intentionally outside the frozen cluster. It is read-only
    # signal evidence and can recover same-story articles that clustering excluded.
    for row in article_pool or []:
        candidate = _candidate_from_row(row, "daily_article_pool", priority=0.10)
        if candidate:
            candidates.append(candidate)

    # The finalized cluster also contains a broader receipt universe; use it only as
    # candidate discovery, never as cluster membership mutation.
    for row in cluster.get("coverage_sources") or []:
        candidate = _candidate_from_row(row, "coverage_receipt", priority=0.05)
        if candidate:
            candidates.append(candidate)
    for row in cluster.get("gdelt_global_source_receipts") or []:
        candidate = _candidate_from_row(row, "global_receipt", priority=0.03)
        if candidate:
            candidates.append(candidate)
    for family in cluster.get("gdelt_global_writeup_families") or []:
        row = {
            "url": family.get("representative_url"),
            "title": family.get("representative_title"),
            "family_id": family.get("family_id"),
            "target_date": family.get("target_date") or "",
        }
        candidate = _candidate_from_row(row, "global_family_representative", priority=0.04)
        if candidate:
            candidates.append(candidate)

    # Canonical URL dedupe, preserving the highest-priority form of each source.
    by_url = {}
    for row in candidates:
        canonical = _canonical_url(row.get("url") or "")
        if not canonical:
            continue
        row = dict(row)
        row["canonical_url"] = canonical
        previous = by_url.get(canonical)
        if previous is None or row["priority"] > previous["priority"]:
            by_url[canonical] = row
    candidates = list(by_url.values())
    if not candidates:
        return []

    profile = _story_profile_text(topic, cluster)
    candidate_texts = [
        _clean_text((row.get("title") or "") + " " + (row.get("description") or ""))
        for row in candidates
    ]

    scores = [0.0] * len(candidates)
    if TfidfVectorizer is not None and cosine_similarity is not None and profile:
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            matrix = vectorizer.fit_transform([profile] + candidate_texts)
            cosine_scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel().tolist()
            scores = [float(value) for value in cosine_scores]
        except Exception:
            scores = [0.0] * len(candidates)

    topic_title = topic.get("topic_title") or ""
    cluster_title = _cluster_label(cluster)
    for idx, row in enumerate(candidates):
        lexical = max(
            _title_overlap_score(row.get("title") or "", topic_title),
            _title_overlap_score(row.get("title") or "", cluster_title),
        )
        semantic = scores[idx] if idx < len(scores) else 0.0
        row["similarity"] = round(max(semantic, lexical), 4)
        row["rank_score"] = round((0.76 * semantic) + (0.24 * lexical) + row.get("priority", 0.0), 4)

    filtered = [
        row for row in candidates
        if row.get("similarity", 0.0) >= SIDECAR_SIMILARITY_MIN
        or row.get("origin") == "existing_story_source"
    ]
    filtered.sort(key=lambda row: (-row.get("rank_score", 0.0), -row.get("similarity", 0.0)))

    # Diversify likely syndicated copies using family id or normalized title. One
    # representative is enough to inspect a duplicated writeup's full text.
    selected = []
    seen_family = set()
    for row in filtered:
        family_key = row.get("family_id") or _normalize_title(row.get("title") or "")[:180]
        if family_key and family_key in seen_family:
            continue
        if family_key:
            seen_family.add(family_key)
        selected.append(row)
        if len(selected) >= MAX_SIDECAR_CANDIDATES:
            break
    return selected


def build_signal_sidecar(
    date_str: str,
    topic: dict,
    cluster: dict,
    evidence: list[dict],
    article_pool: list[dict],
    sidecar_cache: dict,
) -> tuple[list[dict], dict, bool]:
    """
    Read-only, domain-agnostic full-text enrichment for Next Signal.

    Important: collect evidence broadly before deciding what to keep. A useful
    future sentence can sit deeper in an article than an earlier, weaker future
    reference. Nothing collected here can modify the frozen cluster or summary.
    """
    try:
        briefing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        briefing_date = datetime.today().date()

    candidates = _rank_sidecar_candidates(topic, cluster, evidence, article_pool)
    raw_additions = []
    attempted = []
    cache_dirty = False
    network_fetches = 0
    page_attempts = 0

    def collect(candidate: dict, snippet: dict, origin: str, fetch_status: str = "") -> None:
        description = _clean_text(snippet.get("text") or "")
        if not description:
            return
        quality = float(snippet.get("score") or 0.0)
        quality += 1.25 * float(candidate.get("similarity") or 0.0)
        quality += 0.20 if origin == "signal_sidecar_fulltext" else 0.0
        raw_additions.append({
            "origin": origin,
            "title": candidate.get("title") or "",
            "description": description,
            "published_at": candidate.get("published_at") or "",
            "url": candidate.get("url") or "",
            "source": candidate.get("source") or "",
            "sidecar_source_origin": candidate.get("origin") or "",
            "sidecar_similarity": candidate.get("similarity"),
            "_signal_score": round(quality, 4),
            "_fetch_status": fetch_status,
        })

    # Pass 1: inspect metadata for every high-confidence story candidate. Do not
    # stop when a few weak snippets appear; they should not crowd out richer text.
    for candidate in candidates:
        reference_date = _parse_reference_date(candidate.get("published_at") or "", briefing_date)
        metadata_text = _clean_text(
            " ".join(
                part for part in (candidate.get("title") or "", candidate.get("description") or "") if part
            )
        )
        snippets = _future_milestone_snippets(metadata_text, reference_date)
        for snippet in snippets[:MAX_SIDECAR_SNIPPETS_PER_SOURCE]:
            collect(candidate, snippet, "signal_sidecar_metadata")
            attempted.append({
                "url": candidate.get("url"),
                "title": candidate.get("title"),
                "origin": candidate.get("origin"),
                "similarity": candidate.get("similarity"),
                "result": "metadata_milestone",
                "excerpt": snippet["text"],
            })

    # Pass 2: inspect full text from several strong, diversified sources even when
    # metadata already contained a possible milestone. This prevents an early,
    # shallow excerpt from hiding a better timed milestone deeper in the article.
    for candidate in candidates:
        if page_attempts >= MAX_SIDECAR_FETCHES:
            break
        page_attempts += 1
        reference_date = _parse_reference_date(candidate.get("published_at") or "", briefing_date)
        page_text, fetch_meta, changed = _fetch_article_text(candidate.get("url") or "", sidecar_cache)
        cache_dirty = cache_dirty or changed
        if changed:
            network_fetches += 1

        snippets = _future_milestone_snippets(page_text, reference_date) if page_text else []
        if snippets:
            for snippet in snippets[:MAX_SIDECAR_SNIPPETS_PER_SOURCE]:
                collect(candidate, snippet, "signal_sidecar_fulltext", fetch_meta.get("status", ""))
                attempted.append({
                    "url": candidate.get("url"),
                    "title": candidate.get("title"),
                    "origin": candidate.get("origin"),
                    "similarity": candidate.get("similarity"),
                    "result": "fulltext_milestone",
                    "fetch_status": fetch_meta.get("status"),
                    "excerpt": snippet["text"],
                })
        else:
            attempted.append({
                "url": candidate.get("url"),
                "title": candidate.get("title"),
                "origin": candidate.get("origin"),
                "similarity": candidate.get("similarity"),
                "result": "no_timed_milestone",
                "fetch_status": fetch_meta.get("status"),
                "fetch_error": fetch_meta.get("error", ""),
            })

    # Rank after collection, dedupe equivalent excerpts, and preserve source
    # diversity. This keeps the model evidence compact without prematurely
    # truncating discovery.
    raw_additions.sort(
        key=lambda row: (
            -float(row.get("_signal_score") or 0.0),
            0 if row.get("origin") == "signal_sidecar_fulltext" else 1,
            -len(row.get("description") or ""),
        )
    )
    additions = []
    seen_text = []
    per_url = {}
    for row in raw_additions:
        norm = _normalize_title(row.get("description") or "")
        if not norm:
            continue
        if any(norm in old or old in norm for old in seen_text):
            continue
        canonical = _canonical_url(row.get("url") or "")
        if canonical and per_url.get(canonical, 0) >= MAX_SIDECAR_SNIPPETS_PER_SOURCE:
            continue
        clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
        additions.append(clean_row)
        seen_text.append(norm)
        if canonical:
            per_url[canonical] = per_url.get(canonical, 0) + 1
        if len(additions) >= MAX_SIDECAR_MATCHES:
            break

    meta = {
        "candidate_count": len(candidates),
        "page_attempts": page_attempts,
        "network_fetches": network_fetches,
        "evidence_additions": len(additions),
        "attempted": attempted,
    }
    return additions, meta, cache_dirty



def _story_query_terms(topic: dict, cluster: dict, limit: int = 6) -> list[str]:
    """Build a compact, story-specific search query without topic-type rules."""
    title = _normalize_title(topic.get("topic_title") or "")
    label = _normalize_title(_cluster_label(cluster))
    title_tokens = [
        token for token in title.split()
        if len(token) >= 3 and token not in _EXTERNAL_QUERY_STOPWORDS
    ]
    label_tokens = [
        token for token in label.split()
        if len(token) >= 3 and token not in _EXTERNAL_QUERY_STOPWORDS
    ]
    label_set = set(label_tokens)

    ordered = []
    seen = set()
    # Tokens shared by the user-facing title and frozen canonical event are the
    # strongest generic identity anchors.
    for token in title_tokens:
        if token in label_set and token not in seen:
            seen.add(token)
            ordered.append(token)
    # Fill with additional headline anchors when overlap is sparse.
    for token in title_tokens + label_tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
        if len(ordered) >= limit:
            break
    return ordered[:limit]


def _build_external_search_queries(topic: dict, cluster: dict) -> list[str]:
    terms = _story_query_terms(topic, cluster, limit=6)
    if not terms:
        return []
    queries = []
    if len(terms) >= 5:
        queries.append(" ".join(terms[:5]))
    queries.append(" ".join(terms[:3] if len(terms) >= 3 else terms))
    out = []
    for query in queries:
        query = _clean_text(query)
        if query and query not in out:
            out.append(query)
    return out[:2]


def _gdelt_seen_to_iso(value: str) -> str:
    raw = str(value or "").strip()
    match = re.match(r"(20\d{2})(\d{2})(\d{2})(?:T?(\d{2})(\d{2})(\d{2})Z?)?", raw)
    if not match:
        return raw
    year, month, day = match.group(1), match.group(2), match.group(3)
    if match.group(4):
        return f"{year}-{month}-{day}T{match.group(4)}:{match.group(5)}:{match.group(6)}Z"
    return f"{year}-{month}-{day}"


def _search_gdelt_articles(
    query: str,
    briefing_date: date,
    sidecar_cache: dict,
) -> tuple[list[dict], dict, bool]:
    """Primary generic discovery provider. Results are evidence leads only."""
    start_date = briefing_date - timedelta(days=EXTERNAL_SEARCH_WINDOW_DAYS)
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(MAX_EXTERNAL_SEARCH_RESULTS),
        "format": "json",
        "sort": "datedesc",
        "startdatetime": start_date.strftime("%Y%m%d000000"),
        "enddatetime": briefing_date.strftime("%Y%m%d235959"),
    }
    url = GDELT_DOC_API + "?" + urlencode(params)
    raw, fetch_meta, dirty = _fetch_raw_document(url, sidecar_cache, "external-gdelt-search")
    meta = {
        "provider": "GDELT DOC 2.0",
        "query": query,
        "url": url,
        "status": fetch_meta.get("status", "fetch_failed"),
        "error": fetch_meta.get("error", ""),
        "result_count": 0,
    }
    if not raw:
        return [], meta, dirty
    try:
        payload = json.loads(raw)
    except Exception as exc:
        meta["status"] = "invalid_json"
        meta["error"] = _clean_text(str(exc))[:180]
        return [], meta, dirty

    articles = payload.get("articles") if isinstance(payload, dict) else []
    if not isinstance(articles, list):
        articles = []
    rows = []
    for item in articles:
        if not isinstance(item, dict):
            continue
        language = _clean_text(item.get("language") or "")
        if language and language.lower() not in {"english", "en"}:
            continue
        title = _clean_text(item.get("title") or "")
        url_value = item.get("url") or item.get("url_mobile") or ""
        if not title or not url_value:
            continue
        rows.append({
            "title": title,
            "description": "",
            "url": url_value,
            "published_at": _gdelt_seen_to_iso(item.get("seendate") or item.get("seenDate") or ""),
            "source": item.get("domain") or "",
            "domain": item.get("domain") or "",
            "origin": "external_gdelt_search",
            "search_provider": "GDELT DOC 2.0",
        })
    meta["result_count"] = len(rows)
    return rows, meta, dirty


def _search_newsapi_articles(
    query: str,
    briefing_date: date,
    sidecar_cache: dict,
) -> tuple[list[dict], dict, bool]:
    """
    Fallback generic discovery provider. The API key is sent in a header so it is
    never written into URLs, reports, or the sidecar cache.
    """
    start_date = briefing_date - timedelta(days=EXTERNAL_SEARCH_WINDOW_DAYS)
    params = {
        "q": query,
        "from": start_date.isoformat(),
        "to": briefing_date.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": str(MAX_EXTERNAL_SEARCH_RESULTS),
    }
    url = NEWSAPI_URL + "?" + urlencode(params)
    meta = {
        "provider": "NewsAPI",
        "query": query,
        "url": url,
        "status": "unavailable_no_api_key" if not NEWS_API_KEY else "fetch_failed",
        "error": "",
        "result_count": 0,
    }
    if not NEWS_API_KEY:
        return [], meta, False

    canonical = _canonical_url(url)
    cache_key = f"external-newsapi-search::{canonical}"
    cached = sidecar_cache.get(cache_key)
    dirty = False
    if isinstance(cached, dict):
        raw = str(cached.get("raw") or "")
        meta["status"] = cached.get("status", meta["status"])
        meta["error"] = cached.get("error", "")
    else:
        raw = ""
        try:
            request = Request(
                canonical,
                headers={
                    "User-Agent": "NominalNews/1.0",
                    "Accept": "application/json",
                    "X-Api-Key": NEWS_API_KEY,
                },
            )
            with urlopen(request, timeout=8) as response:
                raw_bytes = response.read(MAX_FETCH_BYTES + 1)
                if len(raw_bytes) > MAX_FETCH_BYTES:
                    raw_bytes = raw_bytes[:MAX_FETCH_BYTES]
                charset = response.headers.get_content_charset() or "utf-8"
                raw = raw_bytes.decode(charset, errors="replace")
                cache_value = {
                    "status": "ok" if raw else "empty",
                    "url": canonical,
                    "chars": len(raw),
                    "raw": raw,
                }
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            cache_value = {
                "status": "fetch_failed",
                "url": canonical,
                "error": _clean_text(str(exc))[:180],
                "raw": "",
            }
        except Exception as exc:
            cache_value = {
                "status": "fetch_failed",
                "url": canonical,
                "error": _clean_text(str(exc))[:180],
                "raw": "",
            }
        sidecar_cache[cache_key] = cache_value
        dirty = True
        meta["status"] = cache_value.get("status", "fetch_failed")
        meta["error"] = cache_value.get("error", "")
        raw = str(cache_value.get("raw") or "")

    if not raw:
        return [], meta, dirty
    try:
        payload = json.loads(raw)
    except Exception as exc:
        meta["status"] = "invalid_json"
        meta["error"] = _clean_text(str(exc))[:180]
        return [], meta, dirty

    if isinstance(payload, dict) and payload.get("status") == "error":
        meta["status"] = str(payload.get("code") or "api_error")
        meta["error"] = _clean_text(payload.get("message") or "")[:180]
        return [], meta, dirty

    articles = payload.get("articles") if isinstance(payload, dict) else []
    if not isinstance(articles, list):
        articles = []
    rows = []
    for item in articles:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title") or "")
        url_value = item.get("url") or ""
        if not title or not url_value:
            continue
        source_obj = item.get("source") if isinstance(item.get("source"), dict) else {}
        source_name = _clean_text(source_obj.get("name") or "")
        rows.append({
            "title": title,
            "description": _clean_text(item.get("description") or "")[:MAX_DESCRIPTION_CHARS],
            "url": url_value,
            "published_at": str(item.get("publishedAt") or ""),
            "source": source_name,
            "domain": urlparse(url_value).hostname or "",
            "origin": "external_newsapi_search",
            "search_provider": "NewsAPI",
        })
    meta["result_count"] = len(rows)
    return rows, meta, dirty


def _rank_external_candidates(
    topic: dict,
    cluster: dict,
    rows: list[dict],
    existing_evidence: list[dict],
) -> list[dict]:
    existing_urls = {
        _canonical_url(row.get("url") or "")
        for row in existing_evidence
        if _canonical_url(row.get("url") or "")
    }
    candidates = []
    seen_urls = set()
    for row in rows:
        candidate = _candidate_from_row(row, row.get("origin") or "external_search", priority=0.0)
        if not candidate:
            continue
        canonical = _canonical_url(candidate.get("url") or "")
        if not canonical or canonical in existing_urls or canonical in seen_urls:
            continue
        seen_urls.add(canonical)
        candidate["canonical_url"] = canonical
        candidate["search_provider"] = row.get("search_provider") or ""
        candidates.append(candidate)
    if not candidates:
        return []

    profile = _story_profile_text(topic, cluster)
    candidate_texts = [
        _clean_text((row.get("title") or "") + " " + (row.get("description") or ""))
        for row in candidates
    ]
    scores = [0.0] * len(candidates)
    if TfidfVectorizer is not None and cosine_similarity is not None and profile:
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            matrix = vectorizer.fit_transform([profile] + candidate_texts)
            scores = [float(value) for value in cosine_similarity(matrix[0:1], matrix[1:]).ravel().tolist()]
        except Exception:
            scores = [0.0] * len(candidates)

    topic_title = topic.get("topic_title") or ""
    cluster_title = _cluster_label(cluster)
    for idx, row in enumerate(candidates):
        lexical = max(
            _title_overlap_score(row.get("title") or "", topic_title),
            _title_overlap_score(row.get("title") or "", cluster_title),
        )
        semantic = scores[idx] if idx < len(scores) else 0.0
        row["similarity"] = round(max(semantic, lexical), 4)
        row["rank_score"] = round((0.78 * semantic) + (0.22 * lexical), 4)

    filtered = [row for row in candidates if row.get("similarity", 0.0) >= EXTERNAL_SIMILARITY_MIN]
    filtered.sort(key=lambda row: (-row.get("rank_score", 0.0), -row.get("similarity", 0.0)))

    # One representative per likely syndicated writeup.
    selected = []
    seen_titles = set()
    for row in filtered:
        title_key = _normalize_title(row.get("title") or "")[:180]
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)
        selected.append(row)
        if len(selected) >= MAX_EXTERNAL_CANDIDATES:
            break
    return selected


def build_external_signal_evidence(
    date_str: str,
    topic: dict,
    cluster: dict,
    existing_evidence: list[dict],
    sidecar_cache: dict,
    search_queries: list[str] | None = None,
) -> tuple[list[dict], dict, bool]:
    """
    Provider-resilient, domain-agnostic final-stage discovery for Next Signal.

    GDELT is tried first. If it fails to yield a same-story candidate, NewsAPI is
    used as a fallback when configured. Search results are only leads: candidates
    must still match the finalized event, expose explicit timed future evidence,
    and pass the same strict model validator. No result mutates frozen data.
    """
    try:
        briefing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        briefing_date = datetime.today().date()

    if search_queries is None:
        queries = _build_external_search_queries(topic, cluster)
    else:
        queries = []
        for value in search_queries:
            query = _clean_text(value)
            if query and query not in queries:
                queries.append(query[:180])
            if len(queries) >= 2:
                break
    meta = {
        "provider": "GDELT DOC 2.0 → NewsAPI fallback",
        "queries": queries,
        "searches": [],
        "search_result_count": 0,
        "candidate_count": 0,
        "page_attempts": 0,
        "network_fetches": 0,
        "evidence_additions": 0,
        "attempted": [],
        "status": "no_query" if not queries else "not_found",
        "providers_used": [],
    }
    if not queries:
        return [], meta, False

    cache_dirty = False

    def search_provider(search_fn, provider_name: str) -> list[dict]:
        nonlocal cache_dirty
        discovered_rows = []
        if provider_name not in meta["providers_used"]:
            meta["providers_used"].append(provider_name)
        for query in queries:
            rows, search_meta, changed = search_fn(query, briefing_date, sidecar_cache)
            cache_dirty = cache_dirty or changed
            if changed:
                meta["network_fetches"] += 1
            meta["searches"].append(search_meta)
            discovered_rows.extend(rows)
            if rows:
                ranked_now = _rank_external_candidates(topic, cluster, discovered_rows, existing_evidence)
                if ranked_now:
                    break
        return discovered_rows

    discovered = search_provider(_search_gdelt_articles, "GDELT DOC 2.0")
    candidates = _rank_external_candidates(topic, cluster, discovered, existing_evidence)

    # Fallback only when the primary provider failed to produce a usable same-story
    # lead. This limits external requests while preventing a single provider outage
    # from silently suppressing Next Signal.
    if not candidates:
        fallback_rows = search_provider(_search_newsapi_articles, "NewsAPI")
        discovered.extend(fallback_rows)
        candidates = _rank_external_candidates(topic, cluster, discovered, existing_evidence)

    meta["search_result_count"] = len(discovered)
    meta["candidate_count"] = len(candidates)
    if not candidates:
        meta["status"] = "no_story_match"
        return [], meta, cache_dirty

    raw_additions = []
    page_attempts = 0

    def collect(candidate: dict, snippet: dict, origin: str, fetch_status: str = "") -> None:
        description = _clean_text(snippet.get("text") or "")
        if not description:
            return
        quality = float(snippet.get("score") or 0.0)
        quality += 1.25 * float(candidate.get("similarity") or 0.0)
        quality += 0.20 if origin == "signal_external_fulltext" else 0.0
        raw_additions.append({
            "origin": origin,
            "title": candidate.get("title") or "",
            "description": description,
            "published_at": candidate.get("published_at") or "",
            "url": candidate.get("url") or "",
            "source": candidate.get("source") or "",
            "external_similarity": candidate.get("similarity"),
            "external_search_provider": candidate.get("search_provider") or "external_search",
            "_signal_score": round(quality, 4),
            "_fetch_status": fetch_status,
        })

    # As with the publisher sidecar, collect broadly first and rank later. A weak
    # metadata hit must not crowd out a stronger full-text milestone.
    for candidate in candidates:
        reference_date = _parse_reference_date(candidate.get("published_at") or "", briefing_date)
        provider = candidate.get("search_provider") or "external_search"
        metadata_text = _clean_text(
            " ".join(part for part in (candidate.get("title") or "", candidate.get("description") or "") if part)
        )
        metadata_snippets = _future_milestone_snippets(metadata_text, reference_date)
        for snippet in metadata_snippets[:2]:
            collect(candidate, snippet, "signal_external_metadata")
            meta["attempted"].append({
                "url": candidate.get("url"),
                "title": candidate.get("title"),
                "source": candidate.get("source"),
                "provider": provider,
                "similarity": candidate.get("similarity"),
                "result": "metadata_milestone",
                "excerpt": snippet["text"],
            })

        if page_attempts >= MAX_EXTERNAL_FETCHES:
            continue
        page_attempts += 1
        page_text, fetch_meta, changed = _fetch_article_text(candidate.get("url") or "", sidecar_cache)
        cache_dirty = cache_dirty or changed
        if changed:
            meta["network_fetches"] += 1
        snippets = _future_milestone_snippets(page_text, reference_date) if page_text else []
        if snippets:
            for snippet in snippets[:2]:
                collect(candidate, snippet, "signal_external_fulltext", fetch_meta.get("status", ""))
                meta["attempted"].append({
                    "url": candidate.get("url"),
                    "title": candidate.get("title"),
                    "source": candidate.get("source"),
                    "provider": provider,
                    "similarity": candidate.get("similarity"),
                    "result": "fulltext_milestone",
                    "fetch_status": fetch_meta.get("status"),
                    "excerpt": snippet["text"],
                })
        elif not metadata_snippets:
            meta["attempted"].append({
                "url": candidate.get("url"),
                "title": candidate.get("title"),
                "source": candidate.get("source"),
                "provider": provider,
                "similarity": candidate.get("similarity"),
                "result": "no_timed_milestone",
                "fetch_status": fetch_meta.get("status"),
                "fetch_error": fetch_meta.get("error", ""),
            })

    meta["page_attempts"] = page_attempts

    raw_additions.sort(
        key=lambda row: (
            -float(row.get("_signal_score") or 0.0),
            0 if row.get("origin") == "signal_external_fulltext" else 1,
            -len(row.get("description") or ""),
        )
    )
    additions = []
    seen_text = []
    per_url = {}
    for row in raw_additions:
        norm = _normalize_title(row.get("description") or "")
        if not norm:
            continue
        if any(norm in old or old in norm for old in seen_text):
            continue
        canonical = _canonical_url(row.get("url") or "")
        if canonical and per_url.get(canonical, 0) >= 2:
            continue
        clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
        additions.append(clean_row)
        seen_text.append(norm)
        if canonical:
            per_url[canonical] = per_url.get(canonical, 0) + 1
        if len(additions) >= MAX_EXTERNAL_MATCHES:
            break

    meta["evidence_additions"] = len(additions)
    meta["status"] = "evidence_found" if additions else "no_timed_milestone"
    return additions, meta, cache_dirty


def _clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _canonical_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return raw.lower().rstrip("/")

        path = re.sub(r"/+$", "", parsed.path or "")
        tracking = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "ref", "ref_src",
        }
        kept = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in tracking
        ]
        query = urlencode(sorted(kept))
        return urlunparse(("https", host, path, "", query, ""))
    except Exception:
        return raw.lower().rstrip("/")


def _topic_source_set(topic: dict) -> set[str]:
    return {
        value
        for value in (_canonical_url(url) for url in (topic.get("sources") or []))
        if value
    }


def _cluster_source_set(cluster: dict) -> set[str]:
    urls = []
    for key in ("articles", "related_articles", "coverage_articles"):
        for article in cluster.get(key) or []:
            urls.append(article.get("url") or article.get("url_normalized") or "")
    return {
        value
        for value in (_canonical_url(url) for url in urls)
        if value
    }


def _normalize_title(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _title_overlap_score(left: str, right: str) -> float:
    a = set(_normalize_title(left).split())
    b = set(_normalize_title(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _cluster_label(cluster: dict) -> str:
    return (
        cluster.get("canonical_event")
        or cluster.get("topic")
        or next(
            (
                article.get("title")
                for article in (cluster.get("articles") or [])
                if article.get("title")
            ),
            "",
        )
    )


def match_topic_to_cluster(topic: dict, clusters: list[dict]) -> tuple[int | None, dict]:
    topic_sources = _topic_source_set(topic)
    topic_title = topic.get("topic_title") or ""

    best_idx = None
    best_key = (-1, -1.0)
    best_meta = {"url_overlap": 0, "title_score": 0.0}

    for idx, cluster in enumerate(clusters):
        cluster_sources = _cluster_source_set(cluster)
        overlap = len(topic_sources & cluster_sources)
        title_score = _title_overlap_score(topic_title, _cluster_label(cluster))
        key = (overlap, title_score)
        if key > best_key:
            best_idx = idx
            best_key = key
            best_meta = {"url_overlap": overlap, "title_score": round(title_score, 4)}

    if best_idx is None:
        return None, best_meta

    # URL overlap is the normal path. Title-only fallback must be reasonably strong.
    if best_meta["url_overlap"] <= 0 and best_meta["title_score"] < 0.35:
        return None, best_meta

    return best_idx, best_meta


def _article_identity(article: dict) -> str:
    url = _canonical_url(article.get("url") or article.get("url_normalized") or "")
    if url:
        return "url:" + url
    title = _normalize_title(article.get("title") or "")
    return "title:" + title if title else ""


def _published_sort_value(article: dict) -> str:
    return str(article.get("published_at") or article.get("published_date") or "")


def select_signal_evidence(cluster: dict) -> list[dict]:
    """Use the verified core first, then same-event related/coverage context."""
    selected = []
    seen = set()

    def add(article: dict, origin: str) -> None:
        if len(selected) >= MAX_EVIDENCE_ARTICLES:
            return
        ident = _article_identity(article)
        if not ident or ident in seen:
            return
        title = _clean_text(article.get("title") or "")
        desc = _clean_text(article.get("description") or "")[:MAX_DESCRIPTION_CHARS]
        if not title and not desc:
            return
        seen.add(ident)
        selected.append({
            "origin": origin,
            "title": title,
            "description": desc,
            "published_at": str(article.get("published_at") or article.get("published_date") or ""),
            "url": article.get("url") or article.get("url_normalized") or "",
        })

    # Preserve purifier-approved core order exactly.
    for article in cluster.get("articles") or []:
        add(article, "core")

    # Newer same-event updates are useful for explicit future milestones.
    related = sorted(
        cluster.get("related_articles") or [],
        key=_published_sort_value,
        reverse=True,
    )
    for article in related:
        add(article, "related")

    coverage = sorted(
        cluster.get("coverage_articles") or [],
        key=_published_sort_value,
        reverse=True,
    )
    for article in coverage:
        add(article, "coverage")

    return selected[:MAX_EVIDENCE_ARTICLES]


def _parse_reference_date(value: str, fallback: date) -> date:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    match = re.match(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", raw)
    if not match:
        return fallback
    try:
        parsed = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return max(fallback, parsed)
    except Exception:
        return fallback


def _future_time_cues(text: str, reference_date: date) -> list[str]:
    """Return only temporal cues that plausibly point *after* reference_date."""
    clean = _clean_text(text)
    cues = []

    if _RELATIVE_TIME_RE.search(clean):
        cues.append("relative-window")

    future_framing = bool(_FUTURE_FRAMING_RE.search(clean))

    # Named weekdays are common in retrospective copy ("the election Sunday").
    # Treat 1-4 days ahead as plausibly upcoming; farther weekdays need explicit
    # future framing such as "scheduled", "expected", "next", or "will".
    for match in _WEEKDAY_RE.finditer(clean):
        weekday = _WEEKDAY_NAMES[match.group(0).lower()]
        delta = (weekday - reference_date.weekday()) % 7
        if 1 <= delta <= 4 or (delta >= 5 and future_framing):
            cues.append("future-weekday")
            break

    # Resolve named month/date references when possible. Bare future months count
    # as a window; historical dates are explicitly rejected.
    for match in _MONTH_RE.finditer(clean):
        month_token = match.group("month").lower()
        month = _MONTH_NAMES.get(month_token[:3], _MONTH_NAMES.get(month_token))
        if not month:
            continue
        day_value = match.group("day")
        year_value = match.group("year")
        # "may" is commonly a modal verb ("may produce flooding"). A bare May
        # without a day/year is too ambiguous to be a useful future date cue.
        if month == 5 and not day_value and not year_value:
            continue
        try:
            year = int(year_value) if year_value else reference_date.year
            if day_value:
                candidate = date(year, month, int(day_value))
                if candidate <= reference_date and not year_value:
                    # Do not roll an old month/day into next year merely because some
                    # unrelated future verb appears elsewhere in the text. Require an
                    # explicit "next <month>" cue for that rollover. This removes cheap
                    # prefilter false positives such as "I will ..." plus "in March".
                    local_start = max(0, match.start() - 12)
                    local_prefix = clean[local_start:match.start()].lower()
                    if re.search(r"\bnext\s*$", local_prefix):
                        candidate = date(year + 1, month, int(day_value))
                if candidate > reference_date:
                    cues.append("future-month/date")
                    break
            else:
                # Bare month: only count when the month itself is ahead, or when
                # explicit future framing makes next year's same/past month clear.
                if (year > reference_date.year) or (year == reference_date.year and month > reference_date.month):
                    cues.append("future-month")
                    break
                if not year_value and month <= reference_date.month:
                    local_start = max(0, match.start() - 12)
                    local_prefix = clean[local_start:match.start()].lower()
                    if re.search(r"\bnext\s*$", local_prefix):
                        cues.append("future-month")
                        break
        except Exception:
            continue

    for match in _NUMERIC_DATE_RE.finditer(clean):
        try:
            if match.group("ymd_year"):
                candidate = date(
                    int(match.group("ymd_year")),
                    int(match.group("ymd_month")),
                    int(match.group("ymd_day")),
                )
            else:
                year = int(match.group("md_year")) if match.group("md_year") else reference_date.year
                candidate = date(year, int(match.group("md_month")), int(match.group("md_day")))
                if candidate <= reference_date and not match.group("md_year") and future_framing:
                    candidate = date(year + 1, candidate.month, candidate.day)
            if candidate > reference_date:
                cues.append("future-numeric-date")
                break
        except Exception:
            continue

    return sorted(set(cues))


def _atomic_sentences(text: str) -> list[str]:
    clean = _clean_text(text)
    if not clean:
        return []
    # Protect common abbreviations before sentence splitting, then restore
    # whitespace sometimes lost by scraped article bodies. This prevents "U.S."
    # or "U.N." from being mistaken for a sentence boundary.
    clean = re.sub(r"\b([A-Z])\.([A-Z])\.", r"\1\2", clean)
    clean = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sen|Rep|Gov|St)\.\s+", r"\1 ", clean)
    clean = re.sub(r"(?<=[.!?])(?=[A-Z0-9\"'“‘])", " ", clean)
    return [
        _clean_text(part)
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])", clean)
        if _clean_text(part)
    ]


def _future_anchor_records(text: str, reference_date: date) -> list[dict]:
    """Return explicit future time anchors with stable comparison keys."""
    clean = _clean_text(text)
    if not clean:
        return []

    records = []
    seen = set()

    def add(kind: str, raw: str, start: int, end: int, key: str):
        ident = (kind, key, start, end)
        if ident in seen:
            return
        seen.add(ident)
        records.append({
            "kind": kind,
            "text": _clean_text(raw),
            "start": int(start),
            "end": int(end),
            "key": key,
        })

    for match in _RELATIVE_TIME_RE.finditer(clean):
        raw = match.group(0)
        low = raw.lower()
        weekday_match = _WEEKDAY_RE.search(raw)
        if weekday_match:
            add("relative", raw, match.start(), match.end(), f"weekday:{weekday_match.group(0).lower()}")
        else:
            add("relative", raw, match.start(), match.end(), "relative:" + re.sub(r"\s+", " ", low))

    for match in _WEEKDAY_RE.finditer(clean):
        # Possessive weekdays almost always label a completed event ("Tuesday's
        # meeting") rather than schedule the next one.
        if clean[match.end():match.end() + 2] in {"'s", "’s"}:
            continue
        weekday_name = match.group(0).lower()
        weekday = _WEEKDAY_NAMES[weekday_name]
        delta = (weekday - reference_date.weekday()) % 7
        # Plain same-day weekday mentions are ambiguous; useful same-day future
        # windows should say tonight/later today/etc. and are caught above.
        if 1 <= delta <= 6:
            add("weekday", match.group(0), match.start(), match.end(), f"weekday:{weekday_name}")

    for match in _MONTH_RE.finditer(clean):
        month_token = match.group("month").lower()
        month = _MONTH_NAMES.get(month_token[:3], _MONTH_NAMES.get(month_token))
        if not month:
            continue
        day_value = match.group("day")
        year_value = match.group("year")
        if month == 5 and not day_value and not year_value:
            continue
        try:
            year = int(year_value) if year_value else reference_date.year
            if day_value:
                candidate = date(year, month, int(day_value))
                if candidate <= reference_date and not year_value:
                    prefix = clean[max(0, match.start() - 12):match.start()].lower()
                    if re.search(r"\bnext\s*$", prefix):
                        candidate = date(year + 1, month, int(day_value))
                if candidate > reference_date:
                    add("date", match.group(0), match.start(), match.end(), f"date:{candidate.isoformat()}")
            else:
                candidate_year = year
                if not year_value and month <= reference_date.month:
                    prefix = clean[max(0, match.start() - 12):match.start()].lower()
                    if re.search(r"\bnext\s*$", prefix):
                        candidate_year += 1
                if (candidate_year > reference_date.year) or (
                    candidate_year == reference_date.year and month > reference_date.month
                ):
                    add("month", match.group(0), match.start(), match.end(), f"month:{candidate_year:04d}-{month:02d}")
        except Exception:
            continue

    for match in _NUMERIC_DATE_RE.finditer(clean):
        try:
            if match.group("ymd_year"):
                candidate = date(
                    int(match.group("ymd_year")),
                    int(match.group("ymd_month")),
                    int(match.group("ymd_day")),
                )
            else:
                year = int(match.group("md_year")) if match.group("md_year") else reference_date.year
                candidate = date(year, int(match.group("md_month")), int(match.group("md_day")))
                if candidate <= reference_date and not match.group("md_year"):
                    prefix = clean[max(0, match.start() - 12):match.start()].lower()
                    if re.search(r"\bnext\s*$", prefix):
                        candidate = date(year + 1, candidate.month, candidate.day)
            if candidate > reference_date:
                add("date", match.group(0), match.start(), match.end(), f"date:{candidate.isoformat()}")
        except Exception:
            continue

    records.sort(key=lambda row: (row["start"], row["end"]))
    return records


def _anchor_is_action_timing(sentence: str, anchor: dict) -> bool:
    """Require the time anchor to describe the future action, not the reporting date."""
    if _FINALITY_RE.search(sentence):
        return False

    framing = list(_FUTURE_FRAMING_RE.finditer(sentence))
    # Durations inside standing rules/policies are not themselves scheduled
    # future milestones ("must notify within 48 hours", "caps at 60 days").
    if _RULE_CONTEXT_RE.search(sentence) and not framing:
        return False
    milestones = list(_MILESTONE_RE.finditer(sentence))
    a_start = int(anchor.get("start") or 0)
    a_end = int(anchor.get("end") or a_start)

    # A past/completed event immediately before a date usually owns that date,
    # even when a later clause contains a forecast ("grew Tuesday and was
    # forecast..."). Keep the date only if future framing already appeared before it.
    prior_frames = [frame for frame in framing if frame.end() <= a_start]
    prior_past = [event for event in _PAST_EVENT_RE.finditer(sentence) if event.end() <= a_start]
    if prior_past and not prior_frames:
        nearest = max(prior_past, key=lambda event: event.end())
        if a_start - nearest.end() <= 90:
            return False

    # If a date precedes a reporting verb and the future verb comes after that
    # reporting verb, the date usually timestamps the reporting itself:
    # "On Monday, X said there will be an announcement." Do not reinterpret
    # Monday as the announcement date.
    for report in _REPORTING_VERB_RE.finditer(sentence):
        if a_end <= report.start():
            # A date before a reporting verb normally dates the statement itself.
            # Keep it only when scheduling/forecast language already appeared
            # before the time anchor (e.g. "hearing scheduled Friday, officials said").
            if not any(frame.end() <= a_start for frame in framing):
                return False
        elif report.end() <= a_start:
            # "X said ... on Tuesday that ..." timestamps the statement unless
            # scheduling/forecast language appears between the reporting verb
            # and the date itself.
            between_frames = [
                frame for frame in framing
                if report.end() <= frame.start() <= a_start
            ]
            if not between_frames:
                return False

    # Strongest structure: scheduled/expected/etc. is reasonably close to the
    # time anchor. Keep this wide enough for a single factual sentence such as
    # "rain is expected ... through Thursday" while still sentence-bounding it.
    for frame in framing:
        distance = min(abs(frame.end() - a_start), abs(frame.start() - a_end))
        if distance <= 220:
            return True

    # Concise calendar copy such as "Hearing October 7" or "Vote next week"
    # can omit "scheduled". Keep this fallback deliberately narrow so contextual
    # future dates (for example an election mentioned in a long background
    # sentence) do not trigger a paid model call.
    if len(sentence) <= 110:
        for milestone in milestones:
            distance = min(abs(milestone.end() - a_start), abs(milestone.start() - a_end))
            if distance <= 55 and (milestone.start() <= 35 or a_start <= 35):
                return True

    return False


def _routine_only_future(sentence: str) -> bool:
    return bool(_ROUTINE_ACTIVITY_RE.search(sentence) and not _SUBSTANTIVE_ACTIVITY_RE.search(sentence))


def _grounded_segments_for_row(date_str: str, row: dict) -> list[dict]:
    """Find sentence-level, explicitly timed future actions in one evidence row."""
    try:
        briefing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        briefing_date = datetime.today().date()
    reference_date = _parse_reference_date(row.get("published_at") or "", briefing_date)

    description = _clean_text(row.get("description") or "")
    title = _clean_text(row.get("title") or "")
    text = description or title
    sentences = _atomic_sentences(text)
    if not sentences and title:
        sentences = [title]

    vague_subject_re = re.compile(
        r"^(?:it|they|he|she|this|that|the\s+[a-z][a-z-]{2,}(?:\s+[a-z][a-z-]{2,}){0,2})\b",
        re.I,
    )

    candidates = []
    seen = set()
    for idx, sentence in enumerate(sentences):
        if _FINALITY_RE.search(sentence):
            continue
        if _routine_only_future(sentence):
            continue
        anchors = [
            anchor for anchor in _future_anchor_records(sentence, reference_date)
            if _anchor_is_action_timing(sentence, anchor)
        ]
        if not anchors:
            continue

        # Preserve one preceding sentence only when the timed sentence begins
        # with a vague subject. This helps exact-event validation without letting
        # a date in one sentence bind to a different event in another sentence.
        excerpt = sentence
        vague_context = bool(idx > 0 and vague_subject_re.search(sentence))
        if vague_context:
            excerpt = _clean_text(sentences[idx - 1] + " " + sentence)

        key = _normalize_title(sentence)
        if not key or key in seen:
            continue
        seen.add(key)
        score = 3.0 * len(anchors)
        if _FUTURE_FRAMING_RE.search(sentence):
            score += 2.0
        score += min(2.0, len(_MILESTONE_RE.findall(sentence)))
        if "fulltext" in str(row.get("origin") or ""):
            score += 1.0
        candidates.append({
            "score": score,
            "sentence": sentence,
            "excerpt": excerpt[:MAX_SIDECAR_EXCERPT_CHARS],
            "anchors": anchors,
            "reference_date": reference_date.isoformat(),
            "vague_context": vague_context,
        })

    candidates.sort(key=lambda item: (-item["score"], -len(item["sentence"])))
    return candidates


def prefilter_next_signal(date_str: str, evidence: list[dict]) -> dict:
    """Token-free strict gate: sentence-level timing must be grounded to the future action."""
    candidates = []
    for idx, row in enumerate(evidence, start=1):
        for candidate in _grounded_segments_for_row(date_str, row):
            candidates.append({
                "source_index": idx,
                "reference_date": candidate["reference_date"],
                "time_cues": sorted({anchor["key"] for anchor in candidate["anchors"]}),
                "text": candidate["sentence"][:320],
            })
            break

    if candidates:
        return {
            "passed": True,
            "reason": f"{len(candidates)} evidence item(s) contain a sentence-level future time anchor tied to a future action.",
            "candidates": candidates[:5],
        }

    return {
        "passed": False,
        "reason": "No evidence item contained a grounded future time anchor tied to the future action.",
        "candidates": [],
    }



def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def _save_json_atomic(path: str, payload) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _cache_key(date_str: str, topic: dict, evidence: list[dict], stage: str = "final") -> str:
    material = {
        "date": date_str,
        "title": topic.get("topic_title") or "",
        "summary": topic.get("summary") or "",
        "sources": sorted(topic.get("sources") or []),
        "evidence": [
            {
                "title": row.get("title"),
                "description": row.get("description"),
                "published_at": row.get("published_at"),
                "url": row.get("url"),
            }
            for row in evidence
        ],
    }
    raw = json.dumps(material, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{MODEL}:{PROMPT_VERSION}:{stage}:{digest}"


def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}



_CONTEXT_STOPWORDS = set(_EXTERNAL_QUERY_STOPWORDS) | {
    "said", "says", "say", "will", "would", "could", "may", "might", "expected",
    "scheduled", "set", "next", "future", "president", "minister", "official", "officials",
    "government", "news", "story", "new", "first", "last", "day", "week", "month", "year",
}


def _context_tokens(text: str) -> set[str]:
    return {
        token for token in _normalize_title(text).split()
        if len(token) >= 3 and token not in _CONTEXT_STOPWORDS and not token.isdigit()
    }


def _vague_context_matches_story(topic: dict | None, cluster: dict | None, excerpt: str) -> bool:
    if not topic:
        return True
    title_tokens = _context_tokens(topic.get("topic_title") or "")
    cluster_tokens = _context_tokens(_cluster_label(cluster or {}))
    story_tokens = _context_tokens(_clean_text(" ".join([
        topic.get("topic_title") or "",
        topic.get("summary") or "",
        _cluster_label(cluster or {}),
    ])))
    excerpt_tokens = _context_tokens(excerpt)

    # Prefer tokens repeated by both the finalized headline and frozen cluster
    # identity. They are a cheap proxy for the event's stable identity, without
    # knowing whether the story is weather, law, politics, business, etc.
    identity = title_tokens & cluster_tokens
    if len(identity) >= 4:
        return len(identity & excerpt_tokens) >= 3
    if len(identity) >= 2:
        return len(identity & excerpt_tokens) >= 2
    if identity:
        return bool(identity & excerpt_tokens)

    # Fallback when headline/cluster wording differs heavily: require at least two
    # content-token matches to the full story profile.
    return len(story_tokens & excerpt_tokens) >= 2


def _model_evidence_candidates(
    date_str: str,
    evidence: list[dict],
    limit: int = MAX_MODEL_EVIDENCE,
    topic: dict | None = None,
    cluster: dict | None = None,
) -> list[dict]:
    """Return only sentence-level evidence that passed deterministic timing + context grounding."""
    ranked = []
    for position, row in enumerate(evidence):
        for candidate in _grounded_segments_for_row(date_str, row):
            if candidate.get("vague_context") and not _vague_context_matches_story(
                topic, cluster, candidate.get("excerpt") or ""
            ):
                continue
            derived = dict(row)
            derived["description"] = candidate["excerpt"]
            derived["grounded_sentence"] = candidate["sentence"]
            derived["grounded_time_keys"] = [anchor["key"] for anchor in candidate["anchors"]]
            derived["grounded_reference_date"] = candidate["reference_date"]
            derived["grounded_vague_context"] = bool(candidate.get("vague_context"))
            ranked.append((candidate["score"], position, derived))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    seen = set()
    for _, _, row in ranked:
        ident = (
            _canonical_url(row.get("url") or ""),
            _normalize_title(row.get("grounded_sentence") or row.get("description") or "")[:220],
        )
        if ident in seen:
            continue
        seen.add(ident)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected



def _evidence_lines(evidence: list[dict]) -> list[str]:
    lines = []
    for idx, row in enumerate(evidence, start=1):
        parts = [f"[{idx}] ({row.get('origin', 'source')})"]
        if row.get("published_at"):
            parts.append(f"Published: {row['published_at']}")
        if row.get("source"):
            parts.append(f"Source: {row['source']}")
        if row.get("url"):
            parts.append(f"URL: {row['url']}")
        if row.get("title"):
            parts.append(f"Title: {row['title']}")
        if row.get("description"):
            parts.append(f"Description: {row['description']}")
        lines.append(" | ".join(parts))
    return lines


def _grounding_reference_date(date_str: str, row: dict) -> date:
    raw = str(row.get("grounded_reference_date") or "").strip()
    if raw:
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d").date()
        except Exception:
            pass
    try:
        briefing = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        briefing = datetime.today().date()
    return _parse_reference_date(row.get("published_at") or "", briefing)


def validate_generated_signal_grounding(date_str: str, signal: str, supporting_row: dict) -> tuple[bool, str]:
    """Fail closed when model wording adds timing/certainty not present in its cited evidence."""
    signal = _clean_text(signal)
    source_sentence = _clean_text(
        supporting_row.get("grounded_sentence")
        or supporting_row.get("description")
        or ""
    )
    if not signal or not source_sentence:
        return False, "Generated signal or supporting sentence was empty."

    reference_date = _grounding_reference_date(date_str, supporting_row)
    source_anchors = _future_anchor_records(source_sentence, reference_date)
    signal_anchors = _future_anchor_records(signal, reference_date)
    source_keys = {row["key"] for row in source_anchors}
    signal_keys = {row["key"] for row in signal_anchors}

    if not source_keys:
        return False, "Supporting evidence did not contain an explicit future time anchor."
    if not signal_keys:
        return False, "Generated signal did not preserve an explicit future time anchor."
    if not (source_keys & signal_keys):
        return False, "Generated timing was not present in the cited evidence."

    # Vague timing is never allowed to be invented as a substitute for a real
    # source time window.
    vague = _UNSUPPORTED_VAGUE_TIME_RE.search(signal)
    if vague and vague.group(0).lower() not in source_sentence.lower():
        return False, "Generated signal introduced unsupported vague timing."

    # Do not turn evidence saying this was the final/last opportunity into an
    # invented continuation event.
    if _FINALITY_RE.search(source_sentence) and _CONTINUATION_SIGNAL_RE.search(signal):
        return False, "Generated signal contradicted finality language in the evidence."

    # Preserve uncertainty from forecasts, projections, and qualified future
    # statements. If the source is uncertain, the signal must remain uncertain.
    if _UNCERTAINTY_RE.search(source_sentence) and not _UNCERTAINTY_RE.search(signal):
        return False, "Generated signal removed uncertainty present in the evidence."

    if _routine_only_future(signal):
        return False, "Generated signal described routine itinerary/ceremonial activity rather than a substantive milestone."

    return True, ""


def evaluate_signal_triage(date_str: str, topic: dict, evidence: list[dict]) -> dict:
    """
    One cheap decision over local evidence only.

    Returns SIGNAL when current evidence is sufficient, NO_SIGNAL when there is
    nothing worth pursuing, or RESEARCH_NEEDED only when the evidence strongly
    suggests a substantive upcoming development but a publishable timing/detail
    is still missing. External discovery is allowed only for RESEARCH_NEEDED.
    """
    if openai is None:
        raise RuntimeError("The openai package is required for Next Signal generation.")

    evidence_lines = _evidence_lines(evidence)
    prompt = f"""
You are an evidence-bound editor triaging an optional Next Signal for a news story.

BRIEFING DATE: {date_str}
STORY HEADLINE: {topic.get('topic_title') or ''}
STORY SUMMARY: {topic.get('summary') or ''}

A Next Signal is the next concrete, already-announced point when new information could materially change what a reader knows about THIS exact story.

Choose exactly one status:

SIGNAL
- Current evidence already gives a concrete future milestone/time window.
- It directly advances THIS exact story and is materially informative.
- Routine itinerary, ceremonies, tours, arrivals/departures, speeches, receptions, photo opportunities, or generic future intentions do NOT qualify unless a substantive decision/action/outcome is expected there.
- Preserve uncertainty words such as forecast, expected, could, may, or as early as.

NO_SIGNAL
- There is no useful concrete next milestone in the evidence; OR
- the only future-looking material is vague, aspirational, routine/ceremonial, unrelated, retrospective, an open-ended investigation, or belongs to a different event/person/storm/case.
- Use NO_SIGNAL rather than researching merely because more information might exist somewhere.

RESEARCH_NEEDED
- Use sparingly.
- Current evidence strongly indicates that THIS exact story has a substantive upcoming development, but the evidence is missing the concrete timing/window or another narrow fact required to publish it accurately.
- Examples of an evidence gap: an active impact/forecast is clearly coming but the timing is vague; a formal next proceeding is referenced but its date is omitted.
- Do NOT use RESEARCH_NEEDED for general curiosity, background, routine itinerary, or a story with no demonstrated upcoming milestone.
- If chosen, provide 1 or 2 short provider-neutral search queries that contain the exact story identity plus the missing evidence. Do not name websites, domains, APIs, or sources. Do not invent facts not suggested by the supplied evidence.

For SIGNAL:
- concise reader-facing signal, ideally <=18 words;
- the signal MUST contain an explicit timing phrase that is present in the cited evidence item;
- reuse the evidence's timing wording when practical instead of converting it into a new date/window;
- NEVER add words such as "soon", "shortly", "after the summit", a month, date, or deadline unless that timing is explicitly in the cited evidence;
- source_index must identify the evidence item that explicitly supports both milestone and timing.

EVIDENCE:
{chr(10).join(evidence_lines)}

Return JSON only:
{{"status":"SIGNAL|NO_SIGNAL|RESEARCH_NEEDED","signal":"","source_index":null,"reason":"brief internal reason","research_gap":"","search_queries":[]}}
""".strip()

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use only supplied evidence. Be conservative. Research is exceptional, not the default. "
                    "Never turn a different event mentioned in the same article into this story's signal. Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=260,
    )
    parsed = _extract_json_object(response.choices[0].message["content"])

    status = _clean_text(parsed.get("status") or "").upper().replace(" ", "_")
    if status not in {"SIGNAL", "NO_SIGNAL", "RESEARCH_NEEDED"}:
        status = "NO_SIGNAL"
    signal = _clean_text(parsed.get("signal") or "")
    reason = _clean_text(parsed.get("reason") or "")
    research_gap = _clean_text(parsed.get("research_gap") or "")[:300]
    source_index = parsed.get("source_index")
    try:
        source_index = int(source_index) if source_index is not None else None
    except Exception:
        source_index = None

    raw_queries = parsed.get("search_queries")
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    if not isinstance(raw_queries, list):
        raw_queries = []
    queries = []
    for value in raw_queries:
        query = _clean_text(value)
        if not query:
            continue
        # Queries are evidence-discovery hints only. Keep them compact and strip
        # provider-specific operators so the mechanism stays domain-agnostic.
        query = re.sub(r"\bsite\s*:\s*\S+", " ", query, flags=re.I)
        query = re.sub(r"https?://\S+", " ", query, flags=re.I)
        query = _clean_text(query)
        if query and query not in queries:
            queries.append(query[:180])
        if len(queries) >= 2:
            break

    if status == "SIGNAL":
        if not signal or source_index is None or source_index < 1 or source_index > len(evidence):
            return {
                "status": "NO_SIGNAL",
                "eligible": False,
                "signal": "",
                "source_index": None,
                "reason": "Rejected because the proposed signal lacked a valid supporting evidence index.",
                "research_gap": "",
                "search_queries": [],
            }
        if len(signal.split()) > 22:
            return {
                "status": "NO_SIGNAL",
                "eligible": False,
                "signal": "",
                "source_index": None,
                "reason": "Rejected because the proposed signal was not concise.",
                "research_gap": "",
                "search_queries": [],
            }
        grounded, grounding_reason = validate_generated_signal_grounding(
            date_str, signal, evidence[source_index - 1]
        )
        if not grounded:
            return {
                "status": "NO_SIGNAL",
                "eligible": False,
                "signal": "",
                "source_index": None,
                "reason": f"Deterministic grounding rejection: {grounding_reason}",
                "research_gap": "",
                "search_queries": [],
            }
        return {
            "status": "SIGNAL",
            "eligible": True,
            "signal": signal,
            "source_index": source_index,
            "reason": reason,
            "research_gap": "",
            "search_queries": [],
        }

    if status == "RESEARCH_NEEDED":
        if not queries:
            status = "NO_SIGNAL"
            reason = reason or "Research was suggested but no focused evidence query was supplied."
        else:
            return {
                "status": "RESEARCH_NEEDED",
                "eligible": False,
                "signal": "",
                "source_index": None,
                "reason": reason,
                "research_gap": research_gap,
                "search_queries": queries,
            }

    return {
        "status": "NO_SIGNAL",
        "eligible": False,
        "signal": "",
        "source_index": None,
        "reason": reason or "No useful concrete future milestone identified in local evidence.",
        "research_gap": "",
        "search_queries": [],
    }


def evaluate_next_signal(date_str: str, topic: dict, evidence: list[dict]) -> dict:
    if openai is None:
        raise RuntimeError("The openai package is required for Next Signal generation.")
    evidence_lines = _evidence_lines(evidence)

    prompt = f"""
You are an evidence-bound editor deciding whether a news story has a useful Next Signal.

BRIEFING DATE: {date_str}
STORY HEADLINE: {topic.get('topic_title') or ''}
STORY SUMMARY: {topic.get('summary') or ''}

A Next Signal is the next concrete, already-announced point when new information could materially change what a reader knows about THIS exact story.

A signal is eligible only if ALL are true:
1. The supplied evidence explicitly identifies a future milestone or time window: for example a scheduled vote, hearing, court date/ruling window, meeting, deadline, official report/release, election-result window, launch, or a forecast milestone such as intensification, closest approach, landfall, or onset of hazardous conditions.
2. The milestone directly advances this exact story. It is not merely another event, conference, speech, hazard, person, lawsuit, or news item mentioned in the same article or occurring nearby. If a source discusses multiple events, make sure the timed milestone applies to the subject of the finalized story itself.
3. The milestone is meaningfully informative. It could change the known status, outcome, policy, legal posture, official findings, material consequences, forecast risk, or another substantive state of this story. A future item is NOT useful merely because it is scheduled.
4. Routine itinerary or ceremonial activity is not enough by itself. A tour, arrival/departure, reception, ceremony, conference appearance, speech, photo opportunity, or ordinary schedule item is ineligible unless the evidence makes clear that a substantive decision, negotiation, formal action, release, ruling, vote, or material outcome for THIS story is expected there. A meeting or summit can qualify when it is itself the next substantive decision/negotiation point.
5. Its future timing is supported by the supplied evidence. Do not infer a date from publication metadata or outside knowledge.
6. Prefer direct/primary evidence when present. Publisher evidence is acceptable only when it explicitly states the milestone and timing; do not upgrade speculation into a scheduled fact.
7. For forecasts or other uncertain future states, preserve uncertainty exactly. If the source says "forecast", "expected", "could", "may", or "as early as", the signal must not turn that into certainty.

NOT ELIGIBLE:
- "Outcome of the investigation" when no findings date, hearing, deadline, or release window is given.
- An investigation merely being opened or continuing.
- "More details may emerge," monitoring, negotiations continuing with no scheduled next session, or an unscheduled possible announcement.
- A contextual event that does not directly advance this story.
- Routine travel, ceremonial or social itinerary, tours, departures, receptions, photo opportunities, or appearances with no substantive story-changing action attached.
- A milestone for a different storm, person, court case, policy, or event that merely appears in the same source article.
- A predicted legal, political, or factual outcome. Forecast timing for an active hazard is eligible when the source explicitly supports it and uncertainty is preserved.

OUTPUT RULES:
- If no signal qualifies, return eligible=false and an empty signal.
- If one qualifies, make the reader-facing signal concise, factual, and no more than 18 words.
- When the evidence gives an explicit date/day/time/window, lead with it when practical, e.g. "Tuesday · Senate vote on the bill".
- For an active hazard, a concise signal such as "Thu night–Fri · Forecast to strengthen; earliest approach Friday" is appropriate when explicitly supported.
- source_index must be the numbered evidence item that explicitly supports BOTH the milestone and its timing.
- Do not use any fact not present below.
- The signal must preserve an explicit time anchor from the cited evidence item. Do not invent "soon", "shortly", a calendar month/date, or a relative window.

EVIDENCE:
{chr(10).join(evidence_lines)}

Return JSON only in this exact shape:
{{"eligible": false, "signal": "", "source_index": null, "reason": "brief internal reason"}}
""".strip()

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use only supplied evidence. Be conservative: omission is better than a vague, inferred, "
                    "contextual, or unscheduled Next Signal. Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=220,
    )
    parsed = _extract_json_object(response.choices[0].message["content"])

    eligible = bool(parsed.get("eligible"))
    signal = _clean_text(parsed.get("signal") or "")
    source_index = parsed.get("source_index")
    reason = _clean_text(parsed.get("reason") or "")

    try:
        source_index = int(source_index) if source_index is not None else None
    except Exception:
        source_index = None

    if not eligible:
        signal = ""
        source_index = None

    # Fail closed if the model did not point to a real supporting evidence row.
    if signal and (source_index is None or source_index < 1 or source_index > len(evidence)):
        signal = ""
        eligible = False
        source_index = None
        reason = "Rejected because no valid supporting evidence index was returned."

    # Keep the UI intentionally compact even if the model ignores the word limit.
    if signal and len(signal.split()) > 22:
        signal = ""
        eligible = False
        source_index = None
        reason = "Rejected because the proposed signal was not concise."

    if signal and source_index is not None:
        grounded, grounding_reason = validate_generated_signal_grounding(
            date_str, signal, evidence[source_index - 1]
        )
        if not grounded:
            signal = ""
            eligible = False
            source_index = None
            reason = f"Deterministic grounding rejection: {grounding_reason}"

    return {
        "eligible": bool(eligible and signal),
        "signal": signal,
        "source_index": source_index,
        "reason": reason,
    }


def _default_cluster_file(date_str: str) -> str:
    candidates = [
        f"grouped_articles_final_global_receipts_{date_str}.json",
        f"grouped_articles_final_expanded_{date_str}.json",
        f"grouped_articles_final_{date_str}.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add optional Next Signal fields to existing topic summaries without regenerating summaries."
    )
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    parser.add_argument("--summary-file", type=str, default="")
    parser.add_argument("--cluster-file", type=str, default="")
    parser.add_argument("--article-pool-file", type=str, default="")
    parser.add_argument("--report-file", type=str, default="")
    parser.add_argument("--cache-file", type=str, default=CACHE_PATH)
    parser.add_argument("--sidecar-cache-file", type=str, default=SIDECAR_CACHE_PATH)
    parser.add_argument("--refresh", action="store_true", help="Ignore cached Next Signal decisions.")
    parser.add_argument("--refresh-sidecar", action="store_true", help="Ignore cached article-text retrievals.")
    parser.add_argument("--no-sidecar", action="store_true", help="Disable all signal-only evidence expansion.")
    parser.add_argument("--no-external", action="store_true", help="Disable generic external signal-evidence discovery.")
    parser.add_argument("--dry-run", action="store_true", help="Match topics and rank sidecar candidates but make no API/network calls or file changes.")
    args = parser.parse_args()

    date_str = args.date or datetime.today().strftime("%Y-%m-%d")
    summary_file = args.summary_file or f"topic_summaries_{date_str}.json"
    cluster_file = args.cluster_file or _default_cluster_file(date_str)
    article_pool_file = args.article_pool_file or _default_article_pool_file(date_str)
    report_file = args.report_file or f"next_signal_report_{date_str}.json"

    summaries = _load_json(summary_file, None)
    clusters = _load_json(cluster_file, None)
    article_pool = _load_json(article_pool_file, []) if os.path.exists(article_pool_file) else []
    if not isinstance(summaries, list):
        raise SystemExit(f"Could not load summary list: {summary_file}")
    if not isinstance(clusters, list):
        raise SystemExit(f"Could not load cluster list: {cluster_file}")
    if not isinstance(article_pool, list):
        article_pool = []

    print(f"🧭 Next Signal date: {date_str}")
    print(f"📰 Summary file: {summary_file}")
    print(f"📚 Frozen cluster evidence: {cluster_file}")
    if article_pool:
        print(f"🔭 Signal-only daily article pool: {article_pool_file} ({len(article_pool)} articles)")
    else:
        print(f"🔭 Signal-only daily article pool: unavailable ({article_pool_file}); receipts/full text only")
    print(f"🤖 Model: {MODEL} (gap-guided; max 1 local call/story, second call only after requested research)")

    cache = _load_json(args.cache_file, {})
    if not isinstance(cache, dict):
        cache = {}
    sidecar_cache = {} if args.refresh_sidecar else _load_json(args.sidecar_cache_file, {})
    if not isinstance(sidecar_cache, dict):
        sidecar_cache = {}
    cache_dirty = False
    sidecar_cache_dirty = False
    report = []

    def evaluate_cached(topic: dict, rows: list[dict], stage: str, evaluator) -> tuple[dict, bool, bool]:
        nonlocal cache_dirty
        key = _cache_key(date_str, topic, rows, stage=stage)
        cached = None if args.refresh else cache.get(key)
        if isinstance(cached, dict):
            return dict(cached), True, False
        try:
            decision = evaluator(date_str, topic, rows)
        except Exception as exc:
            if stage == "triage":
                decision = {
                    "status": "NO_SIGNAL",
                    "eligible": False,
                    "signal": "",
                    "source_index": None,
                    "reason": f"API triage failed: {exc}",
                    "research_gap": "",
                    "search_queries": [],
                }
            else:
                decision = {
                    "eligible": False,
                    "signal": "",
                    "source_index": None,
                    "reason": f"API evaluation failed: {exc}",
                }
        cache[key] = decision
        cache_dirty = True
        return decision, False, True

    for idx, topic in enumerate(summaries, start=1):
        cluster_idx, match_meta = match_topic_to_cluster(topic, clusters)
        title = topic.get("topic_title") or f"Topic {idx}"

        # The sole newsletter mutation remains this optional field.
        topic["next_signal"] = ""

        if cluster_idx is None:
            print(f"⚪ {idx}. {title}: no confident cluster match")
            report.append({
                "topic_index": idx,
                "topic_title": title,
                "matched": False,
                "match": match_meta,
                "eligible": False,
                "signal": "",
                "reason": "No confident source-overlap/title match to frozen cluster evidence.",
                "model_called": False,
                "model_call_count": 0,
                "prompt_version": PROMPT_VERSION,
            })
            continue

        cluster = clusters[cluster_idx]
        frozen_evidence = select_signal_evidence(cluster)
        frozen_prefilter = prefilter_next_signal(date_str, frozen_evidence)
        sidecar_candidates = _rank_sidecar_candidates(topic, cluster, frozen_evidence, article_pool)

        if args.dry_run:
            marker = "🔎" if frozen_prefilter["passed"] else "⏭️"
            print(
                f"{marker} {idx}. {title}: cluster {cluster_idx + 1}, "
                f"url_overlap={match_meta['url_overlap']}, evidence={len(frozen_evidence)}, "
                f"frozen_prefilter={'pass' if frozen_prefilter['passed'] else 'skip'}, "
                f"sidecar_candidates={len(sidecar_candidates)}"
            )
            report.append({
                "topic_index": idx,
                "topic_title": title,
                "matched": True,
                "cluster_index": cluster_idx,
                "match": match_meta,
                "evidence_count": len(frozen_evidence),
                "prefilter_version": PREFILTER_VERSION,
                "prefilter_passed": frozen_prefilter["passed"],
                "prefilter_reason": frozen_prefilter["reason"],
                "prefilter_candidates": frozen_prefilter["candidates"],
                "sidecar_candidate_count": len(sidecar_candidates),
                "sidecar_candidates": [
                    {
                        "title": row.get("title"),
                        "url": row.get("url"),
                        "origin": row.get("origin"),
                        "similarity": row.get("similarity"),
                    }
                    for row in sidecar_candidates[:MAX_SIDECAR_CANDIDATES]
                ],
                "eligible": False,
                "signal": "",
                "reason": "Dry run; no API or article-text retrieval performed.",
                "model_called": False,
                "model_call_count": 0,
                "prompt_version": PROMPT_VERSION,
            })
            continue

        # Stage 1 is entirely local/read-only: frozen evidence plus publisher/raw-
        # article sidecar. No GPT call happens until this evidence shows a plausible
        # timed future milestone.
        sidecar_additions = []
        sidecar_meta = {
            "candidate_count": len(sidecar_candidates),
            "page_attempts": 0,
            "network_fetches": 0,
            "evidence_additions": 0,
            "attempted": [],
        }
        if not args.no_sidecar:
            sidecar_additions, sidecar_meta, changed = build_signal_sidecar(
                date_str,
                topic,
                cluster,
                frozen_evidence,
                article_pool,
                sidecar_cache,
            )
            sidecar_cache_dirty = sidecar_cache_dirty or changed

        local_evidence = list(frozen_evidence) + list(sidecar_additions)
        local_prefilter = prefilter_next_signal(date_str, local_evidence)
        local_model_evidence = _model_evidence_candidates(date_str, local_evidence, topic=topic, cluster=cluster)

        triage_decision = None
        triage_cache_hit = False
        final_validation = None
        final_cache_hit = False
        model_call_count = 0
        signal_stage = "none"
        final_decision = {
            "eligible": False,
            "signal": "",
            "source_index": None,
            "reason": "No qualifying Next Signal found.",
        }
        final_evidence = local_model_evidence
        research_gap = ""
        research_queries = []

        # One cheap model call at most for local evidence, and only when the local
        # token-free gate says there is something worth evaluating.
        if local_prefilter["passed"] and local_model_evidence:
            triage_decision, triage_cache_hit, called = evaluate_cached(
                topic,
                local_model_evidence,
                "triage",
                evaluate_signal_triage,
            )
            model_call_count += 1 if called else 0
            status = triage_decision.get("status") or "NO_SIGNAL"
            if status == "SIGNAL" and triage_decision.get("eligible"):
                final_decision = {
                    "eligible": True,
                    "signal": triage_decision.get("signal") or "",
                    "source_index": triage_decision.get("source_index"),
                    "reason": triage_decision.get("reason") or "",
                }
                signal_stage = "local_triage"
            elif status == "RESEARCH_NEEDED":
                research_gap = triage_decision.get("research_gap") or ""
                research_queries = list(triage_decision.get("search_queries") or [])[:2]
            else:
                final_decision["reason"] = triage_decision.get("reason") or "No useful Next Signal in local evidence."
        else:
            final_decision["reason"] = "No locally supported timed future milestone; skipped model and external research."

        # Stage 2 runs ONLY when the local triage explicitly identifies a narrow
        # evidence gap. This prevents external discovery from becoming a default
        # per-story research pass.
        external_additions = []
        external_meta = {
            "provider": "GDELT DOC 2.0 → NewsAPI fallback",
            "queries": research_queries,
            "searches": [],
            "search_result_count": 0,
            "candidate_count": 0,
            "page_attempts": 0,
            "network_fetches": 0,
            "evidence_additions": 0,
            "attempted": [],
            "providers_used": [],
            "status": "not_requested",
        }
        external_prefilter = None

        if (
            signal_stage == "none"
            and triage_decision
            and triage_decision.get("status") == "RESEARCH_NEEDED"
            and research_queries
        ):
            if args.no_sidecar or args.no_external:
                external_meta["status"] = "disabled"
                final_decision["reason"] = "Local evidence needed targeted research, but external enrichment is disabled."
            else:
                external_additions, external_meta, changed = build_external_signal_evidence(
                    date_str,
                    topic,
                    cluster,
                    local_evidence,
                    sidecar_cache,
                    search_queries=research_queries,
                )
                sidecar_cache_dirty = sidecar_cache_dirty or changed
                if external_additions:
                    external_prefilter = prefilter_next_signal(date_str, external_additions)
                if external_prefilter and external_prefilter.get("passed"):
                    external_model_rows = _model_evidence_candidates(date_str, external_additions, topic=topic, cluster=cluster)
                    # Give the final validator the best local candidates plus the
                    # newly discovered evidence, but keep the prompt compact.
                    merged = local_model_evidence + external_model_rows
                    final_model_evidence = []
                    seen = set()
                    for row in merged:
                        key = (
                            _canonical_url(row.get("url") or ""),
                            _normalize_title(row.get("description") or "")[:180],
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        final_model_evidence.append(row)
                        if len(final_model_evidence) >= MAX_MODEL_EVIDENCE:
                            break
                    if final_model_evidence:
                        final_validation, final_cache_hit, called = evaluate_cached(
                            topic,
                            final_model_evidence,
                            "final",
                            evaluate_next_signal,
                        )
                        model_call_count += 1 if called else 0
                        final_decision = final_validation
                        final_evidence = final_model_evidence
                        if final_validation.get("eligible") and _clean_text(final_validation.get("signal") or ""):
                            signal_stage = "external_enrichment"
                    else:
                        final_decision["reason"] = "Targeted research returned no model-ready timed evidence."
                else:
                    final_decision["reason"] = "Targeted research did not find qualifying timed evidence for the identified gap."

        signal = _clean_text(final_decision.get("signal") or "") if final_decision.get("eligible") else ""
        topic["next_signal"] = signal

        source_index = final_decision.get("source_index")
        supporting = None
        if signal and isinstance(source_index, int) and 1 <= source_index <= len(final_evidence):
            supporting = final_evidence[source_index - 1]

        if signal:
            print(f"✅ {idx}. {title}: {signal} [{signal_stage}]")
        elif triage_decision and triage_decision.get("status") == "RESEARCH_NEEDED":
            print(f"🔎 {idx}. {title}: targeted research requested; no publishable signal found")
        elif local_prefilter["passed"]:
            print(f"⚪ {idx}. {title}: local evidence reviewed; no qualifying signal")
        else:
            print(f"⏭️ {idx}. {title}: no local timed milestone; 0 model calls")

        report.append({
            "topic_index": idx,
            "topic_title": title,
            "matched": True,
            "cluster_index": cluster_idx,
            "cluster_label": _cluster_label(cluster),
            "match": match_meta,
            "evidence_count": len(frozen_evidence),
            "prefilter_version": PREFILTER_VERSION,
            "frozen_prefilter_passed": frozen_prefilter["passed"],
            "frozen_prefilter_reason": frozen_prefilter["reason"],
            "local_prefilter_passed": local_prefilter["passed"],
            "local_prefilter_reason": local_prefilter["reason"],
            "local_model_evidence_count": len(local_model_evidence),
            "sidecar_enabled": not args.no_sidecar,
            "sidecar_article_pool_file": article_pool_file if article_pool else "",
            "sidecar_candidate_count": sidecar_meta.get("candidate_count", 0),
            "sidecar_page_attempts": sidecar_meta.get("page_attempts", 0),
            "sidecar_network_fetches": sidecar_meta.get("network_fetches", 0),
            "sidecar_evidence_additions": sidecar_meta.get("evidence_additions", 0),
            "sidecar_attempted": sidecar_meta.get("attempted", []),
            "triage_decision": triage_decision,
            "triage_cache_hit": triage_cache_hit,
            "research_gap": research_gap,
            "research_queries": research_queries,
            "external_enrichment_enabled": bool(not args.no_sidecar and not args.no_external),
            "external_search_provider": external_meta.get("provider", "GDELT DOC 2.0 → NewsAPI fallback"),
            "external_search_queries": external_meta.get("queries", []),
            "external_searches": external_meta.get("searches", []),
            "external_search_providers_used": external_meta.get("providers_used", []),
            "external_search_result_count": external_meta.get("search_result_count", 0),
            "external_candidate_count": external_meta.get("candidate_count", 0),
            "external_page_attempts": external_meta.get("page_attempts", 0),
            "external_network_fetches": external_meta.get("network_fetches", 0),
            "external_evidence_additions": external_meta.get("evidence_additions", 0),
            "external_attempted": external_meta.get("attempted", []),
            "external_status": external_meta.get("status", ""),
            "external_prefilter_passed": bool(external_prefilter and external_prefilter.get("passed")),
            "final_validation_decision": final_validation,
            "final_validation_cache_hit": final_cache_hit,
            "eligible": bool(signal),
            "signal": signal,
            "signal_stage": signal_stage,
            "reason": final_decision.get("reason") or "",
            "supporting_source": supporting,
            "model_called": model_call_count > 0,
            "model_call_count": model_call_count,
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "triage_prompt_version": TRIAGE_PROMPT_VERSION,
            "final_prompt_version": FINAL_PROMPT_VERSION,
        })

    if args.dry_run:
        print("ℹ️ Dry run complete; no files changed.")
        return

    if cache_dirty:
        _save_json_atomic(args.cache_file, cache)
    if sidecar_cache_dirty:
        _save_json_atomic(args.sidecar_cache_file, sidecar_cache)

    # The only mutation to topic summaries is next_signal. Every existing field is preserved.
    _save_json_atomic(summary_file, summaries)
    _save_json_atomic(report_file, report)

    count = sum(1 for row in report if row.get("eligible"))
    local_signals = sum(1 for row in report if row.get("signal_stage") == "local_triage")
    external_signals = sum(1 for row in report if row.get("signal_stage") == "external_enrichment")
    sidecar_fetches = sum(int(row.get("sidecar_network_fetches") or 0) for row in report)
    external_fetches = sum(int(row.get("external_network_fetches") or 0) for row in report)
    model_calls = sum(int(row.get("model_call_count") or 0) for row in report)
    research_requests = sum(
        1 for row in report
        if isinstance(row.get("triage_decision"), dict)
        and row["triage_decision"].get("status") == "RESEARCH_NEEDED"
    )
    zero_call_stories = sum(1 for row in report if int(row.get("model_call_count") or 0) == 0)
    print(f"✅ Added {count} Next Signal{'s' if count != 1 else ''} across {len(summaries)} stories")
    print(f"🔭 Local signals={local_signals}; publisher/raw article fetches={sidecar_fetches}")
    print(f"🌐 Research requested for {research_requests} stories; external signals={external_signals}; external fetches={external_fetches}")
    print(f"💸 GPT calls={model_calls}; zero-call stories={zero_call_stories}/{len(summaries)}")
    print(f"🧾 Wrote Next Signal audit → {report_file}")


if __name__ == "__main__":
    main()
