#!/usr/bin/env python3
"""
audit_gdelt_global_discovery.py

Read-only, cache-compatible shadow audit for upstream global story discovery
using GDELT's Global Similarity Graph (GSG).

Current audit behavior:
- Samples the target UTC date plus the existing Nominal News ±1-day window.
- Collapses exact and near-syndicated copies into write-up families.
- Splits broad GSG neighborhoods into exact developments anchored to the
  target date; adjacent-day coverage counts only when it matches that same
  development.
- Matches candidates to published/upstream Nominal News events using exact
  URL/title evidence first, then strict event-specific anchors or multiple
  discriminative shared terms. Generic people, institutions, and procedural
  verbs cannot establish a match by themselves.
- Produces a bounded, read-only ingestion preview with two candidate types:
  genuinely new discoveries and upstream stories that were underrepresented.
- Emits only URLs absent from the normalized local corpus, with four English
  representatives per event whenever the preview includes that event.
- Carries GDELT breadth as shadow metadata and computes an exploratory blended
  ranking without modifying Nominal News production ranking.
- Keeps English articles as all user-facing representatives. Directly linked
  non-English coverage may contribute only to aggregate attention metrics.
- Optional --pipeline-mode writes a separate candidate feed while preserving
  the normal shadow-audit behavior and stable filenames by default.

No OpenAI, NewsAPI, BigQuery, or other paid API calls are made.
No production pipeline file is modified.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable
from urllib.parse import urlparse

import requests

try:
    import tldextract  # Optional; safe fallback below.
except Exception:
    tldextract = None

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception:
    np = None
    SentenceTransformer = None


VERSION = "4.3"
GSG_URL = (
    "https://data.gdeltproject.org/gdeltv3/gsg/"
    "{stamp}.gsg.json.gz"
)
USER_AGENT = (
    "NominalNews-GDELT-Discovery-Audit/4.3 "
    "(read-only research; bounded direct downloads)"
)
MODEL_NAME = "all-MiniLM-L6-v2"
ENGLISH_NAMES = {"english", "eng", "en"}

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into",
    "after", "before", "over", "under", "amid", "about", "says", "say",
    "said", "new", "latest", "live", "news", "report", "reports", "update",
    "updates", "video", "watch", "why", "how", "what", "when", "where",
    "who", "its", "their", "his", "her", "our", "your", "are", "was",
    "were", "has", "have", "had", "will", "would", "could", "should",
    "may", "might", "more", "most", "than", "one", "two", "day", "days",
    "year", "years", "today", "tomorrow", "yesterday", "world", "global",
}

GENERIC_TITLE_TOKENS = {
    "bulletin", "briefing", "roundup", "headlines", "homepage", "newsletter",
}


@dataclass(frozen=True)
class Limits:
    max_files: int
    max_bytes: int
    max_edges: int
    max_raw_candidates: int


class UnionFind:
    def __init__(self, items: Iterable[Any]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: Any) -> Any:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: Any, b: Any) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only GDELT GSG global-discovery audit."
    )
    p.add_argument("--date", required=True, help="Target UTC date: YYYY-MM-DD")
    p.add_argument(
        "--window-days",
        type=int,
        default=1,
        help="Days before and after target date. Default 1 (±1 day).",
    )
    p.add_argument(
        "--slot-minutes",
        type=int,
        default=60,
        choices=(15, 30, 60),
        help="Sampling cadence. Default 60 minutes.",
    )
    p.add_argument(
        "--files-per-slot",
        type=int,
        default=1,
        help="Maximum successful GSG files per slot. Default 1.",
    )
    p.add_argument(
        "--probe-minutes",
        type=int,
        default=6,
        help="Minutes after each slot timestamp to probe. Default 6.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=72,
        help="Hard selected-file ceiling. Default 72 for ±1-day hourly sampling.",
    )
    p.add_argument(
        "--max-download-mb",
        type=float,
        default=30.0,
        help="Hard newly-downloaded compressed-data ceiling. Default 30 MB.",
    )
    p.add_argument(
        "--max-edges",
        type=int,
        default=300_000,
        help="Hard accepted-edge ceiling. Default 300,000.",
    )
    p.add_argument(
        "--max-raw-candidates",
        type=int,
        default=600,
        help="Maximum seed neighborhoods entering canonical merge. Default 600.",
    )
    p.add_argument(
        "--min-sim",
        type=float,
        default=0.55,
        help="Minimum GSG lead-text similarity for non-title edges.",
    )
    p.add_argument(
        "--min-shared-words",
        type=int,
        default=7,
        help="Minimum shared lead-text words for non-title edges.",
    )
    p.add_argument(
        "--min-domains",
        type=int,
        default=5,
        help="Minimum distinct outlet domains for a candidate. Default 5.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Maximum canonical candidates in output. Default 25.",
    )
    p.add_argument(
        "--max-ingestion-candidates",
        type=int,
        default=12,
        help=(
            "Maximum candidate events in the bounded ingestion preview. "
            "Default 12."
        ),
    )
    p.add_argument(
        "--max-representatives",
        type=int,
        default=4,
        help="English net-new representative URLs per preview candidate. Default 4.",
    )
    p.add_argument(
        "--max-ingestion-records",
        type=int,
        default=48,
        help="Hard total net-new article-record ceiling. Default 48.",
    )
    p.add_argument(
        "--family-lead-sim",
        type=float,
        default=0.88,
        help="Lead-text similarity for near-syndication family collapse.",
    )
    p.add_argument(
        "--family-title-jaccard",
        type=float,
        default=0.72,
        help="Headline-token overlap for near-syndication collapse.",
    )
    p.add_argument(
        "--event-merge-sim",
        type=float,
        default=0.74,
        help="MiniLM similarity floor for duplicate seed-event merging.",
    )
    p.add_argument(
        "--target-cluster-sim",
        type=float,
        default=0.67,
        help=(
            "Minimum full-title similarity for grouping target-day write-ups "
            "into one development. Default 0.67."
        ),
    )
    p.add_argument(
        "--adjacent-support-sim",
        type=float,
        default=0.78,
        help=(
            "Minimum full-title similarity for adjacent-day support. "
            "Default 0.78."
        ),
    )
    p.add_argument(
        "--min-event-cohesion",
        type=float,
        default=0.52,
        help=(
            "Minimum target-anchored title-cohesion p20 for output candidates. "
            "Default 0.52."
        ),
    )
    p.add_argument(
        "--min-preview-writeups",
        type=int,
        default=3,
        help=(
            "Minimum target-day write-up families for ingestion preview. "
            "Default 3."
        ),
    )
    p.add_argument(
        "--published-match-threshold",
        type=float,
        default=0.54,
        help="Semantic match threshold to a published Nominal News event.",
    )
    p.add_argument(
        "--upstream-match-threshold",
        type=float,
        default=0.52,
        help="Semantic match threshold to an earlier filtered cluster.",
    )
    p.add_argument(
        "--cache-dir",
        default=".gdelt_gsg_cache",
        help="Local GSG file cache directory.",
    )
    p.add_argument(
        "--summary-file",
        default=None,
        help="Published topic summaries JSON; defaults by date.",
    )
    p.add_argument(
        "--final-file",
        default=None,
        help="Published final-cluster JSON; defaults by date.",
    )
    p.add_argument(
        "--upstream-file",
        default=None,
        help="Earlier filtered clusters JSON; defaults by date.",
    )
    p.add_argument(
        "--normalized-file",
        default=None,
        help=(
            "Normalized local article corpus used for net-new URL checks; "
            "defaults to articles_raw_normalized_{date}.json."
        ),
    )
    p.add_argument(
        "--cached-only",
        action="store_true",
        help="Use existing cache only; make no network requests.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=35,
        help="Per-request read timeout in seconds. Default 35.",
    )
    p.add_argument(
        "--pipeline-mode",
        action="store_true",
        help=(
            "Generate a pre-merge candidate feed using clustered_articles as "
            "the upstream comparison. Published/final outputs are deliberately "
            "ignored so stale files cannot influence discovery."
        ),
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path. Manual-audit default is unchanged.",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path. Manual-audit default is unchanged.",
    )
    return p.parse_args()


def canonical_url(url: str) -> str:
    try:
        p = urlparse((url or "").strip())
        host = (p.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = re.sub(r"/+$", "", p.path or "")
        if not host:
            return ""
        return f"https://{host}{path}"
    except Exception:
        return ""


def outlet_domain(url: str) -> str:
    try:
        host = (urlparse(url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return ""

        if tldextract is not None:
            ext = tldextract.extract(host)
            if ext.domain and ext.suffix:
                return f"{ext.domain}.{ext.suffix}"

        parts = host.split(".")
        common_second_level = {
            "co", "com", "org", "net", "gov", "ac", "edu", "news",
        }
        if len(parts) >= 3 and parts[-2] in common_second_level:
            return ".".join(parts[-3:])
        return ".".join(parts[-2:]) if len(parts) >= 2 else host
    except Exception:
        return ""


def normalize_language(value: Any) -> str:
    return re.sub(r"[^a-z]", "", str(value or "").lower())


def is_english(value: Any) -> bool:
    return normalize_language(value) in ENGLISH_NAMES


def clean_title(title: str) -> str:
    text = re.sub(r"<[^>]+>", " ", title or "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(title: str) -> str:
    text = clean_title(title).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_tokens(title: str) -> set[str]:
    return {
        t for t in normalize_title(title).split()
        if len(t) >= 3 and t not in STOPWORDS
    }


def title_jaccard(a: str, b: str) -> float:
    aa = title_tokens(a)
    bb = title_tokens(b)
    return len(aa & bb) / len(aa | bb) if aa and bb else 0.0


def generic_exact_title(title: str) -> bool:
    toks = title_tokens(title)
    if len(toks) < 4 or len(normalize_title(title)) < 30:
        return True
    return len(toks & GENERIC_TITLE_TOKENS) >= 2


def meaningful_anchor_tokens(titles: list[str]) -> set[str]:
    per_title = [title_tokens(t) for t in titles if clean_title(t)]
    if not per_title:
        return set()
    counts = Counter(tok for toks in per_title for tok in toks)
    repeat_floor = 2 if len(per_title) >= 2 else 1
    repeated = {tok for tok, n in counts.items() if n >= repeat_floor}
    representative = per_title[0] if per_title else set()
    return repeated | {t for t in representative if len(t) >= 5}


def stable_id(prefix: str, values: Iterable[str]) -> str:
    raw = "|".join(sorted(set(values))).encode("utf-8", errors="ignore")
    return f"{prefix}_{hashlib.sha1(raw).hexdigest()[:12]}"



_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def normalize_day(value: Any) -> str:
    """Best-effort YYYY-MM-DD extraction from GSG date fields."""
    text = str(value or "").strip()
    if not text:
        return ""

    m = re.search(r"(?<!\d)(20\d{2})[-/]?(\d{2})[-/]?(\d{2})(?!\d)", text)
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3))
            ).strftime("%Y-%m-%d")
        except ValueError:
            return ""
    return ""


def day_from_url(url: str) -> str:
    """Extract a publication date from common URL date patterns."""
    text = str(url or "")
    m = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)", text)
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3))
            ).strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = re.search(
        r"/(20\d{2})/(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)/(\d{1,2})(?:/|$)",
        text,
        flags=re.I,
    )
    if m:
        month = _MONTHS.get(m.group(2).lower())
        try:
            if month:
                return datetime(
                    int(m.group(1)), month, int(m.group(3))
                ).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return ""


def node_explicit_days(raw_date: Any, raw_url: str) -> set[str]:
    days = set()
    for value in (normalize_day(raw_date), day_from_url(raw_url)):
        if value:
            days.add(value)
    return days


def observed_day(slot_id: str) -> str:
    raw = str(slot_id or "")[:8]
    if len(raw) == 8 and raw.isdigit():
        try:
            return datetime.strptime(raw, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return ""
    return ""


def family_day_role(family: dict, target_day: str) -> str:
    """
    TARGET_EXPLICIT is strongest. If GSG lacks an article date, a target-day
    observation is used as a conservative fallback. Explicit adjacent dates
    are never promoted by observation time.
    """
    explicit = set(family.get("explicit_days") or set())
    observed = set(family.get("observed_days") or set())
    if target_day in explicit:
        return "TARGET_EXPLICIT"
    if explicit:
        return "ADJACENT_EXPLICIT"
    if target_day in observed:
        return "TARGET_OBSERVED"
    return "ADJACENT_OBSERVED"


def ordered_window_dates(target: datetime, window_days: int) -> list[datetime]:
    out = [target]
    for delta in range(1, max(0, window_days) + 1):
        out.append(target - timedelta(days=delta))
        out.append(target + timedelta(days=delta))
    return out


def slot_times(date_obj: datetime, cadence_minutes: int) -> list[datetime]:
    start = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    out = []
    cur = start
    while cur < end:
        out.append(cur)
        cur += timedelta(minutes=cadence_minutes)
    return out


def download_candidate(
    session: requests.Session,
    url: str,
    dest: Path,
    remaining_bytes: int,
    timeout: int,
) -> tuple[bool, int, str]:
    if dest.exists() and dest.stat().st_size > 20:
        return True, 0, "cache"
    if remaining_bytes <= 0:
        return False, 0, "byte_cap"

    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    try:
        with session.get(
            url,
            stream=True,
            timeout=(7, timeout),
            allow_redirects=True,
        ) as response:
            if response.status_code == 404:
                return False, 0, "missing"
            if response.status_code in {429, 500, 502, 503, 504}:
                return False, 0, f"http_{response.status_code}"
            response.raise_for_status()

            declared = int(response.headers.get("content-length") or 0)
            if declared and declared > remaining_bytes:
                return False, 0, "byte_cap"

            written = 0
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > remaining_bytes:
                        raise RuntimeError("byte_cap")
                    f.write(chunk)

        if written < 20:
            tmp.unlink(missing_ok=True)
            return False, 0, "empty"

        with open(tmp, "rb") as f:
            if f.read(2) != b"\x1f\x8b":
                tmp.unlink(missing_ok=True)
                return False, 0, "not_gzip"

        tmp.replace(dest)
        return True, written, "download"
    except RuntimeError as exc:
        tmp.unlink(missing_ok=True)
        return False, 0, str(exc)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return False, 0, type(exc).__name__


def acquire_files(
    target: datetime,
    args: argparse.Namespace,
    limits: Limits,
) -> tuple[list[Path], dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    selected: list[Path] = []
    selected_seen: set[str] = set()
    downloaded_bytes = 0
    status_counts = Counter()
    per_day = defaultdict(Counter)
    budget_exhausted = False

    for day in ordered_window_dates(target, args.window_days):
        day_str = day.strftime("%Y-%m-%d")
        cache_root = Path(args.cache_dir) / day_str
        cache_root.mkdir(parents=True, exist_ok=True)

        for slot in slot_times(day, args.slot_minutes):
            if len(selected) >= limits.max_files or downloaded_bytes >= limits.max_bytes:
                budget_exhausted = True
                break

            found_this_slot = 0
            for offset in range(max(1, args.probe_minutes)):
                stamp_dt = slot + timedelta(minutes=offset)
                stamp = stamp_dt.strftime("%Y%m%d%H%M00")
                dest = cache_root / f"{stamp}.gsg.json.gz"
                key = str(dest.resolve())

                if key in selected_seen:
                    continue

                if dest.exists() and dest.stat().st_size > 20:
                    selected.append(dest)
                    selected_seen.add(key)
                    status_counts["cache"] += 1
                    per_day[day_str]["cache"] += 1
                    found_this_slot += 1
                elif not args.cached_only:
                    ok, added, status = download_candidate(
                        session=session,
                        url=GSG_URL.format(stamp=stamp),
                        dest=dest,
                        remaining_bytes=limits.max_bytes - downloaded_bytes,
                        timeout=args.timeout,
                    )
                    status_counts[status] += 1
                    per_day[day_str][status] += 1
                    if ok:
                        selected.append(dest)
                        selected_seen.add(key)
                        downloaded_bytes += added
                        found_this_slot += 1
                    elif status == "byte_cap":
                        budget_exhausted = True
                        break

                if found_this_slot >= max(1, args.files_per_slot):
                    break

            if budget_exhausted:
                break
        if budget_exhausted:
            break

    return selected, {
        "selected_file_count": len(selected),
        "downloaded_bytes_this_run": downloaded_bytes,
        "downloaded_mb_this_run": round(downloaded_bytes / (1024 * 1024), 2),
        "status_counts": dict(status_counts),
        "per_day": {k: dict(v) for k, v in sorted(per_day.items())},
        "budget_exhausted": budget_exhausted,
        "cache_root": str(Path(args.cache_dir)),
    }


def update_node(
    nodes: dict[str, dict],
    record: dict,
    side: str,
    slot_id: str,
) -> None:
    raw_url = str(record.get(f"{side}Url") or "").strip()
    url = canonical_url(raw_url)
    if not url:
        return

    title = clean_title(str(record.get(f"{side}Title") or ""))
    lang = str(record.get(f"{side}Lang") or "").strip()
    date = str(record.get(f"{side}Date") or "").strip()
    image = str(record.get(f"{side}Image") or "").strip()
    domain = outlet_domain(raw_url)

    candidate = {
        "url": raw_url,
        "canonical_url": url,
        "title": title,
        "language": lang,
        "date": date,
        "image": image,
        "domain": domain,
        "slots": {slot_id},
        "explicit_days": node_explicit_days(date, raw_url),
        "observed_days": ({observed_day(slot_id)} if observed_day(slot_id) else set()),
    }

    old = nodes.get(url)
    if old is None:
        nodes[url] = candidate
        return

    old.setdefault("slots", set()).add(slot_id)
    old.setdefault("explicit_days", set()).update(candidate["explicit_days"])
    old.setdefault("observed_days", set()).update(candidate["observed_days"])

    prefer_candidate = (
        is_english(lang) and not is_english(old.get("language"))
    ) or (title and not old.get("title"))

    if prefer_candidate:
        candidate["slots"] = set(old.get("slots") or set()) | {slot_id}
        candidate["explicit_days"] = (
            set(old.get("explicit_days") or set()) | candidate["explicit_days"]
        )
        candidate["observed_days"] = (
            set(old.get("observed_days") or set()) | candidate["observed_days"]
        )
        nodes[url] = candidate


def parse_gsg_files(
    files: list[Path],
    args: argparse.Namespace,
    limits: Limits,
) -> tuple[dict, dict, dict]:
    nodes: dict[str, dict] = {}
    adjacency: dict[str, dict[str, dict]] = defaultdict(dict)
    stats = Counter()
    file_errors = []

    for path in files:
        if stats["accepted_edge_records"] >= limits.max_edges:
            break

        slot_id = path.name.split(".", 1)[0]

        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    stats["records_seen"] += 1
                    if stats["accepted_edge_records"] >= limits.max_edges:
                        break
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        rec = json.loads(line)
                    except Exception:
                        stats["invalid_json"] += 1
                        continue

                    edge_type = str(rec.get("Type") or rec.get("type") or "").lower()
                    sim = float(rec.get("simScore") or 0.0)
                    words = int(rec.get("simWords") or 0)

                    if edge_type == "title":
                        sim = 1.0
                    elif edge_type != "sim":
                        stats["unknown_type"] += 1
                        continue
                    elif sim < args.min_sim or words < args.min_shared_words:
                        stats["below_threshold"] += 1
                        continue

                    from_raw = str(rec.get("fromUrl") or "").strip()
                    to_raw = str(rec.get("toUrl") or "").strip()
                    from_url = canonical_url(from_raw)
                    to_url = canonical_url(to_raw)

                    if not from_url or not to_url or from_url == to_url:
                        stats["invalid_url"] += 1
                        continue

                    from_domain = outlet_domain(from_raw)
                    to_domain = outlet_domain(to_raw)
                    if not from_domain or not to_domain:
                        stats["invalid_domain"] += 1
                        continue
                    if from_domain == to_domain:
                        stats["same_outlet"] += 1
                        continue

                    from_en = is_english(rec.get("fromLang"))
                    to_en = is_english(rec.get("toLang"))
                    if not (from_en or to_en):
                        stats["no_english_anchor"] += 1
                        continue

                    update_node(nodes, rec, "from", slot_id)
                    update_node(nodes, rec, "to", slot_id)

                    prior = adjacency[from_url].get(to_url)
                    if prior:
                        prior["sim"] = max(prior["sim"], sim)
                        prior["shared_words"] = max(prior["shared_words"], words)
                        prior["types"].add(edge_type)
                        prior["slots"].add(slot_id)
                    else:
                        edge = {
                            "sim": round(sim, 4),
                            "shared_words": words,
                            "types": {edge_type},
                            "slots": {slot_id},
                        }
                        adjacency[from_url][to_url] = edge
                        adjacency[to_url][from_url] = edge

                    stats["accepted_edge_records"] += 1
                    if from_en and to_en:
                        stats["english_english_edge_records"] += 1
                    else:
                        stats["cross_language_edge_records"] += 1

        except Exception as exc:
            file_errors.append({
                "file": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            })

    unique_edges = sum(len(v) for v in adjacency.values()) // 2
    stats["nodes"] = len(nodes)
    stats["unique_edges"] = unique_edges
    stats["edge_limit_reached"] = int(
        stats["accepted_edge_records"] >= limits.max_edges
    )
    return nodes, adjacency, {
        **dict(stats),
        "file_errors": file_errors,
    }


def iter_unique_edges(adjacency: dict[str, dict[str, dict]]):
    seen = set()
    for u, neighbors in adjacency.items():
        for v, edge in neighbors.items():
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            yield u, v, edge


def build_writeup_families(
    nodes: dict[str, dict],
    adjacency: dict[str, dict[str, dict]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict], dict[str, str], dict]:
    uf = UnionFind(nodes.keys())
    stats = Counter()

    exact_title_owner: dict[str, str] = {}
    for url, node in nodes.items():
        title = node.get("title") or ""
        norm = normalize_title(title)
        if not norm or generic_exact_title(title):
            continue
        owner = exact_title_owner.get(norm)
        if owner:
            uf.union(url, owner)
            stats["exact_title_unions"] += 1
        else:
            exact_title_owner[norm] = url

    for u, v, edge in iter_unique_edges(adjacency):
        types = edge.get("types") or set()
        if "title" in types:
            uf.union(u, v)
            stats["title_edge_unions"] += 1
            continue

        if (
            float(edge.get("sim", 0.0)) >= args.family_lead_sim
            and int(edge.get("shared_words", 0)) >= max(10, args.min_shared_words)
        ):
            a = nodes.get(u, {}).get("title") or ""
            b = nodes.get(v, {}).get("title") or ""
            if title_jaccard(a, b) >= args.family_title_jaccard:
                uf.union(u, v)
                stats["near_syndication_unions"] += 1

    groups: dict[str, list[str]] = defaultdict(list)
    for url in nodes:
        groups[uf.find(url)].append(url)

    families: dict[str, dict] = {}
    url_to_family: dict[str, str] = {}

    for urls in groups.values():
        family_id = stable_id("wf", urls)
        member_nodes = [nodes[u] for u in urls]
        english_nodes = [n for n in member_nodes if is_english(n.get("language"))]
        candidates = english_nodes or member_nodes

        representative = max(
            candidates,
            key=lambda n: (
                len(adjacency.get(n["canonical_url"], {})),
                len(n.get("title") or ""),
            ),
        )

        domains = {n.get("domain") for n in member_nodes if n.get("domain")}
        languages = {
            normalize_language(n.get("language"))
            for n in member_nodes if n.get("language")
        }
        slots = set()
        titles = set()
        explicit_days = set()
        observed_days = set()
        for n in member_nodes:
            slots.update(n.get("slots") or set())
            explicit_days.update(n.get("explicit_days") or set())
            observed_days.update(n.get("observed_days") or set())
            norm = normalize_title(n.get("title") or "")
            if norm:
                titles.add(norm)

        family = {
            "family_id": family_id,
            "urls": set(urls),
            "domains": domains,
            "languages": languages,
            "slots": slots,
            "explicit_days": explicit_days,
            "observed_days": observed_days,
            "unique_title_count": len(titles),
            "url_count": len(urls),
            "outlet_count": len(domains),
            "english_url_count": len(english_nodes),
            "cross_language_url_count": sum(
                1 for n in member_nodes if not is_english(n.get("language"))
            ),
            "representative": {
                "title": representative.get("title"),
                "url": representative.get("url"),
                "canonical_url": representative.get("canonical_url"),
                "domain": representative.get("domain"),
                "language": representative.get("language"),
                "image": representative.get("image"),
            },
            "_members": member_nodes,
        }
        families[family_id] = family
        for url in urls:
            url_to_family[url] = family_id

    stats["family_count"] = len(families)
    stats["multi_outlet_family_count"] = sum(
        1 for f in families.values() if f["outlet_count"] > 1
    )
    stats["largest_family_outlets"] = max(
        (f["outlet_count"] for f in families.values()),
        default=0,
    )
    return families, url_to_family, dict(stats)


def build_family_graph(
    adjacency: dict[str, dict[str, dict]],
    url_to_family: dict[str, str],
) -> dict[str, dict[str, dict]]:
    graph: dict[str, dict[str, dict]] = defaultdict(dict)

    for u, v, edge in iter_unique_edges(adjacency):
        fu = url_to_family.get(u)
        fv = url_to_family.get(v)
        if not fu or not fv or fu == fv:
            continue

        prior = graph[fu].get(fv)
        if prior:
            prior["max_sim"] = max(prior["max_sim"], float(edge.get("sim", 0.0)))
            prior["max_shared_words"] = max(
                prior["max_shared_words"], int(edge.get("shared_words", 0))
            )
            prior["edge_count"] += 1
            prior["slots"].update(edge.get("slots") or set())
        else:
            fam_edge = {
                "max_sim": float(edge.get("sim", 0.0)),
                "max_shared_words": int(edge.get("shared_words", 0)),
                "edge_count": 1,
                "slots": set(edge.get("slots") or set()),
            }
            graph[fu][fv] = fam_edge
            graph[fv][fu] = fam_edge

    return graph


def raw_candidate_from_seed(
    seed_id: str,
    families: dict[str, dict],
    family_graph: dict[str, dict[str, dict]],
) -> dict | None:
    seed = families.get(seed_id)
    if not seed or not seed.get("representative", {}).get("title"):
        return None
    if not is_english(seed.get("representative", {}).get("language")):
        return None

    family_ids = {seed_id, *family_graph.get(seed_id, {}).keys()}
    urls = set()
    domains = set()
    slots = set()
    languages = set()
    titles = []

    for family_id in family_ids:
        fam = families[family_id]
        urls.update(fam["urls"])
        domains.update(fam["domains"])
        slots.update(fam["slots"])
        languages.update(fam["languages"])
        title = fam.get("representative", {}).get("title") or ""
        if title:
            titles.append(title)

    ordered_titles = [seed["representative"]["title"]]
    ordered_titles.extend(
        sorted(
            [t for t in titles if t != ordered_titles[0]],
            key=len,
            reverse=True,
        )[:9]
    )

    family_outlet_counts = [families[f]["outlet_count"] for f in family_ids]
    preliminary_score = (
        len(family_ids) * 4.0
        + math.log1p(len(domains)) * 2.0
        + math.log1p(len(slots)) * 1.5
    )

    return {
        "seed_family_id": seed_id,
        "family_ids": set(family_ids),
        "urls": urls,
        "domains": domains,
        "slots": slots,
        "languages": languages,
        "titles": ordered_titles,
        "anchor_tokens": meaningful_anchor_tokens(ordered_titles),
        "preliminary_score": preliminary_score,
        "largest_family_outlet_count": max(family_outlet_counts, default=0),
    }


def fallback_similarity(a: str, b: str) -> float:
    aa = title_tokens(a)
    bb = title_tokens(b)
    return len(aa & bb) / len(aa | bb) if aa and bb else 0.0


def candidate_text(candidate: dict) -> str:
    return ". ".join(candidate.get("titles", [])[:8])


def load_model() -> Any | None:
    if SentenceTransformer is None or np is None:
        return None
    try:
        print(f"🧠 Loading local semantic model: {MODEL_NAME}")
        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        print(f"⚠️ Local semantic model unavailable: {type(exc).__name__}: {exc}")
        return None



IDENTITY_GENERIC = {
    "the", "a", "an", "latest", "breaking", "watch", "live", "news",
    "report", "reports", "update", "updates", "judge", "court", "jury",
    "trial", "case", "official", "officials", "president", "prime minister",
    "senator", "governor", "minister", "police", "authorities", "source",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "us", "u s", "uk", "u k", "eu",
}

DEVELOPMENT_GENERIC = {
    "case", "story", "news", "report", "reports", "latest", "update",
    "updates", "official", "officials", "source", "sources", "people",
    "person", "world", "today", "tomorrow", "yesterday", "year", "years",
}

PENDING_MARKERS = {
    "await", "could", "deliber", "deadlock", "expect", "may", "might",
    "pend", "plan", "possibl", "risk", "resum", "seek", "set", "talk",
    "yet",
}

FINALITY_MARKERS = {
    "acquit", "announc", "approv", "approve", "approved", "arrest",
    "block", "buy", "bought", "cancel", "cancell", "canceled",
    "cancelled", "charg", "clos", "confirm", "convict", "declar", "die",
    "died", "dies", "death", "end", "fin", "fine", "fined", "launch",
    "lose", "lost",
    "mistrial", "penaliz", "reject", "releas", "resign", "rescu", "rul",
    "rule",
    "sentenc", "sign", "suspend", "win", "wins", "won",
}

ACTION_GROUP_PATTERNS = {
    "PENALTY": {
        "ban", "dock", "fine", "penaliz", "punish", "sanction", "suspend", "strip",
    },
    "DEPARTURE": {
        "depart", "exit", "leav", "leave", "left", "out", "part", "quit", "resign",
    },
    "DEATH": {
        "dead", "death", "die", "died", "dies", "pass",
    },
    "LEGAL_CHARGE": {
        "arrest", "charg", "indict", "prosecut", "sentenc",
    },
    "ATTACK": {
        "attack", "bomb", "kill", "shoot", "shot", "strike", "wound",
    },
    "ACQUISITION": {
        "acquir", "buy", "bought", "merg", "purchas",
    },
    "ELECTION_RESULT": {
        "elect", "primary", "vote", "win", "won",
    },
    "COURT_RULING": {
        "block", "dismiss", "order", "rul", "rule",
    },
    "LAUNCH_RELEASE": {
        "debut", "launch", "releas", "unveil",
    },
    "CLOSURE": {
        "clos", "shutdown", "shut",
    },
}


# Comparison-only exclusions. These terms can help a semantic model recognize
# the general domain of a story, but they are too generic to prove that two
# records describe the same event.
MATCH_GENERIC_WORDS = {
    # Generic institutions / offices
    "administration", "agency", "authorities", "authority", "committee",
    "congress", "department", "federal", "government", "governor", "house",
    "institution", "judge", "judges", "judicial", "jury", "lawmakers",
    "leader", "minister", "military", "official", "officials", "panel",
    "police", "president", "secretary", "senate", "state", "supreme",
    "court", "courts", "white",

    # Generic legal / procedural language
    "appeal", "appeals", "appealed", "appealing", "ask", "asks", "asked",
    "asking", "bid", "bids", "block", "blocks", "blocked", "blocking",
    "case", "cases", "challenge", "challenges", "challenged", "decision",
    "decisions", "hearing", "hearings", "law", "laws", "lawsuit",
    "lawsuits", "legal", "lift", "lifts", "motion", "motions", "move",
    "moves", "order", "orders", "ordered", "request", "requests",
    "requested", "rule", "rules", "ruled", "ruling", "rulings", "seek",
    "seeks", "seeking", "trial", "trials", "verdict", "verdicts",

    # Generic reporting / timing language
    "according", "breaking", "latest", "new", "news", "report", "reports",
    "reported", "reporting", "says", "said", "statement", "update",
    "updates",
}

# A shared action group remains useful as a compatibility check, but the
# generic COURT_RULING group is never sufficient evidence on its own.
MATCH_GENERIC_ACTION_GROUPS = {"COURT_RULING"}


def development_action_groups(title: str) -> set[str]:
    stems = {stemish(t) for t in normalize_title(title).split()}
    groups = {
        group
        for group, patterns in ACTION_GROUP_PATTERNS.items()
        if stems & patterns
    }
    norm = normalize_title(title)
    if "parts ways" in norm:
        groups.add("DEPARTURE")
    if "cause of death" in norm:
        groups.add("DEATH")
    return groups



def stemish(token: str) -> str:
    t = re.sub(r"[^a-z0-9]", "", (token or "").lower())
    if len(t) <= 4:
        return t
    for suffix in ("ingly", "edly", "ments", "ment", "ation", "ations"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[:-len(suffix)]
    for suffix in ("ing", "ied", "ies", "ed", "es", "s"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            base = t[:-len(suffix)]
            if suffix == "ied":
                base += "y"
            return base
    return t


def extract_identity_terms(title: str) -> set[str]:
    """Automatic proper-name/location/org cues from a headline."""
    text = clean_title(title)
    terms = set()

    pattern = re.compile(
        r"\b(?:[A-Z]{2,}|[A-Z][A-Za-z0-9'’.-]+)"
        r"(?:\s+(?:[A-Z]{2,}|[A-Z][A-Za-z0-9'’.-]+)){0,3}\b"
    )
    for match in pattern.finditer(text):
        phrase = normalize_title(match.group(0))
        words = [w for w in phrase.split() if w not in STOPWORDS]
        while words and words[0] in IDENTITY_GENERIC:
            words.pop(0)
        while words and words[-1] in IDENTITY_GENERIC:
            words.pop()
        if not words:
            continue
        normalized = " ".join(words)
        if normalized and normalized not in IDENTITY_GENERIC:
            terms.add(normalized)
        for word in words:
            if len(word) >= 4 and word not in IDENTITY_GENERIC:
                terms.add(word)

    # Acronyms and distinctive mixed-case product/organization tokens.
    for raw in re.findall(r"\b[A-Za-z][A-Za-z0-9.-]{2,}\b", text):
        if raw.isupper() or any(ch.isdigit() for ch in raw):
            token = normalize_title(raw)
            if token and token not in STOPWORDS:
                terms.add(token)

    return terms


def development_tokens(title: str) -> set[str]:
    identity_words = {
        word
        for term in extract_identity_terms(title)
        for word in term.split()
    }
    out = set()
    for token in normalize_title(title).split():
        if token in STOPWORDS or token in DEVELOPMENT_GENERIC:
            continue
        if token in identity_words:
            continue
        stem = stemish(token)
        if len(stem) >= 3 and stem not in STOPWORDS and stem not in DEVELOPMENT_GENERIC:
            out.add(stem)
    return out


def development_text(title: str) -> str:
    tokens = sorted(development_tokens(title))
    return " ".join(tokens) if tokens else normalize_title(title)


def development_stage(title: str) -> str:
    norm = normalize_title(title)
    toks = {stemish(t) for t in norm.split()}
    pending = bool(toks & PENDING_MARKERS) or bool(
        re.search(r"\b(without|no) (?:a )?(?:verdict|decision|agreement)\b", norm)
    )
    final = bool(toks & FINALITY_MARKERS)
    if final:
        return "FINAL"
    if pending:
        return "PENDING"
    return "NEUTRAL"



def _comparison_generic_stems() -> set[str]:
    """Static generic stems excluded from event-match evidence."""
    stems = set()
    for value in (
        set(STOPWORDS)
        | set(IDENTITY_GENERIC)
        | set(DEVELOPMENT_GENERIC)
        | set(MATCH_GENERIC_WORDS)
    ):
        for token in normalize_title(str(value)).split():
            stem = stemish(token)
            if stem:
                stems.add(stem)
    return stems


def _identity_phrase_key(term: str) -> str:
    parts = [
        stemish(token)
        for token in normalize_title(term).split()
        if len(stemish(token)) >= 3
    ]
    return " ".join(parts)


def _raw_comparison_terms(titles: Iterable[str]) -> set[str]:
    generic = _comparison_generic_stems()
    out = set()
    for title in titles:
        for token in normalize_title(title).split():
            if token.isdigit():
                continue
            stem = stemish(token)
            if len(stem) >= 3 and stem not in generic:
                out.add(stem)
    return out


def build_reference_match_context(references: list[dict]) -> dict:
    """
    Build event-level document-frequency exclusions.

    A person, institution, or term repeated across many different candidate
    references (for example, a president's surname) is treated as generic for
    matching purposes. It may still contribute to semantic similarity, but it
    cannot prove event identity.
    """
    term_df = Counter()
    identity_df = Counter()

    for ref in references:
        texts = list(dict.fromkeys(
            clean_title(t)
            for t in ([ref.get("title") or ""] + list(ref.get("texts") or []))
            if clean_title(t)
        ))
        terms = _raw_comparison_terms(texts)
        term_df.update(terms)

        identities = set()
        for title in texts:
            for term in extract_identity_terms(title):
                key = _identity_phrase_key(term)
                if key:
                    identities.add(key)
        identity_df.update(identities)

    n_refs = max(1, len(references))
    term_floor = max(4, int(math.ceil(n_refs * 0.06)))
    identity_floor = max(3, int(math.ceil(n_refs * 0.04)))

    return {
        "reference_count": len(references),
        "generic_term_floor": term_floor,
        "generic_identity_floor": identity_floor,
        "dynamic_generic_terms": {
            term for term, count in term_df.items()
            if count >= term_floor
        },
        "dynamic_generic_identities": {
            term for term, count in identity_df.items()
            if count >= identity_floor
        },
    }


def discriminative_match_terms(
    titles: Iterable[str],
    context: dict | None,
) -> set[str]:
    """
    Return automatically derived event-specific terms.

    Static procedural/institutional vocabulary and high-document-frequency
    terms are excluded. Person and organization names may remain when they are
    not ubiquitous, but two shared terms or a stronger anchor are required.
    """
    dynamic = set((context or {}).get("dynamic_generic_terms") or set())
    return {
        term for term in _raw_comparison_terms(titles)
        if term not in dynamic
    }


def event_specific_anchor_phrases(
    titles: Iterable[str],
    context: dict | None,
) -> set[str]:
    """
    Derive multiword event anchors without hardcoding story names.

    Anchors include rare multiword identities and compact lexical phrases
    containing at least two discriminative stems. Generic institutions such as
    "Supreme Court" and procedural phrases such as "asks court to block" are
    excluded from anchor evidence.
    """
    titles = [clean_title(t) for t in titles if clean_title(t)]
    generic = _comparison_generic_stems()
    dynamic_terms = set((context or {}).get("dynamic_generic_terms") or set())
    dynamic_identities = set(
        (context or {}).get("dynamic_generic_identities") or set()
    )
    excluded = generic | dynamic_terms
    anchors = set()

    # Rare multiword people, organizations, locations, or products.
    for title in titles:
        for term in extract_identity_terms(title):
            key = _identity_phrase_key(term)
            parts = [p for p in key.split() if p and p not in excluded]
            if (
                len(parts) >= 2
                and key not in dynamic_identities
                and len(set(parts)) >= 2
            ):
                anchors.add("identity:" + " ".join(parts))

    # Contiguous lexical windows. Generic connector/procedural words are
    # omitted from the normalized anchor, but at least two specific stems must
    # remain in the original short window.
    for title in titles:
        raw_tokens = normalize_title(title).split()
        stems = [stemish(token) for token in raw_tokens]
        for width in (2, 3, 4):
            for start in range(0, max(0, len(stems) - width + 1)):
                window = stems[start:start + width]
                specific = [
                    stem for stem in window
                    if len(stem) >= 3
                    and not stem.isdigit()
                    and stem not in excluded
                ]
                if len(set(specific)) >= 2:
                    anchors.add("lexical:" + " ".join(specific))

    return anchors


def _identity_match_stems(titles: Iterable[str]) -> set[str]:
    out = set()
    for title in titles:
        for term in extract_identity_terms(title):
            for token in normalize_title(term).split():
                stem = stemish(token)
                if len(stem) >= 3:
                    out.add(stem)
    return out


def strict_pipeline_reference_metrics(
    candidate_titles: list[str],
    reference_texts: list[str],
    vector_cache: dict[str, Any] | None,
    *,
    candidate_urls: Iterable[str] | None = None,
    reference_urls: Iterable[str] | None = None,
    context: dict | None = None,
) -> dict:
    """
    Strict event comparison used only for pipeline-status classification.

    The existing looser reference_match_metrics_cached() remains unchanged for
    post-split GDELT candidate deduplication.
    """
    base = reference_match_metrics_cached(
        candidate_titles,
        reference_texts,
        vector_cache,
    )

    candidate_titles = list(dict.fromkeys(
        clean_title(t) for t in candidate_titles if clean_title(t)
    ))
    reference_texts = list(dict.fromkeys(
        clean_title(t) for t in reference_texts if clean_title(t)
    ))

    candidate_url_set = {
        canonical_url(u) for u in (candidate_urls or [])
        if canonical_url(u)
    }
    reference_url_set = {
        canonical_url(u) for u in (reference_urls or [])
        if canonical_url(u)
    }
    exact_urls = sorted(candidate_url_set & reference_url_set)

    candidate_exact_titles = {
        normalize_title(t) for t in candidate_titles
        if normalize_title(t) and not generic_exact_title(t)
    }
    reference_exact_titles = {
        normalize_title(t) for t in reference_texts
        if normalize_title(t) and not generic_exact_title(t)
    }
    exact_titles = sorted(candidate_exact_titles & reference_exact_titles)

    candidate_terms = discriminative_match_terms(candidate_titles, context)
    reference_terms = discriminative_match_terms(reference_texts, context)
    shared_terms = sorted(candidate_terms & reference_terms)

    candidate_anchors = event_specific_anchor_phrases(
        candidate_titles, context
    )
    reference_anchors = event_specific_anchor_phrases(
        reference_texts, context
    )
    shared_anchors = sorted(candidate_anchors & reference_anchors)

    candidate_identity_stems = _identity_match_stems(candidate_titles)
    reference_identity_stems = _identity_match_stems(reference_texts)
    shared_identity_stems = (
        candidate_identity_stems
        & reference_identity_stems
        & set(shared_terms)
    )
    shared_non_identity_terms = sorted(
        set(shared_terms) - shared_identity_stems
    )

    candidate_action_groups = set().union(
        *(development_action_groups(t) for t in candidate_titles)
    )
    reference_action_groups = set().union(
        *(development_action_groups(t) for t in reference_texts)
    )
    shared_action_groups = sorted(
        candidate_action_groups & reference_action_groups
    )
    specific_action_groups = [
        group for group in shared_action_groups
        if group not in MATCH_GENERIC_ACTION_GROUPS
    ]

    exact_url_match = bool(exact_urls)
    exact_title_match = bool(exact_titles)
    shared_anchor_match = bool(shared_anchors)

    # Two discriminative shared terms are required when no exact/multiword
    # evidence exists. At least one should describe something beyond a shared
    # person/org name, unless a specific action/outcome group also agrees.
    discriminative_term_match = (
        len(shared_terms) >= 2
        and (
            bool(shared_non_identity_terms)
            or bool(specific_action_groups)
            or float(base.get("development_similarity", 0.0) or 0.0) >= 0.58
        )
    )

    stage_conflict = bool(base.get("stage_conflict"))
    development_compatible = (
        exact_url_match
        or exact_title_match
        or bool(shared_action_groups)
        or float(base.get("development_similarity", 0.0) or 0.0) >= 0.46
        or (
            shared_anchor_match
            and (
                bool(shared_non_identity_terms)
                or len(shared_terms) >= 3
            )
            and float(base.get("similarity", 0.0) or 0.0) >= 0.42
        )
    )

    specific_evidence = (
        exact_url_match
        or exact_title_match
        or shared_anchor_match
        or discriminative_term_match
    )

    event_gate = bool(
        specific_evidence
        and development_compatible
        and not stage_conflict
    )

    if exact_url_match:
        basis = "EXACT_URL"
    elif exact_title_match:
        basis = "EXACT_TITLE"
    elif shared_anchor_match:
        basis = "EVENT_ANCHOR"
    elif discriminative_term_match:
        basis = "DISCRIMINATIVE_TERMS"
    else:
        basis = "NONE"

    return {
        **base,
        "event_gate": event_gate,
        "exact_url_match": exact_url_match,
        "exact_title_match": exact_title_match,
        "exact_url_count": len(exact_urls),
        "exact_title_count": len(exact_titles),
        "shared_event_anchors": shared_anchors[:12],
        "shared_discriminative_terms": shared_terms[:16],
        "shared_non_identity_terms": shared_non_identity_terms[:12],
        "shared_action_groups": shared_action_groups,
        "specific_action_groups": specific_action_groups,
        "specific_evidence": specific_evidence,
        "match_basis": basis,
    }


def _reference_match_sort_key(row: dict) -> tuple:
    """Prefer exact and event-specific evidence before raw semantic score."""
    return (
        int(bool(row.get("exact_url_match"))),
        int(bool(row.get("exact_title_match"))),
        int(bool(row.get("event_gate"))),
        int(bool(row.get("shared_event_anchors"))),
        len(row.get("shared_discriminative_terms") or []),
        float(row.get("similarity", 0.0) or 0.0),
        float(row.get("development_similarity", 0.0) or 0.0),
    )



# ---------------------------------------------------------------------------
# Reusable runtime matching API
# ---------------------------------------------------------------------------
# These helpers let the optional final-ranking shadow match every purifier-
# approved event against the full bounded GDELT audit catalog. They do not run
# downloads, change audit CLI behavior, or modify production files.


def runtime_event_payload_from_cluster(
    cluster: dict,
    *,
    event_id: str | None = None,
    max_titles: int = 16,
) -> dict:
    core = list(cluster.get("articles") or [])
    related = list(cluster.get("related_articles") or [])

    articles = []
    seen_articles = set()
    for article in [*core, *related]:
        raw_url = article.get("url_normalized") or article.get("url") or ""
        key = canonical_url(raw_url) or normalize_title(article.get("title") or "")
        if not key or key in seen_articles:
            continue
        seen_articles.add(key)
        articles.append(article)

    raw_titles = [cluster.get("canonical_event") or ""]
    raw_titles.extend(article.get("title") or "" for article in core)
    raw_titles.extend(article.get("title") or "" for article in related)
    titles = list(dict.fromkeys(
        clean_title(value) for value in raw_titles if clean_title(value)
    ))[:max_titles]

    urls = {
        str(article.get("url_normalized") or article.get("url") or "").strip()
        for article in articles
        if str(article.get("url_normalized") or article.get("url") or "").strip()
    }
    candidate_counts = Counter(
        str(article.get("gdelt_candidate_id") or "").strip()
        for article in articles
        if str(article.get("gdelt_candidate_id") or "").strip()
    )

    return {
        "event_id": event_id,
        "title": titles[0] if titles else "",
        "titles": titles,
        "urls": urls,
        "candidate_counts": candidate_counts,
    }


def _runtime_catalog_titles(candidate: dict, limit: int = 12) -> list[str]:
    values = [candidate.get("canonical_title") or ""]
    values.extend(
        row.get("title") or ""
        for row in (candidate.get("representatives") or [])
        if isinstance(row, dict)
    )
    values.extend(
        row.get("representative_title") or ""
        for row in (candidate.get("writeup_families") or [])[:10]
        if isinstance(row, dict)
    )
    return list(dict.fromkeys(
        clean_title(value) for value in values if clean_title(value)
    ))[:limit]


def _runtime_catalog_urls(candidate: dict) -> set[str]:
    values = [candidate.get("canonical_url") or ""]
    values.extend(
        row.get("url") or ""
        for row in (candidate.get("representatives") or [])
        if isinstance(row, dict)
    )
    values.extend(
        row.get("representative_url") or ""
        for row in (candidate.get("writeup_families") or [])[:12]
        if isinstance(row, dict)
    )
    return {str(value).strip() for value in values if str(value or "").strip()}


def _runtime_catalog_attention(
    candidate: dict,
    fallback_percentile: float,
) -> dict:
    raw = dict(candidate.get("global_attention") or {})
    percentile = raw.get("discovery_percentile")
    if percentile is None:
        percentile = (
            (candidate.get("shadow_ranking_signals") or {}).get("global_signal")
        )
    if percentile is None:
        percentile = fallback_percentile

    def int_value(*keys):
        for key in keys:
            value = raw.get(key)
            if value is None:
                value = candidate.get(key)
            try:
                if value is not None:
                    return int(value)
            except Exception:
                pass
        return 0

    def float_value(*keys):
        for key in keys:
            value = raw.get(key)
            if value is None:
                value = candidate.get(key)
            try:
                if value is not None:
                    return float(value)
            except Exception:
                pass
        return 0.0

    return {
        "discovery_percentile": max(0.0, min(1.0, float(percentile or 0.0))),
        "discovery_score": round(float_value("discovery_score"), 3),
        "target_day_writeup_family_count": int_value(
            "target_day_writeup_family_count"
        ),
        "target_day_outlet_count": int_value(
            "target_day_outlet_count", "target_day_domain_count"
        ),
        "all_supporting_writeup_family_count": int_value(
            "all_supporting_writeup_family_count", "writeup_family_count"
        ),
        "all_supporting_outlet_count": int_value(
            "all_supporting_outlet_count", "unique_domain_count"
        ),
        "active_window_count": int_value("active_window_count"),
        "language_count": int_value("language_count"),
        "cross_language_url_count": int_value("cross_language_url_count"),
        "target_date": str(
            raw.get("target_date") or candidate.get("target_date") or ""
        ).strip(),
    }


def prepare_runtime_event_matcher(
    audit_payload: dict,
    event_payloads: list[dict],
    model: Any | None,
) -> dict:
    """
    Prepare one local semantic/lexical matcher for a bounded audit catalog and
    a candidate-event pool. No network requests or paid calls are made here.
    """
    raw_candidates = audit_payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return {
            "status": "UNAVAILABLE",
            "reason": "audit payload has no candidates",
            "catalog": [],
        }

    denominator = max(1, len(raw_candidates) - 1)
    catalog = []
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        titles = _runtime_catalog_titles(candidate)
        if not candidate_id or not titles:
            continue
        fallback_percentile = (
            (len(raw_candidates) - 1 - index) / denominator
            if len(raw_candidates) > 1 else 1.0
        )
        catalog.append({
            "candidate_id": candidate_id,
            "canonical_title": titles[0],
            "titles": titles,
            "urls": _runtime_catalog_urls(candidate),
            "pipeline_status": str(
                candidate.get("pipeline_status") or ""
            ).strip(),
            "preview_type": str(
                candidate.get("ingestion_preview_type")
                or candidate.get("preview_type")
                or ""
            ).strip(),
            "global_attention": _runtime_catalog_attention(
                candidate,
                fallback_percentile,
            ),
        })

    if not catalog:
        return {
            "status": "UNAVAILABLE",
            "reason": "audit payload has no usable candidates",
            "catalog": [],
        }

    references = []
    texts = []
    for candidate in catalog:
        references.append({
            "title": candidate["canonical_title"],
            "texts": candidate["titles"],
        })
        texts.extend(candidate["titles"])

    for event in event_payloads:
        event_titles = list(event.get("titles") or [])
        if not event_titles:
            continue
        references.append({
            "title": event.get("title") or event_titles[0],
            "texts": event_titles,
        })
        texts.extend(event_titles)

    texts.extend(
        development_text(value)
        for value in list(texts)
        if development_text(value)
    )
    vector_cache = build_text_vector_cache(texts, model)
    context = build_reference_match_context(references)

    return {
        "status": "OK",
        "catalog": catalog,
        "catalog_by_id": {row["candidate_id"]: row for row in catalog},
        "model": model,
        "vector_cache": vector_cache,
        "context": context,
        "semantic_mode": (
            MODEL_NAME if vector_cache is not None else "token_jaccard_fallback"
        ),
    }


def _extend_runtime_event_vectors(matcher: dict, titles: list[str]) -> None:
    model = matcher.get("model")
    vector_cache = matcher.get("vector_cache")
    if model is None or vector_cache is None:
        return

    requested = []
    for title in titles:
        clean = clean_title(title)
        if clean:
            requested.append(clean)
            dev = development_text(clean)
            if dev:
                requested.append(dev)
    missing = [
        value for value in dict.fromkeys(requested)
        if value not in vector_cache
    ]
    if not missing:
        return
    added = build_text_vector_cache(missing, model)
    if added:
        vector_cache.update(added)


def runtime_event_match_options(
    event_payload: dict,
    matcher: dict,
    *,
    threshold: float = 0.54,
    max_matches: int = 5,
    min_direct_articles: int = 2,
) -> list[dict]:
    """Return strict full-catalog matches for one candidate or approved event."""
    if matcher.get("status") != "OK":
        return []

    event_titles = list(event_payload.get("titles") or [])
    event_urls = set(event_payload.get("urls") or set())
    candidate_counts = Counter(event_payload.get("candidate_counts") or {})
    if not event_titles:
        return []

    _extend_runtime_event_vectors(matcher, event_titles)
    options = []

    for candidate in matcher.get("catalog") or []:
        candidate_id = candidate["candidate_id"]
        direct_count = int(candidate_counts.get(candidate_id, 0) or 0)
        direct_match = direct_count >= max(1, min_direct_articles)

        metrics = strict_pipeline_reference_metrics(
            candidate["titles"],
            event_titles,
            matcher.get("vector_cache"),
            candidate_urls=candidate.get("urls") or set(),
            reference_urls=event_urls,
            context=matcher.get("context"),
        )
        exact_match = bool(
            metrics.get("exact_url_match")
            or metrics.get("exact_title_match")
        )
        semantic_match = bool(
            metrics.get("event_gate")
            and float(metrics.get("similarity", 0.0) or 0.0) >= threshold
        )
        if not (direct_match or exact_match or semantic_match):
            continue

        meta = dict(candidate.get("global_attention") or {})
        global_signal = max(
            0.0,
            min(1.0, float(meta.get("discovery_percentile", 0.0) or 0.0)),
        )
        audit_key = tuple(_reference_match_sort_key(metrics))
        option = {
            "candidate_id": candidate_id,
            "canonical_title": candidate["canonical_title"],
            "pipeline_status": candidate.get("pipeline_status") or "",
            "preview_type": candidate.get("preview_type") or "",
            **meta,
            "retained_article_count": direct_count,
            "match_source": (
                "RETAINED_DISCOVERY_ARTICLES"
                if direct_match else "FULL_AUDIT_CATALOG"
            ),
            "match_basis": (
                "CANDIDATE_ID"
                if direct_match else str(metrics.get("match_basis") or "NONE")
            ),
            "match_threshold": threshold,
            "match_similarity": round(
                float(metrics.get("similarity", 0.0) or 0.0), 4
            ),
            "development_similarity": round(
                float(metrics.get("development_similarity", 0.0) or 0.0), 4
            ),
            "exact_url_match": bool(metrics.get("exact_url_match")),
            "exact_title_match": bool(metrics.get("exact_title_match")),
            "event_gate": bool(metrics.get("event_gate")),
            "stage_conflict": bool(metrics.get("stage_conflict")),
            "shared_event_anchors": list(
                metrics.get("shared_event_anchors") or []
            )[:8],
            "shared_discriminative_terms": list(
                metrics.get("shared_discriminative_terms") or []
            )[:10],
            "shared_action_groups": list(
                metrics.get("shared_action_groups") or []
            )[:8],
        }
        # When two catalog candidates have equally strong retained discovery
        # support inside the approved event, prefer the candidate representing
        # the broader GDELT event before using semantic-match detail as a
        # tie-breaker. This prevents a small duplicate candidate from masking
        # the dominant global candidate for the same discrete development.
        option["_priority"] = (
            int(direct_match),
            direct_count,
            global_signal,
            *audit_key,
        )
        options.append(option)

    options.sort(key=lambda row: tuple(row.get("_priority") or ()), reverse=True)
    return options[: max(1, max_matches)]


def assign_runtime_event_matches(
    event_payloads: list[dict],
    matcher: dict,
    *,
    threshold: float = 0.54,
    max_matches_per_event: int = 5,
) -> dict:
    """
    Greedily enforce one catalog candidate per event and one event per catalog
    candidate. Exact retained-candidate evidence always outranks semantic-only
    evidence.
    """
    pairs = []
    options_by_event = {}

    for event in event_payloads:
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            continue
        options = runtime_event_match_options(
            event,
            matcher,
            threshold=threshold,
            max_matches=max_matches_per_event,
        )
        options_by_event[event_id] = options
        for option in options:
            pairs.append({
                "event_id": event_id,
                "candidate_id": option.get("candidate_id"),
                "option": option,
                "priority": tuple(option.get("_priority") or ()),
            })

    pairs.sort(key=lambda row: row["priority"], reverse=True)
    assigned_event_ids = set()
    assigned_candidate_ids = set()
    assignments = {}

    for pair in pairs:
        event_id = pair["event_id"]
        candidate_id = pair["candidate_id"]
        if event_id in assigned_event_ids or candidate_id in assigned_candidate_ids:
            continue
        assigned_event_ids.add(event_id)
        assigned_candidate_ids.add(candidate_id)
        assignments[event_id] = pair["option"]

    return {
        "assignments": assignments,
        "options_by_event": options_by_event,
        "stats": {
            "event_count": len(event_payloads),
            "catalog_candidate_count": len(matcher.get("catalog") or []),
            "assigned_event_count": len(assignments),
            "unique_candidate_count_used": len(assigned_candidate_ids),
            "unmatched_event_count": max(0, len(event_payloads) - len(assignments)),
        },
    }

def family_representative_for_day(family: dict, target_day: str) -> dict:
    members = [
        n for n in (family.get("_members") or [])
        if is_english(n.get("language")) and clean_title(n.get("title") or "")
    ]
    if not members:
        return family.get("representative") or {}

    explicit_target = [
        n for n in members if target_day in set(n.get("explicit_days") or set())
    ]
    if explicit_target:
        pool = explicit_target
    else:
        no_explicit_target_observed = [
            n for n in members
            if not set(n.get("explicit_days") or set())
            and target_day in set(n.get("observed_days") or set())
        ]
        pool = no_explicit_target_observed or members

    chosen = max(
        pool,
        key=lambda n: (
            int(target_day in set(n.get("explicit_days") or set())),
            len(clean_title(n.get("title") or "")),
        ),
    )
    return {
        "title": chosen.get("title"),
        "url": chosen.get("url"),
        "canonical_url": chosen.get("canonical_url"),
        "domain": chosen.get("domain"),
        "language": chosen.get("language"),
        "image": chosen.get("image"),
    }


def family_signal_rows(
    family_ids: Iterable[str],
    families: dict[str, dict],
    target_day: str,
    model: Any | None,
) -> dict[str, dict]:
    ids = [f for f in sorted(set(family_ids)) if f in families]
    rows = {}
    titles = []
    dev_texts = []

    for family_id in ids:
        family = families[family_id]
        rep = family_representative_for_day(family, target_day)
        title = clean_title(rep.get("title") or family.get("representative", {}).get("title") or "")
        if not title:
            continue
        row = {
            "family_id": family_id,
            "title": title,
            "representative": rep,
            "identity_terms": extract_identity_terms(title),
            "development_tokens": development_tokens(title),
            "development_stage": development_stage(title),
            "day_role": family_day_role(family, target_day),
            "outlet_count": int(family.get("outlet_count", 0) or 0),
            "active_window_count": len(family.get("slots") or set()),
        }
        rows[family_id] = row
        titles.append(title)
        dev_texts.append(development_text(title))

    if model is not None and np is not None and rows:
        try:
            encoded = np.asarray(
                model.encode(
                    titles + dev_texts,
                    normalize_embeddings=True,
                    batch_size=64,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            n = len(titles)
            for idx, family_id in enumerate(rows):
                rows[family_id]["title_vector"] = encoded[idx]
                rows[family_id]["development_vector"] = encoded[n + idx]
        except Exception:
            pass

    return rows


def identity_overlap(a: dict, b: dict) -> set[str]:
    aa = set(a.get("identity_terms") or set())
    bb = set(b.get("identity_terms") or set())
    overlap = aa & bb
    # Prefer multiword identities, but a distinctive shared proper token is useful.
    return {
        term for term in overlap
        if " " in term or len(term) >= 4
    }


def pair_development_metrics(a: dict, b: dict) -> dict:
    avec = a.get("title_vector")
    bvec = b.get("title_vector")
    davec = a.get("development_vector")
    dbvec = b.get("development_vector")

    if avec is not None and bvec is not None:
        full_sim = float(avec @ bvec)
    else:
        full_sim = fallback_similarity(a.get("title", ""), b.get("title", ""))

    if davec is not None and dbvec is not None:
        dev_sim = float(davec @ dbvec)
    else:
        dev_sim = fallback_similarity(
            development_text(a.get("title", "")),
            development_text(b.get("title", "")),
        )

    at = set(a.get("development_tokens") or set())
    bt = set(b.get("development_tokens") or set())
    union = at | bt
    dev_jaccard = len(at & bt) / len(union) if union else 0.0
    ids = identity_overlap(a, b)

    stage_transition = {
        a.get("development_stage"),
        b.get("development_stage"),
    } == {"PENDING", "FINAL"}

    return {
        "full_similarity": round(full_sim, 4),
        "development_similarity": round(dev_sim, 4),
        "development_jaccard": round(dev_jaccard, 4),
        "shared_development_tokens": sorted(at & bt),
        "shared_identity_terms": sorted(ids),
        "stage_transition": stage_transition,
    }


def same_target_development(
    a: dict,
    b: dict,
    args: argparse.Namespace,
) -> tuple[bool, dict]:
    metrics = pair_development_metrics(a, b)
    full_sim = metrics["full_similarity"]
    dev_sim = metrics["development_similarity"]
    dev_jacc = metrics["development_jaccard"]
    shared_dev = len(metrics["shared_development_tokens"])
    shared_identity = bool(metrics["shared_identity_terms"])

    if metrics["stage_transition"]:
        return False, metrics

    vectors_available = (
        a.get("title_vector") is not None
        and b.get("title_vector") is not None
    )
    if not vectors_available:
        passes = (
            shared_identity
            and full_sim >= 0.20
            and (dev_jacc >= 0.12 or shared_dev >= 1)
        )
        return passes, metrics

    passes = (
        shared_identity
        and full_sim >= args.target_cluster_sim
        and (dev_sim >= 0.50 or dev_jacc >= 0.14 or shared_dev >= 2)
    ) or (
        full_sim >= 0.84
        and dev_sim >= 0.58
        and (dev_jacc >= 0.18 or shared_dev >= 2)
    )
    return passes, metrics


def same_adjacent_development(
    anchor: dict,
    other: dict,
    args: argparse.Namespace,
) -> tuple[bool, dict]:
    metrics = pair_development_metrics(anchor, other)
    if metrics["stage_transition"]:
        return False, metrics

    full_sim = metrics["full_similarity"]
    dev_sim = metrics["development_similarity"]
    dev_jacc = metrics["development_jaccard"]
    shared_dev = len(metrics["shared_development_tokens"])
    shared_identity = bool(metrics["shared_identity_terms"])

    vectors_available = (
        anchor.get("title_vector") is not None
        and other.get("title_vector") is not None
    )
    if not vectors_available:
        passes = (
            shared_identity
            and full_sim >= 0.25
            and dev_jacc >= 0.15
            and shared_dev >= 2
        )
        return passes, metrics

    passes = (
        shared_identity
        and full_sim >= args.adjacent_support_sim
        and dev_sim >= 0.62
        and (dev_jacc >= 0.20 or shared_dev >= 2)
    ) or (
        full_sim >= 0.88
        and dev_sim >= 0.68
        and (dev_jacc >= 0.24 or shared_dev >= 3)
    )
    return passes, metrics


def choose_cluster_medoid(
    family_ids: list[str],
    signals: dict[str, dict],
) -> str:
    if len(family_ids) == 1:
        return family_ids[0]

    best = family_ids[0]
    best_score = -1e9
    for family_id in family_ids:
        row = signals[family_id]
        sims = []
        for other_id in family_ids:
            if other_id == family_id:
                continue
            metrics = pair_development_metrics(row, signals[other_id])
            sims.append(metrics["full_similarity"])
        score = (
            (mean(sims) if sims else 0.0)
            + 0.015 * math.log1p(row.get("outlet_count", 0))
            + 0.010 * math.log1p(row.get("active_window_count", 0))
        )
        if score > best_score:
            best_score = score
            best = family_id
    return best


def split_group_into_target_developments(
    group: dict,
    families: dict[str, dict],
    target_day: str,
    model: Any | None,
    args: argparse.Namespace,
    signal_cache: dict[str, dict] | None = None,
) -> tuple[list[dict], dict]:
    """
    Split an overmerged seed group around developments actually observed on the
    target date. Adjacent-day write-ups can support only a clearly matching
    target-day development; they cannot create the candidate themselves.
    """
    family_ids = set(group.get("family_ids") or set())
    if signal_cache is not None:
        signals = {
            family_id: signal_cache[family_id]
            for family_id in family_ids
            if family_id in signal_cache
        }
    else:
        signals = family_signal_rows(family_ids, families, target_day, model)
    target_ids = [
        family_id for family_id, row in signals.items()
        if row.get("day_role") in {"TARGET_EXPLICIT", "TARGET_OBSERVED"}
    ]
    adjacent_ids = [family_id for family_id in signals if family_id not in target_ids]

    target_ids.sort(
        key=lambda f: (
            signals[f].get("outlet_count", 0),
            signals[f].get("active_window_count", 0),
        ),
        reverse=True,
    )

    clusters: list[dict] = []
    pair_checks = 0

    for family_id in target_ids:
        best_idx = None
        best_score = -1.0
        for idx, cluster in enumerate(clusters):
            medoid_id = cluster["medoid_id"]
            ok, metrics = same_target_development(
                signals[medoid_id], signals[family_id], args
            )
            pair_checks += 1
            if not ok:
                continue
            score = (
                metrics["full_similarity"] * 0.65
                + metrics["development_similarity"] * 0.35
            )
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None:
            clusters.append({
                "target_family_ids": [family_id],
                "adjacent_family_ids": [],
                "medoid_id": family_id,
                "adjacent_assignments": [],
            })
        else:
            clusters[best_idx]["target_family_ids"].append(family_id)

    # Recompute each medoid once after assignment. Avoids cubic work in large
    # global stories while preserving the same local clustering decision.
    for cluster in clusters:
        cluster["medoid_id"] = choose_cluster_medoid(
            cluster["target_family_ids"], signals
        )

    # Very small target fragments are allowed to attach to a stronger cluster
    # only if they pass the same exact-development gate against its medoid.
    merged_clusters: list[dict] = []
    for cluster in sorted(
        clusters,
        key=lambda c: sum(
            signals[f].get("outlet_count", 0)
            for f in c["target_family_ids"]
        ),
        reverse=True,
    ):
        if not merged_clusters:
            merged_clusters.append(cluster)
            continue
        best_idx = None
        best_score = -1.0
        for idx, existing in enumerate(merged_clusters):
            ok, metrics = same_target_development(
                signals[existing["medoid_id"]],
                signals[cluster["medoid_id"]],
                args,
            )
            pair_checks += 1
            if not ok:
                continue
            score = metrics["full_similarity"] + 0.25 * metrics["development_similarity"]
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            merged_clusters.append(cluster)
        else:
            existing = merged_clusters[best_idx]
            existing["target_family_ids"].extend(cluster["target_family_ids"])
            existing["medoid_id"] = choose_cluster_medoid(
                existing["target_family_ids"], signals
            )

    clusters = merged_clusters

    for family_id in adjacent_ids:
        scored = []
        for idx, cluster in enumerate(clusters):
            medoid = signals[cluster["medoid_id"]]
            ok, metrics = same_adjacent_development(
                medoid, signals[family_id], args
            )
            pair_checks += 1
            score = (
                metrics["full_similarity"] * 0.65
                + metrics["development_similarity"] * 0.35
            )
            scored.append((score, idx, ok, metrics))

        scored.sort(reverse=True, key=lambda row: row[0])
        if not scored or not scored[0][2]:
            continue
        margin = scored[0][0] - scored[1][0] if len(scored) > 1 else 1.0
        if len(scored) > 1 and margin < 0.05:
            continue

        _, idx, _, metrics = scored[0]
        clusters[idx]["adjacent_family_ids"].append(family_id)
        clusters[idx]["adjacent_assignments"].append({
            "family_id": family_id,
            "title": signals[family_id]["title"],
            "full_similarity": metrics["full_similarity"],
            "development_similarity": metrics["development_similarity"],
            "development_jaccard": metrics["development_jaccard"],
        })

    output_groups = []
    for cluster in clusters:
        ids = set(cluster["target_family_ids"]) | set(cluster["adjacent_family_ids"])
        if not ids:
            continue
        output_groups.append({
            "prototype": group.get("prototype"),
            "members": group.get("members") or [],
            "family_ids": ids,
            "target_family_ids": set(cluster["target_family_ids"]),
            "adjacent_family_ids": set(cluster["adjacent_family_ids"]),
            "target_medoid_id": cluster["medoid_id"],
            "split_diagnostics": {
                "target_anchor_family_count": len(cluster["target_family_ids"]),
                "adjacent_support_family_count": len(cluster["adjacent_family_ids"]),
                "adjacent_assignments": cluster["adjacent_assignments"],
            },
            "_signals": {family_id: signals[family_id] for family_id in ids},
        })

    return output_groups, {
        "input_family_count": len(family_ids),
        "target_anchor_family_count": len(target_ids),
        "adjacent_family_count": len(adjacent_ids),
        "output_development_count": len(output_groups),
        "pair_checks": pair_checks,
        "discarded_adjacent_family_count": max(
            0,
            len(adjacent_ids) - sum(
                len(c.get("adjacent_family_ids") or set())
                for c in output_groups
            ),
        ),
    }


def merge_duplicate_seed_candidates(
    raw_candidates: list[dict],
    model: Any | None,
    args: argparse.Namespace,
) -> list[dict]:
    raw_candidates = sorted(
        raw_candidates,
        key=lambda c: c["preliminary_score"],
        reverse=True,
    )[: max(1, args.max_raw_candidates)]

    texts = [candidate_text(c) for c in raw_candidates]
    vectors = None
    if model is not None and texts:
        try:
            vectors = np.asarray(
                model.encode(texts, normalize_embeddings=True),
                dtype=np.float32,
            )
        except Exception:
            vectors = None

    canonical: list[dict] = []

    for idx, cand in enumerate(raw_candidates):
        best_cluster = None
        best_strength = -1.0

        for cluster in canonical:
            prototype = raw_candidates[cluster["prototype_index"]]
            overlap = len(cand["family_ids"] & cluster["family_ids"])
            min_size = max(1, min(len(cand["family_ids"]), len(cluster["family_ids"])))
            overlap_coeff = overlap / min_size
            union_size = len(cand["family_ids"] | cluster["family_ids"])
            family_jaccard = overlap / union_size if union_size else 0.0

            if vectors is not None:
                sem = float(vectors[idx] @ vectors[cluster["prototype_index"]])
            else:
                sem = fallback_similarity(candidate_text(cand), candidate_text(prototype))

            anchor_overlap = len(
                cand.get("anchor_tokens", set())
                & prototype.get("anchor_tokens", set())
            )
            token_overlap = fallback_similarity(candidate_text(cand), candidate_text(prototype))

            merge_ok = (
                overlap_coeff >= 0.45
                or family_jaccard >= 0.28
                or (
                    sem >= args.event_merge_sim
                    and anchor_overlap >= 2
                    and token_overlap >= 0.12
                )
                or (sem >= 0.86 and token_overlap >= 0.18)
            )

            if not merge_ok:
                continue

            strength = max(
                overlap_coeff,
                family_jaccard,
                sem * 0.8 + min(anchor_overlap, 4) * 0.04,
            )
            if strength > best_strength:
                best_strength = strength
                best_cluster = cluster

        if best_cluster is None:
            canonical.append({
                "prototype_index": idx,
                "member_indices": [idx],
                "family_ids": set(cand["family_ids"]),
            })
        else:
            best_cluster["member_indices"].append(idx)
            best_cluster["family_ids"].update(cand["family_ids"])

    return [
        {
            "prototype": raw_candidates[c["prototype_index"]],
            "members": [raw_candidates[i] for i in c["member_indices"]],
            "family_ids": c["family_ids"],
        }
        for c in canonical
    ]


def discovery_score(metrics: dict) -> tuple[float, dict]:
    target_families = int(metrics["target_day_writeup_family_count"])
    target_domains = int(metrics["target_day_domain_count"])
    adjacent_families = int(metrics["adjacent_support_family_count"])
    domains = int(metrics["unique_domain_count"])
    windows = int(metrics["active_window_count"])
    titles = int(metrics["headline_variant_count"])
    languages = int(metrics["language_count"])
    cross = int(metrics["cross_language_url_count"])
    largest_share = float(metrics["largest_family_outlet_share"])
    cohesion = float(metrics.get("event_title_cohesion_p20", 0.0) or 0.0)
    dev_cohesion = float(metrics.get("development_cohesion_p20", 0.0) or 0.0)

    components = {
        "target_writeup_breadth": round(
            12.0 * math.log2(1 + target_families), 3
        ),
        "target_outlet_breadth": round(
            4.0 * math.log2(1 + target_domains), 3
        ),
        "adjacent_same_development_support": round(
            3.0 * math.log2(1 + adjacent_families), 3
        ),
        "all_outlet_breadth": round(1.5 * math.log2(1 + domains), 3),
        "time_recurrence": round(2.5 * math.log2(1 + windows), 3),
        "headline_diversity": round(1.5 * math.log2(1 + titles), 3),
        "language_breadth": round(1.5 * math.log2(1 + languages), 3),
        "cross_language": round(0.75 * math.log2(1 + cross), 3),
        "cohesion_bonus": round(
            5.0 * max(0.0, cohesion - 0.50)
            + 3.0 * max(0.0, dev_cohesion - 0.45),
            3,
        ),
    }

    penalty = 0.0
    if target_families == 1:
        penalty += 9.0
    elif target_families == 2:
        penalty += 3.0
    if largest_share >= 0.80:
        penalty += (largest_share - 0.80) * 15.0

    components["syndication_penalty"] = round(penalty, 3)
    score = (
        sum(v for k, v in components.items() if k != "syndication_penalty")
        - penalty
    )
    return round(score, 4), components


def canonical_event_from_group(
    group: dict,
    families: dict[str, dict],
    family_title_vectors: dict[str, Any] | None,
    target_day: str,
    args: argparse.Namespace,
) -> dict | None:
    family_ids = set(group.get("family_ids") or set())
    target_ids = set(group.get("target_family_ids") or set())
    adjacent_ids = set(group.get("adjacent_family_ids") or set())
    signals = group.get("_signals") or family_signal_rows(
        family_ids, families, target_day, None
    )

    target_ids = {f for f in target_ids if f in families and f in signals}
    adjacent_ids = {f for f in adjacent_ids if f in families and f in signals}
    family_ids = target_ids | adjacent_ids
    if not target_ids or not family_ids:
        return None

    medoid_id = group.get("target_medoid_id")
    if medoid_id not in target_ids:
        medoid_id = choose_cluster_medoid(sorted(target_ids), signals)
    medoid = signals[medoid_id]
    representative = medoid.get("representative") or {}
    if not representative.get("title"):
        return None

    # Remove any adjacent family that became weak once the target-date medoid
    # was fixed. This second check prevents broad merged neighborhoods from
    # leaking into the candidate after canonical grouping.
    checked_adjacent = set()
    adjacent_diagnostics = []
    for family_id in sorted(adjacent_ids):
        ok, metrics = same_adjacent_development(
            medoid, signals[family_id], args
        )
        if ok:
            checked_adjacent.add(family_id)
        adjacent_diagnostics.append({
            "family_id": family_id,
            "title": signals[family_id].get("title"),
            "kept": bool(ok),
            **metrics,
        })
    adjacent_ids = checked_adjacent
    family_ids = target_ids | adjacent_ids

    def row_vector(family_id: str, key: str):
        value = signals.get(family_id, {}).get(key)
        if value is not None:
            return value
        if key == "title_vector" and family_title_vectors:
            return family_title_vectors.get(family_id)
        return None

    medoid_title_vec = row_vector(medoid_id, "title_vector")
    medoid_dev_vec = row_vector(medoid_id, "development_vector")

    target_title_sims = []
    all_title_sims = []
    all_dev_sims = []
    for family_id in sorted(family_ids):
        title_vec = row_vector(family_id, "title_vector")
        dev_vec = row_vector(family_id, "development_vector")
        if title_vec is not None and medoid_title_vec is not None:
            sim = float(title_vec @ medoid_title_vec)
        else:
            sim = fallback_similarity(
                signals[family_id].get("title", ""), medoid.get("title", "")
            )
        all_title_sims.append(sim)
        if family_id in target_ids:
            target_title_sims.append(sim)

        if dev_vec is not None and medoid_dev_vec is not None:
            dev_sim = float(dev_vec @ medoid_dev_vec)
        else:
            dev_sim = fallback_similarity(
                development_text(signals[family_id].get("title", "")),
                development_text(medoid.get("title", "")),
            )
        all_dev_sims.append(dev_sim)

    target_cohesion = (
        float(np.quantile(target_title_sims, 0.20))
        if np is not None and target_title_sims
        else (min(target_title_sims) if target_title_sims else 0.0)
    )
    event_cohesion = (
        float(np.quantile(all_title_sims, 0.20))
        if np is not None and all_title_sims
        else (min(all_title_sims) if all_title_sims else 0.0)
    )
    development_cohesion = (
        float(np.quantile(all_dev_sims, 0.20))
        if np is not None and all_dev_sims
        else (min(all_dev_sims) if all_dev_sims else 0.0)
    )

    # Multiple target-day write-ups must be mutually coherent. MiniLM is the
    # normal path. A conservative lexical fallback keeps the audit usable if
    # the optional local model is unavailable.
    cohesion_method = (
        "local_semantic_embeddings"
        if medoid_title_vec is not None
        else "lexical_jaccard_fallback"
    )
    cohesion_floor = (
        args.min_event_cohesion
        if cohesion_method == "local_semantic_embeddings"
        else 0.18
    )
    if len(target_ids) >= 2 and target_cohesion < cohesion_floor:
        return None
    if len(family_ids) >= 3 and event_cohesion < cohesion_floor:
        return None

    family_rows = [families[f] for f in family_ids]
    urls = set()
    domains = set()
    slots = set()
    languages = set()
    normalized_titles = set()
    cross_language_urls = 0
    family_details = []

    target_domains = set()
    target_urls = set()
    target_explicit_count = 0

    for family_id in sorted(family_ids):
        fam = families[family_id]
        signal = signals[family_id]
        urls.update(fam["urls"])
        domains.update(fam["domains"])
        slots.update(fam["slots"])
        languages.update(fam["languages"])
        cross_language_urls += fam["cross_language_url_count"]

        title = signal.get("title") or fam.get("representative", {}).get("title") or ""
        norm = normalize_title(title)
        if norm:
            normalized_titles.add(norm)

        role = signal.get("day_role") or family_day_role(fam, target_day)
        if family_id in target_ids:
            target_domains.update(fam["domains"])
            target_urls.update(fam["urls"])
            if role == "TARGET_EXPLICIT":
                target_explicit_count += 1

        family_details.append({
            "family_id": fam["family_id"],
            "representative_title": title,
            "representative_url": signal.get("representative", {}).get("url")
                or fam.get("representative", {}).get("url"),
            "day_role": role,
            "outlet_count": fam["outlet_count"],
            "url_count": fam["url_count"],
            "unique_title_count": fam["unique_title_count"],
            "language_count": len(fam["languages"]),
            "active_window_count": len(fam["slots"]),
        })

    slot_days = {s[:8] for s in slots if len(s) >= 8}
    largest_family_outlets = max(
        (f["outlet_count"] for f in family_rows), default=0
    )
    largest_share = largest_family_outlets / max(1, len(domains))

    # Representatives are target-date first. Adjacent-day links are evidence,
    # not the public-facing identity of the candidate.
    ordered_ids = sorted(
        target_ids,
        key=lambda f: (
            pair_development_metrics(medoid, signals[f])["full_similarity"],
            families[f]["outlet_count"],
        ),
        reverse=True,
    )
    ordered_ids.extend(
        sorted(
            adjacent_ids,
            key=lambda f: (
                pair_development_metrics(medoid, signals[f])["full_similarity"],
                families[f]["outlet_count"],
            ),
            reverse=True,
        )
    )

    representatives = []
    used_domains = set()
    for family_id in ordered_ids:
        rep = signals[family_id].get("representative") or {}
        domain = rep.get("domain")
        if not domain or domain in used_domains or not rep.get("title"):
            continue
        used_domains.add(domain)
        representatives.append({
            "title": rep.get("title"),
            "url": rep.get("url"),
            "domain": domain,
            "language": rep.get("language"),
            "image": rep.get("image"),
            "day_role": signals[family_id].get("day_role"),
            "writeup_family_outlets": families[family_id]["outlet_count"],
        })
        if len(representatives) >= 6:
            break

    metrics = {
        "target_day_writeup_family_count": len(target_ids),
        "target_day_domain_count": len(target_domains),
        "target_day_url_count": len(target_urls),
        "target_day_explicit_family_count": target_explicit_count,
        "adjacent_support_family_count": len(adjacent_ids),
        "writeup_family_count": len(family_rows),
        "unique_domain_count": len(domains),
        "unique_url_count": len(urls),
        "headline_variant_count": len(normalized_titles),
        "active_window_count": len(slots),
        "active_day_count": len(slot_days),
        "language_count": len(languages),
        "cross_language_url_count": cross_language_urls,
        "largest_family_outlet_count": largest_family_outlets,
        "largest_family_outlet_share": round(largest_share, 4),
        "target_title_cohesion_p20": round(target_cohesion, 4),
        "event_title_cohesion_p20": round(event_cohesion, 4),
        "development_cohesion_p20": round(development_cohesion, 4),
        "cohesion_method": cohesion_method,
        "cohesion_gate_floor": round(cohesion_floor, 4),
    }
    score, components = discovery_score(metrics)

    return {
        "candidate_id": stable_id("event", family_ids),
        "target_date": target_day,
        "canonical_title": representative.get("title"),
        "canonical_url": representative.get("url"),
        "canonical_domain": representative.get("domain"),
        "discovery_score": score,
        "score_components": components,
        **metrics,
        "representatives": representatives,
        "writeup_families": sorted(
            family_details,
            key=lambda f: (
                f["day_role"].startswith("TARGET"),
                f["outlet_count"],
                f["active_window_count"],
            ),
            reverse=True,
        ),
        "development_split": {
            **(group.get("split_diagnostics") or {}),
            "adjacent_revalidation": adjacent_diagnostics,
        },
        "_family_ids": family_ids,
        "_target_family_ids": target_ids,
        "_titles": [signals[f].get("title", "") for f in ordered_ids],
        "_urls": set(urls),
    }


def clusters_from_json(data: Any) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("clusters") or []
    return []


def safe_load_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None



def iter_article_rows(data: Any) -> Iterable[dict]:
    """Yield article-like dictionaries from common Nominal News JSON shapes."""
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("url") or item.get("url_normalized"):
                yield item
            if isinstance(item.get("articles"), list):
                yield from iter_article_rows(item.get("articles"))
            if isinstance(item.get("coverage_articles"), list):
                yield from iter_article_rows(item.get("coverage_articles"))
    elif isinstance(data, dict):
        for key in ("articles", "coverage_articles", "clusters"):
            value = data.get(key)
            if isinstance(value, list):
                yield from iter_article_rows(value)


def load_normalized_corpus_urls(path: Path) -> tuple[set[str], dict]:
    """Load canonical URLs already present in the normalized local corpus."""
    data = safe_load_json(path)
    urls = set()
    article_count = 0
    if data is not None:
        for article in iter_article_rows(data):
            article_count += 1
            raw = article.get("url_normalized") or article.get("url") or ""
            normalized = canonical_url(raw)
            if normalized:
                urls.add(normalized)
    return urls, {
        "normalized_file": str(path),
        "normalized_file_found": path.exists(),
        "normalized_article_count": article_count,
        "normalized_unique_url_count": len(urls),
    }


def load_reference_sets(
    summary_path: Path,
    final_path: Path,
    upstream_path: Path,
) -> tuple[list[dict], list[dict], dict]:
    summaries = safe_load_json(summary_path)
    final_data = safe_load_json(final_path)
    upstream_data = safe_load_json(upstream_path)

    final_clusters = clusters_from_json(final_data)
    summary_rows = summaries if isinstance(summaries, list) else []

    published = []
    total_published = max(len(summary_rows), len(final_clusters))
    for i in range(total_published):
        summary = summary_rows[i] if i < len(summary_rows) else {}
        cluster = final_clusters[i] if i < len(final_clusters) else {}
        title = clean_title(
            str(summary.get("topic_title") or cluster.get("canonical_event") or cluster.get("topic") or "")
        )
        texts = []
        for value in (
            title,
            summary.get("summary"),
            cluster.get("canonical_event"),
        ):
            value = clean_title(str(value or ""))
            if value:
                texts.append(value)
        for article in cluster.get("articles", [])[:12]:
            article_title = clean_title(str(article.get("title") or ""))
            if article_title:
                texts.append(article_title)
        if title or texts:
            cluster_articles = cluster.get("articles", []) if isinstance(cluster, dict) else []
            cluster_domains = {
                outlet_domain(a.get("url_normalized") or a.get("url") or "")
                for a in cluster_articles
                if outlet_domain(a.get("url_normalized") or a.get("url") or "")
            }
            related_articles = (
                cluster.get("related_articles", [])
                if isinstance(cluster, dict)
                else []
            )
            exact_articles = [*cluster_articles, *related_articles]
            reference_urls = {
                canonical_url(a.get("url_normalized") or a.get("url") or "")
                for a in exact_articles
                if canonical_url(a.get("url_normalized") or a.get("url") or "")
            }
            reference_exact_titles = {
                normalize_title(a.get("title") or "")
                for a in exact_articles
                if normalize_title(a.get("title") or "")
            }

            published.append({
                "reference_id": f"published_{i + 1}",
                "rank": i + 1,
                "title": title or (texts[0] if texts else f"Published {i + 1}"),
                "texts": list(dict.fromkeys(texts)),
                "urls": reference_urls,
                "exact_titles": reference_exact_titles,
                "attention_score": float(cluster.get("attention_score", 0.0) or 0.0),
                "attention_article_count": int(cluster.get("attention_article_count", len(cluster_articles)) or len(cluster_articles)),
                "attention_domain_count": int(cluster.get("attention_domain_count", len(cluster_domains)) or len(cluster_domains)),
                "core_article_count": len(cluster_articles),
                "core_domain_count": len(cluster_domains),
            })

    upstream = []
    for i, cluster in enumerate(clusters_from_json(upstream_data)):
        texts = []
        for article in cluster.get("articles", [])[:16]:
            title = clean_title(str(article.get("title") or ""))
            if title:
                texts.append(title)
        if not texts:
            continue
        upstream_articles = cluster.get("articles", [])
        upstream_domains = {
            outlet_domain(a.get("url_normalized") or a.get("url") or "")
            for a in upstream_articles
            if outlet_domain(a.get("url_normalized") or a.get("url") or "")
        }
        upstream_urls = {
            canonical_url(a.get("url_normalized") or a.get("url") or "")
            for a in upstream_articles
            if canonical_url(a.get("url_normalized") or a.get("url") or "")
        }
        upstream_exact_titles = {
            normalize_title(a.get("title") or "")
            for a in upstream_articles
            if normalize_title(a.get("title") or "")
        }

        upstream.append({
            "reference_id": f"upstream_{i + 1}",
            "rank": i + 1,
            "title": clean_title(str(cluster.get("canonical_event") or cluster.get("topic") or texts[0])),
            "texts": list(dict.fromkeys(texts)),
            "urls": upstream_urls,
            "exact_titles": upstream_exact_titles,
            "article_count": len(upstream_articles),
            "domain_count": len(upstream_domains),
        })

    return published, upstream, {
        "summary_file": str(summary_path),
        "summary_file_found": summary_path.exists(),
        "final_file": str(final_path),
        "final_file_found": final_path.exists(),
        "upstream_file": str(upstream_path),
        "upstream_file_found": upstream_path.exists(),
        "published_reference_count": len(published),
        "upstream_reference_count": len(upstream),
    }


def build_text_vector_cache(
    texts: Iterable[str],
    model: Any | None,
) -> dict[str, Any] | None:
    if model is None or np is None:
        return None
    unique = list(dict.fromkeys(clean_title(t) for t in texts if clean_title(t)))
    if not unique:
        return None
    try:
        vectors = np.asarray(
            model.encode(
                unique,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        return {text: vectors[i] for i, text in enumerate(unique)}
    except Exception:
        return None


def reference_match_metrics_cached(
    candidate_titles: list[str],
    reference_texts: list[str],
    vector_cache: dict[str, Any] | None,
) -> dict:
    candidate_titles = list(dict.fromkeys(
        clean_title(t) for t in candidate_titles if clean_title(t)
    ))
    reference_texts = list(dict.fromkeys(
        clean_title(t) for t in reference_texts if clean_title(t)
    ))
    if not candidate_titles or not reference_texts:
        return {
            "similarity": 0.0,
            "development_similarity": 0.0,
            "shared_identity_terms": [],
            "shared_development_tokens": [],
            "event_gate": False,
            "stage_conflict": False,
        }

    max_sim = 0.0
    support = 0.0
    best_a = candidate_titles[0]
    best_b = reference_texts[0]

    if vector_cache:
        a_rows = [(t, vector_cache.get(t)) for t in candidate_titles]
        b_rows = [(t, vector_cache.get(t)) for t in reference_texts]
        a_rows = [(t, v) for t, v in a_rows if v is not None]
        b_rows = [(t, v) for t, v in b_rows if v is not None]
        if a_rows and b_rows:
            a = np.asarray([v for _, v in a_rows], dtype=np.float32)
            b = np.asarray([v for _, v in b_rows], dtype=np.float32)
            sims = a @ b.T
            flat = int(np.argmax(sims))
            ai, bi = np.unravel_index(flat, sims.shape)
            max_sim = float(sims[ai, bi])
            best_a = a_rows[ai][0]
            best_b = b_rows[bi][0]
            top_per_candidate = np.max(sims, axis=1)
            support = float(np.mean(
                np.sort(top_per_candidate)[-min(3, len(top_per_candidate)):]
            ))
    else:
        for a_title in candidate_titles:
            for b_title in reference_texts:
                sim = fallback_similarity(a_title, b_title)
                if sim > max_sim:
                    max_sim = sim
                    best_a = a_title
                    best_b = b_title
        support = max_sim

    combined_similarity = max(max_sim, 0.70 * max_sim + 0.30 * support)

    candidate_dev = list(dict.fromkeys(
        development_text(t) for t in candidate_titles if development_text(t)
    ))
    reference_dev = list(dict.fromkeys(
        development_text(t) for t in reference_texts if development_text(t)
    ))
    dev_similarity = 0.0
    if vector_cache:
        a_rows = [vector_cache.get(t) for t in candidate_dev]
        b_rows = [vector_cache.get(t) for t in reference_dev]
        a_rows = [v for v in a_rows if v is not None]
        b_rows = [v for v in b_rows if v is not None]
        if a_rows and b_rows:
            dev_similarity = float(np.max(
                np.asarray(a_rows, dtype=np.float32)
                @ np.asarray(b_rows, dtype=np.float32).T
            ))
    if dev_similarity == 0.0:
        dev_similarity = max(
            (fallback_similarity(a, b) for a in candidate_dev for b in reference_dev),
            default=0.0,
        )

    candidate_identity = set().union(
        *(extract_identity_terms(t) for t in candidate_titles)
    )
    reference_identity = set().union(
        *(extract_identity_terms(t) for t in reference_texts)
    )
    shared_identity = {
        term for term in (candidate_identity & reference_identity)
        if " " in term or len(term) >= 4
    }

    candidate_action = set().union(
        *(development_tokens(t) for t in candidate_titles)
    )
    reference_action = set().union(
        *(development_tokens(t) for t in reference_texts)
    )
    shared_action = candidate_action & reference_action

    full_token_overlap = title_tokens(best_a) & title_tokens(best_b)
    strong_identity = (
        any(" " in term for term in shared_identity)
        or len(shared_identity) >= 2
        or (
            len(shared_identity) == 1
            and len(next(iter(shared_identity))) >= 6
            and combined_similarity >= 0.65
        )
    )

    if vector_cache:
        identity_pass = strong_identity or (
            combined_similarity >= 0.74 and len(full_token_overlap) >= 3
        )
        development_pass = (
            dev_similarity >= 0.50
            or len(shared_action) >= 2
        )
    else:
        identity_pass = strong_identity or (
            combined_similarity >= 0.35 and len(full_token_overlap) >= 3
        )
        development_pass = (
            dev_similarity >= 0.12
            or len(shared_action) >= 1
        )

    candidate_stage = development_stage(best_a)
    reference_stage = development_stage(best_b)
    stage_conflict = {
        candidate_stage, reference_stage
    } == {"PENDING", "FINAL"}

    return {
        "similarity": round(combined_similarity, 4),
        "development_similarity": round(dev_similarity, 4),
        "shared_identity_terms": sorted(shared_identity)[:12],
        "shared_development_tokens": sorted(shared_action)[:12],
        "event_gate": bool(identity_pass and development_pass and not stage_conflict),
        "stage_conflict": stage_conflict,
        "best_candidate_title": best_a,
        "best_reference_text": best_b,
    }


def compare_candidates(
    candidates: list[dict],
    published: list[dict],
    upstream: list[dict],
    model: Any | None,
    args: argparse.Namespace,
) -> dict:
    all_texts = []
    for candidate in candidates:
        titles = candidate.get("_titles", [])[:12]
        all_texts.extend(titles)
        all_texts.extend(development_text(t) for t in titles)
    for ref in [*published, *upstream]:
        texts = ref.get("texts", [])
        all_texts.extend(texts)
        all_texts.extend(development_text(t) for t in texts)

    vector_cache = build_text_vector_cache(all_texts, model)
    match_context = build_reference_match_context([*published, *upstream])

    semantic_only_rejections = 0
    evidence_counts = Counter()

    for candidate in candidates:
        candidate_titles = candidate.get("_titles", [])[:12]
        candidate_urls = set(candidate.get("_urls") or set())

        best_published = None
        for ref in published:
            metrics = strict_pipeline_reference_metrics(
                candidate_titles,
                ref["texts"],
                vector_cache,
                candidate_urls=candidate_urls,
                reference_urls=ref.get("urls") or set(),
                context=match_context,
            )
            row = {
                "reference_id": ref["reference_id"],
                "topic_title": ref["title"],
                **metrics,
            }
            if (
                best_published is None
                or _reference_match_sort_key(row)
                > _reference_match_sort_key(best_published)
            ):
                best_published = row

        best_upstream = None
        for ref in upstream:
            metrics = strict_pipeline_reference_metrics(
                candidate_titles,
                ref["texts"],
                vector_cache,
                candidate_urls=candidate_urls,
                reference_urls=ref.get("urls") or set(),
                context=match_context,
            )
            row = {
                "reference_id": ref["reference_id"],
                "cluster_title": ref["title"],
                **metrics,
            }
            if (
                best_upstream is None
                or _reference_match_sort_key(row)
                > _reference_match_sort_key(best_upstream)
            ):
                best_upstream = row

        candidate["published_match"] = best_published
        candidate["upstream_match"] = best_upstream

        published_exact = bool(
            best_published
            and (
                best_published.get("exact_url_match")
                or best_published.get("exact_title_match")
            )
        )
        upstream_exact = bool(
            best_upstream
            and (
                best_upstream.get("exact_url_match")
                or best_upstream.get("exact_title_match")
            )
        )

        published_ok = bool(
            best_published
            and (
                published_exact
                or (
                    best_published.get("event_gate")
                    and best_published["similarity"]
                    >= args.published_match_threshold
                )
            )
        )
        upstream_ok = bool(
            best_upstream
            and (
                upstream_exact
                or (
                    best_upstream.get("event_gate")
                    and best_upstream["similarity"]
                    >= args.upstream_match_threshold
                )
            )
        )

        if published_ok:
            candidate["pipeline_status"] = "MATCHED_PUBLISHED"
            evidence_counts[
                f"published_{best_published.get('match_basis', 'UNKNOWN').lower()}"
            ] += 1
        elif upstream_ok:
            candidate["pipeline_status"] = "SEEN_UPSTREAM_NOT_PUBLISHED"
            evidence_counts[
                f"upstream_{best_upstream.get('match_basis', 'UNKNOWN').lower()}"
            ] += 1
        else:
            candidate["pipeline_status"] = "POSSIBLE_MISS"
            if (
                (
                    best_published
                    and best_published["similarity"]
                    >= args.published_match_threshold
                )
                or (
                    best_upstream
                    and best_upstream["similarity"]
                    >= args.upstream_match_threshold
                )
            ):
                semantic_only_rejections += 1

    return {
        "semantic_model": (
            MODEL_NAME if vector_cache is not None
            else "token_jaccard_fallback"
        ),
        "published_match_threshold": args.published_match_threshold,
        "upstream_match_threshold": args.upstream_match_threshold,
        "match_requirement": (
            "Exact canonical URL/title evidence is evaluated first. Otherwise "
            "the semantic threshold must be paired with either a shared "
            "event-specific multiword anchor or at least two automatically "
            "derived discriminative terms. Generic people, institutions, and "
            "procedural verbs cannot establish a match by themselves; "
            "pending/final development conflicts are rejected."
        ),
        "reference_match_context": {
            "reference_count": match_context.get("reference_count", 0),
            "dynamic_generic_term_count": len(
                match_context.get("dynamic_generic_terms") or set()
            ),
            "dynamic_generic_identity_count": len(
                match_context.get("dynamic_generic_identities") or set()
            ),
            "generic_term_floor": match_context.get("generic_term_floor"),
            "generic_identity_floor": match_context.get(
                "generic_identity_floor"
            ),
        },
        "match_evidence_counts": dict(evidence_counts),
        "embedded_unique_text_count": len(vector_cache or {}),
        "semantic_only_match_rejections": semantic_only_rejections,
        "status_counts": dict(
            Counter(c["pipeline_status"] for c in candidates)
        ),
    }



def _candidate_titles(candidate: dict, limit: int = 12) -> list[str]:
    titles = [candidate.get("canonical_title") or ""]
    titles.extend(candidate.get("_titles") or [])
    return list(dict.fromkeys(
        clean_title(t) for t in titles if clean_title(t)
    ))[:limit]


def _post_split_duplicate_metrics(
    a: dict,
    b: dict,
    vector_cache: dict[str, Any] | None,
) -> dict:
    a_families = set(a.get("_family_ids") or set())
    b_families = set(b.get("_family_ids") or set())
    a_target = set(a.get("_target_family_ids") or set())
    b_target = set(b.get("_target_family_ids") or set())

    family_overlap = len(a_families & b_families)
    target_overlap = len(a_target & b_target)
    family_overlap_coeff = family_overlap / max(
        1, min(len(a_families), len(b_families))
    )
    target_overlap_coeff = target_overlap / max(
        1, min(len(a_target), len(b_target))
    )

    event = reference_match_metrics_cached(
        _candidate_titles(a), _candidate_titles(b), vector_cache
    )
    canonical_jaccard = title_jaccard(
        a.get("canonical_title") or "",
        b.get("canonical_title") or "",
    )

    shared_identity_count = len(event.get("shared_identity_terms") or [])
    shared_development_count = len(event.get("shared_development_tokens") or [])
    shared_action_groups = sorted(
        development_action_groups(a.get("canonical_title") or "")
        & development_action_groups(b.get("canonical_title") or "")
    )
    stage_conflict = bool(event.get("stage_conflict"))

    semantic_duplicate = (
        event.get("event_gate")
        and not stage_conflict
        and float(event.get("similarity", 0.0) or 0.0) >= 0.70
        and float(event.get("development_similarity", 0.0) or 0.0) >= 0.48
        and shared_identity_count >= 1
        and (
            shared_development_count >= 1
            or canonical_jaccard >= 0.16
        )
    )
    very_strong_duplicate = (
        not stage_conflict
        and float(event.get("similarity", 0.0) or 0.0) >= 0.84
        and float(event.get("development_similarity", 0.0) or 0.0) >= 0.58
        and shared_identity_count >= 1
    )
    lexical_event_duplicate = (
        not stage_conflict
        and bool(shared_action_groups)
        and canonical_jaccard >= 0.16
        and (
            shared_identity_count >= 2
            or any(
                " " in term
                for term in (event.get("shared_identity_terms") or [])
            )
        )
    )
    overlap_duplicate = (
        not stage_conflict
        and (
            family_overlap_coeff >= 0.22
            or target_overlap_coeff >= 0.18
        )
    )

    duplicate = bool(
        overlap_duplicate
        or semantic_duplicate
        or very_strong_duplicate
        or lexical_event_duplicate
    )
    strength = max(
        family_overlap_coeff,
        target_overlap_coeff,
        0.75 * float(event.get("similarity", 0.0) or 0.0)
        + 0.25 * float(event.get("development_similarity", 0.0) or 0.0),
    )

    return {
        "duplicate": duplicate,
        "strength": round(strength, 4),
        "family_overlap_count": family_overlap,
        "target_family_overlap_count": target_overlap,
        "family_overlap_coefficient": round(family_overlap_coeff, 4),
        "target_family_overlap_coefficient": round(target_overlap_coeff, 4),
        "canonical_title_jaccard": round(canonical_jaccard, 4),
        "shared_action_groups": shared_action_groups,
        **event,
    }


def merge_duplicate_development_candidates(
    candidates: list[dict],
    families: dict[str, dict],
    model: Any | None,
    args: argparse.Namespace,
    target_day: str,
) -> tuple[list[dict], dict]:
    """
    Final deduplication after target-date development splitting.

    This intentionally uses a highest-scoring prototype for each group rather
    than transitive unioning, which reduces the risk of chaining adjacent
    developments together.
    """
    if len(candidates) <= 1:
        return candidates, {
            "input_candidate_count": len(candidates),
            "output_candidate_count": len(candidates),
            "merged_candidate_count": 0,
            "merged_group_count": 0,
            "pair_checks": 0,
            "reconstruction_failures": 0,
        }

    ordered = sorted(
        candidates,
        key=lambda c: (
            float(c.get("discovery_score", 0.0) or 0.0),
            int(c.get("target_day_writeup_family_count", 0) or 0),
            int(c.get("target_day_domain_count", 0) or 0),
        ),
        reverse=True,
    )

    all_texts = []
    for candidate in ordered:
        titles = _candidate_titles(candidate)
        all_texts.extend(titles)
        all_texts.extend(development_text(t) for t in titles)
    vector_cache = build_text_vector_cache(all_texts, model)

    groups: list[dict] = []
    pair_checks = 0

    for candidate in ordered:
        best_group = None
        best_metrics = None
        for group in groups:
            prototype = group["prototype"]
            metrics = _post_split_duplicate_metrics(
                prototype, candidate, vector_cache
            )
            pair_checks += 1
            if not metrics["duplicate"]:
                continue
            if best_metrics is None or metrics["strength"] > best_metrics["strength"]:
                best_group = group
                best_metrics = metrics

        if best_group is None:
            groups.append({
                "prototype": candidate,
                "members": [candidate],
                "matches": [],
            })
        else:
            best_group["members"].append(candidate)
            best_group["matches"].append({
                "candidate_id": candidate.get("candidate_id"),
                "canonical_title": candidate.get("canonical_title"),
                "metrics": best_metrics,
            })

    merged: list[dict] = []
    reconstruction_failures = 0
    merged_group_count = 0

    for group in groups:
        members = group["members"]
        if len(members) == 1:
            merged.append(members[0])
            continue

        merged_group_count += 1
        family_ids = set().union(
            *(set(c.get("_family_ids") or set()) for c in members)
        )
        target_ids = set().union(
            *(set(c.get("_target_family_ids") or set()) for c in members)
        )
        adjacent_ids = family_ids - target_ids
        signals = family_signal_rows(
            family_ids, families, target_day, model
        )
        target_ids = {f for f in target_ids if f in signals}
        adjacent_ids = {f for f in adjacent_ids if f in signals}

        rebuilt = None
        reconstruction_succeeded = False
        if target_ids:
            medoid = choose_cluster_medoid(sorted(target_ids), signals)
            rebuilt = canonical_event_from_group(
                {
                    "prototype": group["prototype"],
                    "members": members,
                    "family_ids": target_ids | adjacent_ids,
                    "target_family_ids": target_ids,
                    "adjacent_family_ids": adjacent_ids,
                    "target_medoid_id": medoid,
                    "split_diagnostics": {
                        "post_split_duplicate_merge": True,
                        "input_candidate_count": len(members),
                    },
                    "_signals": signals,
                },
                families,
                None,
                target_day,
                args,
            )
            reconstruction_succeeded = rebuilt is not None

        if rebuilt is None:
            reconstruction_failures += 1
            rebuilt = dict(group["prototype"])

        rebuilt["post_split_dedup"] = {
            "merged": True,
            "input_candidate_count": len(members),
            "input_candidate_ids": [c.get("candidate_id") for c in members],
            "input_titles": [c.get("canonical_title") for c in members],
            "prototype_candidate_id": group["prototype"].get("candidate_id"),
            "pair_matches": group["matches"],
            "reconstruction_succeeded": reconstruction_succeeded,
        }
        merged.append(rebuilt)

    return merged, {
        "input_candidate_count": len(candidates),
        "output_candidate_count": len(merged),
        "merged_candidate_count": len(candidates) - len(merged),
        "merged_group_count": merged_group_count,
        "pair_checks": pair_checks,
        "reconstruction_failures": reconstruction_failures,
        "semantic_model": MODEL_NAME if vector_cache is not None else "token_jaccard_fallback",
    }


def _rank_percentiles(
    rows: list[dict],
    value_getter,
) -> dict[int, float]:
    ordered = sorted(
        range(len(rows)),
        key=lambda i: float(value_getter(rows[i]) or 0.0),
        reverse=True,
    )
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: 1.0}
    return {
        idx: round(1.0 - (rank / (len(ordered) - 1)), 4)
        for rank, idx in enumerate(ordered)
    }


def attach_global_attention(candidates: list[dict]) -> None:
    percentiles = _rank_percentiles(
        candidates,
        lambda c: c.get("discovery_score", 0.0),
    )
    for idx, candidate in enumerate(candidates):
        candidate["global_attention"] = {
            "source": "GDELT Global Similarity Graph",
            "target_date": candidate.get("target_date"),
            "target_day_writeup_family_count": int(
                candidate.get("target_day_writeup_family_count", 0) or 0
            ),
            "target_day_outlet_count": int(
                candidate.get("target_day_domain_count", 0) or 0
            ),
            "adjacent_same_development_writeup_count": int(
                candidate.get("adjacent_support_family_count", 0) or 0
            ),
            "all_supporting_writeup_family_count": int(
                candidate.get("writeup_family_count", 0) or 0
            ),
            "all_supporting_outlet_count": int(
                candidate.get("unique_domain_count", 0) or 0
            ),
            "active_window_count": int(
                candidate.get("active_window_count", 0) or 0
            ),
            "language_count": int(candidate.get("language_count", 0) or 0),
            "cross_language_url_count": int(
                candidate.get("cross_language_url_count", 0) or 0
            ),
            "discovery_score": float(candidate.get("discovery_score", 0.0) or 0.0),
            "discovery_percentile": percentiles.get(idx, 0.0),
            "shadow_only": True,
        }


def _blend_shadow_signals(global_signal: float, local_signal: float) -> float:
    high = max(global_signal, local_signal)
    low = min(global_signal, local_signal)
    return round(80.0 * high + 20.0 * low, 3)


def build_shadow_blended_ranking(
    candidates: list[dict],
    published: list[dict],
) -> tuple[list[dict], dict]:
    """
    Exploratory ranking only. Either a strong local or global signal can carry
    an event, while agreement between both earns a modest confirmation bonus.
    """
    published_percentiles = _rank_percentiles(
        published,
        lambda r: r.get("attention_score", 0.0),
    )
    published_local = {
        row.get("reference_id"): published_percentiles.get(idx, 0.0)
        for idx, row in enumerate(published)
    }

    rows = []
    best_global_by_published: dict[str, dict] = {}

    for candidate in candidates:
        global_signal = float(
            (candidate.get("global_attention") or {}).get(
                "discovery_percentile", 0.0
            ) or 0.0
        )
        published_match = candidate.get("published_match") or {}
        upstream_match = candidate.get("upstream_match") or {}
        status = candidate.get("pipeline_status")

        if status == "MATCHED_PUBLISHED" and published_match.get("reference_id"):
            ref_id = published_match["reference_id"]
            current = best_global_by_published.get(ref_id)
            if current is None or global_signal > current["global_signal"]:
                best_global_by_published[ref_id] = {
                    "candidate": candidate,
                    "global_signal": global_signal,
                }
            local_signal = published_local.get(ref_id, 0.0)
        elif status == "SEEN_UPSTREAM_NOT_PUBLISHED":
            # Presence upstream is useful corroboration, not an attention score.
            local_signal = min(
                0.25,
                0.10 + 0.15 * float(upstream_match.get("similarity", 0.0) or 0.0),
            )
        else:
            local_signal = 0.0

        candidate["shadow_ranking_signals"] = {
            "global_signal": round(global_signal, 4),
            "local_signal": round(local_signal, 4),
            "blended_score": _blend_shadow_signals(global_signal, local_signal),
            "production_ranking_changed": False,
        }

    for idx, row in enumerate(published):
        ref_id = row.get("reference_id")
        local_signal = published_local.get(ref_id, 0.0)
        matched = best_global_by_published.get(ref_id)
        global_signal = matched["global_signal"] if matched else 0.0
        candidate = matched["candidate"] if matched else None
        rows.append({
            "event_type": "PUBLISHED",
            "reference_id": ref_id,
            "title": row.get("title"),
            "production_rank": row.get("rank"),
            "local_attention_score": row.get("attention_score"),
            "local_signal": round(local_signal, 4),
            "global_signal": round(global_signal, 4),
            "gdelt_candidate_id": candidate.get("candidate_id") if candidate else None,
            "blended_score": _blend_shadow_signals(global_signal, local_signal),
        })

    for candidate in candidates:
        status = candidate.get("pipeline_status")
        if status == "MATCHED_PUBLISHED":
            continue
        preview_type = (
            "UNDERREPRESENTED_UPSTREAM"
            if status == "SEEN_UPSTREAM_NOT_PUBLISHED"
            else "NEW_DISCOVERY"
        )
        signals = candidate.get("shadow_ranking_signals") or {}
        rows.append({
            "event_type": preview_type,
            "candidate_id": candidate.get("candidate_id"),
            "title": candidate.get("canonical_title"),
            "pipeline_status": status,
            "local_signal": signals.get("local_signal", 0.0),
            "global_signal": signals.get("global_signal", 0.0),
            "blended_score": signals.get("blended_score", 0.0),
        })

    rows.sort(
        key=lambda r: (
            float(r.get("blended_score", 0.0) or 0.0),
            float(r.get("global_signal", 0.0) or 0.0),
            float(r.get("local_signal", 0.0) or 0.0),
        ),
        reverse=True,
    )
    candidate_by_id = {c.get("candidate_id"): c for c in candidates}
    for rank, row in enumerate(rows, start=1):
        row["shadow_rank"] = rank
        candidate = candidate_by_id.get(row.get("candidate_id"))
        if candidate is not None:
            candidate["shadow_blended_rank"] = rank

    return rows, {
        "formula": (
            "80 × stronger(local, global) + 20 × weaker(local, global). "
            "This is exploratory only and never changes production order."
        ),
        "published_event_count": len(published),
        "candidate_event_count": sum(
            1 for r in rows if r.get("event_type") != "PUBLISHED"
        ),
        "row_count": len(rows),
    }



def _receipt_language_label(raw: object) -> str:
    value = str(raw or "").strip()
    normalized = normalize_language(value)
    if normalized in ENGLISH_NAMES:
        return "English"
    if not value:
        return "Unknown"
    return value.replace("_", " ").replace("-", " ").title()


def _receipt_member_priority(member: dict, target_day: str, family: dict) -> tuple:
    explicit = set(member.get("explicit_days") or set())
    observed = set(member.get("observed_days") or set())
    language = member.get("language")
    url = str(member.get("url") or "")
    return (
        int(is_english(language)),
        int(target_day in explicit),
        int((not explicit) and target_day in observed),
        int(url.lower().startswith("https://")),
        int(bool(clean_title(member.get("title") or ""))),
        int(family.get("outlet_count", 0) or 0),
        len(clean_title(member.get("title") or "")),
    )


def build_global_source_receipt_catalog(
    candidates: list[dict],
    families: dict[str, dict],
    target_day: str,
) -> dict:
    """
    Build two complementary receipt views for every bounded GDELT candidate:

    1. one receipt per outlet domain for outlet counts and political-bias
       accounting; and
    2. one row per distinct write-up family, with every outlet copy nested
       beneath it for the user-facing article list.

    All-language receipts are retained for auditability. Only English receipts
    are marked displayable in the current English-first product.
    """
    catalog = {}
    total_all = 0
    total_english = 0
    total_non_english = 0
    total_family_receipts = 0
    incomplete_candidates = []

    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not candidate_id:
            continue

        target_family_ids = sorted(
            set(candidate.get("_target_family_ids") or set())
        )
        candidate_family_meta = {
            str(row.get("family_id")): row
            for row in (candidate.get("writeup_families") or [])
            if row.get("family_id")
        }

        # Event-level outlet view: one visible receipt per unique domain.
        by_domain = {}
        # Write-up view: every target-day family retains its own outlet copies.
        family_rows = []

        for family_id in target_family_ids:
            family = families.get(family_id)
            if not family:
                continue

            family_meta = candidate_family_meta.get(str(family_id)) or {}
            family_by_domain = {}

            for member in family.get("_members") or []:
                domain = str(member.get("domain") or "").lower().strip()
                url = str(member.get("url") or "").strip()
                if not domain or not url:
                    continue

                language_raw = member.get("language")
                language_label = _receipt_language_label(language_raw)
                english = is_english(language_raw)
                explicit = set(member.get("explicit_days") or set())
                observed = set(member.get("observed_days") or set())

                if target_day in explicit:
                    day_basis = "TARGET_EXPLICIT"
                elif not explicit and target_day in observed:
                    day_basis = "TARGET_OBSERVED"
                elif explicit:
                    day_basis = "TARGET_FAMILY_ADJACENT_URL"
                else:
                    day_basis = "TARGET_FAMILY_OBSERVED_ELSEWHERE"

                receipt = {
                    "domain": domain,
                    "url": url,
                    "canonical_url": member.get("canonical_url") or canonical_url(url),
                    "source": domain,
                    "title": clean_title(member.get("title") or ""),
                    "language": language_label,
                    "language_raw": str(language_raw or ""),
                    "displayable_english": bool(english),
                    "writeup_family_id": family_id,
                    "writeup_family_outlet_count": int(
                        family.get("outlet_count", 0) or 0
                    ),
                    "day_basis": day_basis,
                    "target_date": target_day,
                    "origin": "gdelt_gsg",
                    "syndicated": bool(
                        int(family.get("outlet_count", 0) or 0) > 1
                    ),
                }

                priority = _receipt_member_priority(member, target_day, family)

                current_family = family_by_domain.get(domain)
                if current_family is None or priority > current_family[0]:
                    family_by_domain[domain] = (priority, receipt)

                current_event = by_domain.get(domain)
                if current_event is None or priority > current_event[0]:
                    by_domain[domain] = (priority, receipt)

            family_receipts = [row[1] for row in family_by_domain.values()]
            family_receipts.sort(
                key=lambda row: (
                    not bool(row.get("displayable_english")),
                    row.get("domain") or "",
                )
            )
            family_english = [
                row for row in family_receipts if row.get("displayable_english")
            ]
            family_non_english = [
                row for row in family_receipts
                if not row.get("displayable_english")
            ]

            representative_title = (
                family_meta.get("representative_title")
                or (family.get("representative") or {}).get("title")
                or (family_receipts[0].get("title") if family_receipts else "")
            )
            representative_url = (
                family_meta.get("representative_url")
                or (family.get("representative") or {}).get("url")
                or (family_english[0].get("url") if family_english else "")
                or (family_receipts[0].get("url") if family_receipts else "")
            )

            expected_family_outlets = int(
                family.get("outlet_count", 0) or 0
            )
            family_rows.append({
                "family_id": family_id,
                "representative_title": representative_title,
                "representative_url": representative_url,
                "day_role": family_meta.get("day_role") or "TARGET_FAMILY",
                "outlet_count": expected_family_outlets,
                "receipt_outlet_count": len(family_receipts),
                "url_count": int(family.get("url_count", 0) or 0),
                "language_count": len(family.get("languages") or set()),
                "active_window_count": len(family.get("slots") or set()),
                "english_receipt_count": len(family_english),
                "non_english_receipt_count": len(family_non_english),
                "displayable_english": bool(family_english or representative_url),
                "receipt_catalog_complete": (
                    len(family_receipts) == expected_family_outlets
                ),
                "receipts": family_receipts,
            })
            total_family_receipts += len(family_receipts)

        receipts = [row[1] for row in by_domain.values()]
        receipts.sort(
            key=lambda row: (
                not bool(row.get("displayable_english")),
                row.get("domain") or "",
            )
        )
        english_receipts = [
            row for row in receipts if row.get("displayable_english")
        ]
        non_english_receipts = [
            row for row in receipts if not row.get("displayable_english")
        ]

        expected_outlets = int(
            candidate.get("target_day_domain_count", 0) or 0
        )
        expected_writeups = int(
            candidate.get("target_day_writeup_family_count", 0) or 0
        )
        outlet_complete = len(receipts) == expected_outlets
        writeup_complete = (
            len(family_rows) == expected_writeups
            and all(row.get("representative_url") for row in family_rows)
        )
        complete = outlet_complete and writeup_complete
        if not complete:
            incomplete_candidates.append({
                "candidate_id": candidate_id,
                "expected_outlets": expected_outlets,
                "receipt_outlets": len(receipts),
                "expected_writeups": expected_writeups,
                "receipt_writeups": len(family_rows),
                "outlet_catalog_complete": outlet_complete,
                "writeup_catalog_complete": writeup_complete,
            })

        catalog[candidate_id] = {
            "candidate_id": candidate_id,
            "canonical_title": candidate.get("canonical_title") or "",
            "target_date": target_day,
            "target_day_writeup_family_count": expected_writeups,
            "target_day_outlet_count": expected_outlets,
            "all_language_receipt_count": len(receipts),
            "english_receipt_count": len(english_receipts),
            "non_english_receipt_count": len(non_english_receipts),
            "receipt_catalog_complete": complete,
            "outlet_catalog_complete": outlet_complete,
            "writeup_catalog_complete": writeup_complete,
            "writeup_families": sorted(
                family_rows,
                key=lambda row: (
                    row.get("outlet_count", 0),
                    row.get("active_window_count", 0),
                ),
                reverse=True,
            ),
            "receipts": receipts,
        }

        total_all += len(receipts)
        total_english += len(english_receipts)
        total_non_english += len(non_english_receipts)

    return {
        "schema_version": "1.1",
        "date": target_day,
        "candidate_count": len(catalog),
        "all_language_receipt_count": total_all,
        "english_receipt_count": total_english,
        "non_english_receipt_count": total_non_english,
        "family_receipt_count": total_family_receipts,
        "incomplete_candidate_count": len(incomplete_candidates),
        "incomplete_candidates": incomplete_candidates,
        "english_display_policy": (
            "All-language receipts are retained for auditability. Only English "
            "receipts are eligible for the current user-facing source list."
        ),
        "candidates": catalog,
    }


def clean_for_json(candidate: dict) -> dict:
    out = dict(candidate)
    out.pop("_family_ids", None)
    out.pop("_target_family_ids", None)
    out.pop("_titles", None)
    out.pop("_urls", None)
    return out



def _target_day_member_pool(family: dict, target_day: str) -> list[dict]:
    rows = []
    for member in family.get("_members") or []:
        if not is_english(member.get("language")):
            continue
        if not clean_title(member.get("title") or ""):
            continue
        if not member.get("url"):
            continue

        explicit = set(member.get("explicit_days") or set())
        observed = set(member.get("observed_days") or set())
        if explicit:
            if target_day not in explicit:
                continue
            day_strength = 2
        elif target_day in observed:
            day_strength = 1
        else:
            continue

        row = dict(member)
        row["_day_strength"] = day_strength
        rows.append(row)

    rows.sort(
        key=lambda n: (
            int(n.get("_day_strength", 0)),
            len(clean_title(n.get("title") or "")),
            int(bool(n.get("image"))),
        ),
        reverse=True,
    )
    return rows


def select_net_new_representatives(
    candidate: dict,
    families: dict[str, dict],
    target_day: str,
    normalized_urls: set[str],
    reserved_urls: set[str],
    limit: int,
) -> tuple[list[dict], dict]:
    """Prefer one target-day English URL from separate families and domains."""
    target_ids = [
        f for f in set(candidate.get("_target_family_ids") or set())
        if f in families
    ]
    target_ids.sort(
        key=lambda f: (
            int(families[f].get("outlet_count", 0) or 0),
            len(families[f].get("slots") or set()),
            int(families[f].get("unique_title_count", 0) or 0),
        ),
        reverse=True,
    )

    used_urls = set()
    used_domains = set()
    used_families = set()
    selected = []
    candidate_target_urls = set()

    family_members: dict[str, list[dict]] = {}
    for family_id in target_ids:
        members = _target_day_member_pool(families[family_id], target_day)
        family_members[family_id] = members
        for member in members:
            cu = canonical_url(member.get("url") or "")
            if cu:
                candidate_target_urls.add(cu)

    def add_member(family_id: str, member: dict) -> bool:
        raw_url = member.get("url") or ""
        cu = canonical_url(raw_url)
        dom = member.get("domain") or outlet_domain(raw_url)
        if not cu or not dom:
            return False
        if cu in normalized_urls or cu in reserved_urls or cu in used_urls:
            return False
        if dom in used_domains:
            return False

        selected.append({
            "title": clean_title(member.get("title") or ""),
            "description": "",
            "url": raw_url,
            "source": dom,
            "published_date": target_day,
            "language": "en",
            "origin": "gdelt_discovery",
            "gdelt_candidate_id": candidate.get("candidate_id"),
            "gdelt_writeup_family_id": family_id,
            "gdelt_writeup_family_outlets": int(
                families[family_id].get("outlet_count", 0) or 0
            ),
        })
        used_urls.add(cu)
        used_domains.add(dom)
        used_families.add(family_id)
        return True

    # Pass 1: maximize independent write-up-family and outlet diversity.
    for family_id in target_ids:
        for member in family_members.get(family_id, []):
            if add_member(family_id, member):
                break
        if len(selected) >= limit:
            break

    # Pass 2: if fewer than four families are available, permit another URL
    # from a family already used, but still require a different outlet/domain.
    if len(selected) < limit:
        for family_id in target_ids:
            for member in family_members.get(family_id, []):
                if add_member(family_id, member) and len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break

    return selected[:limit], {
        "target_family_count": len(target_ids),
        "target_url_count": len(candidate_target_urls),
        "already_in_normalized_corpus_count": len(
            candidate_target_urls & normalized_urls
        ),
        "selected_distinct_family_count": len(used_families),
        "selected_distinct_domain_count": len(used_domains),
    }


def build_ingestion_preview(
    candidates: list[dict],
    families: dict[str, dict],
    normalized_urls: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    """
    Produce a bounded, read-only preview of net-new candidate articles.

    Preview types:
      - NEW_DISCOVERY: no matching published or upstream event.
      - UNDERREPRESENTED_UPSTREAM: the event appeared upstream but lacked
        enough focused support to survive publication.
    """
    preview = []
    exclusions = Counter()
    reserved_urls = set()
    required_records = max(1, int(args.max_representatives))
    max_records = max(1, int(args.max_ingestion_records))

    ordered = sorted(
        candidates,
        key=lambda c: (
            float((c.get("shadow_ranking_signals") or {}).get("blended_score", 0.0) or 0.0),
            float(c.get("discovery_score", 0.0) or 0.0),
            int(c.get("target_day_writeup_family_count", 0) or 0),
        ),
        reverse=True,
    )

    for candidate in ordered:
        status = candidate.get("pipeline_status")
        if status == "MATCHED_PUBLISHED":
            exclusions["already_published"] += 1
            continue
        if status == "POSSIBLE_MISS":
            preview_type = "NEW_DISCOVERY"
        elif status == "SEEN_UPSTREAM_NOT_PUBLISHED":
            preview_type = "UNDERREPRESENTED_UPSTREAM"
        else:
            exclusions["unsupported_pipeline_status"] += 1
            continue

        if int(candidate.get("target_day_writeup_family_count", 0) or 0) < args.min_preview_writeups:
            exclusions["too_few_target_day_writeups"] += 1
            continue
        if int(candidate.get("target_day_domain_count", 0) or 0) < args.min_domains:
            exclusions["too_few_target_day_outlets"] += 1
            continue
        preview_cohesion_floor = float(
            candidate.get("cohesion_gate_floor", args.min_event_cohesion)
            or args.min_event_cohesion
        )
        if float(candidate.get("target_title_cohesion_p20", 0.0) or 0.0) < preview_cohesion_floor:
            exclusions["weak_target_day_cohesion"] += 1
            continue
        canonical_tokens = title_tokens(candidate.get("canonical_title") or "")
        if len(canonical_tokens) < 3 or len(canonical_tokens & GENERIC_TITLE_TOKENS) >= 2:
            exclusions["generic_canonical_title"] += 1
            continue

        if len(preview) >= max(1, args.max_ingestion_candidates):
            exclusions["candidate_cap_reached"] += 1
            break
        if sum(p["article_count"] for p in preview) + required_records > max_records:
            exclusions["article_record_cap_reached"] += 1
            break

        articles, selection_stats = select_net_new_representatives(
            candidate=candidate,
            families=families,
            target_day=candidate.get("target_date") or "",
            normalized_urls=normalized_urls,
            reserved_urls=reserved_urls,
            limit=required_records,
        )
        if len(articles) < required_records:
            exclusions["insufficient_net_new_english_representatives"] += 1
            continue

        for article in articles:
            cu = canonical_url(article.get("url") or "")
            if cu:
                reserved_urls.add(cu)

        preview_rank = len(preview) + 1
        candidate["ingestion_preview_rank"] = preview_rank
        candidate["ingestion_preview_type"] = preview_type
        global_attention = dict(candidate.get("global_attention") or {})

        preview.append({
            "preview_rank": preview_rank,
            "preview_type": preview_type,
            "candidate_id": candidate.get("candidate_id"),
            "canonical_title": candidate.get("canonical_title"),
            "pipeline_status": status,
            "discovery_score": candidate.get("discovery_score"),
            "shadow_blended_rank": candidate.get("shadow_blended_rank"),
            "shadow_blended_score": (
                candidate.get("shadow_ranking_signals") or {}
            ).get("blended_score"),
            "global_attention": global_attention,
            "existing_local_match": {
                "published": candidate.get("published_match"),
                "upstream": candidate.get("upstream_match"),
            },
            "selection": selection_stats,
            "article_count": len(articles),
            "articles": articles,
            "note": (
                "Audit preview only. All article URLs are absent from the "
                "normalized local corpus. A future optional ingestion stage "
                "would still send them through existing normalization, "
                "clustering, purity, and publication gates."
            ),
        })

    type_counts = Counter(p["preview_type"] for p in preview)
    return preview, {
        "candidate_count": len(preview),
        "article_record_count": sum(p["article_count"] for p in preview),
        "preview_type_counts": dict(type_counts),
        "max_candidates": args.max_ingestion_candidates,
        "max_total_article_records": max_records,
        "required_net_new_representatives_per_candidate": required_records,
        "eligibility": {
            "pipeline_statuses": [
                "POSSIBLE_MISS",
                "SEEN_UPSTREAM_NOT_PUBLISHED",
            ],
            "minimum_target_day_writeups": args.min_preview_writeups,
            "minimum_target_day_outlets": args.min_domains,
            "minimum_target_title_cohesion_p20": args.min_event_cohesion,
            "representatives_must_be_absent_from_normalized_corpus": True,
        },
        "exclusions": dict(exclusions),
    }


def write_csv(path: Path, candidates: list[dict]) -> None:
    fields = [
        "rank",
        "ingestion_preview_rank",
        "ingestion_preview_type",
        "shadow_blended_rank",
        "shadow_blended_score",
        "global_attention_percentile",
        "pipeline_status",
        "canonical_title",
        "canonical_domain",
        "discovery_score",
        "target_day_writeup_family_count",
        "target_day_domain_count",
        "adjacent_support_family_count",
        "writeup_family_count",
        "unique_domain_count",
        "unique_url_count",
        "headline_variant_count",
        "target_title_cohesion_p20",
        "event_title_cohesion_p20",
        "development_cohesion_p20",
        "active_window_count",
        "active_day_count",
        "language_count",
        "cross_language_url_count",
        "largest_family_outlet_count",
        "largest_family_outlet_share",
        "published_topic",
        "published_similarity",
        "published_event_gate",
        "upstream_cluster",
        "upstream_similarity",
        "upstream_event_gate",
        "canonical_url",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            published = candidate.get("published_match") or {}
            upstream = candidate.get("upstream_match") or {}
            writer.writerow({
                "rank": rank,
                "ingestion_preview_rank": candidate.get("ingestion_preview_rank"),
                "ingestion_preview_type": candidate.get("ingestion_preview_type"),
                "shadow_blended_rank": candidate.get("shadow_blended_rank"),
                "shadow_blended_score": (candidate.get("shadow_ranking_signals") or {}).get("blended_score"),
                "global_attention_percentile": (candidate.get("global_attention") or {}).get("discovery_percentile"),
                "pipeline_status": candidate.get("pipeline_status"),
                "canonical_title": candidate.get("canonical_title"),
                "canonical_domain": candidate.get("canonical_domain"),
                "discovery_score": candidate.get("discovery_score"),
                "target_day_writeup_family_count": candidate.get("target_day_writeup_family_count"),
                "target_day_domain_count": candidate.get("target_day_domain_count"),
                "adjacent_support_family_count": candidate.get("adjacent_support_family_count"),
                "writeup_family_count": candidate.get("writeup_family_count"),
                "unique_domain_count": candidate.get("unique_domain_count"),
                "unique_url_count": candidate.get("unique_url_count"),
                "headline_variant_count": candidate.get("headline_variant_count"),
                "target_title_cohesion_p20": candidate.get("target_title_cohesion_p20"),
                "event_title_cohesion_p20": candidate.get("event_title_cohesion_p20"),
                "development_cohesion_p20": candidate.get("development_cohesion_p20"),
                "active_window_count": candidate.get("active_window_count"),
                "active_day_count": candidate.get("active_day_count"),
                "language_count": candidate.get("language_count"),
                "cross_language_url_count": candidate.get("cross_language_url_count"),
                "largest_family_outlet_count": candidate.get("largest_family_outlet_count"),
                "largest_family_outlet_share": candidate.get("largest_family_outlet_share"),
                "published_topic": published.get("topic_title"),
                "published_similarity": published.get("similarity"),
                "published_event_gate": published.get("event_gate"),
                "upstream_cluster": upstream.get("cluster_title"),
                "upstream_similarity": upstream.get("similarity"),
                "upstream_event_gate": upstream.get("event_gate"),
                "canonical_url": candidate.get("canonical_url"),
            })


def main() -> int:
    started = time.perf_counter()
    args = parse_args()

    try:
        target = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print("❌ Invalid --date. Use YYYY-MM-DD.")
        return 2

    limits = Limits(
        max_files=max(1, args.max_files),
        max_bytes=max(1, int(args.max_download_mb * 1024 * 1024)),
        max_edges=max(1, args.max_edges),
        max_raw_candidates=max(1, args.max_raw_candidates),
    )

    output_json = Path(
        args.output_json or f"gdelt_global_discovery_audit_{args.date}.json"
    )
    output_csv = Path(
        args.output_csv or f"gdelt_global_discovery_audit_{args.date}.csv"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"🌍 GDELT shadow discovery audit for {args.date} (schema {VERSION})")
    print(
        f"   window=±{args.window_days} day(s), cadence={args.slot_minutes}m, "
        f"files≤{limits.max_files}, new download≤{args.max_download_mb:.0f}MB, "
        f"edges≤{limits.max_edges:,}"
    )
    print("   English representatives only; linked non-English coverage may count.")
    if args.pipeline_mode:
        print(
            "   mode=pipeline candidate feed; existing clustered articles are "
            "comparison-only and production outputs are ignored"
        )

    files, download_stats = acquire_files(target, args, limits)
    if not files:
        output = {
            "audit_schema_version": VERSION,
            "date": args.date,
            "status": "NO_GSG_FILES_FOUND",
            "execution_mode": (
                "PIPELINE_CANDIDATE_FEED" if args.pipeline_mode else "SHADOW_AUDIT"
            ),
            "read_only": True,
            "production_files_modified": [],
            "download": download_stats,
        }
        output_json.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"⚠️ No GSG files found. Wrote diagnostic → {output_json}")
        return 0

    print(
        f"📦 Using {len(files)} GSG files "
        f"({download_stats['downloaded_mb_this_run']} MB newly downloaded)"
    )

    nodes, adjacency, parse_stats = parse_gsg_files(files, args, limits)
    families, url_to_family, family_stats = build_writeup_families(
        nodes, adjacency, args
    )
    family_graph = build_family_graph(adjacency, url_to_family)

    raw_candidates = []
    for seed_id in families:
        candidate = raw_candidate_from_seed(seed_id, families, family_graph)
        if candidate and len(candidate["domains"]) >= args.min_domains:
            raw_candidates.append(candidate)

    model = load_model()
    canonical_groups = merge_duplicate_seed_candidates(raw_candidates, model, args)

    all_group_family_ids = {
        family_id
        for group in canonical_groups
        for family_id in (group.get("family_ids") or set())
    }
    signal_cache = family_signal_rows(
        all_group_family_ids, families, args.date, model
    )
    family_title_vectors = {
        family_id: row.get("title_vector")
        for family_id, row in signal_cache.items()
        if row.get("title_vector") is not None
    } or None

    split_groups = []
    split_stats = Counter()
    for group in canonical_groups:
        developments, diag = split_group_into_target_developments(
            group,
            families,
            args.date,
            model,
            args,
            signal_cache=signal_cache,
        )
        split_groups.extend(developments)
        split_stats["canonical_groups_processed"] += 1
        split_stats["target_anchored_developments"] += len(developments)
        if not developments:
            split_stats["groups_without_target_development"] += 1
        split_stats["discarded_adjacent_families"] += int(
            diag.get("discarded_adjacent_family_count", 0) or 0
        )
        if len(developments) > 1:
            split_stats["groups_split_into_multiple_developments"] += 1

    canonical_candidates = []
    cohesion_rejected = 0
    for group in split_groups:
        candidate = canonical_event_from_group(
            group,
            families,
            family_title_vectors,
            args.date,
            args,
        )
        if candidate and candidate["target_day_domain_count"] >= args.min_domains:
            canonical_candidates.append(candidate)
        else:
            cohesion_rejected += 1

    canonical_candidates, post_split_dedup_stats = merge_duplicate_development_candidates(
        canonical_candidates,
        families,
        model,
        args,
        args.date,
    )

    canonical_candidates.sort(
        key=lambda c: (
            c["discovery_score"],
            c["writeup_family_count"],
            c["unique_domain_count"],
            c["active_window_count"],
        ),
        reverse=True,
    )
    canonical_candidates = canonical_candidates[: max(1, args.top_n)]
    attach_global_attention(canonical_candidates)

    global_source_receipt_catalog = (
        build_global_source_receipt_catalog(
            canonical_candidates, families, args.date
        )
        if args.pipeline_mode
        else None
    )

    if args.pipeline_mode:
        # Do not let stale downstream files from a prior run influence the
        # pre-merge candidate feed. The current KMeans output is the only local
        # event comparison at this point in the pipeline.
        missing_root = Path(".gdelt_pipeline_reference_void") / args.date
        summary_path = Path(
            args.summary_file or missing_root / "topic_summaries.json"
        )
        final_path = Path(
            args.final_file or missing_root / "grouped_articles_final.json"
        )
        upstream_path = Path(
            args.upstream_file or f"clustered_articles_{args.date}.json"
        )
    else:
        summary_path = Path(
            args.summary_file or f"topic_summaries_{args.date}.json"
        )
        final_path = Path(
            args.final_file or f"grouped_articles_final_{args.date}.json"
        )
        upstream_path = Path(
            args.upstream_file or f"grouped_articles_filtered_{args.date}.json"
        )

    normalized_path = Path(
        args.normalized_file or f"articles_raw_normalized_{args.date}.json"
    )
    normalized_urls, normalized_stats = load_normalized_corpus_urls(
        normalized_path
    )

    published, upstream, reference_stats = load_reference_sets(
        summary_path, final_path, upstream_path
    )
    reference_stats.update(normalized_stats)

    comparison_stats = compare_candidates(
        canonical_candidates,
        published,
        upstream,
        model,
        args,
    )
    shadow_ranking, shadow_ranking_stats = build_shadow_blended_ranking(
        canonical_candidates,
        published,
    )
    if normalized_stats.get("normalized_file_found"):
        ingestion_preview, ingestion_preview_stats = build_ingestion_preview(
            canonical_candidates,
            families,
            normalized_urls,
            args,
        )
    else:
        ingestion_preview = []
        ingestion_preview_stats = {
            "candidate_count": 0,
            "article_record_count": 0,
            "preview_type_counts": {},
            "max_candidates": args.max_ingestion_candidates,
            "max_total_article_records": args.max_ingestion_records,
            "required_net_new_representatives_per_candidate": args.max_representatives,
            "status": "NORMALIZED_CORPUS_MISSING",
            "exclusions": {"normalized_file_missing": len(canonical_candidates)},
        }

    cleaned = [clean_for_json(c) for c in canonical_candidates]
    elapsed = round(time.perf_counter() - started, 2)

    output = {
        "audit_schema_version": VERSION,
        "date": args.date,
        "status": "OK",
        "execution_mode": (
            "PIPELINE_CANDIDATE_FEED" if args.pipeline_mode else "SHADOW_AUDIT"
        ),
        "read_only": True,
        "production_files_modified": [],
        "method": {
            "dataset": "GDELT Global Similarity Graph",
            "sampling": (
                "Bounded hourly sampling across the target date ±1 day by default."
            ),
            "english_policy": (
                "Every accepted graph edge contains an English article. "
                "Only English articles appear as representatives; directly linked "
                "non-English coverage contributes only to aggregate breadth metrics."
            ),
            "writeup_family_definition": (
                "Exact headlines, GSG title links, and very-high-similarity headline "
                "variants are collapsed before story ranking."
            ),
            "canonical_event_definition": (
                "Merged seed neighborhoods are split around developments observed "
                "on the target date. Adjacent-day write-ups count only when local "
                "semantic, identity, action/outcome, and development-stage checks "
                "tie them to that same target-date development."
            ),
            "comparison_definition": (
                "Exact canonical URL/title evidence is checked first. Otherwise "
                "a semantic match must also share an event-specific multiword "
                "anchor or at least two discriminative terms after generic "
                "people, institutions, and procedural language are excluded."
            ),
            "post_split_deduplication_definition": (
                "After target-date developments are split, duplicate event candidates "
                "are merged using family overlap plus strict semantic identity and "
                "action/outcome agreement."
            ),
            "ingestion_preview_definition": (
                "The bounded preview includes NEW_DISCOVERY and "
                "UNDERREPRESENTED_UPSTREAM events. It emits only target-date English "
                "URLs absent from the normalized local corpus and never appends them "
                "to production data."
            ),
            "shadow_ranking_definition": (
                "An exploratory blended ranking combines normalized local attention "
                "and GDELT global breadth. It is diagnostic only and never changes "
                "the production page order."
            ),
            "thresholds": {
                "minimum_gsg_similarity": args.min_sim,
                "minimum_shared_words": args.min_shared_words,
                "minimum_target_day_domains": args.min_domains,
                "near_syndication_lead_similarity": args.family_lead_sim,
                "near_syndication_title_jaccard": args.family_title_jaccard,
                "duplicate_seed_merge_similarity": args.event_merge_sim,
                "target_day_development_similarity": args.target_cluster_sim,
                "adjacent_day_support_similarity": args.adjacent_support_sim,
                "minimum_event_title_cohesion_p20": args.min_event_cohesion,
            },
        },
        "limits": {
            "max_files": limits.max_files,
            "max_new_download_mb": args.max_download_mb,
            "max_accepted_edge_records": limits.max_edges,
            "max_raw_candidates": limits.max_raw_candidates,
            "top_n": args.top_n,
            "max_ingestion_candidates": args.max_ingestion_candidates,
            "max_total_ingestion_article_records": args.max_ingestion_records,
            "net_new_representatives_per_ingestion_candidate": args.max_representatives,
        },
        "download": download_stats,
        "parse": parse_stats,
        "writeup_families": family_stats,
        "candidate_pipeline": {
            "raw_seed_candidate_count": len(raw_candidates),
            "canonical_seed_group_count": len(canonical_groups),
            "target_anchored_development_count_before_quality_gate": len(split_groups),
            "cohesion_or_target_domain_rejected_count": cohesion_rejected,
            "post_split_deduplication": post_split_dedup_stats,
            "canonical_candidate_count": len(cleaned),
            "development_split_stats": dict(split_stats),
        },
        "references": reference_stats,
        "comparison": comparison_stats,
        "shadow_ranking_stats": shadow_ranking_stats,
        "shadow_blended_ranking": shadow_ranking,
        "ingestion_preview_stats": ingestion_preview_stats,
        "ingestion_preview": ingestion_preview,
        "runtime_seconds": elapsed,
        "candidates": cleaned,
    }
    if global_source_receipt_catalog is not None:
        output["global_source_receipt_catalog"] = (
            global_source_receipt_catalog
        )

    output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_csv, cleaned)

    print(
        f"🔗 Parsed {parse_stats.get('unique_edges', 0):,} unique cross-outlet "
        f"edges across {parse_stats.get('nodes', 0):,} URLs"
    )
    print(
        f"🧾 Collapsed into {family_stats.get('family_count', 0):,} "
        f"write-up families"
    )
    print(
        f"🧭 {len(raw_candidates):,} seed groups → "
        f"{len(canonical_groups):,} canonical seed groups → "
        f"{len(split_groups):,} target-date developments → "
        f"{post_split_dedup_stats.get('output_candidate_count', 0):,} "
        f"post-split unique events → top {len(cleaned)} candidates"
    )
    if post_split_dedup_stats.get("merged_candidate_count", 0):
        print(
            f"🧹 Final event dedup merged "
            f"{post_split_dedup_stats['merged_candidate_count']} duplicate candidates "
            f"across {post_split_dedup_stats['merged_group_count']} groups"
        )
    if not normalized_stats.get("normalized_file_found"):
        print(
            f"⚠️ Missing {normalized_path}; net-new ingestion preview was disabled."
        )

    for rank, candidate in enumerate(cleaned, start=1):
        preview_mark = (
            f" preview#{candidate.get('ingestion_preview_rank')}:"
            f"{candidate.get('ingestion_preview_type')}"
            if candidate.get("ingestion_preview_rank")
            else ""
        )
        print(
            f"  {rank:>2}. score={candidate['discovery_score']:>6.2f} | "
            f"target={candidate['target_day_writeup_family_count']:>2} write-ups/"
            f"{candidate['target_day_domain_count']:>3} outlets | "
            f"adj={candidate['adjacent_support_family_count']:>2} | "
            f"coh={candidate['target_title_cohesion_p20']:.2f} | "
            f"{candidate['pipeline_status']:<27}{preview_mark} | "
            f"{candidate['canonical_title'][:66]}"
        )

    if global_source_receipt_catalog is not None:
        print(
            f"🔗 GSG receipt catalog: "
            f"{global_source_receipt_catalog['all_language_receipt_count']:,} "
            f"outlet receipts across "
            f"{global_source_receipt_catalog['candidate_count']} candidates "
            f"({global_source_receipt_catalog['english_receipt_count']:,} English; "
            f"{global_source_receipt_catalog['non_english_receipt_count']:,} non-English)"
        )
        if global_source_receipt_catalog["incomplete_candidate_count"]:
            print(
                f"⚠️ Receipt catalog incomplete for "
                f"{global_source_receipt_catalog['incomplete_candidate_count']} "
                f"candidate(s); aggregate counts remain authoritative."
            )

    print(
        f"📥 Net-new ingestion preview: {ingestion_preview_stats['candidate_count']} "
        f"candidate events / {ingestion_preview_stats['article_record_count']} "
        f"English records | types="
        f"{ingestion_preview_stats.get('preview_type_counts', {})}"
    )
    print(f"✅ Wrote audit → {output_json}")
    print(f"✅ Wrote review table → {output_csv}")
    print(f"⏱️ Runtime: {elapsed}s")
    if args.pipeline_mode:
        print(
            "🛡️ Candidate feed only; no baseline cluster or downstream "
            "production file was modified."
        )
    else:
        print("🛡️ No production JSON or pipeline stage was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
