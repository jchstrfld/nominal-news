# final_cohesion_check.py — high-precision, token-efficient final pass
# - Loads grouped_articles_filtered_{date}.json by default
# - Supports optional --input-file / --output-file overrides for isolated shadow tests
# - Can write an optional post-purity GDELT global-ranking shadow comparison
# - Runs exact + semantic de-duplication (token-free)
# - Optionally validates clusters with GPT (can be skipped via --no-openai)
# - Adds source diversity & bias distribution
# - Caps to top-K clusters
# - Writes grouped_articles_final_{date}.json
#
# Usage examples:
#   python final_cohesion_check.py --date 2025-09-24 --no-openai --top-k 10 --print-report
#   python final_cohesion_check.py --date 2025-09-24
#   python final_cohesion_check.py --date 2025-09-24 --input-file shadow_filtered.json --output-file shadow_final.json
#   python final_cohesion_check.py --date 2025-09-24 --gdelt-ranking-shadow
#   python final_cohesion_check.py --date 2025-09-24 --gdelt-ranking-shadow --gdelt-audit-file gdelt_discovery_candidates_2025-09-24.json
#
# Notes:
# - No OpenAI tokens are used if you pass --no-openai (or no key is present).
# - To see changes on your webpage (index.html) you typically need to re-run summarization,
#   because the page reads topic_summaries_{date}.json. See run notes at the end of file.

from __future__ import annotations

import json
import os
import copy
import sys
import re
import hashlib
import importlib.util
from bisect import bisect_right
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
    purity_report = False
    input_file = None
    output_file = None
    gdelt_ranking_shadow = False
    gdelt_ranking_shadow_file = None
    gdelt_audit_file = None
    gdelt_global_output_file = None

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
        elif a == "--purity-report":
            purity_report = True
            i += 1
        elif a == "--input-file":
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                print("❌ No value provided after --input-file")
                sys.exit(1)
            input_file = args[i + 1]
            i += 2
        elif a == "--output-file":
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                print("❌ No value provided after --output-file")
                sys.exit(1)
            output_file = args[i + 1]
            i += 2
        elif a == "--gdelt-ranking-shadow":
            gdelt_ranking_shadow = True
            i += 1
        elif a == "--gdelt-ranking-shadow-file":
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                print("❌ No value provided after --gdelt-ranking-shadow-file")
                sys.exit(1)
            gdelt_ranking_shadow_file = args[i + 1]
            gdelt_ranking_shadow = True
            i += 2
        elif a == "--gdelt-audit-file":
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                print("❌ No value provided after --gdelt-audit-file")
                sys.exit(1)
            gdelt_audit_file = args[i + 1]
            gdelt_ranking_shadow = True
            i += 2
        elif a == "--gdelt-global-output-file":
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                print("❌ No value provided after --gdelt-global-output-file")
                sys.exit(1)
            gdelt_global_output_file = args[i + 1]
            gdelt_ranking_shadow = True
            i += 2
        else:
            i += 1

    if not date_str:
        date_str = datetime.today().strftime("%Y-%m-%d")

    return (
        date_str,
        no_openai,
        top_k,
        print_report,
        purity_report,
        input_file,
        output_file,
        gdelt_ranking_shadow,
        gdelt_ranking_shadow_file,
        gdelt_audit_file,
        gdelt_global_output_file,
    )

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
EVENT_MAX_CALLS = int(os.getenv("NN_EVENT_MAX_CALLS", "48"))  # hard ceiling; rank-safe stopping usually ends earlier
MIN_PUBLISH_ARTICLES = int(os.getenv("NN_MIN_PUBLISH_ARTICLES", "4"))
MIN_PUBLISH_DOMAINS = int(os.getenv("NN_MIN_PUBLISH_DOMAINS", "3"))
_EVENT_TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
_EVENT_FINISH_REASONS = Counter()
EVENT_CACHE_FILE = os.getenv("NN_EVENT_CACHE_FILE", "eventness_cache.json")

# Bump whenever eventness prompt/acceptance semantics change.
EVENT_CACHE_VERSION = "v6-discrete-event-anchor-2026-09"

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

HIGH_TRUST_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "nytimes.com",
    "bbc.com",
    "wsj.com",
    "ft.com",
    "economist.com",
    "washingtonpost.com",
    "npr.org",
    "abcnews.go.com",
    "cbsnews.com",
    "nbcnews.com",
    "theguardian.com",
    "dw.com",
    "aljazeera.com",
}


def attention_metadata_from_articles(articles: list[dict]) -> dict:
    """
    Ranking-only attention footprint captured from the immediate pre-final
    (grouped_articles_filtered) parent cluster.

    This metadata never changes event membership and never adds articles back
    after purity filtering.
    """
    urls = []
    domains = []

    for a in articles:
        raw_url = a.get("url_normalized") or a.get("url") or ""
        if not raw_url:
            continue

        key = canonicalize_url(raw_url)
        if key:
            urls.append(key)

        dom = domain_from_url(raw_url)
        if dom:
            domains.append(dom)

    return {
        "_attention_urls": sorted(set(urls)),
        "_attention_domains": sorted(set(domains)),
        "attention_article_count": len(set(urls)) if urls else len(articles),
        "attention_domain_count": len(set(domains)),
    }


def merge_attention_metadata(into: dict, src: dict) -> None:
    """
    Union attention lineage when two candidate clusters are merged.
    Ranking metadata only; does not affect purity/article membership.
    """
    urls = set(into.get("_attention_urls", []))
    urls.update(src.get("_attention_urls", []))

    domains = set(into.get("_attention_domains", []))
    domains.update(src.get("_attention_domains", []))

    into["_attention_urls"] = sorted(urls)
    into["_attention_domains"] = sorted(domains)

    if urls:
        into["attention_article_count"] = len(urls)
    else:
        into["attention_article_count"] = max(
            int(into.get("attention_article_count", 0) or 0),
            int(src.get("attention_article_count", 0) or 0),
        )

    into["attention_domain_count"] = len(domains)


def attention_score(cluster: dict) -> float:
    """
    Final page-order score for already-validated events.

    Primary signal: breadth in the immediate pre-final filtered lineage.
      - unique domains dominate
      - article volume is secondary
      - verified-core breadth/size provide a small confirmation signal

    This score is ranking-only. Purity/event validation still determines which
    clusters exist and which articles belong to them.
    """
    arts = cluster.get("articles", [])
    verified_size = len(arts)

    verified_div = cluster.get("source_diversity") or {}
    verified_domains = int(verified_div.get("unique_domains", 0) or 0)

    attention_articles = int(
        cluster.get("attention_article_count", verified_size) or verified_size
    )
    attention_domains = int(
        cluster.get("attention_domain_count", verified_domains) or verified_domains
    )

    score = 0.0
    score += attention_domains * 4.0
    score += min(attention_articles, 20) * 0.75

    # Small verification-strength confirmation; deliberately subordinate to
    # the pre-final attention footprint.
    score += verified_domains * 0.25
    score += min(verified_size, 12) * 0.10

    return round(score, 4)



def _clamp01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _local_attention_percentile(score: float, sorted_reference_scores: list[float]) -> float:
    """
    Map a raw local attention score onto the fixed pre-review candidate
    distribution. The reference distribution never changes during purification,
    so a candidate's pre-clean value remains a valid upper bound.
    """
    if not sorted_reference_scores:
        return 0.0
    try:
        value = float(score)
    except Exception:
        value = 0.0
    return _clamp01(bisect_right(sorted_reference_scores, value) / len(sorted_reference_scores))


def _gdelt_global_attention_from_articles(
    articles: list[dict],
    *,
    min_retained_articles: int = 2,
) -> dict:
    """
    Recover event-level GDELT attention metadata only from discovery articles
    that remain attached to the candidate/event.

    Requiring at least two retained representatives from the same GDELT
    candidate prevents an accidental single-article merge from granting global
    ranking credit. The publication purifier remains the authority on event
    eligibility.
    """
    by_candidate: dict[str, dict] = {}

    for article in articles or []:
        candidate_id = str(article.get("gdelt_candidate_id") or "").strip()
        meta = article.get("gdelt_global_attention")
        if not candidate_id or not isinstance(meta, dict):
            continue

        rec = by_candidate.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "canonical_title": str(
                    article.get("gdelt_canonical_title") or ""
                ).strip(),
                "preview_type": str(
                    article.get("gdelt_preview_type") or ""
                ).strip(),
                "retained_article_urls": set(),
                "discovery_percentile": 0.0,
                "discovery_score": 0.0,
                "target_day_writeup_family_count": 0,
                "target_day_outlet_count": 0,
                "all_supporting_writeup_family_count": 0,
                "all_supporting_outlet_count": 0,
                "active_window_count": 0,
                "language_count": 0,
                "cross_language_url_count": 0,
                "target_date": str(meta.get("target_date") or "").strip(),
            },
        )

        article_url = canonicalize_url(
            article.get("url_normalized") or article.get("url") or ""
        )
        if article_url:
            rec["retained_article_urls"].add(article_url)

        if not rec["canonical_title"]:
            rec["canonical_title"] = str(
                article.get("gdelt_canonical_title") or ""
            ).strip()
        if not rec["preview_type"]:
            rec["preview_type"] = str(
                article.get("gdelt_preview_type") or ""
            ).strip()

        for key in [
            "discovery_percentile",
            "discovery_score",
            "target_day_writeup_family_count",
            "target_day_outlet_count",
            "all_supporting_writeup_family_count",
            "all_supporting_outlet_count",
            "active_window_count",
            "language_count",
            "cross_language_url_count",
        ]:
            try:
                rec[key] = max(float(rec.get(key, 0) or 0), float(meta.get(key, 0) or 0))
            except Exception:
                pass

    eligible = []
    for rec in by_candidate.values():
        retained_count = len(rec.pop("retained_article_urls", set()))
        rec["retained_article_count"] = retained_count
        rec["discovery_percentile"] = _clamp01(rec.get("discovery_percentile", 0.0))

        for key in [
            "target_day_writeup_family_count",
            "target_day_outlet_count",
            "all_supporting_writeup_family_count",
            "all_supporting_outlet_count",
            "active_window_count",
            "language_count",
            "cross_language_url_count",
        ]:
            rec[key] = int(rec.get(key, 0) or 0)

        rec["discovery_score"] = round(float(rec.get("discovery_score", 0.0) or 0.0), 3)
        if retained_count >= max(1, min_retained_articles):
            eligible.append(rec)

    eligible.sort(
        key=lambda item: (
            float(item.get("discovery_percentile", 0.0) or 0.0),
            float(item.get("discovery_score", 0.0) or 0.0),
            int(item.get("target_day_writeup_family_count", 0) or 0),
            int(item.get("target_day_outlet_count", 0) or 0),
        ),
        reverse=True,
    )

    primary = eligible[0] if eligible else None
    return {
        "eligible": bool(primary),
        "minimum_retained_articles": max(1, min_retained_articles),
        "global_signal": (
            round(float(primary.get("discovery_percentile", 0.0)), 4)
            if primary else None
        ),
        "primary_candidate_id": (
            primary.get("candidate_id") if primary else None
        ),
        "candidates": eligible,
    }



_GDELT_RUNTIME_MODULE = None
_GDELT_RUNTIME_ERROR = None
GDELT_CATALOG_POTENTIAL_MATCH_THRESHOLD = 0.48
GDELT_CATALOG_FINAL_MATCH_THRESHOLD = 0.54


def _load_gdelt_runtime_module():
    """Lazy-load the adjacent audit's reusable matcher only in shadow mode."""
    global _GDELT_RUNTIME_MODULE, _GDELT_RUNTIME_ERROR
    if _GDELT_RUNTIME_MODULE is not None:
        return _GDELT_RUNTIME_MODULE
    if _GDELT_RUNTIME_ERROR is not None:
        return None

    path = Path(__file__).resolve().parent / "audit_gdelt_global_discovery.py"
    if not path.exists():
        _GDELT_RUNTIME_ERROR = f"missing {path.name}"
        return None

    module_name = "_nominal_news_gdelt_runtime_matcher"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to create module spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        required = [
            "runtime_event_payload_from_cluster",
            "prepare_runtime_event_matcher",
            "runtime_event_match_options",
            "assign_runtime_event_matches",
        ]
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError("missing runtime matcher API: " + ", ".join(missing))
        _GDELT_RUNTIME_MODULE = module
        return module
    except Exception as exc:
        sys.modules.pop(module_name, None)
        _GDELT_RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def _load_gdelt_audit_payload(
    date_str: str,
    override_path: str | None,
) -> tuple[dict | None, dict]:
    if override_path:
        paths = [Path(override_path)]
    else:
        paths = [
            Path(f"gdelt_discovery_candidates_{date_str}.json"),
            Path(f"gdelt_global_discovery_audit_{date_str}.json"),
        ]
        paths.sort(
            key=lambda path: path.stat().st_mtime if path.exists() else -1.0,
            reverse=True,
        )

    errors = []
    for path in paths:
        if not path.exists():
            errors.append(f"missing:{path}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid_json:{path}:{type(exc).__name__}")
            continue
        payload_date = str(payload.get("date") or "").strip()
        if payload_date and payload_date != date_str:
            errors.append(f"date_mismatch:{path}:{payload_date}")
            continue
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            errors.append(f"missing_candidates:{path}")
            continue
        return payload, {
            "status": "OK",
            "file": str(path),
            "candidate_count": len(candidates),
            "audit_schema_version": payload.get("audit_schema_version"),
            "errors": errors,
        }

    return None, {
        "status": "UNAVAILABLE",
        "file": None,
        "candidate_count": 0,
        "errors": errors,
    }


def _gdelt_option_evidence(option: dict | None, catalog_file: str | None) -> dict:
    if not option:
        return {
            "eligible": False,
            "matching_mode": "FULL_AUDIT_CATALOG",
            "catalog_file": catalog_file,
            "global_signal": None,
            "primary_candidate_id": None,
            "candidates": [],
        }
    clean = {key: value for key, value in option.items() if not key.startswith("_")}
    return {
        "eligible": True,
        "matching_mode": "FULL_AUDIT_CATALOG",
        "catalog_file": catalog_file,
        "global_signal": round(float(clean.get("discovery_percentile", 0.0) or 0.0), 4),
        "primary_candidate_id": clean.get("candidate_id"),
        "candidates": [clean],
    }


def _combine_gdelt_shadow_evidence(
    retained_evidence: dict | None,
    catalog_evidence: dict | None,
) -> dict:
    retained_evidence = retained_evidence or {}
    catalog_evidence = catalog_evidence or {}
    rows = []
    for evidence in [retained_evidence, catalog_evidence]:
        if evidence.get("eligible"):
            rows.extend(
                dict(row) for row in (evidence.get("candidates") or [])
                if isinstance(row, dict)
            )
    if not rows:
        return catalog_evidence or retained_evidence

    by_id = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        previous = by_id.get(candidate_id)
        if previous is None:
            by_id[candidate_id] = row
            continue
        chosen = row if row.get("match_source") else previous
        other = previous if chosen is row else row
        chosen = dict(chosen)
        chosen["retained_article_count"] = max(
            int(chosen.get("retained_article_count", 0) or 0),
            int(other.get("retained_article_count", 0) or 0),
        )
        by_id[candidate_id] = chosen

    merged = list(by_id.values())
    merged.sort(
        key=lambda row: (
            int(row.get("match_basis") == "CANDIDATE_ID"),
            int(row.get("retained_article_count", 0) or 0),
            float(row.get("discovery_percentile", 0.0) or 0.0),
            float(row.get("match_similarity", 0.0) or 0.0),
        ),
        reverse=True,
    )
    primary = merged[0]
    return {
        "eligible": True,
        "matching_mode": (
            "RETAINED_ARTICLES_AND_FULL_CATALOG"
            if retained_evidence.get("eligible") and catalog_evidence.get("eligible")
            else (
                "FULL_AUDIT_CATALOG"
                if catalog_evidence.get("eligible")
                else "RETAINED_DISCOVERY_ARTICLES"
            )
        ),
        "catalog_file": catalog_evidence.get("catalog_file"),
        "global_signal": round(float(primary.get("discovery_percentile", 0.0) or 0.0), 4),
        "primary_candidate_id": primary.get("candidate_id"),
        "candidates": merged,
    }


def _prepare_gdelt_full_catalog_matcher(
    date_str: str,
    audit_override: str | None,
    clusters: list[dict],
) -> tuple[object | None, dict | None, dict]:
    payload, diag = _load_gdelt_audit_payload(date_str, audit_override)
    module = _load_gdelt_runtime_module()
    if payload is None or module is None:
        if module is None:
            diag["matcher_status"] = "UNAVAILABLE"
            diag["matcher_error"] = _GDELT_RUNTIME_ERROR
        return module, None, diag

    event_payloads = [
        module.runtime_event_payload_from_cluster(
            cluster,
            event_id=_shadow_event_id(cluster),
        )
        for cluster in clusters
    ]
    matcher = module.prepare_runtime_event_matcher(
        payload,
        event_payloads,
        _get_sem_embedder(),
    )
    diag["matcher_status"] = matcher.get("semantic_mode") or matcher.get("status")
    diag["match_threshold_potential"] = GDELT_CATALOG_POTENTIAL_MATCH_THRESHOLD
    diag["match_threshold_final"] = GDELT_CATALOG_FINAL_MATCH_THRESHOLD
    return module, matcher, diag


def _gdelt_catalog_options_for_cluster(
    cluster: dict,
    module,
    matcher: dict | None,
    *,
    threshold: float,
    max_matches: int,
) -> list[dict]:
    if module is None or not matcher or matcher.get("status") != "OK":
        return []
    payload = module.runtime_event_payload_from_cluster(
        cluster,
        event_id=_shadow_event_id(cluster),
    )
    return module.runtime_event_match_options(
        payload,
        matcher,
        threshold=threshold,
        max_matches=max_matches,
    )


def _resolve_all_approved_gdelt_matches(
    approved_events: list[dict],
    module,
    matcher: dict | None,
    local_reference_scores: list[float],
    *,
    apply: bool,
) -> dict:
    """Enforce one full-audit candidate per approved event and vice versa."""
    event_payloads = []
    for event in approved_events:
        payload = (
            module.runtime_event_payload_from_cluster(
                event,
                event_id=_shadow_event_id(event),
            )
            if module is not None else None
        )
        if payload is not None:
            event_payloads.append(payload)

    assignment_result = (
        module.assign_runtime_event_matches(
            event_payloads,
            matcher,
            threshold=GDELT_CATALOG_FINAL_MATCH_THRESHOLD,
            max_matches_per_event=5,
        )
        if module is not None and matcher and matcher.get("status") == "OK"
        else {"assignments": {}, "stats": {}}
    )
    assignments = assignment_result.get("assignments") or {}
    used_candidate_ids = {
        option.get("candidate_id")
        for option in assignments.values()
        if option.get("candidate_id")
    }

    by_event = {}
    for event in approved_events:
        event_id = _shadow_event_id(event)
        option = assignments.get(event_id)
        catalog_evidence = _gdelt_option_evidence(
            option,
            (matcher or {}).get("catalog_file"),
        )
        retained = _gdelt_global_attention_from_articles(
            event.get("related_articles", event.get("articles", [])),
            min_retained_articles=2,
        )

        if option:
            assigned_id = option.get("candidate_id")
            same_rows = [
                row for row in (retained.get("candidates") or [])
                if row.get("candidate_id") == assigned_id
            ]
            retained = {
                "eligible": bool(same_rows),
                "global_signal": retained.get("global_signal") if same_rows else None,
                "primary_candidate_id": assigned_id if same_rows else None,
                "candidates": same_rows,
            }
        elif retained.get("eligible"):
            retained_id = retained.get("primary_candidate_id")
            if not retained_id or retained_id in used_candidate_ids:
                retained = {
                    "eligible": False,
                    "global_signal": None,
                    "primary_candidate_id": None,
                    "candidates": [],
                }
            else:
                used_candidate_ids.add(retained_id)

        evidence = _combine_gdelt_shadow_evidence(retained, catalog_evidence)
        local_signal = _local_attention_percentile(
            float(event.get("attention_score", 0.0) or 0.0),
            local_reference_scores,
        )
        global_signal = evidence.get("global_signal") if evidence.get("eligible") else None
        shadow_score = _gdelt_shadow_blended_score(local_signal, global_signal)
        by_event[event_id] = {
            "local_signal": local_signal,
            "global_signal": global_signal,
            "shadow_score": shadow_score,
            "evidence": evidence,
        }
        if apply:
            event["_gdelt_shadow_local_signal"] = local_signal
            event["_gdelt_shadow_global_signal"] = global_signal
            event["_gdelt_shadow_score"] = shadow_score
            event["_gdelt_shadow_evidence"] = evidence

    stats = dict(assignment_result.get("stats") or {})
    stats.update({
        "approved_event_count": len(approved_events),
        "matched_approved_event_count": sum(
            1 for row in by_event.values() if row["evidence"].get("eligible")
        ),
        "unmatched_approved_event_count": sum(
            1 for row in by_event.values() if not row["evidence"].get("eligible")
        ),
    })
    return {"by_event": by_event, "stats": stats}

def _gdelt_shadow_blended_score(
    local_signal: float,
    global_signal: float | None,
) -> float:
    """
    Shadow-only ranking:
      - no GDELT evidence: preserve 100% of the local signal
      - both signals: 80% of the stronger signal + 20% confirmation from the
        weaker signal

    Absence from the bounded GDELT sample never penalizes a locally prominent
    event.
    """
    local = _clamp01(local_signal)
    if global_signal is None:
        return round(local * 100.0, 3)

    global_value = _clamp01(global_signal)
    stronger = max(local, global_value)
    weaker = min(local, global_value)
    return round((0.80 * stronger + 0.20 * weaker) * 100.0, 3)


def _shadow_event_id(cluster: dict) -> str:
    signature = _cluster_sig_urls(cluster)
    if not signature:
        signature = str(cluster.get("canonical_event") or cluster.get("topic") or "")
    return "event_" + hashlib.sha1(signature.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _shadow_event_title(cluster: dict) -> str:
    canonical = str(cluster.get("canonical_event") or "").strip()
    if canonical:
        return canonical
    for article in cluster.get("articles", []):
        title = str(article.get("title") or "").strip()
        if title:
            return title
    return str(cluster.get("topic") or "Untitled event")


def _write_gdelt_ranking_shadow(
    *,
    path: str,
    date_str: str,
    input_file: str,
    output_file: str,
    approved_events: list[dict],
    top_k: int,
    review_queue_size: int,
    reviewed_candidates: int,
    event_calls: int,
    event_cache_hits: int,
    purifier_stop_reason: str,
    gdelt_catalog_diag: dict,
    gdelt_assignment_stats: dict,
    global_output_file: str | None = None,
) -> None:
    """Write a separate comparison file; never alter production page order."""
    local_sorted = sorted(
        approved_events,
        key=lambda item: (
            float(item.get("attention_score", 0.0) or 0.0),
            importance_score(item),
        ),
        reverse=True,
    )
    shadow_sorted = sorted(
        approved_events,
        key=lambda item: (
            float(item.get("_gdelt_shadow_score", 0.0) or 0.0),
            float(item.get("attention_score", 0.0) or 0.0),
            importance_score(item),
        ),
        reverse=True,
    )

    local_rank = {_shadow_event_id(c): i for i, c in enumerate(local_sorted, start=1)}
    shadow_rank = {_shadow_event_id(c): i for i, c in enumerate(shadow_sorted, start=1)}

    def row(cluster: dict) -> dict:
        event_id = _shadow_event_id(cluster)
        evidence = cluster.get("_gdelt_shadow_evidence") or {
            "eligible": False,
            "global_signal": None,
            "primary_candidate_id": None,
            "candidates": [],
        }
        core_articles = cluster.get("articles", [])
        related_articles = cluster.get("related_articles", core_articles)
        core_gdelt = sum(
            1 for a in core_articles if a.get("origin") == "gdelt_discovery"
        )
        related_gdelt = sum(
            1 for a in related_articles if a.get("origin") == "gdelt_discovery"
        )

        if core_gdelt and core_gdelt == len(core_articles):
            source_type = "GDELT_DISCOVERY"
        elif evidence.get("eligible"):
            source_type = "LOCAL_WITH_GDELT_CONFIRMATION"
        else:
            source_type = "LOCAL_ONLY"

        return {
            "event_id": event_id,
            "title": _shadow_event_title(cluster),
            "topic": cluster.get("topic"),
            "source_type": source_type,
            "local_rank": local_rank.get(event_id),
            "global_shadow_rank": shadow_rank.get(event_id),
            "local_attention_score": round(
                float(cluster.get("attention_score", 0.0) or 0.0), 4
            ),
            "local_signal": round(
                float(cluster.get("_gdelt_shadow_local_signal", 0.0) or 0.0), 4
            ),
            "global_signal": cluster.get("_gdelt_shadow_global_signal"),
            "global_shadow_score": round(
                float(cluster.get("_gdelt_shadow_score", 0.0) or 0.0), 3
            ),
            "core_article_count": len(core_articles),
            "core_domain_count": int(
                (cluster.get("source_diversity") or {}).get("unique_domains", 0) or 0
            ),
            "related_article_count": len(related_articles),
            "related_domain_count": int(
                cluster.get("related_domain_count", 0) or 0
            ),
            "retained_gdelt_core_articles": core_gdelt,
            "retained_gdelt_related_articles": related_gdelt,
            "gdelt_global_attention": evidence,
        }

    rows_by_id = {
        _shadow_event_id(c): row(c)
        for c in approved_events
    }
    local_rows = [rows_by_id[_shadow_event_id(c)] for c in local_sorted]
    shadow_rows = [rows_by_id[_shadow_event_id(c)] for c in shadow_sorted]

    limit = max(1, top_k)
    local_top_ids = [_shadow_event_id(c) for c in local_sorted[:limit]]
    shadow_top_ids = [_shadow_event_id(c) for c in shadow_sorted[:limit]]
    local_top_set = set(local_top_ids)
    shadow_top_set = set(shadow_top_ids)

    promoted = [rows_by_id[eid] for eid in shadow_top_ids if eid not in local_top_set]
    displaced = [rows_by_id[eid] for eid in local_top_ids if eid not in shadow_top_set]

    rank_changes = []
    for event_id in sorted(local_top_set | shadow_top_set):
        item = rows_by_id[event_id]
        if item["local_rank"] != item["global_shadow_rank"]:
            rank_changes.append({
                "event_id": event_id,
                "title": item["title"],
                "local_rank": item["local_rank"],
                "global_shadow_rank": item["global_shadow_rank"],
                "rank_change": (
                    item["local_rank"] - item["global_shadow_rank"]
                    if item["local_rank"] is not None
                    and item["global_shadow_rank"] is not None
                    else None
                ),
            })

    rank_secure = purifier_stop_reason in {
        "top_k_local_and_global_rank_secured",
        "queue_exhausted_rank_complete",
    }

    payload = {
        "schema_version": "1.1",
        "date": date_str,
        "status": "OK",
        "shadow_only": True,
        "live_page_order_changed_by_shadow_score": False,
        "local_ranking_formula_unchanged": True,
        "input_file": input_file,
        "local_output_file": output_file,
        "global_output_file": global_output_file,
        "eligibility": (
            "Only purifier-approved DISCRETE_EVENT clusters are compared. "
            "GDELT cannot bypass the publication gate."
        ),
        "formula": {
            "local_only": "100 × local attention percentile",
            "local_and_global": (
                "100 × (0.80 × stronger(local, GDELT) + "
                "0.20 × weaker(local, GDELT))"
            ),
            "local_signal": (
                "Percentile of the event's local attention score in the fixed "
                "pre-review candidate distribution."
            ),
            "global_signal": (
                "GDELT discovery percentile assigned only after the event "
                "passes the publication purifier. Exact retained-candidate "
                "evidence is preferred; otherwise every approved event is "
                "matched against the full bounded audit catalog using the "
                "audit's strict identity/action/development gate."
            ),
            "absence_rule": (
                "No GDELT match does not reduce a local event's score."
            ),
        },
        "gdelt_catalog": {
            **(gdelt_catalog_diag or {}),
            "assignment": gdelt_assignment_stats or {},
            "one_candidate_per_approved_event": True,
            "one_approved_event_per_candidate": True,
        },
        "review": {
            "queue_candidate_count": review_queue_size,
            "reviewed_candidate_count": reviewed_candidates,
            "approved_event_count": len(approved_events),
            "live_calls": event_calls,
            "cache_hits": event_cache_hits,
            "stop_reason": purifier_stop_reason,
            "local_and_global_top_k_rank_secured": rank_secure,
        },
        "top_k": limit,
        "local_top_k": local_rows[:limit],
        "global_shadow_top_k": shadow_rows[:limit],
        "changes": {
            "promoted_into_global_top_k": promoted,
            "displaced_from_global_top_k": displaced,
            "rank_changes_within_union": rank_changes,
        },
        "all_purifier_approved_events_by_global_shadow_rank": shadow_rows,
    }

    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def importance_score(cluster: dict) -> float:
    """
    Final ranking score after purity filtering.
    Prioritizes broad, cross-outlet attention over merely coherent local/random events.
    Ranking-only: does not drop clusters.
    """
    arts = cluster.get("articles", [])
    size = len(arts)

    div = cluster.get("source_diversity") or {}
    uniq_domains = int(div.get("unique_domains", 0) or 0)
    entropy = float(div.get("entropy", 0.0) or 0.0)
    source_conc = float(cluster.get("source_concentration", 0.0) or 0.0)

    bias_dist = cluster.get("bias_distribution") or {}
    known_biases = [k for k in bias_dist.keys() if k != "Unknown"]
    bias_breadth = len(known_biases)

    today_ratio = float(cluster.get("today_ratio", 1.0) or 1.0)

    high_trust_hits = 0
    times = []
    for a in arts:
        raw_url = a.get("url_normalized") or a.get("url") or ""
        dom = urlparse(raw_url).netloc.replace("www.", "").lower()
        if dom in HIGH_TRUST_DOMAINS:
            high_trust_hits += 1

        raw = a.get("published_at") or ""
        try:
            times.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        except Exception:
            pass

    if len(times) >= 2:
        span_hours = max(1.0, (max(times) - min(times)).total_seconds() / 3600)
        velocity = min(3.0, len(times) / span_hours)
    else:
        velocity = 0.0

    score = 0.0
    score += min(size, 12) * 0.8
    score += uniq_domains * 1.15
    score += entropy * 1.4
    score += bias_breadth * 0.9
    score += velocity * 1.2
    score += today_ratio * 2.0
    score += min(high_trust_hits, 6) * 0.9

    if source_conc >= 0.40:
        score -= 1.5
    if source_conc >= 0.60:
        score -= 2.5

    if cluster.get("eventness_label") in {"SINGLE_EVENT", "SINGLE_EVENT_MATH"}:
        score += 1.0

    return score

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
                merge_attention_metadata(clusters[w], clusters[l])
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

def purify_cluster_with_gpt(articles: list[dict]) -> dict:
    """
    One bounded publication-gate call for one candidate cluster.

    The model must first distinguish:
      - DISCRETE_EVENT: one current event/development can be stated as a single
        neutral "What happened now?" sentence.
      - TOPIC_WAVE: a shared subject, anniversary, conflict state, person,
        institution, or collection of separate developments rather than one
        publishable current event.

    For a DISCRETE_EVENT it returns two nested memberships:
      - core_indexes: strict same-event reporting used for summarization
      - related_indexes: CORE plus direct reactions, immediate consequences,
        tributes, and analysis explicitly anchored to that exact event

    KEEP versus CLEAN is derived from returned membership; the model does not
    decide it. The call both purifies and validates. There is no second GPT call.
    """
    if openai is None or not os.getenv("OPENAI_API_KEY"):
        return {
            "action": "ERROR",
            "event_type": "",
            "event": "",
            "core_indexes": [],
            "related_indexes": [],
            "why": "OpenAI disabled",
        }

    indexed = []
    for i, article in enumerate(articles, start=1):
        title = (article.get("title") or "").strip()
        if not title:
            continue
        domain = domain_from_url(
            article.get("url_normalized") or article.get("url") or ""
        ) or "unknown-domain"
        indexed.append((i, domain, title))

    if len(indexed) < MIN_PUBLISH_ARTICLES:
        return {
            "action": "REJECT",
            "event_type": "TOPIC_WAVE",
            "event": "",
            "core_indexes": [],
            "related_indexes": [],
            "why": "Too few titled articles",
        }

    titles_block = "\n".join(
        f"[{i}] ({domain}) {title}"
        for i, domain, title in indexed
    )

    prompt = (
        "You are the final publication gate for a high-precision news briefing.\n"
        "First classify the candidate as exactly one of these:\n"
        "DISCRETE_EVENT: one identifiable current event or development that can be stated as one neutral "
        "'What happened now?' sentence.\n"
        "TOPIC_WAVE: a shared topic, anniversary, historical subject, person, institution, country, conflict, "
        "or multiple separate developments without one qualifying current event.\n\n"
        "If noisy titles contain one qualifying DISCRETE_EVENT, salvage that event by selecting only its memberships. "
        "Do not reject merely because unrelated titles are present.\n\n"
        "CORE titles must independently report the same discrete event and be safe to summarize together. "
        "Every CORE title must answer the same 'What happened now?' sentence.\n"
        "RELATED titles may include CORE plus direct reactions, immediate consequences, tributes, or analysis "
        "explicitly anchored to that exact event. A title is not RELATED merely because it shares a person, "
        "country, war, institution, political issue, or broad subject.\n\n"
        "Anniversary coverage, memorial collections, archival articles, historical retrospectives, or broad conflict "
        "status are TOPIC_WAVE unless the titles concern the same current ceremony, release, filing, finding, strike, "
        "ruling, briefing, agreement, announcement, or similarly identifiable development. Separate attacks, "
        "lawsuits, policy actions, and other incidents remain separate events.\n\n"
        "Choose the discrete event with the greatest distinct-domain support; use article count only as a tie-breaker. "
        f"The CORE must contain at least {MIN_PUBLISH_ARTICLES} titles from at least "
        f"{MIN_PUBLISH_DOMAINS} distinct outlet domains.\n"
        "For DISCRETE_EVENT, populate both index arrays using the numbered titles; CORE must be a subset of RELATED. "
        "For TOPIC_WAVE, return both arrays empty.\n\n"
        "Return exactly one compact JSON object with these keys and no markdown: "
        "event_type, event, core_indexes, related_indexes, why. "
        "Allowed event_type values are DISCRETE_EVENT and TOPIC_WAVE.\n\n"
        "Numbered titles:\n" + titles_block
    )

    try:
        resp = openai.ChatCompletion.create(
            model=EVENT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Be strict, literal, concise, and return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=180,
            request_timeout=45,
        )
    except Exception as exc:
        return {
            "action": "ERROR",
            "event_type": "",
            "event": "",
            "core_indexes": [],
            "related_indexes": [],
            "why": f"OpenAI validation error: {type(exc).__name__}",
        }

    choice = resp.choices[0]
    finish_reason = str(getattr(choice, "finish_reason", None) or "unknown")
    _EVENT_FINISH_REASONS[finish_reason] += 1

    usage = getattr(resp, "usage", None)
    if usage is not None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                _EVENT_TOKEN_USAGE[key] += int(usage.get(key, 0) or 0)
            except Exception:
                pass

    if finish_reason != "stop":
        return {
            "action": "ERROR",
            "event_type": "",
            "event": "",
            "core_indexes": [],
            "related_indexes": [],
            "why": f"Unexpected finish_reason={finish_reason}",
            "finish_reason": finish_reason,
        }

    content = (choice.message["content"] or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        left = content.index("{")
        right = content.rindex("}") + 1
        obj = json.loads(content[left:right])
    except Exception:
        return {
            "action": "ERROR",
            "event_type": "",
            "event": "",
            "core_indexes": [],
            "related_indexes": [],
            "why": "Invalid structured validation response",
            "finish_reason": finish_reason,
        }

    event_type = str(obj.get("event_type") or "").strip().upper()
    if event_type not in {"DISCRETE_EVENT", "TOPIC_WAVE"}:
        return {
            "action": "ERROR",
            "event_type": event_type,
            "event": "",
            "core_indexes": [],
            "related_indexes": [],
            "why": "Invalid event_type",
            "finish_reason": finish_reason,
        }

    valid_article_indexes = {i for i, _, _ in indexed}

    def normalize_indexes(values) -> list[int]:
        out = []
        for value in values or []:
            try:
                idx = int(value)
            except Exception:
                continue
            if idx in valid_article_indexes and idx not in out:
                out.append(idx)
        return sorted(out)

    core_indexes = normalize_indexes(obj.get("core_indexes"))
    related_indexes = normalize_indexes(obj.get("related_indexes"))
    event = str(obj.get("event") or "").strip()[:300]
    why = str(obj.get("why") or "").strip()[:300]

    if event_type == "TOPIC_WAVE":
        return {
            "action": "REJECT",
            "event_type": event_type,
            "event": event,
            "core_indexes": [],
            "related_indexes": [],
            "why": why or "No qualifying discrete current event",
            "finish_reason": finish_reason,
        }

    # DISCRETE_EVENT: trust only the memberships explicitly returned.
    related_indexes = sorted(set(related_indexes) | set(core_indexes))
    if not event or not core_indexes or not related_indexes:
        return {
            "action": "ERROR",
            "event_type": event_type,
            "event": event,
            "core_indexes": core_indexes,
            "related_indexes": related_indexes,
            "why": why or "Incomplete discrete-event membership",
            "finish_reason": finish_reason,
        }

    action = (
        "KEEP"
        if set(related_indexes) == valid_article_indexes
        else "CLEAN"
    )

    return {
        "action": action,
        "event_type": event_type,
        "event": event,
        "core_indexes": core_indexes,
        "related_indexes": related_indexes,
        "why": why or "Discrete event identified",
        "finish_reason": finish_reason,
    }


_RELATED_ANCHOR_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "after", "before",
    "into", "over", "under", "says", "said", "say", "new", "latest", "live",
    "amid", "about", "more", "will", "has", "have", "had", "was", "were",
    "are", "its", "their", "his", "her", "what", "when", "where", "why",
    "how", "news", "report", "reports", "update", "updates", "video", "watch",
    "today", "yesterday", "tomorrow", "year", "years", "day", "days",
}

_RELATED_PHRASE_STOPWORDS = _RELATED_ANCHOR_STOPWORDS | {
    "a", "an", "of", "in", "on", "at", "to", "as", "by", "or", "but",
    "is", "be", "been", "being", "it", "they", "he", "she", "we", "you",
    "i", "do", "does", "did",
}


def _related_phrase_sequence(text: str) -> list[str]:
    """Normalize text into a compact sequence suitable for phrase matching."""
    text = (text or "").lower().replace("’", "'")
    text = re.sub(r"[-–—/]+", " ", text)
    raw = re.findall(r"[a-z0-9][a-z0-9']*", text)
    return [
        token.strip("'")
        for token in raw
        if len(token.strip("'")) >= 2
        and token.strip("'") not in _RELATED_PHRASE_STOPWORDS
    ]


def _related_anchor_tokens(text: str) -> set[str]:
    return {
        token
        for token in _related_phrase_sequence(text)
        if len(token) >= 4
    }


def _related_ngram_phrases(text: str, min_n: int = 2, max_n: int = 4) -> set[str]:
    tokens = _related_phrase_sequence(text)
    phrases = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            phrases.add(" ".join(tokens[i:i + n]))
    return phrases


def _proper_name_phrases(text: str) -> set[str]:
    """
    Extract multi-token capitalized names/institutions without a fixed entity list.
    These are useful identity anchors, but identity alone is not enough to prove
    that an article covers the same development.
    """
    spans = re.findall(
        r"\b(?:[A-Z][A-Za-z'’.-]*|[A-Z]{2,})"
        r"(?:\s+(?:[A-Z][A-Za-z'’.-]*|[A-Z]{2,})){1,4}\b",
        text or "",
    )

    out = set()
    for span in spans:
        tokens = _related_phrase_sequence(span)
        for n in range(2, min(4, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                out.add(" ".join(tokens[i:i + n]))
    return out


def _event_anchor_terms(core_articles: list[dict], event_sentence: str) -> set[str]:
    """Derive event-specific lexical anchors from the selected core itself."""
    counts = Counter()
    for article in core_articles:
        title_tokens = _related_anchor_tokens(article.get("title") or "")
        counts.update(title_tokens)

    event_tokens = _related_anchor_tokens(event_sentence)
    min_core_support = max(2, (len(core_articles) + 2) // 3)

    anchors = {
        token
        for token, count in counts.items()
        if count >= min_core_support or (token in event_tokens and count >= 1)
    }
    anchors.update(token for token in event_tokens if counts.get(token, 0) >= 1)
    return anchors


def _event_anchor_phrases(
    core_articles: list[dict],
    event_sentence: str,
) -> tuple[set[str], set[str]]:
    """
    Return (event_phrases, identity_phrases), derived only from the core and
    canonical event sentence.

    Repeated core phrases and event-sentence phrases supported by a core title
    become anchors. Proper-name phrases are tracked separately so a shared
    person/institution cannot, by itself, rescue a different event.
    """
    phrase_counts = Counter()
    core_phrase_union = set()
    identity_phrases = set()

    for article in core_articles:
        title = article.get("title") or ""
        title_phrases = _related_ngram_phrases(title)
        phrase_counts.update(title_phrases)
        core_phrase_union.update(title_phrases)
        identity_phrases.update(_proper_name_phrases(title))

    event_phrases = _related_ngram_phrases(event_sentence)
    identity_phrases.update(_proper_name_phrases(event_sentence))

    supported_identity = identity_phrases & core_phrase_union
    all_anchors = {
        phrase for phrase, count in phrase_counts.items() if count >= 2
    }
    all_anchors.update(event_phrases & core_phrase_union)

    return all_anchors - supported_identity, supported_identity


def _related_membership_text(article: dict) -> str:
    title = (article.get("title") or "").strip()
    desc = (article.get("description") or "").strip()
    desc = re.sub(r"<[^>]+>", " ", desc)
    desc = re.sub(r"\s+", " ", desc)[:300]
    return f"{title}. {desc}".strip()


def filter_related_articles_to_core(
    core_articles: list[dict],
    candidate_articles: list[dict],
    event_sentence: str,
    *,
    gpt_related_articles: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """
    Free, conservative event-attention verification over the ENTIRE original
    candidate—not only the indexes GPT marked RELATED.

    CORE is never altered. The local pass may:
      - restore a GPT-omitted article when it is strongly anchored to the exact
        event; and
      - remove a GPT-selected article that shares only the person/topic but not
        the same development.

    This affects attention/ranking metadata only. Summary membership remains the
    strict GPT-selected CORE.
    """
    diag = {
        "applied": False,
        "from": len(candidate_articles),
        "to": len(candidate_articles),
        "gpt_selected": len(gpt_related_articles or []),
        "restored": [],
        "removed": [],
    }

    if not core_articles:
        return [], {**diag, "reason": "missing_core"}

    def article_key(article: dict) -> str:
        raw = article.get("url_normalized") or article.get("url") or ""
        return canonicalize_url(raw) or (article.get("title") or "").strip().lower()

    core_keys = {article_key(article) for article in core_articles}
    core_keys.discard("")
    gpt_related_keys = {
        article_key(article)
        for article in (gpt_related_articles or [])
        if article_key(article)
    }

    # Preserve original candidate order while removing duplicate records.
    deduped_candidates = []
    seen = set()
    for article in candidate_articles:
        key = article_key(article)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(article)

    non_core = [
        article for article in deduped_candidates
        if article_key(article) not in core_keys
    ]
    if not non_core:
        core_ordered = [
            article for article in deduped_candidates
            if article_key(article) in core_keys
        ]
        return core_ordered, {
            **diag,
            "applied": True,
            "from": len(deduped_candidates),
            "to": len(core_ordered),
        }

    sem = _get_sem_embedder()
    if sem is None:
        return list(core_articles), {
            **diag,
            "applied": True,
            "from": len(deduped_candidates),
            "to": len(core_articles),
            "reason": "semantic_model_unavailable_core_only",
            "removed": [
                {
                    "title": (article.get("title") or "").strip(),
                    "gpt_selected": article_key(article) in gpt_related_keys,
                    "reason": "unverified",
                }
                for article in non_core
            ],
        }

    core_texts = [_related_membership_text(article) for article in core_articles]
    candidate_texts = [_related_membership_text(article) for article in non_core]

    try:
        core_X = np.asarray(
            sem.encode(core_texts, normalize_embeddings=True),
            dtype=np.float32,
        )
        candidate_X = np.asarray(
            sem.encode(candidate_texts, normalize_embeddings=True),
            dtype=np.float32,
        )
        event_vec = np.asarray(
            sem.encode([event_sentence or " "], normalize_embeddings=True),
            dtype=np.float32,
        )[0]
    except Exception:
        return list(core_articles), {
            **diag,
            "applied": True,
            "from": len(deduped_candidates),
            "to": len(core_articles),
            "reason": "semantic_encoding_failed_core_only",
            "removed": [
                {
                    "title": (article.get("title") or "").strip(),
                    "gpt_selected": article_key(article) in gpt_related_keys,
                    "reason": "unverified",
                }
                for article in non_core
            ],
        }

    core_vec = core_X.mean(axis=0)
    core_vec = core_vec / max(np.linalg.norm(core_vec), 1e-12)

    token_anchors = _event_anchor_terms(core_articles, event_sentence)
    event_phrases, identity_phrases = _event_anchor_phrases(
        core_articles,
        event_sentence,
    )
    identity_tokens = {
        token
        for phrase in identity_phrases
        for token in phrase.split()
        if len(token) >= 4
    }
    event_token_anchors = token_anchors - identity_tokens

    kept_non_core_keys = set()
    restored = []
    removed = []

    for article, vec in zip(non_core, candidate_X):
        key = article_key(article)
        was_gpt_selected = key in gpt_related_keys

        peer_sims = core_X @ vec
        max_peer = float(np.max(peer_sims)) if len(peer_sims) else 0.0
        peer_support = int(np.sum(peer_sims >= 0.48))
        core_sim = float(vec @ core_vec)
        event_sim = float(vec @ event_vec)
        semantic_peak = max(core_sim, event_sim)

        full_text = _related_membership_text(article)
        title_phrases = _related_ngram_phrases(article.get("title") or "")
        full_phrases = _related_ngram_phrases(full_text)
        article_tokens = _related_anchor_tokens(full_text)

        token_hits = sorted(token_anchors & article_tokens)
        event_token_hits = sorted(event_token_anchors & article_tokens)
        title_event_phrase_hits = sorted(event_phrases & title_phrases)
        full_event_phrase_hits = sorted(event_phrases & full_phrases)
        title_identity_hits = sorted(identity_phrases & title_phrases)
        full_identity_hits = sorted(identity_phrases & full_phrases)

        keep_reason = ""

        # Event/action phrase in the headline is the strongest general signal.
        if (
            title_event_phrase_hits
            and len(token_hits) >= 2
            and max_peer >= 0.44
            and semantic_peak >= 0.40
        ):
            keep_reason = "title_event_phrase"

        # Event phrase in the description is useful, but requires more support.
        elif (
            full_event_phrase_hits
            and len(event_token_hits) >= 1
            and len(token_hits) >= 3
            and max_peer >= 0.49
            and semantic_peak >= 0.42
        ):
            keep_reason = "full_event_phrase"

        # Strong distributed lexical + semantic agreement can recover paraphrases.
        elif (
            len(event_token_hits) >= 2
            and len(token_hits) >= 4
            and max_peer >= 0.52
            and semantic_peak >= 0.44
            and peer_support >= 1
        ):
            keep_reason = "multi_anchor_semantic"

        # A shared named person/institution is not enough by itself. It can
        # support membership only when at least one event token and strong
        # semantic agreement are also present.
        elif (
            title_identity_hits
            and len(event_token_hits) >= 1
            and max_peer >= (0.56 if was_gpt_selected else 0.60)
            and semantic_peak >= (0.46 if was_gpt_selected else 0.50)
        ):
            keep_reason = "identity_plus_event"

        elif (
            full_identity_hits
            and len(event_token_hits) >= 2
            and max_peer >= (0.56 if was_gpt_selected else 0.60)
            and semantic_peak >= (0.46 if was_gpt_selected else 0.50)
        ):
            keep_reason = "full_identity_plus_event"

        # Near-duplicate paraphrases may omit a repeated phrase but must still
        # share multiple anchors and be exceptionally close to a core report.
        elif (
            max_peer >= 0.72
            and semantic_peak >= 0.52
            and len(token_hits) >= 3
            and len(event_token_hits) >= 1
        ):
            keep_reason = "near_duplicate_semantic"

        if keep_reason:
            kept_non_core_keys.add(key)
            if not was_gpt_selected:
                restored.append({
                    "title": (article.get("title") or "").strip(),
                    "reason": keep_reason,
                    "core_sim": round(core_sim, 3),
                    "event_sim": round(event_sim, 3),
                    "max_peer_sim": round(max_peer, 3),
                    "event_token_hits": event_token_hits,
                    "event_phrase_hits": title_event_phrase_hits or full_event_phrase_hits,
                    "identity_phrase_hits": title_identity_hits or full_identity_hits,
                })
        else:
            removed.append({
                "title": (article.get("title") or "").strip(),
                "gpt_selected": was_gpt_selected,
                "core_sim": round(core_sim, 3),
                "event_sim": round(event_sim, 3),
                "max_peer_sim": round(max_peer, 3),
                "peer_support": peer_support,
                "token_hits": token_hits,
                "event_token_hits": event_token_hits,
                "event_phrase_hits": title_event_phrase_hits or full_event_phrase_hits,
                "identity_phrase_hits": title_identity_hits or full_identity_hits,
            })

    final_keys = core_keys | kept_non_core_keys
    filtered = [
        article for article in deduped_candidates
        if article_key(article) in final_keys
    ]

    # Ensure every CORE article is present even if absent from the candidate list.
    filtered_keys = {article_key(article) for article in filtered}
    for article in core_articles:
        key = article_key(article)
        if key and key not in filtered_keys:
            filtered.append(article)
            filtered_keys.add(key)

    return filtered, {
        "applied": True,
        "from": len(deduped_candidates),
        "to": len(filtered),
        "gpt_selected": len(gpt_related_keys),
        "token_anchor_count": len(token_anchors),
        "event_phrase_count": len(event_phrases),
        "identity_phrase_count": len(identity_phrases),
        "restored": restored,
        "removed": removed,
    }


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

def dominant_entity_ratio(nlp, titles: list[str]) -> float:
    """Fraction of usable titles containing the most common named entity."""
    if not nlp or not titles:
        return 1.0

    counts = Counter()
    usable = 0
    for t in titles[:18]:
        ents = _entities(nlp, t)
        if not ents:
            continue
        usable += 1
        for e in ents:
            counts[e] += 1

    if usable < 4 or not counts:
        return 0.0

    return counts.most_common(1)[0][1] / usable


def event_action_consistency(nlp, titles: list[str]) -> float:
    """Measures whether titles share the same leading event action/frame."""
    if not nlp or not titles:
        return 0.0

    actions = []
    for t in titles[:18]:
        doc = nlp(t)
        verbs = [
            tok.lemma_.lower()
            for tok in doc
            if tok.pos_ in {"VERB", "AUX"}
            and not tok.is_stop
            and len(tok.lemma_) > 2
        ]
        if verbs:
            actions.append(verbs[0])

    if len(actions) < 4:
        return 0.0

    counts = Counter(actions)
    return counts.most_common(1)[0][1] / len(actions)


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

def split_two_substantial_event_components(
    articles: list[dict],
    *,
    min_cluster_size: int = 8,
    min_part_size: int = 4,
    min_part_frac: float = 0.30,
    max_centroid_sim: float = 0.72,
    min_p10_gain: float = 0.06,
    min_std_gain: float = 0.025,
) -> tuple[list[list[dict]], dict]:
    """
    Conservative math-only two-event splitter.

    Attempts a cosine-space k=2 partition only for clusters large enough to
    plausibly contain two real sub-events. Splits only when:
      - both parts are substantial,
      - the two part centroids are meaningfully distinct,
      - and partitioning materially improves cohesion.

    Returns ([part1, part2], diagnostics) when split; otherwise ([articles], diagnostics).
    """
    n = len(articles)
    diag = {
        "event_split": False,
        "split_from": n,
        "split_sizes": [n],
    }

    if n < min_cluster_size:
        return [articles], diag

    sem = _get_sem_embedder()
    if sem is None:
        return [articles], diag

    texts = []
    valid_idx = []

    for i, a in enumerate(articles):
        t = (a.get("title") or "").strip()
        d = (a.get("description") or "").strip()
        if d:
            d = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", d))[:250]
        txt = (t + ". " + d).strip() if t else d
        if txt:
            texts.append(txt)
            valid_idx.append(i)

    if len(texts) < min_cluster_size:
        return [articles], diag

    try:
        X = sem.encode(texts, normalize_embeddings=True)
        X = np.asarray(X, dtype=np.float32)
    except Exception:
        return [articles], diag

    # Baseline cohesion.
    _, p10_before, std_before = _cohesion_stats(X)

    # Deterministic farthest-pair initialization for cosine k=2.
    a0 = 0
    s0 = (X @ X[a0:a0+1].T).reshape(-1)
    a = int(np.argmin(s0))
    sa = (X @ X[a:a+1].T).reshape(-1)
    b = int(np.argmin(sa))

    c1 = X[a].copy()
    c2 = X[b].copy()
    labels = None

    for _ in range(8):
        s1 = (X @ c1.reshape(-1, 1)).reshape(-1)
        s2 = (X @ c2.reshape(-1, 1)).reshape(-1)
        new_labels = (s2 > s1).astype(np.int32)

        if new_labels.sum() == 0 or new_labels.sum() == len(new_labels):
            return [articles], diag

        if labels is not None and np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels

        c1 = X[labels == 0].mean(axis=0)
        c2 = X[labels == 1].mean(axis=0)
        c1 = c1 / max(np.linalg.norm(c1), 1e-12)
        c2 = c2 / max(np.linalg.norm(c2), 1e-12)

    if labels is None:
        return [articles], diag

    idx1 = np.where(labels == 0)[0]
    idx2 = np.where(labels == 1)[0]

    n1, n2 = len(idx1), len(idx2)
    if n1 < min_part_size or n2 < min_part_size:
        return [articles], diag

    if min(n1, n2) / len(X) < min_part_frac:
        return [articles], diag

    X1 = X[idx1]
    X2 = X[idx2]

    _, p10_1, std_1 = _cohesion_stats(X1)
    _, p10_2, std_2 = _cohesion_stats(X2)

    p10_after = (n1 * p10_1 + n2 * p10_2) / (n1 + n2)
    std_after = (n1 * std_1 + n2 * std_2) / (n1 + n2)

    cent1 = X1.mean(axis=0)
    cent2 = X2.mean(axis=0)
    cent1 = cent1 / max(np.linalg.norm(cent1), 1e-12)
    cent2 = cent2 / max(np.linalg.norm(cent2), 1e-12)
    centroid_sim = float(cent1 @ cent2)

    cohesion_improved = (
        (p10_after - p10_before >= min_p10_gain)
        or
        (std_before - std_after >= min_std_gain)
    )

    if centroid_sim > max_centroid_sim or not cohesion_improved:
        return [articles], diag

    original_idx1 = {valid_idx[int(i)] for i in idx1}
    original_idx2 = {valid_idx[int(i)] for i in idx2}

    part1 = [a for i, a in enumerate(articles) if i in original_idx1]
    part2 = [a for i, a in enumerate(articles) if i in original_idx2]

    # Largest part first for stable ordering.
    parts = sorted([part1, part2], key=len, reverse=True)

    diag = {
        "event_split": True,
        "split_from": n,
        "split_sizes": [len(parts[0]), len(parts[1])],
        "split_centroid_sim": round(centroid_sim, 3),
        "split_p10_before": round(p10_before, 3),
        "split_p10_after": round(p10_after, 3),
        "split_std_before": round(std_before, 3),
        "split_std_after": round(std_after, 3),
    }

    return parts, diag


def extract_dominant_event_component(
    articles: list[dict],
    *,
    min_cluster_size: int = 6,
    min_component_size: int = 5,
    sim_threshold: float = 0.54,
    min_component_frac: float = 0.55,
    min_p10_gain: float = 0.04,
    min_std_gain: float = 0.025,
) -> tuple[list[dict], dict]:
    """
    Math-only article-level cleanup.

    Builds a semantic similarity graph inside one candidate cluster and extracts
    the largest connected component only when that component:
      - contains at least min_component_size articles,
      - contains at least min_component_frac of the cluster,
      - and materially improves cohesion.

    This removes unrelated sub-stories while preserving the dominant event.
    Returns (articles, diagnostics).
    """
    n = len(articles)
    diag = {
        "component_trimmed": False,
        "component_from": n,
        "component_to": n,
    }

    if n < min_cluster_size:
        return articles, diag

    sem = _get_sem_embedder()
    if sem is None:
        return articles, diag

    texts = []
    valid_idx = []

    for i, a in enumerate(articles):
        t = (a.get("title") or "").strip()
        d = (a.get("description") or "").strip()
        if d:
            d = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", d))[:250]
        txt = (t + ". " + d).strip() if t else d
        if txt:
            texts.append(txt)
            valid_idx.append(i)

    if len(texts) < min_component_size:
        return articles, diag

    try:
        X = sem.encode(texts, normalize_embeddings=True)
        X = np.asarray(X, dtype=np.float32)
    except Exception:
        return articles, diag

    S = X @ X.T
    m = len(texts)

    # Connected components over strong semantic links.
    seen = set()
    components = []

    for i in range(m):
        if i in seen:
            continue

        stack = [i]
        seen.add(i)
        comp = []

        while stack:
            cur = stack.pop()
            comp.append(cur)

            nbrs = np.where(S[cur] >= sim_threshold)[0]
            for j in nbrs:
                j = int(j)
                if j == cur or j in seen:
                    continue
                seen.add(j)
                stack.append(j)

        components.append(comp)

    if not components:
        return articles, diag

    components.sort(key=len, reverse=True)
    best = components[0]

    if len(best) == m:
        return articles, diag

    if len(best) < min_component_size:
        return articles, diag

    if (len(best) / max(1, m)) < min_component_frac:
        return articles, diag

    # Compare cohesion before vs. after extraction.
    _, p10_before, std_before = _cohesion_stats(X)
    X_best = X[best, :]
    _, p10_after, std_after = _cohesion_stats(X_best)

    improved = (
        (p10_after - p10_before >= min_p10_gain)
        or
        (std_before - std_after >= min_std_gain)
    )

    if not improved:
        return articles, diag

    keep_original_idx = {valid_idx[i] for i in best}
    kept = [a for i, a in enumerate(articles) if i in keep_original_idx]

    diag = {
        "component_trimmed": True,
        "component_from": n,
        "component_to": len(kept),
        "component_p10_before": round(p10_before, 3),
        "component_p10_after": round(p10_after, 3),
        "component_std_before": round(std_before, 3),
        "component_std_after": round(std_after, 3),
    }

    return kept, diag


def filter_articles_to_dominant_event(
    articles: list[dict],
    *,
    min_cluster_size: int = 5,
    min_keep: int = 4,
    core_top_k: int = 4,
    min_core_sim: float = 0.50,
    min_peer_sim: float = 0.52,
    min_peer_support: int = 2,
    max_remove_frac: float = 0.30,
) -> tuple[list[dict], dict]:
    """
    Final conservative article-to-event membership cleanup.

    Builds a dominant event core from the most mutually central articles, then
    removes only articles that are weak against the dominant event core AND lack
    support from enough other core-like articles.

    This avoids letting a contaminated whole-cluster centroid rescue an article
    that belongs to a related-but-distinct event.

    Safeguards:
      - never acts on clusters smaller than min_cluster_size
      - never removes more than max_remove_frac
      - always keeps at least min_keep articles
      - if safeguards are exceeded, leaves the cluster unchanged
    """
    n = len(articles)

    diag = {
        "membership_trimmed": False,
        "membership_from": n,
        "membership_to": n,
    }

    if n < min_cluster_size:
        return articles, diag

    sem = _get_sem_embedder()
    if sem is None:
        return articles, diag

    texts = []
    valid_idx = []

    for i, a in enumerate(articles):
        t = (a.get("title") or "").strip()
        d = (a.get("description") or "").strip()

        if d:
            d = re.sub(
                r"\s+",
                " ",
                re.sub(r"<[^>]+>", " ", d)
            )[:250]

        txt = (t + ". " + d).strip() if t else d

        if txt:
            texts.append(txt)
            valid_idx.append(i)

    if len(texts) < min_cluster_size:
        return articles, diag

    try:
        X = sem.encode(
            texts,
            normalize_embeddings=True
        )
        X = np.asarray(X, dtype=np.float32)
    except Exception:
        return articles, diag

    m = len(X)
    S = X @ X.T

    # Semantic centrality: mean similarity to all other articles.
    S_no_diag = S.copy()
    np.fill_diagonal(S_no_diag, np.nan)
    centrality = np.nanmean(S_no_diag, axis=1)

    k = min(core_top_k, max(3, m // 2))
    core_idx = np.argsort(centrality)[-k:]

    # Dominant-event core centroid.
    core_vec = X[core_idx].mean(axis=0)
    core_vec = core_vec / max(
        np.linalg.norm(core_vec),
        1e-12
    )

    core_sims = X @ core_vec

    # Core-like reference set:
    # articles that are at least reasonably aligned with the dominant core.
    core_like_idx = [
        i for i in range(m)
        if core_sims[i] >= min_core_sim
    ]

    # Ensure the actual selected central core is always represented.
    core_like_idx = sorted(
        set(core_like_idx) | set(core_idx.tolist())
    )

    remove_local = []
    article_diags = []

    for i in range(m):
        # Similarity to other core-like articles.
        peer_sims = []

        for j in core_like_idx:
            if i == j:
                continue
            peer_sims.append(float(S[i, j]))

        peer_support = sum(
            1 for s in peer_sims
            if s >= min_peer_sim
        )

        weak_core = float(core_sims[i]) < min_core_sim
        weak_peer_support = peer_support < min_peer_support

        should_remove = (
            weak_core
            and weak_peer_support
        )

        article_diags.append({
            "local_index": i,
            "title": (
                articles[valid_idx[i]].get("title") or ""
            ).strip(),
            "core_sim": round(float(core_sims[i]), 3),
            "peer_support": int(peer_support),
            "removed": bool(should_remove),
        })

        if should_remove:
            remove_local.append(i)

    if not remove_local:
        return articles, diag

    max_remove = int(np.floor(n * max_remove_frac))
    max_remove = min(
        max_remove,
        n - min_keep
    )

    if max_remove <= 0:
        return articles, diag

    # If too many articles fail, remove only the weakest ones rather than
    # discarding the whole cleanup attempt.
    if len(remove_local) > max_remove:
        remove_local = sorted(
            remove_local,
            key=lambda i: (
                float(core_sims[i]),
                sum(
                    1
                    for j in core_like_idx
                    if i != j and float(S[i, j]) >= min_peer_sim
                ),
            ),
        )[:max_remove]

    remove_original = {
        valid_idx[i]
        for i in remove_local
    }

    kept = [
        a for i, a in enumerate(articles)
        if i not in remove_original
    ]

    if len(kept) < min_keep:
        return articles, diag

    removed_details = [
        d for d in article_diags
        if d["local_index"] in remove_local
    ]

    diag = {
        "membership_trimmed": True,
        "membership_from": n,
        "membership_to": len(kept),
        "membership_removed": len(remove_original),
        "membership_min_core_sim": round(
            float(min(core_sims)),
            3
        ),
        "membership_peer_threshold": min_peer_sim,
        "membership_min_peer_support": min_peer_support,
        "membership_removed_articles": removed_details,
    }

    return kept, diag

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
        merge_attention_metadata(into, src)

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
# Final-decision diagnostics
# ----------------------------
def _print_decision_diag(c: dict, rank_pos: int, decision: str, reason: str) -> None:
    if not c.get("_purity_report_enabled"):
        return
    print(
        f"[decision] pre_rank={rank_pos + 1} "
        f"topic={c.get('topic')} "
        f"size={len(c.get('articles', []))} "
        f"math_pure={c.get('_math_pure', False)} "
        f"tight_math={c.get('_tight_event_math', False)} "
        f"broad_math={c.get('_broad_event_math', False)} "
        f"gpt_label={c.get('_gpt_label', '') or 'N/A'} "
        f"gpt_source={c.get('_gpt_source', '') or 'N/A'} "
        f"decision={decision} "
        f"reason={reason}"
    )

# ----------------------------
# Output cleaning
# ----------------------------
_OUTPUT_TRANSIENT_KEYS = [
    "_purity_report_enabled",
    "_math_pure",
    "_tight_event_math",
    "_broad_event_math",
    "_gpt_label",
    "_gpt_source",
    "_attention_urls",
    "_attention_domains",
    "_gdelt_shadow_local_upper_signal",
    "_gdelt_shadow_global_upper_signal",
    "_gdelt_shadow_potential_score",
    "_gdelt_shadow_potential_evidence",
    "_gdelt_catalog_match_options",
    "_gdelt_shadow_local_signal",
    "_gdelt_shadow_global_signal",
    "_gdelt_shadow_score",
    "_gdelt_shadow_evidence",
]


def _clusters_for_output(clusters: list[dict]) -> list[dict]:
    """Deep-copy clusters and remove run-only diagnostic fields."""
    cleaned = copy.deepcopy(clusters)
    for cluster in cleaned:
        for key in _OUTPUT_TRANSIENT_KEYS:
            cluster.pop(key, None)
    return cleaned


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    (
        date_str,
        no_openai,
        top_k,
        print_report,
        purity_report,
        input_override,
        output_override,
        gdelt_ranking_shadow,
        gdelt_ranking_shadow_override,
        gdelt_audit_override,
        gdelt_global_output_override,
    ) = _parse_args()

    # Env & OpenAI
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    if openai is not None:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    input_file = (
        input_override
        or f"grouped_articles_filtered_{date_str}.json"
    )
    output_file = (
        output_override
        or (
            f"grouped_articles_final_gdelt_shadow_{date_str}.json"
            if gdelt_ranking_shadow
            else f"grouped_articles_final_{date_str}.json"
        )
    )
    gdelt_ranking_shadow_file = (
        gdelt_ranking_shadow_override
        or f"gdelt_global_ranking_shadow_{date_str}.json"
    )
    gdelt_global_output_file = (
        gdelt_global_output_override
        or f"grouped_articles_final_global_shadow_{date_str}.json"
    )
    gdelt_runtime_module = None
    gdelt_catalog_matcher = None
    gdelt_catalog_diag = {
        "status": "DISABLED",
        "file": None,
        "candidate_count": 0,
        "errors": [],
    }
    gdelt_assignment_stats = {}

    if gdelt_ranking_shadow:
        print(
            f"🌐 GDELT global-ranking shadow enabled; "
            f"local-order output is isolated at {output_file}"
        )

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

        attention_meta = attention_metadata_from_articles(arts)

        final_candidates.append({
            "topic": cluster.get("topic", cluster.get("topic_title", "Merged Topic")),
            "articles": arts_kept,
            "source_diversity": diversity,
            "source_concentration": compute_source_concentration(arts_kept),
            "bias_distribution": bias_dist,
            **attention_meta,
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

    # Article-level dominant-event extraction.
    # Preserve the cluster unless a large internal component is clearly cleaner.
    component_cleaned = []

    for c in deduped:
        original_articles = c.get("articles", [])
        core_articles, component_diag = extract_dominant_event_component(original_articles)

        if component_diag.get("component_trimmed"):
            c = dict(c)
            c["articles"] = core_articles
            c["source_diversity"] = source_diversity(core_articles)
            c["source_concentration"] = compute_source_concentration(core_articles)
            c["bias_distribution"] = aggregate_bias_distribution(core_articles)
            c["today_ratio"] = round(cluster_today_ratio(core_articles, date_str), 3)
            c["component_extraction"] = component_diag

        component_cleaned.append(c)

    deduped = component_cleaned

    # Conservative two-event split.
    # If one candidate still contains two substantial, internally coherent event
    # components, promote each component to its own candidate before GPT/ranking.
    #
    # One additional validation pass is allowed on each split child:
    #   1) dominant-component extraction
    #   2) one more two-event split
    # This is intentionally bounded to avoid recursive shredding.
    split_candidates = []

    for c in deduped:
        parts, split_diag = split_two_substantial_event_components(c.get("articles", []))

        if not split_diag.get("event_split"):
            split_candidates.append(c)
            continue

        for part_num, part_articles in enumerate(parts, start=1):
            c_part = dict(c)
            c_part["articles"] = part_articles
            c_part["source_diversity"] = source_diversity(part_articles)
            c_part["source_concentration"] = compute_source_concentration(part_articles)
            c_part["bias_distribution"] = aggregate_bias_distribution(part_articles)
            c_part["today_ratio"] = round(cluster_today_ratio(part_articles, date_str), 3)
            c_part["event_split"] = {
                **split_diag,
                "part": part_num,
                "validation_depth": 1,
            }

            # Bounded child cleanup pass: dominant component first.
            child_core, child_component_diag = extract_dominant_event_component(part_articles)
            if child_component_diag.get("component_trimmed"):
                c_part["articles"] = child_core
                c_part["source_diversity"] = source_diversity(child_core)
                c_part["source_concentration"] = compute_source_concentration(child_core)
                c_part["bias_distribution"] = aggregate_bias_distribution(child_core)
                c_part["today_ratio"] = round(cluster_today_ratio(child_core, date_str), 3)
                c_part["child_component_extraction"] = child_component_diag

            # Bounded child split pass: at most one extra split.
            child_parts, child_split_diag = split_two_substantial_event_components(c_part.get("articles", []))

            if child_split_diag.get("event_split"):
                for child_num, child_articles in enumerate(child_parts, start=1):
                    c_child = dict(c_part)
                    c_child["articles"] = child_articles
                    c_child["source_diversity"] = source_diversity(child_articles)
                    c_child["source_concentration"] = compute_source_concentration(child_articles)
                    c_child["bias_distribution"] = aggregate_bias_distribution(child_articles)
                    c_child["today_ratio"] = round(cluster_today_ratio(child_articles, date_str), 3)
                    c_child["child_event_split"] = {
                        **child_split_diag,
                        "parent_part": part_num,
                        "part": child_num,
                        "validation_depth": 2,
                    }
                    split_candidates.append(c_child)
            else:
                split_candidates.append(c_part)

    deduped = split_candidates

    # Final, low-cost GPT pass: only on suspicious big buckets
    event_cache = _load_event_cache()
    event_calls = 0

    # Pre-rank to decide which ones are worth validating (top 15 by current score)
    pre_ranked = sorted(deduped, key=matter_score, reverse=True)
    for c in pre_ranked:
        c["_purity_report_enabled"] = bool(purity_report)

    if purity_report:
        print("\n=== Cluster purity diagnostics ===")
        for idx, c in enumerate(pre_ranked, start=1):
            arts = c.get("articles", [])
            titles = [a.get("title", "") for a in arts if a.get("title")]
            p10, std = cluster_cohesion_fast(arts)
            nn_mean, nn_p10 = cluster_nn_tightness(arts)
            ent_coh = cluster_entity_cohesion(nlp, titles)
            multi_lump = _cluster_is_thematic_multi_lump(arts)

            print(
                f"\n[{idx}] topic={c.get('topic')} "
                f"size={len(arts)} "
                f"p10={p10:.3f} "
                f"std={std:.3f} "
                f"nn_p10={nn_p10:.3f} "
                f"entity_coh={ent_coh:.3f} "
                f"multi_lump={multi_lump} "
                f"component_trim={((c.get('component_extraction') or {}).get('component_from', len(arts)))}"
                f"→{((c.get('component_extraction') or {}).get('component_to', len(arts)))} "
                f"event_split={((c.get('event_split') or {}).get('split_from', len(arts)))}"
                f"→{((c.get('event_split') or {}).get('split_sizes', [len(arts)]))} "
                f"child_trim={((c.get('child_component_extraction') or {}).get('component_from', len(arts)))}"
                f"→{((c.get('child_component_extraction') or {}).get('component_to', len(arts)))} "
                f"child_split={((c.get('child_event_split') or {}).get('split_from', len(arts)))}"
                f"→{((c.get('child_event_split') or {}).get('split_sizes', [len(arts)]))}"
            )

            for a in arts[:8]:
                print("   -", (a.get("title") or "").strip())

    kept = []

    for rank_pos, c in enumerate(pre_ranked):

        arts = c.get("articles", [])
        titles = [a.get("title","") for a in arts if a.get("title")]

        # Only consider GPT for larger clusters that still look "bucket-ish"
        # (math-based signals; no lists)
        p10, std = cluster_cohesion_fast(arts)
        nn_mean, nn_p10 = cluster_nn_tightness(arts)
        ent_coh = cluster_entity_cohesion(nlp, titles) if "cluster_entity_cohesion" in globals() else 1.0
        dom_ent = dominant_entity_ratio(nlp, titles) if "dominant_entity_ratio" in globals() else 0.0
        action_consistency = event_action_consistency(nlp, titles) if "event_action_consistency" in globals() else 0.0
        multi_lump = _cluster_is_thematic_multi_lump(arts)

        tr = float(c.get("today_ratio", 1.0) or 1.0)

        # Conservative math-only purity gate.
        # Drops obvious hodgepodge clusters before GPT/summarization.
        #
        # A strong dominant entity plus weak secondary entity is NOT, by itself,
        # evidence of multiple events. Legitimate single events often have exactly
        # that structure when outlets cover consequences or follow-up angles.
        small_low_cohesion_mixed = (
            len(arts) >= 5 and
            p10 < 0.60 and
            nn_p10 < 0.35 and
            ent_coh < 0.15
        )

        obvious_mixed_math = (
            multi_lump or
            small_low_cohesion_mixed or
            (
                len(arts) >= 6 and (
                    (ent_coh == 0.0 and nn_p10 < 0.32) or
                    (p10 < 0.48 and ent_coh < 0.08) or
                    (nn_p10 < 0.32 and ent_coh < 0.08)
                )
            )
        )

        if obvious_mixed_math:
            c["eventness_label"] = "MIXED_MATH"
            c["_math_pure"] = False
            _print_decision_diag(c, rank_pos, "REJECT", "obvious_mixed_math")
            continue

        # Math-pure auto-accept:
        # Path 1 keeps the original tight-event behavior.
        tight_event_math = (
            len(arts) >= 5 and
            p10 >= 0.72 and
            nn_p10 >= 0.60 and
            std <= 0.08 and
            ent_coh >= 0.18 and
            dom_ent >= 0.70 and
            action_consistency >= 0.40
        )

        # Path 2 conservatively accepts larger evolving events with strong
        # cohesion and entity anchoring, even when coverage spans reactions,
        # consequences, and follow-up developments.
        broad_event_math = (
            len(arts) >= 8 and
            p10 >= 0.66 and
            nn_p10 >= 0.55 and
            std <= 0.10 and
            ent_coh >= 0.45 and
            dom_ent >= 0.75 and
            not multi_lump
        )

        math_pure = tight_event_math or broad_event_math

        c["_tight_event_math"] = bool(tight_event_math)
        c["_broad_event_math"] = bool(broad_event_math)
        c["_math_pure"] = bool(math_pure)

        if math_pure:
            # Strong mathematical cohesion is useful evidence, but does not prove
            # that every article refers to one specific event.
            c["eventness_label"] = "MATH_COHERENT"

        # No GPT validation here.
        # Math filtering/splitting removes obvious contamination first.
        # Exact GPT event validation happens once, after membership cleanup
        # and attention ranking, so API budget is spent only on finalists.
        _print_decision_diag(
            c,
            rank_pos,
            "KEEP",
            "passed_math_filter_pending_final_validation",
        )
        kept.append(c)

    deduped = kept

    # Final article-to-event membership cleanup.
    # This acts only on already accepted clusters and removes isolated residual
    # articles that are weak against both the dominant event core and centroid.
    membership_cleaned = []

    for c in deduped:
        original_articles = c.get("articles", [])
        kept_articles, membership_diag = filter_articles_to_dominant_event(original_articles)

        if membership_diag.get("membership_trimmed"):
            c = dict(c)
            c["articles"] = kept_articles
            c["source_diversity"] = source_diversity(kept_articles)
            c["source_concentration"] = compute_source_concentration(kept_articles)
            c["bias_distribution"] = aggregate_bias_distribution(kept_articles)
            c["today_ratio"] = round(cluster_today_ratio(kept_articles, date_str), 3)
            c["membership_extraction"] = membership_diag

        if len(c.get("articles", [])) >= 4:
            membership_cleaned.append(c)

    deduped = membership_cleaned

    # Final page order:
    # purity determines eligibility; attention determines prominence.
    for c in deduped:
        c["attention_score"] = attention_score(c)

    ranked = sorted(
        deduped,
        key=lambda c: (
            float(c.get("attention_score", 0.0) or 0.0),
            importance_score(c),  # tie-break only
        ),
        reverse=True,
    )

    # GPT validation queue:
    # Default behavior remains local-attention-first. In explicit GDELT shadow
    # mode, globally strong candidates may move earlier in the review queue, but
    # production output is still sorted by the existing local attention score.
    shadow_local_reference_scores = sorted(
        float(c.get("attention_score", 0.0) or 0.0)
        for c in ranked
    )

    if gdelt_ranking_shadow:
        (
            gdelt_runtime_module,
            gdelt_catalog_matcher,
            gdelt_catalog_diag,
        ) = _prepare_gdelt_full_catalog_matcher(
            date_str,
            gdelt_audit_override,
            ranked,
        )
        if gdelt_catalog_matcher is not None:
            gdelt_catalog_matcher["catalog_file"] = gdelt_catalog_diag.get("file")
            print(
                f"🌐 Full-audit GDELT matcher: "
                f"{gdelt_catalog_diag.get('candidate_count', 0)} candidates "
                f"from {gdelt_catalog_diag.get('file')} "
                f"({gdelt_catalog_diag.get('matcher_status')})"
            )
        else:
            detail = gdelt_catalog_diag.get("matcher_error") or "; ".join(
                gdelt_catalog_diag.get("errors") or []
            )
            print(
                "⚠️ Full-audit GDELT matcher unavailable; "
                "using retained discovery-article evidence only"
                + (f": {detail}" if detail else "")
            )

        for c in ranked:
            local_upper = _local_attention_percentile(
                float(c.get("attention_score", 0.0) or 0.0),
                shadow_local_reference_scores,
            )
            retained_evidence = _gdelt_global_attention_from_articles(
                c.get("articles", []),
                min_retained_articles=2,
            )
            potential_options = _gdelt_catalog_options_for_cluster(
                c,
                gdelt_runtime_module,
                gdelt_catalog_matcher,
                threshold=GDELT_CATALOG_POTENTIAL_MATCH_THRESHOLD,
                max_matches=3,
            )
            catalog_evidence = _gdelt_option_evidence(
                potential_options[0] if potential_options else None,
                gdelt_catalog_diag.get("file"),
            )
            global_evidence = _combine_gdelt_shadow_evidence(
                retained_evidence,
                catalog_evidence,
            )
            global_upper = (
                global_evidence.get("global_signal")
                if global_evidence.get("eligible")
                else None
            )
            c["_gdelt_shadow_local_upper_signal"] = local_upper
            c["_gdelt_shadow_global_upper_signal"] = global_upper
            c["_gdelt_shadow_potential_score"] = _gdelt_shadow_blended_score(
                local_upper,
                global_upper,
            )
            c["_gdelt_shadow_potential_evidence"] = global_evidence

        review_queue = sorted(
            ranked,
            key=lambda c: (
                float(c.get("_gdelt_shadow_potential_score", 0.0) or 0.0),
                float(c.get("attention_score", 0.0) or 0.0),
                cluster_cohesion_fast(c.get("articles", []))[0],
                cluster_nn_tightness(c.get("articles", []))[1],
                importance_score(c),
            ),
            reverse=True,
        )

        # Suffix maxima make stopping rank-safe for BOTH the unchanged local
        # order and the GDELT global shadow order, regardless of queue ordering.
        shadow_suffix_local_upper = [0.0] * (len(review_queue) + 1)
        shadow_suffix_global_upper = [0.0] * (len(review_queue) + 1)
        for idx in range(len(review_queue) - 1, -1, -1):
            shadow_suffix_local_upper[idx] = max(
                shadow_suffix_local_upper[idx + 1],
                float(review_queue[idx].get("attention_score", 0.0) or 0.0),
            )
            shadow_suffix_global_upper[idx] = max(
                shadow_suffix_global_upper[idx + 1],
                float(
                    review_queue[idx].get("_gdelt_shadow_potential_score", 0.0)
                    or 0.0
                ),
            )
    else:
        review_queue = sorted(
            ranked,
            key=lambda c: (
                float(c.get("attention_score", 0.0) or 0.0),
                cluster_cohesion_fast(c.get("articles", []))[0],
                cluster_nn_tightness(c.get("articles", []))[1],
                importance_score(c),
            ),
            reverse=True,
        )
        shadow_suffix_local_upper = []
        shadow_suffix_global_upper = []

    # Final structured publication gate:
    # one call identifies a strict summary CORE and broader event-specific RELATED
    # coverage. A local core-anchored pass then verifies every RELATED-only article
    # before that material can affect attention ranking.
    event_cache_hits = 0
    reviewed_candidates = 0
    purifier_stop_reason = "not_run"
    vetted = []

    if not no_openai and os.getenv("OPENAI_API_KEY"):
        purifier_stop_reason = "queue_exhausted"

        for queue_pos, c in enumerate(review_queue):
            # Once enough valid events exist, stop only when unreviewed
            # candidates cannot enter the top K. In GDELT shadow mode, BOTH the
            # unchanged local ranking and the global shadow ranking must be
            # secured before stopping.
            if len(vetted) >= top_k:
                provisional_local = sorted(
                    vetted,
                    key=lambda item: (
                        float(item.get("attention_score", 0.0) or 0.0),
                        importance_score(item),
                    ),
                    reverse=True,
                )
                cutoff_attention = float(
                    provisional_local[top_k - 1].get("attention_score", 0.0)
                    or 0.0
                )

                if gdelt_ranking_shadow:
                    provisional_assignment = _resolve_all_approved_gdelt_matches(
                        vetted,
                        gdelt_runtime_module,
                        gdelt_catalog_matcher,
                        shadow_local_reference_scores,
                        apply=False,
                    )
                    provisional_shadow = sorted(
                        vetted,
                        key=lambda item: (
                            float(
                                provisional_assignment["by_event"]
                                .get(_shadow_event_id(item), {})
                                .get("shadow_score", 0.0)
                            ),
                            float(item.get("attention_score", 0.0) or 0.0),
                            importance_score(item),
                        ),
                        reverse=True,
                    )
                    cutoff_event_id = _shadow_event_id(provisional_shadow[top_k - 1])
                    cutoff_shadow = float(
                        provisional_assignment["by_event"]
                        .get(cutoff_event_id, {})
                        .get("shadow_score", 0.0)
                    )
                    remaining_local_upper = shadow_suffix_local_upper[queue_pos]
                    remaining_shadow_upper = shadow_suffix_global_upper[queue_pos]

                    if (
                        remaining_local_upper < cutoff_attention
                        and remaining_shadow_upper < cutoff_shadow
                    ):
                        purifier_stop_reason = "top_k_local_and_global_rank_secured"
                        break
                else:
                    remaining_upper_bound = float(
                        c.get("attention_score", 0.0) or 0.0
                    )
                    if remaining_upper_bound < cutoff_attention:
                        purifier_stop_reason = "top_k_rank_secured"
                        break

            original_articles = c.get("articles", [])
            sig = _cluster_sig_urls(c)
            cache_key = f"{EVENT_CACHE_VERSION}::{EVENT_MODEL}::{sig}"
            cached = event_cache.get(cache_key)

            valid_cached = (
                isinstance(cached, dict)
                and cached.get("action") in {"KEEP", "CLEAN", "REJECT"}
                and cached.get("event_type") in {"DISCRETE_EVENT", "TOPIC_WAVE"}
            )

            if valid_cached:
                result = cached
                event_cache_hits += 1
            else:
                if event_calls >= EVENT_MAX_CALLS:
                    purifier_stop_reason = "live_call_cap_reached_before_rank_secured"
                    if purity_report:
                        print(
                            f"[safety] topic={c.get('topic')} "
                            f"decision=STOP reason=gpt_live_call_cap_reached"
                        )
                    break

                result = purify_cluster_with_gpt(original_articles)
                event_calls += 1

                if result.get("action") in {"KEEP", "CLEAN", "REJECT"}:
                    # Cache URL membership rather than only numeric indexes. The
                    # cluster signature is order-independent, so CORE/RELATED
                    # membership remains stable if article order changes.
                    result = dict(result)

                    def urls_for_indexes(index_key: str) -> list[str]:
                        zero_based = {
                            int(i) - 1
                            for i in result.get(index_key, [])
                            if isinstance(i, int) or str(i).isdigit()
                        }
                        return sorted({
                            canonicalize_url(
                                article.get("url_normalized") or article.get("url") or ""
                            )
                            for idx, article in enumerate(original_articles)
                            if idx in zero_based
                            and (article.get("url_normalized") or article.get("url"))
                        })

                    result["core_urls"] = urls_for_indexes("core_indexes")
                    result["related_urls"] = urls_for_indexes("related_indexes")
                    event_cache[cache_key] = result

            reviewed_candidates += 1
            action = str(result.get("action") or "ERROR").upper()
            event_type = str(result.get("event_type") or "").upper()

            if action == "ERROR":
                if purity_report:
                    print(
                        f"[safety] topic={c.get('topic')} "
                        f"action=ERROR decision=SKIP "
                        f"reason={result.get('why', '')}"
                    )
                continue

            if action == "REJECT" or event_type != "DISCRETE_EVENT":
                if purity_report:
                    print(
                        f"[safety] topic={c.get('topic')} "
                        f"event_type={event_type or 'UNKNOWN'} "
                        f"action=REJECT decision=REJECT "
                        f"reason={result.get('why', '')}"
                    )
                continue

            def select_membership(url_key: str, index_key: str) -> list[dict]:
                urls = {
                    str(u).strip()
                    for u in result.get(url_key, [])
                    if str(u).strip()
                }
                if urls:
                    return [
                        article
                        for article in original_articles
                        if canonicalize_url(
                            article.get("url_normalized") or article.get("url") or ""
                        ) in urls
                    ]

                zero_based = {
                    int(i) - 1
                    for i in result.get(index_key, [])
                    if isinstance(i, int) or str(i).isdigit()
                }
                return [
                    article
                    for idx, article in enumerate(original_articles)
                    if idx in zero_based
                ]

            core_articles = select_membership("core_urls", "core_indexes")
            gpt_related_articles = select_membership("related_urls", "related_indexes")

            # CORE is always part of RELATED. Preserve original cluster order.
            core_ids = {id(article) for article in core_articles}
            related_ids = {id(article) for article in gpt_related_articles}
            if core_ids - related_ids:
                gpt_related_articles = [
                    article
                    for article in original_articles
                    if id(article) in (related_ids | core_ids)
                ]

            core_domains = {
                domain_from_url(a.get("url_normalized") or a.get("url") or "")
                for a in core_articles
            }
            core_domains.discard("")

            if (
                len(core_articles) < MIN_PUBLISH_ARTICLES
                or len(core_domains) < MIN_PUBLISH_DOMAINS
            ):
                if purity_report:
                    print(
                        f"[safety] topic={c.get('topic')} action={action} "
                        f"decision=REJECT reason=insufficient_core_support "
                        f"core_articles={len(core_articles)} core_domains={len(core_domains)}"
                    )
                continue

            related_articles, related_filter_diag = filter_related_articles_to_core(
                core_articles,
                original_articles,
                result.get("event") or "",
                gpt_related_articles=gpt_related_articles,
            )

            related_domains = {
                domain_from_url(a.get("url_normalized") or a.get("url") or "")
                for a in related_articles
            }
            related_domains.discard("")

            original_titled_keys = {
                canonicalize_url(a.get("url_normalized") or a.get("url") or "")
                or (a.get("title") or "").strip().lower()
                for a in original_articles
                if (a.get("title") or "").strip()
            }
            final_related_keys = {
                canonicalize_url(a.get("url_normalized") or a.get("url") or "")
                or (a.get("title") or "").strip().lower()
                for a in related_articles
                if (a.get("title") or "").strip()
            }
            final_action = (
                "KEEP"
                if final_related_keys == original_titled_keys
                else "CLEAN"
            )

            published = dict(c)
            published["articles"] = core_articles
            published["related_articles"] = related_articles
            published["source_diversity"] = source_diversity(core_articles)
            published["source_concentration"] = compute_source_concentration(core_articles)
            published["bias_distribution"] = aggregate_bias_distribution(core_articles)
            published["today_ratio"] = round(
                cluster_today_ratio(core_articles, date_str), 3
            )
            published["eventness_label"] = "SINGLE_EVENT"
            published["event_type"] = "DISCRETE_EVENT"
            published["canonical_event"] = result.get("event") or ""
            published["related_article_count"] = len(related_articles)
            published["related_domain_count"] = len(related_domains)
            published["related_source_diversity"] = source_diversity(related_articles)

            core_ids = {id(article) for article in core_articles}
            related_ids = {id(article) for article in related_articles}
            published["publication_purification"] = {
                "action": final_action,
                "gpt_membership_action": action,
                "event_type": "DISCRETE_EVENT",
                "from": len(original_articles),
                "to": len(core_articles),
                "core_to": len(core_articles),
                "gpt_related_to": len(gpt_related_articles),
                "related_to": len(related_articles),
                "related_only_titles": [
                    (article.get("title") or "").strip()
                    for article in original_articles
                    if id(article) in related_ids and id(article) not in core_ids
                ],
                "removed_titles": [
                    (article.get("title") or "").strip()
                    for article in original_articles
                    if id(article) not in related_ids
                ],
                "local_related_filter": related_filter_diag,
                "why": result.get("why") or "",
            }

            # Rank from locally verified event-specific RELATED coverage while
            # summarizing strictly from CORE.
            related_attention = attention_metadata_from_articles(related_articles)
            published.update(related_attention)
            published["attention_score"] = attention_score(published)

            if gdelt_ranking_shadow:
                local_signal = _local_attention_percentile(
                    float(published.get("attention_score", 0.0) or 0.0),
                    shadow_local_reference_scores,
                )
                retained_evidence = _gdelt_global_attention_from_articles(
                    related_articles,
                    min_retained_articles=2,
                )
                catalog_options = _gdelt_catalog_options_for_cluster(
                    published,
                    gdelt_runtime_module,
                    gdelt_catalog_matcher,
                    threshold=GDELT_CATALOG_FINAL_MATCH_THRESHOLD,
                    max_matches=5,
                )
                published["_gdelt_catalog_match_options"] = catalog_options
                catalog_evidence = _gdelt_option_evidence(
                    catalog_options[0] if catalog_options else None,
                    gdelt_catalog_diag.get("file"),
                )
                global_evidence = _combine_gdelt_shadow_evidence(
                    retained_evidence,
                    catalog_evidence,
                )
                global_signal = (
                    global_evidence.get("global_signal")
                    if global_evidence.get("eligible")
                    else None
                )
                published["_gdelt_shadow_local_signal"] = local_signal
                published["_gdelt_shadow_global_signal"] = global_signal
                published["_gdelt_shadow_score"] = _gdelt_shadow_blended_score(
                    local_signal,
                    global_signal,
                )
                published["_gdelt_shadow_evidence"] = global_evidence

            vetted.append(published)

            if purity_report:
                local_removed = len(related_filter_diag.get("removed", []))
                local_restored = len(related_filter_diag.get("restored", []))
                print(
                    f"[safety] topic={c.get('topic')} event_type=DISCRETE_EVENT "
                    f"action={final_action} decision=KEEP "
                    f"core={len(core_articles)}/{len(core_domains)}dom "
                    f"related={len(related_articles)}/{len(related_domains)}dom "
                    f"local_related_restored={local_restored} "
                    f"local_related_removed={local_removed}"
                )

        if purifier_stop_reason == "queue_exhausted":
            purifier_stop_reason = (
                "queue_exhausted_rank_complete"
                if len(vetted) >= top_k
                else "queue_exhausted_before_target"
            )

        if gdelt_ranking_shadow and vetted:
            resolved = _resolve_all_approved_gdelt_matches(
                vetted,
                gdelt_runtime_module,
                gdelt_catalog_matcher,
                shadow_local_reference_scores,
                apply=True,
            )
            gdelt_assignment_stats = resolved.get("stats") or {}
            print(
                "🌐 Full-audit GDELT assignments: "
                f"{gdelt_assignment_stats.get('matched_approved_event_count', 0)} "
                f"of {len(vetted)} approved events matched; "
                f"{gdelt_assignment_stats.get('unmatched_approved_event_count', 0)} unmatched"
            )

        ranked = sorted(
            vetted,
            key=lambda c: (
                float(c.get("attention_score", 0.0) or 0.0),
                importance_score(c),
            ),
            reverse=True,
        )

        finish_reason_text = ", ".join(
            f"{reason}:{count}"
            for reason, count in sorted(_EVENT_FINISH_REASONS.items())
        ) or "none"

        print(
            "🧾 Event purifier: "
            f"{event_calls} live calls, {event_cache_hits} cache hits, "
            f"{reviewed_candidates}/{len(review_queue)} candidates reviewed, "
            f"{_EVENT_TOKEN_USAGE['prompt_tokens']} input tokens, "
            f"{_EVENT_TOKEN_USAGE['completion_tokens']} output tokens, "
            f"finish_reasons={finish_reason_text}, "
            f"stop={purifier_stop_reason}, cap={EVENT_MAX_CALLS}"
        )

    if gdelt_ranking_shadow:
        if vetted:
            _write_gdelt_ranking_shadow(
                path=gdelt_ranking_shadow_file,
                date_str=date_str,
                input_file=input_file,
                output_file=output_file,
                approved_events=vetted,
                top_k=top_k,
                review_queue_size=len(review_queue),
                reviewed_candidates=reviewed_candidates,
                event_calls=event_calls,
                event_cache_hits=event_cache_hits,
                purifier_stop_reason=purifier_stop_reason,
                gdelt_catalog_diag=gdelt_catalog_diag,
                gdelt_assignment_stats=gdelt_assignment_stats,
                global_output_file=gdelt_global_output_file,
            )

            shadow_top = sorted(
                vetted,
                key=lambda item: (
                    float(item.get("_gdelt_shadow_score", 0.0) or 0.0),
                    float(item.get("attention_score", 0.0) or 0.0),
                    importance_score(item),
                ),
                reverse=True,
            )[: max(1, top_k)]
            local_top_ids = {
                _shadow_event_id(item)
                for item in ranked[: max(1, top_k)]
            }

            # Materialize the purifier-approved global top-K as an isolated
            # shadow file. This never changes the production final JSON.
            local_rank_by_id = {
                _shadow_event_id(item): pos
                for pos, item in enumerate(ranked, start=1)
            }
            global_output = _clusters_for_output(shadow_top)
            for pos, (original, exported) in enumerate(
                zip(shadow_top, global_output),
                start=1,
            ):
                event_id = _shadow_event_id(original)
                exported["global_attention_rank"] = pos
                exported["local_attention_rank"] = local_rank_by_id.get(event_id)
                exported["global_attention_score"] = round(
                    float(original.get("_gdelt_shadow_score", 0.0) or 0.0),
                    4,
                )
                exported["ranking_mode"] = "GLOBAL_COVERAGE_SHADOW"

            Path(gdelt_global_output_file).write_text(
                json.dumps(global_output, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            promoted_titles = [
                _shadow_event_title(item)
                for item in shadow_top
                if _shadow_event_id(item) not in local_top_ids
            ]
            print(
                f"🌐 Wrote GDELT global-ranking shadow → "
                f"{gdelt_ranking_shadow_file}"
            )
            print(
                f"🌍 Wrote purifier-approved global top-{max(1, top_k)} → "
                f"{gdelt_global_output_file}"
            )
            if promoted_titles:
                print(
                    "   Shadow top-K promotions: "
                    + " | ".join(promoted_titles[:5])
                )
            else:
                print("   Shadow top-K promotions: none")
        else:
            payload = {
                "schema_version": "1.0",
                "date": date_str,
                "status": "PURIFIER_NOT_RUN_OR_NO_APPROVED_EVENTS",
                "shadow_only": True,
                "live_page_order_changed_by_shadow_score": False,
                "local_ranking_formula_unchanged": True,
                "input_file": input_file,
                "local_output_file": output_file,
            }
            Path(gdelt_ranking_shadow_file).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(
                f"⚠️ GDELT ranking shadow had no purifier-approved events → "
                f"{gdelt_ranking_shadow_file}"
            )

    # Save cache if we made any calls
    if event_calls > 0:
        _save_event_cache(event_cache)

    # Cap top-K AFTER filtering
    capped = ranked[: max(1, top_k)]

    # Remove run-only diagnostic keys before writing JSON.
    capped = _clusters_for_output(capped)

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(capped, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(capped)} high-precision clusters to {output_file}")

    # Optional console report (token-free)
    if print_report:
        print("\n=== Final clusters (console report) ===")
        for idx, c in enumerate(capped, start=1):
            arts = c.get("articles", [])
            membership_diag = c.get("membership_extraction") or {}
            membership_text = ""
            if membership_diag.get("membership_trimmed"):
                membership_text = (
                    f" membership="
                    f"{membership_diag.get('membership_from')}→{membership_diag.get('membership_to')}"
                )

            print(
                f"\n[{idx}] size={len(arts)} "
                f"entropy={(c.get('source_diversity') or {}).get('entropy',0)} "
                f"attention={c.get('attention_article_count', len(arts))}/"
                f"{c.get('attention_domain_count', 0)}dom "
                f"attention_score={c.get('attention_score', 0)}"
                f"{membership_text}"
            )
            for a in arts[:5]:
                print("   -", (a.get("title") or a.get("url") or "").strip())
        print("\n(Only first 5 article titles per cluster shown.)")


if __name__ == "__main__":
    main()
