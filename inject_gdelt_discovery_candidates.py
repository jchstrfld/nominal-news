#!/usr/bin/env python3
"""
inject_gdelt_discovery_candidates.py

Optional, fail-open bridge between the GDELT discovery audit and the existing
Nominal News merge/purity pipeline.

Reads:
  - clustered_articles_{date}.json
  - gdelt_discovery_candidates_{date}.json

Writes only when the candidate feed passes validation:
  - clustered_articles_with_gdelt_{date}.json
  - gdelt_discovery_injection_{date}.json

Safety properties:
  - Never overwrites clustered_articles_{date}.json.
  - Appends pre-grouped GDELT seeds after all existing clusters, preserving the
    existing baseline cluster order and baseline-to-baseline merge behavior.
  - Requires four English, net-new URLs from four domains and four write-up
    families for each injected event.
  - Carries GDELT breadth as shadow metadata only; it does not directly alter
    production ranking.
  - If the audit/candidate feed is missing or invalid, removes any stale
    augmented output, writes a SKIPPED manifest, and exits successfully unless
    --strict is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


BIAS_ORDER = [
    "Far Left",
    "Left",
    "Center",
    "Right",
    "Far Right",
    "Unknown",
]
VALID_BIASES = set(BIAS_ORDER)
VALID_PREVIEW_TYPES = {
    "NEW_DISCOVERY",
    "UNDERREPRESENTED_UPSTREAM",
}
TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "spm",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate and append bounded GDELT discovery seed clusters."
    )
    p.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    p.add_argument("--input-file", default=None)
    p.add_argument("--audit-file", default=None)
    p.add_argument("--output-file", default=None)
    p.add_argument("--manifest-file", default=None)
    p.add_argument("--bias-file", default="bias_overrides.json")
    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--max-records", type=int, default=48)
    p.add_argument("--articles-per-candidate", type=int, default=4)
    p.add_argument(
        "--no-update-bias-overrides",
        action="store_true",
        help="Label unknown domains Unknown without appending them to bias_overrides.json.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero for a skipped/invalid optional feed.",
    )
    return p.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def version_tuple(value: Any) -> tuple[int, ...]:
    nums = re.findall(r"\d+", str(value or ""))
    return tuple(int(x) for x in nums[:3]) if nums else ()


def normalize_url(url: str) -> str:
    """Match the existing normalizer's URL shape as closely as possible."""
    if not url:
        return ""
    try:
        p = urlparse(url.strip())
        netloc = (p.netloc or "").lower()
        if netloc.startswith("m."):
            netloc = netloc[2:]
        q_pairs = [
            (k, v)
            for k, v in parse_qsl(p.query, keep_blank_values=True)
            if k.lower() not in TRACKING_KEYS
        ]
        query = urlencode(q_pairs, doseq=True)
        path = p.path.rstrip("/") or "/"
        return urlunparse((p.scheme or "https", netloc, path, "", query, ""))
    except Exception:
        return (url or "").strip()


def canonical_url_key(url: str) -> str:
    try:
        p = urlparse(normalize_url(url))
        host = (p.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = re.sub(r"/+$", "", p.path or "")
        q_pairs = sorted(parse_qsl(p.query, keep_blank_values=True))
        query = urlencode(q_pairs, doseq=True)
        return f"{host}{path}?{query}" if query else f"{host}{path}"
    except Exception:
        return (url or "").strip().lower()


def domain_from_url(url: str) -> str:
    try:
        host = (urlparse(url or "").hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def canonical_bias(value: Any) -> str:
    raw = str(value or "Unknown").strip().title().replace("-", " ")
    raw = re.sub(r"\s+", " ", raw)
    return raw if raw in VALID_BIASES else "Unknown"


def load_bias_overrides(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = load_json(path)
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}

    lookup: dict[str, str] = {}
    for domain, meta in raw.items():
        key = str(domain or "").strip().lower()
        if key.startswith("www."):
            key = key[4:]
        if not key:
            continue
        value = meta.get("bias") if isinstance(meta, dict) else meta
        lookup[key] = canonical_bias(value)
    return lookup, raw


def compute_bias_distribution(articles: list[dict]) -> dict[str, int]:
    counts = Counter(canonical_bias(a.get("bias")) for a in articles)
    total = sum(counts.values())
    if not total:
        return {}
    exact = {k: counts[k] * 100.0 / total for k in BIAS_ORDER}
    floors = {k: int(exact[k]) for k in BIAS_ORDER}
    left = 100 - sum(floors.values())
    for key in sorted(
        BIAS_ORDER,
        key=lambda k: exact[k] - floors[k],
        reverse=True,
    ):
        if left <= 0:
            break
        floors[key] += 1
        left -= 1
    return {k: floors[k] for k in BIAS_ORDER if floors.get(k, 0) > 0}


def skip_optional(
    *,
    reason: str,
    output_path: Path,
    manifest_path: Path,
    input_path: Path,
    audit_path: Path,
    strict: bool,
    details: dict[str, Any] | None = None,
) -> int:
    output_path.unlink(missing_ok=True)
    manifest = {
        "status": "SKIPPED",
        "reason": reason,
        "date": None,
        "base_cluster_file": str(input_path),
        "candidate_feed_file": str(audit_path),
        "augmented_cluster_file": str(output_path),
        "details": details or {},
    }
    atomic_write_json(manifest_path, manifest)
    print(f"⚠️ GDELT discovery injection skipped: {reason}")
    print("↪ Continuing safely with the original clustered-articles file.")
    return 2 if strict else 0


def validate_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def main() -> int:
    args = parse_args()
    if not validate_date(args.date):
        print("❌ Invalid --date. Use YYYY-MM-DD.")
        return 2

    input_path = Path(args.input_file or f"clustered_articles_{args.date}.json")
    audit_path = Path(
        args.audit_file or f"gdelt_discovery_candidates_{args.date}.json"
    )
    output_path = Path(
        args.output_file or f"clustered_articles_with_gdelt_{args.date}.json"
    )
    manifest_path = Path(
        args.manifest_file or f"gdelt_discovery_injection_{args.date}.json"
    )
    bias_path = Path(args.bias_file)

    # Stale augmented data must never survive a failed optional run.
    output_path.unlink(missing_ok=True)

    if not input_path.exists():
        return skip_optional(
            reason="base_cluster_file_missing",
            output_path=output_path,
            manifest_path=manifest_path,
            input_path=input_path,
            audit_path=audit_path,
            strict=args.strict,
        )
    if not audit_path.exists():
        return skip_optional(
            reason="candidate_feed_missing",
            output_path=output_path,
            manifest_path=manifest_path,
            input_path=input_path,
            audit_path=audit_path,
            strict=args.strict,
        )

    try:
        base_clusters = load_json(input_path)
        audit = load_json(audit_path)
    except Exception as e:
        return skip_optional(
            reason="json_load_failed",
            output_path=output_path,
            manifest_path=manifest_path,
            input_path=input_path,
            audit_path=audit_path,
            strict=args.strict,
            details={"error": f"{type(e).__name__}: {e}"},
        )

    if not isinstance(base_clusters, list) or not all(
        isinstance(c, dict) and isinstance(c.get("articles"), list)
        for c in base_clusters
    ):
        return skip_optional(
            reason="invalid_base_cluster_schema",
            output_path=output_path,
            manifest_path=manifest_path,
            input_path=input_path,
            audit_path=audit_path,
            strict=args.strict,
        )

    if not isinstance(audit, dict):
        return skip_optional(
            reason="invalid_candidate_feed_schema",
            output_path=output_path,
            manifest_path=manifest_path,
            input_path=input_path,
            audit_path=audit_path,
            strict=args.strict,
        )

    checks = {
        "status_ok": audit.get("status") == "OK",
        "date_matches": audit.get("date") == args.date,
        "schema_supported": version_tuple(audit.get("audit_schema_version")) >= (4, 1),
        "read_only_origin": audit.get("read_only") is True,
        "no_prior_production_writes": not bool(audit.get("production_files_modified")),
        "preview_is_list": isinstance(audit.get("ingestion_preview"), list),
    }
    if not all(checks.values()):
        return skip_optional(
            reason="candidate_feed_validation_failed",
            output_path=output_path,
            manifest_path=manifest_path,
            input_path=input_path,
            audit_path=audit_path,
            strict=args.strict,
            details={"checks": checks},
        )

    max_candidates = max(0, min(int(args.max_candidates), 12))
    max_records = max(0, min(int(args.max_records), 48))
    per_candidate = max(4, int(args.articles_per_candidate))
    if per_candidate != 4:
        # Production rollout is intentionally fixed to the validated 4-record seed.
        per_candidate = 4

    existing_urls = {
        canonical_url_key(a.get("url_normalized") or a.get("url") or "")
        for c in base_clusters
        for a in c.get("articles", [])
        if canonical_url_key(a.get("url_normalized") or a.get("url") or "")
    }
    reserved_urls = set(existing_urls)
    seen_candidate_ids: set[str] = set()
    bias_lookup, bias_raw = load_bias_overrides(bias_path)
    unknown_domains: set[str] = set()
    seed_clusters: list[dict] = []
    accepted: list[dict] = []
    rejected: list[dict] = []

    preview = sorted(
        audit.get("ingestion_preview") or [],
        key=lambda x: int(x.get("preview_rank", 10**9) or 10**9),
    )

    for item in preview:
        if len(seed_clusters) >= max_candidates:
            break
        if sum(len(c["articles"]) for c in seed_clusters) + per_candidate > max_records:
            break

        candidate_id = str(item.get("candidate_id") or "").strip()
        title = re.sub(r"\s+", " ", str(item.get("canonical_title") or "")).strip()
        preview_type = str(item.get("preview_type") or "").strip()
        global_attention = item.get("global_attention") or {}
        selection = item.get("selection") or {}
        articles = item.get("articles") or []

        reasons = []
        if not candidate_id or candidate_id in seen_candidate_ids:
            reasons.append("missing_or_duplicate_candidate_id")
        if len(title) < 20:
            reasons.append("canonical_title_too_short")
        if preview_type not in VALID_PREVIEW_TYPES:
            reasons.append("unsupported_preview_type")
        if not isinstance(global_attention, dict):
            reasons.append("global_attention_missing")
        else:
            if global_attention.get("target_date") != args.date:
                reasons.append("global_attention_date_mismatch")
            if global_attention.get("shadow_only") is not True:
                reasons.append("global_attention_not_shadow_only")
        if int(selection.get("selected_distinct_domain_count", 0) or 0) < 4:
            reasons.append("fewer_than_four_selected_domains")
        if int(selection.get("selected_distinct_family_count", 0) or 0) < 4:
            reasons.append("fewer_than_four_selected_writeup_families")
        if not isinstance(articles, list) or len(articles) < per_candidate:
            reasons.append("fewer_than_four_articles")

        prepared: list[dict] = []
        local_urls: set[str] = set()
        local_domains: set[str] = set()
        local_families: set[str] = set()

        if not reasons:
            for article in articles:
                if len(prepared) >= per_candidate:
                    break
                if not isinstance(article, dict):
                    continue
                article_title = re.sub(
                    r"\s+", " ", str(article.get("title") or "")
                ).strip()
                raw_url = str(article.get("url") or "").strip()
                language = str(article.get("language") or "").strip().lower()
                article_candidate_id = str(
                    article.get("gdelt_candidate_id") or ""
                ).strip()
                family_id = str(
                    article.get("gdelt_writeup_family_id") or ""
                ).strip()
                url_key = canonical_url_key(raw_url)
                domain = domain_from_url(raw_url)

                if len(article_title) < 12:
                    continue
                if not raw_url.startswith(("http://", "https://")):
                    continue
                if language not in {"en", "eng", "english"}:
                    continue
                if article_candidate_id != candidate_id:
                    continue
                if not family_id or family_id in local_families:
                    continue
                if not domain or domain in local_domains:
                    continue
                if not url_key or url_key in reserved_urls or url_key in local_urls:
                    continue

                bias = bias_lookup.get(domain, "Unknown")
                if bias == "Unknown" and domain not in bias_raw:
                    unknown_domains.add(domain)

                record = dict(article)
                record.update({
                    "title": article_title,
                    "description": str(article.get("description") or ""),
                    "url": raw_url,
                    "url_normalized": normalize_url(raw_url),
                    "source": str(article.get("source") or domain),
                    "published_at": str(article.get("published_at") or ""),
                    "published_date": args.date,
                    "language": "en",
                    "bias": bias,
                    "origin": "gdelt_discovery",
                    "gdelt_candidate_id": candidate_id,
                    "gdelt_preview_type": preview_type,
                    "gdelt_canonical_title": title,
                    "gdelt_discovery_score": float(
                        item.get("discovery_score", 0.0) or 0.0
                    ),
                    "gdelt_global_attention": dict(global_attention),
                    "gdelt_global_attention_shadow_only": True,
                })
                prepared.append(record)
                local_urls.add(url_key)
                local_domains.add(domain)
                local_families.add(family_id)

        if len(prepared) != per_candidate:
            reasons.append("four_unique_valid_records_not_available")

        if reasons:
            rejected.append({
                "candidate_id": candidate_id,
                "canonical_title": title,
                "reasons": sorted(set(reasons)),
            })
            continue

        seen_candidate_ids.add(candidate_id)
        reserved_urls.update(local_urls)
        seed = {
            "topic": f"GDELT discovery: {title}",
            "articles": prepared,
            "bias_distribution": compute_bias_distribution(prepared),
            "cluster_origin": "gdelt_discovery",
            "gdelt_candidate_id": candidate_id,
            "gdelt_preview_type": preview_type,
            "gdelt_canonical_title": title,
            "gdelt_discovery_score": float(item.get("discovery_score", 0.0) or 0.0),
            "gdelt_global_attention_shadow": dict(global_attention),
            "gdelt_seed_article_count": len(prepared),
            "gdelt_seed_domain_count": len(local_domains),
            "gdelt_seed_writeup_family_count": len(local_families),
        }
        seed_clusters.append(seed)
        accepted.append({
            "candidate_id": candidate_id,
            "preview_type": preview_type,
            "canonical_title": title,
            "article_count": len(prepared),
            "domain_count": len(local_domains),
            "writeup_family_count": len(local_families),
        })

    if not seed_clusters:
        return skip_optional(
            reason="no_candidate_passed_injection_validation",
            output_path=output_path,
            manifest_path=manifest_path,
            input_path=input_path,
            audit_path=audit_path,
            strict=args.strict,
            details={"rejected_candidates": rejected},
        )

    # Existing clusters remain byte-for-byte equivalent as JSON values and in
    # their original order. GDELT seeds are appended only after them.
    augmented = [*base_clusters, *seed_clusters]
    atomic_write_json(output_path, augmented)

    added_override_domains: list[str] = []
    if not args.no_update_bias_overrides and unknown_domains:
        current_lookup, current_raw = load_bias_overrides(bias_path)
        normalized_existing = {
            str(key or "").strip().lower().removeprefix("www.")
            for key in current_raw
        }
        for domain in sorted(unknown_domains):
            if domain not in normalized_existing:
                current_raw[domain] = {
                    "bias": "Unknown",
                    "sources": ["gdelt-discovery-auto-seen"],
                    "notes": "",
                }
                added_override_domains.append(domain)
        if added_override_domains:
            atomic_write_json(bias_path, current_raw)

    manifest = {
        "status": "OK",
        "date": args.date,
        "mode": "optional_gdelt_discovery",
        "base_cluster_file": str(input_path),
        "candidate_feed_file": str(audit_path),
        "augmented_cluster_file": str(output_path),
        "base_cluster_sha256": sha256_file(input_path),
        "candidate_feed_sha256": sha256_file(audit_path),
        "baseline_cluster_count": len(base_clusters),
        "seed_cluster_count": len(seed_clusters),
        "output_cluster_count": len(augmented),
        "injected_article_count": sum(len(c["articles"]) for c in seed_clusters),
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "unknown_bias_domains_added": added_override_domains,
        "safety": {
            "baseline_input_overwritten": False,
            "baseline_cluster_order_preserved": True,
            "seeds_appended_after_baseline": True,
            "minimum_articles_per_seed": 4,
            "minimum_domains_per_seed": 4,
            "minimum_writeup_families_per_seed": 4,
            "global_attention_affects_live_rank": False,
        },
    }
    atomic_write_json(manifest_path, manifest)

    print(
        f"✅ Prepared optional GDELT discovery input: "
        f"{len(base_clusters)} baseline + {len(seed_clusters)} seed clusters"
    )
    print(
        f"📥 Injected {manifest['injected_article_count']} articles across "
        f"{len(seed_clusters)} pre-grouped candidate events"
    )
    if rejected:
        print(f"🧹 Rejected {len(rejected)} candidates during injection validation")
    if added_override_domains:
        print(
            f"🧩 Added {len(added_override_domains)} newly seen domains to "
            f"{bias_path} as Unknown"
        )
    print(f"💾 Wrote augmented clusters → {output_path}")
    print(f"🧾 Wrote injection manifest → {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
