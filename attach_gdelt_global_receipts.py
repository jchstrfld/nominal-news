#!/usr/bin/env python3
"""
attach_gdelt_global_receipts.py

Shadow-only bridge from the GDELT Global Similarity Graph candidate catalog to
purifier-approved, globally ranked Nominal News events.

It never changes event membership, core articles, related articles, ranking,
or production files. It adds:
- all-language GSG receipt metadata for auditability;
- English-only source links for the current user-facing receipts list;
- separate global outlet/write-up counts.

No network or OpenAI calls are made.
The bridge fails closed if the ranking assignment and receipt catalog come
from different audit runs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

try:
    from bias_labeler import lookup_bias_by_domain, canonicalize
except Exception:
    lookup_bias_by_domain = None

    def canonicalize(value):
        return value or "Unknown"


ORIGIN_PRIORITY = {
    "core": 0,
    "local": 1,
    "gdelt": 2,
    "coverage": 3,
    "gdelt_gsg": 4,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attach GSG source receipts to an isolated global shadow output."
    )
    p.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    p.add_argument(
        "--input-file",
        default=None,
        help=(
            "Expanded purifier-approved global shadow JSON. Defaults to "
            "grouped_articles_final_global_expanded_shadow_{date}.json."
        ),
    )
    p.add_argument(
        "--candidate-file",
        default=None,
        help=(
            "Pipeline-mode GDELT candidate feed containing "
            "global_source_receipt_catalog."
        ),
    )
    p.add_argument(
        "--ranking-file",
        default=None,
        help="GDELT global-ranking shadow JSON used to assign candidate IDs.",
    )
    p.add_argument(
        "--output-file",
        default=None,
        help="Separate shadow output path.",
    )
    return p.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clusters_from(data):
    if isinstance(data, list):
        return data, "list"
    if isinstance(data, dict) and isinstance(data.get("clusters"), list):
        return data["clusters"], "dict"
    raise ValueError("Input must be a cluster list or a dict with a clusters list.")


def domain_from_url(url: str) -> str:
    try:
        host = (urlparse(url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def stable_membership_hash(clusters: list[dict]) -> str:
    payload = []
    for cluster in clusters:
        payload.append({
            "articles": [a.get("url") for a in cluster.get("articles", [])],
            "related_articles": [
                a.get("url") for a in cluster.get("related_articles", [])
            ],
            "global_attention_rank": cluster.get("global_attention_rank"),
        })
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_bias(url: str, fallback: str = "Unknown") -> str:
    if lookup_bias_by_domain is None:
        return canonicalize(fallback or "Unknown")
    try:
        found = lookup_bias_by_domain(url)
    except Exception:
        found = None
    return canonicalize(found or fallback or "Unknown")


def ranking_candidate_map(ranking: dict) -> dict[int, dict]:
    rows = (
        ranking.get("global_shadow_top_k")
        or ranking.get("global_top_k")
        or []
    )
    out = {}
    for row in rows:
        try:
            rank = int(row.get("global_shadow_rank"))
        except Exception:
            continue
        attention = row.get("gdelt_global_attention") or {}
        candidate_id = attention.get("primary_candidate_id")
        if not candidate_id:
            candidates = attention.get("candidates") or []
            if candidates:
                candidate_id = candidates[0].get("candidate_id")
        out[rank] = {
            "candidate_id": candidate_id,
            "event_id": row.get("event_id"),
            "topic": row.get("topic"),
            "title": row.get("title"),
            "source_type": row.get("source_type"),
            "global_signal": row.get("global_signal"),
            "global_shadow_score": row.get("global_shadow_score"),
        }
    return out


def source_priority(source: dict) -> tuple:
    origin = source.get("origin") or "coverage"
    return (
        ORIGIN_PRIORITY.get(origin, 9),
        int(bool(source.get("syndicated"))),
        int((source.get("bias") or "Unknown") == "Unknown"),
        int(not bool(source.get("url"))),
    )


def merge_english_receipts(
    existing: list[dict],
    english_receipts: list[dict],
) -> list[dict]:
    by_domain = {}

    for raw in existing or []:
        row = dict(raw)
        domain = (row.get("domain") or domain_from_url(row.get("url") or ""))
        domain = domain.lower().strip()
        if not domain:
            continue
        row["domain"] = domain
        row["bias"] = normalize_bias(row.get("url") or f"https://{domain}/", row.get("bias"))
        current = by_domain.get(domain)
        if current is None or source_priority(row) < source_priority(current):
            by_domain[domain] = row

    for raw in english_receipts:
        row = dict(raw)
        domain = (row.get("domain") or domain_from_url(row.get("url") or ""))
        domain = domain.lower().strip()
        if not domain:
            continue

        row["domain"] = domain
        row["source"] = row.get("source") or domain
        row["origin"] = "gdelt_gsg"
        row["syndicated"] = bool(row.get("syndicated"))
        row["bias"] = normalize_bias(
            row.get("url") or f"https://{domain}/",
            row.get("bias") or "Unknown",
        )
        row["gdelt_gsg_receipt"] = True

        current = by_domain.get(domain)
        if current is None:
            by_domain[domain] = row
        else:
            # Preserve the existing core/local/DOC receipt as the visible link,
            # but annotate that GSG independently observed the same outlet.
            current["gdelt_gsg_confirmed"] = True
            current.setdefault(
                "gdelt_writeup_family_id", row.get("writeup_family_id")
            )
            current.setdefault(
                "gdelt_writeup_family_outlet_count",
                row.get("writeup_family_outlet_count"),
            )

    return sorted(
        by_domain.values(),
        key=lambda row: (
            ORIGIN_PRIORITY.get(row.get("origin") or "coverage", 9),
            row.get("domain") or "",
        ),
    )


def prepare_writeup_families(bundle: dict) -> tuple[list[dict], set[str]]:
    """
    Normalize the family-level receipt catalog while preserving every distinct
    write-up and every outlet copy beneath that write-up.
    """
    prepared = []
    unknown_domains = set()

    for raw_family in bundle.get("writeup_families") or []:
        family = dict(raw_family)
        receipts = []
        seen_domains = set()

        for raw in family.get("receipts") or []:
            row = dict(raw)
            domain = (
                row.get("domain")
                or domain_from_url(row.get("url") or "")
            )
            domain = domain.lower().strip()
            url = (row.get("url") or "").strip()
            if not domain or not url or domain in seen_domains:
                continue
            seen_domains.add(domain)

            row["domain"] = domain
            row["source"] = row.get("source") or domain
            row["origin"] = "gdelt_gsg"
            row["syndicated"] = bool(
                row.get("syndicated")
                or int(family.get("outlet_count", 0) or 0) > 1
            )
            row["bias"] = normalize_bias(
                url or f"https://{domain}/",
                row.get("bias") or "Unknown",
            )
            row["gdelt_gsg_receipt"] = True
            if row["bias"] == "Unknown":
                unknown_domains.add(domain)
            receipts.append(row)

        receipts.sort(
            key=lambda row: (
                not bool(row.get("displayable_english")),
                0 if row.get("day_basis") == "TARGET_EXPLICIT" else 1,
                row.get("domain") or "",
            )
        )

        representative_url = (family.get("representative_url") or "").strip()
        representative_title = (family.get("representative_title") or "").strip()

        representative = None
        if representative_url:
            for row in receipts:
                if (row.get("url") or "").strip() == representative_url:
                    representative = row
                    break
        if representative is None:
            representative = next(
                (row for row in receipts if row.get("displayable_english")),
                receipts[0] if receipts else None,
            )
        if representative is not None:
            representative_url = representative_url or representative.get("url") or ""
            representative_title = (
                representative_title
                or representative.get("title")
                or ""
            )

        family["representative_url"] = representative_url
        family["representative_title"] = representative_title
        family["receipts"] = receipts
        family["receipt_outlet_count"] = len(receipts)
        family["english_receipt_count"] = sum(
            1 for row in receipts if row.get("displayable_english")
        )
        family["non_english_receipt_count"] = (
            len(receipts) - family["english_receipt_count"]
        )
        family["displayable_english"] = bool(
            family["english_receipt_count"] and representative_url
        )
        family["receipt_catalog_complete"] = bool(
            len(receipts) == int(family.get("outlet_count", 0) or 0)
        )
        prepared.append(family)

    prepared.sort(
        key=lambda row: (
            int(row.get("outlet_count", 0) or 0),
            int(row.get("active_window_count", 0) or 0),
        ),
        reverse=True,
    )
    return prepared, unknown_domains


def candidate_id_from_cluster(cluster: dict) -> str | None:
    ids = {
        a.get("gdelt_candidate_id")
        for a in (
            list(cluster.get("articles") or [])
            + list(cluster.get("related_articles") or [])
        )
        if a.get("gdelt_candidate_id")
    }
    return next(iter(ids)) if len(ids) == 1 else None


def main() -> int:
    args = parse_args()
    d = args.date

    input_path = Path(
        args.input_file
        or f"grouped_articles_final_global_expanded_shadow_{d}.json"
    )
    candidate_path = Path(
        args.candidate_file or f"gdelt_discovery_candidates_{d}.json"
    )
    ranking_path = Path(
        args.ranking_file or f"gdelt_global_ranking_shadow_{d}.json"
    )
    output_path = Path(
        args.output_file
        or f"grouped_articles_final_global_receipts_shadow_{d}.json"
    )

    for path in (input_path, candidate_path, ranking_path):
        if not path.exists():
            print(f"❌ Missing required file: {path}")
            return 1

    raw_input = load_json(input_path)
    input_clusters, shape = clusters_from(raw_input)
    candidate_doc = load_json(candidate_path)
    ranking_doc = load_json(ranking_path)

    receipt_catalog = candidate_doc.get("global_source_receipt_catalog") or {}
    catalog_candidates = receipt_catalog.get("candidates") or {}
    if not catalog_candidates:
        print(
            "❌ Candidate feed has no global_source_receipt_catalog. "
            "Rerun the updated audit in --pipeline-mode first."
        )
        return 1

    rank_map = ranking_candidate_map(ranking_doc)

    # Fail closed when the ranking shadow and candidate feed were generated
    # from different audit runs. Candidate IDs are the authoritative join key;
    # silently attaching only the IDs that still happen to exist would create
    # a partially inconsistent source list.
    assigned_candidate_ids = {
        row.get("candidate_id")
        for row in rank_map.values()
        if row.get("candidate_id")
    }
    missing_assigned_ids = sorted(
        assigned_candidate_ids - set(catalog_candidates.keys())
    )
    if missing_assigned_ids:
        print(
            "❌ Ranking shadow and candidate receipt catalog are out of sync. "
            "Rerun final_cohesion_check.py with the current candidate feed, "
            "then rerun this bridge."
        )
        print(
            "   Missing assigned candidate IDs: "
            + ", ".join(missing_assigned_ids)
        )
        return 2

    before_hash = stable_membership_hash(input_clusters)
    clusters = copy.deepcopy(input_clusters)

    used_candidate_ids = set()
    attached = 0
    unmatched = 0
    total_global_outlets = 0
    total_english_receipts = 0
    total_non_english_receipts = 0
    total_writeup_families = 0
    unknown_bias_domains = set()

    for index, cluster in enumerate(clusters, start=1):
        if cluster.get("event_type") not in (None, "DISCRETE_EVENT"):
            cluster["gdelt_receipt_bridge"] = {
                "status": "SKIPPED_NOT_DISCRETE_EVENT"
            }
            continue

        try:
            global_rank = int(cluster.get("global_attention_rank") or index)
        except Exception:
            global_rank = index

        assignment = rank_map.get(global_rank) or {}

        # Rank-based attachment is only safe when the shadow event identity
        # still matches the expanded cluster occupying that rank.
        assigned_topic = (assignment.get("topic") or "").strip()
        cluster_topic = (cluster.get("topic") or "").strip()
        if assigned_topic and cluster_topic and assigned_topic != cluster_topic:
            print(
                f"❌ Rank {global_rank} identity mismatch: ranking topic "
                f"{assigned_topic!r} != cluster topic {cluster_topic!r}."
            )
            print(
                "   Regenerate the global shadow expansion before attaching receipts."
            )
            return 2

        candidate_id = assignment.get("candidate_id") or candidate_id_from_cluster(cluster)
        bundle = catalog_candidates.get(candidate_id) if candidate_id else None

        if not bundle:
            unmatched += 1
            cluster["gdelt_receipt_bridge"] = {
                "status": "NO_GDELT_CANDIDATE_MATCH",
                "global_attention_rank": global_rank,
            }
            print(
                f"  • Rank {global_rank}: no GSG receipt candidate for "
                f"{(cluster.get('canonical_event') or cluster.get('topic') or '')[:62]}"
            )
            continue

        if candidate_id in used_candidate_ids:
            print(f"❌ Candidate {candidate_id} assigned to more than one event.")
            return 1
        used_candidate_ids.add(candidate_id)

        all_receipts = [dict(r) for r in bundle.get("receipts") or []]
        receipt_domains = {
            (r.get("domain") or "").lower().strip()
            for r in all_receipts if r.get("domain")
        }
        expected_outlets = int(bundle.get("target_day_outlet_count", 0) or 0)
        if bundle.get("receipt_catalog_complete") and len(receipt_domains) != expected_outlets:
            print(
                f"❌ Receipt integrity mismatch for {candidate_id}: "
                f"expected {expected_outlets}, found {len(receipt_domains)}."
            )
            return 1

        for row in all_receipts:
            row["bias"] = normalize_bias(
                row.get("url") or f"https://{row.get('domain')}/",
                row.get("bias") or "Unknown",
            )
            if row["bias"] == "Unknown" and row.get("domain"):
                unknown_bias_domains.add(row["domain"])

        writeup_families, family_unknowns = prepare_writeup_families(bundle)
        unknown_bias_domains.update(family_unknowns)
        expected_writeups = int(
            bundle.get("target_day_writeup_family_count", 0) or 0
        )
        if bundle.get("writeup_catalog_complete") and len(writeup_families) != expected_writeups:
            print(
                f"❌ Write-up integrity mismatch for {candidate_id}: "
                f"expected {expected_writeups}, found {len(writeup_families)}."
            )
            return 1

        english_receipts = [
            r for r in all_receipts if r.get("displayable_english")
        ]
        non_english_receipts = [
            r for r in all_receipts if not r.get("displayable_english")
        ]

        merged_sources = merge_english_receipts(
            cluster.get("coverage_sources") or [],
            english_receipts,
        )
        linked_domains = {
            s.get("domain") for s in merged_sources if s.get("domain")
        }

        cluster["coverage_sources"] = merged_sources
        cluster["coverage_domains"] = sorted(linked_domains)
        cluster["coverage_outlet_count"] = len(linked_domains)

        # Preserve every GSG receipt, including non-English sources, in a
        # separate audit field. Only English receipts enter coverage_sources.
        cluster["gdelt_global_source_receipts"] = all_receipts
        cluster["gdelt_global_writeup_families"] = writeup_families
        cluster["gdelt_global_coverage"] = {
            "candidate_id": candidate_id,
            "canonical_title": bundle.get("canonical_title") or "",
            "target_date": bundle.get("target_date") or d,
            "global_outlet_count": expected_outlets,
            "global_unique_writeup_count": int(
                bundle.get("target_day_writeup_family_count", 0) or 0
            ),
            "english_source_receipt_count": len(english_receipts),
            "non_english_source_receipt_count": len(non_english_receipts),
            "linked_english_outlet_count": len({
                r.get("domain") for r in english_receipts if r.get("domain")
            }),
            "combined_linked_outlet_count": len(linked_domains),
            "receipt_catalog_complete": bool(
                bundle.get("receipt_catalog_complete")
            ),
            "outlet_catalog_complete": bool(
                bundle.get("outlet_catalog_complete")
            ),
            "writeup_catalog_complete": bool(
                bundle.get("writeup_catalog_complete")
            ),
            "linked_writeup_family_count": len(writeup_families),
            "english_displayable_writeup_count": sum(
                1 for row in writeup_families
                if row.get("displayable_english")
            ),
            "english_display_policy": receipt_catalog.get(
                "english_display_policy"
            ),
            "ranking_assignment": assignment,
        }
        cluster["global_coverage_outlet_count"] = expected_outlets
        cluster["global_unique_writeup_count"] = int(
            bundle.get("target_day_writeup_family_count", 0) or 0
        )
        cluster["global_english_source_count"] = len(english_receipts)
        cluster["global_non_english_source_count"] = len(non_english_receipts)
        cluster["gdelt_receipt_bridge"] = {
            "status": "ATTACHED",
            "candidate_id": candidate_id,
            "event_membership_changed": False,
            "ranking_changed": False,
            "production_file_modified": False,
        }

        attached += 1
        total_global_outlets += expected_outlets
        total_english_receipts += len(english_receipts)
        total_non_english_receipts += len(non_english_receipts)
        total_writeup_families += len(writeup_families)

        print(
            f"  • Rank {global_rank}: {expected_outlets} global outlets / "
            f"{cluster['global_unique_writeup_count']} write-ups / "
            f"{len(writeup_families)} linked write-ups / "
            f"{len(english_receipts)} English outlet links — "
            f"{(cluster.get('canonical_event') or cluster.get('topic') or '')[:54]}"
        )

    after_hash = stable_membership_hash(clusters)
    if after_hash != before_hash:
        print("❌ Safety check failed: event membership or order changed.")
        return 1

    if shape == "list":
        output_data = clusters
    else:
        output_data = dict(raw_input)
        output_data["clusters"] = clusters

    save_json(output_path, output_data)

    print(f"✅ Attached GSG receipts to {attached} purifier-approved events")
    print(f"   unmatched events: {unmatched}")
    print(
        f"   catalog totals used: {total_global_outlets} global outlet receipts; "
        f"{total_writeup_families} write-up families; "
        f"{total_english_receipts} English; {total_non_english_receipts} non-English"
    )
    print(f"   unknown-bias receipt domains: {len(unknown_bias_domains)}")
    print(f"✅ Wrote receipt shadow → {output_path}")
    print("🛡️ Event membership, ranking, input, and production files were unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
