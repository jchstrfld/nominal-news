
import argparse
import json
import os
from urllib.parse import urlparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    return p.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def clusters_from(obj):
    if isinstance(obj, dict):
        return obj.get("clusters", [])
    return obj


def article_url(a):
    return (a.get("url_normalized") or a.get("url") or "").strip()


def domain(url):
    try:
        host = (urlparse(url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def cluster_urls(cluster):
    return {
        article_url(a)
        for a in cluster.get("articles", [])
        if article_url(a)
    }


def unique_domains(articles):
    return {
        domain(article_url(a))
        for a in articles
        if domain(article_url(a))
    }


def label(cluster, fallback):
    arts = cluster.get("articles", [])
    for a in arts:
        title = (a.get("title") or "").strip()
        if title:
            return title
    return cluster.get("topic") or fallback


def parent_matches(final_cluster, earlier_clusters):
    """
    Find actual lineage contributors by exact surviving URL overlap.
    No semantic matching is used.
    """
    furls = cluster_urls(final_cluster)
    matches = []

    for idx, c in enumerate(earlier_clusters):
        curls = cluster_urls(c)
        overlap = furls & curls
        if not overlap:
            continue

        matches.append({
            "index": idx,
            "topic": c.get("topic"),
            "label": label(c, f"Cluster {idx}"),
            "overlap_count": len(overlap),
            "final_overlap_ratio": len(overlap) / max(len(furls), 1),
            "parent_overlap_ratio": len(overlap) / max(len(curls), 1),
            "article_count": len(c.get("articles", [])),
            "domain_count": len(unique_domains(c.get("articles", []))),
            "overlap_urls": sorted(overlap),
        })

    matches.sort(
        key=lambda x: (
            x["overlap_count"],
            x["final_overlap_ratio"],
            x["parent_overlap_ratio"],
        ),
        reverse=True,
    )
    return matches


def lineage_union(matches, earlier_clusters):
    """
    Union all parent clusters that contributed a surviving final URL.
    Diagnostic only: this may intentionally include articles later removed for
    impurity, which is exactly what the lineage audit is measuring.
    """
    selected = [earlier_clusters[m["index"]] for m in matches]
    by_url = {}

    for c in selected:
        for a in c.get("articles", []):
            u = article_url(a)
            if u and u not in by_url:
                by_url[u] = a

    articles = list(by_url.values())
    return articles, unique_domains(articles)


def main():
    args = parse_args()
    d = args.date

    merged_path = f"grouped_articles_{d}.json"
    filtered_path = f"grouped_articles_filtered_{d}.json"
    final_path = f"grouped_articles_final_{d}.json"
    output_path = f"lineage_attention_audit_{d}.json"

    for path in (merged_path, filtered_path, final_path):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")

    merged = clusters_from(load_json(merged_path))
    filtered = clusters_from(load_json(filtered_path))
    final = clusters_from(load_json(final_path))

    rows = []

    print("\n🧬 LINEAGE ATTENTION AUDIT")
    print("=" * 112)
    print("Counts are exact URL-lineage footprints; no purity logic or semantic thresholds are changed.\n")

    for rank, fc in enumerate(final, start=1):
        final_articles = fc.get("articles", [])
        final_domains = unique_domains(final_articles)

        merged_parents = parent_matches(fc, merged)
        filtered_parents = parent_matches(fc, filtered)

        merged_union, merged_domains = lineage_union(merged_parents, merged)
        filtered_union, filtered_domains = lineage_union(filtered_parents, filtered)

        primary_merged = merged_parents[0] if merged_parents else None
        primary_filtered = filtered_parents[0] if filtered_parents else None

        row = {
            "rank": rank,
            "topic": fc.get("topic"),
            "label": label(fc, f"Final {rank}"),

            "merged_lineage": {
                "parent_count": len(merged_parents),
                "article_count": len(merged_union),
                "domain_count": len(merged_domains),
                "gain_vs_final_articles": len(merged_union) - len(final_articles),
                "gain_vs_final_domains": len(merged_domains) - len(final_domains),
                "primary_parent": primary_merged,
                "parents": merged_parents,
            },

            "filtered_lineage": {
                "parent_count": len(filtered_parents),
                "article_count": len(filtered_union),
                "domain_count": len(filtered_domains),
                "gain_vs_final_articles": len(filtered_union) - len(final_articles),
                "gain_vs_final_domains": len(filtered_domains) - len(final_domains),
                "primary_parent": primary_filtered,
                "parents": filtered_parents,
            },

            "final_verified": {
                "article_count": len(final_articles),
                "domain_count": len(final_domains),
            },

            "event_split": fc.get("event_split"),
            "eventness_label": fc.get("eventness_label"),
        }
        rows.append(row)

        m = row["merged_lineage"]
        f = row["filtered_lineage"]
        v = row["final_verified"]

        print(
            f"#{rank:<2} "
            f"merged={m['article_count']:>3}/{m['domain_count']:>2} dom  →  "
            f"filtered={f['article_count']:>3}/{f['domain_count']:>2} dom  →  "
            f"final={v['article_count']:>2}/{v['domain_count']:>2} dom   "
            f"{row['label'][:62]}"
        )

        if len(merged_parents) > 1 or len(filtered_parents) > 1:
            print(
                f"     lineage parents: merged={len(merged_parents)} "
                f"filtered={len(filtered_parents)}"
            )

    # Two diagnostic orderings:
    # 1) immediate pre-final footprint (safer)
    # 2) earlier merged footprint (broader, but potentially noisier)
    filtered_order = sorted(
        rows,
        key=lambda r: (
            r["filtered_lineage"]["domain_count"],
            r["filtered_lineage"]["article_count"],
            r["final_verified"]["domain_count"],
        ),
        reverse=True,
    )

    merged_order = sorted(
        rows,
        key=lambda r: (
            r["merged_lineage"]["domain_count"],
            r["merged_lineage"]["article_count"],
            r["final_verified"]["domain_count"],
        ),
        reverse=True,
    )

    print("\n🏁 ORDER BY IMMEDIATE PRE-FINAL LINEAGE")
    print("=" * 112)
    for i, r in enumerate(filtered_order, start=1):
        x = r["filtered_lineage"]
        print(
            f"#{i:<2} {x['domain_count']:>2} domains / {x['article_count']:>3} articles  "
            f"{r['label'][:76]}"
        )

    print("\n🏁 ORDER BY EARLIER MERGED LINEAGE")
    print("=" * 112)
    for i, r in enumerate(merged_order, start=1):
        x = r["merged_lineage"]
        print(
            f"#{i:<2} {x['domain_count']:>2} domains / {x['article_count']:>3} articles  "
            f"{r['label'][:76]}"
        )

    save_json(
        output_path,
        {
            "date": d,
            "method": (
                "Exact URL lineage from each final verified event back to every "
                "earlier cluster that contributed at least one surviving final URL. "
                "Counts are unions of those actual parent clusters."
            ),
            "current_order": rows,
            "filtered_lineage_order": filtered_order,
            "merged_lineage_order": merged_order,
        },
    )

    print(f"\n✅ Wrote read-only lineage audit → {output_path}")


if __name__ == "__main__":
    main()
