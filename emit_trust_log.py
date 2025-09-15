# emit_trust_log.py — summarize clusters for QC (no GPT)
import json, argparse, os
from datetime import datetime
from collections import Counter
from urllib.parse import urlparse

def domain_from_url(u):
    try:
        return urlparse(u or "").netloc.replace("www.", "").lower()
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str)
    args = ap.parse_args()
    date_str = args.date or datetime.today().strftime("%Y-%m-%d")

    grouped_path = f"grouped_articles_final_{date_str}.json"
    if not os.path.exists(grouped_path):
        print(f"❌ Missing {grouped_path}")
        return

    with open(grouped_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    report = []
    for idx, c in enumerate(clusters):
        arts = c.get("articles", [])
        n = len(arts)
        domains = [domain_from_url(a.get("url", "")) for a in arts if a.get("url")]
        dom_counts = Counter(d for d in domains if d)
        top_dom, top_cnt = (dom_counts.most_common(1)[0] if dom_counts else ("", 0))
        bias_counts = Counter((a.get("bias") or "Center") for a in arts)
        total_bias = sum(bias_counts.values()) or 1
        bias_pct = {k: round(v*100/total_bias, 1) for k, v in bias_counts.items()}
        sample_titles = [a.get("title","")[:140] for a in arts[:5]]

        report.append({
            "cluster_index": idx,
            "num_articles": n,
            "unique_domains": len(dom_counts),
            "top_domain": {"domain": top_dom, "share": round((top_cnt/(len(domains) or 1))*100, 1)},
            "bias_distribution_pct": bias_pct,
            "example_titles": sample_titles
        })

    out_path = f"trust_log_{date_str}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "clusters": report}, f, indent=2, ensure_ascii=False)

    print(f"🧪 Wrote {out_path} with {len(report)} clusters")

if __name__ == "__main__":
    main()
