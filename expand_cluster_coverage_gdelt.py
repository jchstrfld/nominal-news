# expand_cluster_coverage_gdelt.py
# Phase 1: Expand VERIFIED clusters with additional coverage from GDELT
# Runs AFTER final_cohesion_check.py and BEFORE summarize_grouped_topics.py
# Free / rate-limited only. No OpenAI calls.

import argparse
import json
import os
import time
import requests
import re
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from bias_labeler import lookup_bias_by_domain, canonicalize
UNMAPPED_BIAS_FILE = "unmapped_bias.json"


# ----------------------------
# Config (safe defaults)
# ----------------------------
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_QUERIES_PER_RUN = 20           # keep low to avoid 429
REQUEST_SLEEP_SECONDS = 8.0        # be polite; avoid rate limits
MAX_RECORDS_PER_QUERY = 250
MAX_ARTICLES_PER_CLUSTER = 120     # hard cap to avoid bloat
MAX_PER_DOMAIN = 3                 # structural diversity guardrail

# ----------------------------
# Helpers
# ----------------------------
GDELT_CACHE_FILE = "gdelt_doc_cache.json"

def load_cache():
    if os.path.exists(GDELT_CACHE_FILE):
        with open(GDELT_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(GDELT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def extract_entities(title: str):
    # crude but effective: consecutive Capitalized Words
    import re
    ents = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", title)
    # drop very short / generic
    return [e for e in ents if len(e.split()) <= 3][:4]

def gdelt_safe_query(title: str) -> str:
    import re
    t = title.lower()
    t = re.sub(r"\s[-–|].*$", "", t)       # remove source suffix
    t = re.sub(r"[^a-z0-9\s]", " ", t)     # strip punctuation
    words = [w for w in t.split() if len(w) >= 3]  # <-- key fix
    return " ".join(words[:12])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    return p.parse_args()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_date_window(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    start = (d - timedelta(days=3)).strftime("%Y%m%d000000")
    end = (d + timedelta(days=3)).strftime("%Y%m%d235959")
    return start, end

def domain_from_url(url):
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""

def build_queries_from_cluster(cluster):
    # Primary: 3 simplified titles (this is what got you +6 earlier)
    queries = []
    for a in cluster.get("articles", []):
        title = (a.get("title") or "").strip()
        if not title:
            continue
        q = gdelt_safe_query(title)
        if q and q not in queries:
            queries.append(q)
        if len(queries) >= 3:
            break

    # Fallback: if titles are weak, query top entity alone
    if not queries:
        for a in cluster.get("articles", []):
            title = (a.get("title") or "").strip()
            if not title:
                continue
            ents = extract_entities(title)  # if you added this earlier
            if ents:
                queries.append(ents[0])
                break

    return queries

def gdelt_query(query, start, end):
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": MAX_RECORDS_PER_QUERY,
        "startdatetime": start,
        "enddatetime": end,
        "sort": "HybridRel",
    }

    r = requests.get(GDELT_ENDPOINT, params=params, timeout=30)

    if r.status_code == 429:
        # exponential backoff
        raise RuntimeError("GDELT_429")

    if r.status_code != 200:
        raise RuntimeError(f"GDELT HTTP {r.status_code}")

    text = (r.text or "").strip()
    if not text.startswith("{"):
        # GDELT sometimes returns plain text like:
        # "Your search contained a keyword that was too short."
        return {"articles": []}

    try:
        return r.json()
    except Exception:
        # Sometimes GDELT returns malformed JSON under load even if it starts with "{"
        print(f"⚠️ GDELT returned invalid JSON; treating as 0 hits.")
        return {"articles": []}

def normalize_gdelt_article(a):
    url = a.get("url")
    if not url:
        return None

    bias = lookup_bias_by_domain(url)
    bias = canonicalize(bias) if bias else "Unknown"

    return {
        "title": a.get("title"),
        "url": url,
        "description": a.get("seendate"),
        "published_at": a.get("seendate"),
        "source": domain_from_url(url),
        "bias": bias,
        "gdelt": True
    }

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    date_str = args.date or datetime.today().strftime("%Y-%m-%d")

    input_file = f"grouped_articles_final_{date_str}.json"
    output_file = f"grouped_articles_final_expanded_{date_str}.json"

    if not os.path.exists(input_file):
        print(f"❌ Missing {input_file}. Run final_cohesion_check.py first.")
        return

    grouped = load_json(input_file)
    clusters = grouped["clusters"] if isinstance(grouped, dict) else grouped

    start, end = get_date_window(date_str)
    cache = load_cache()
    cache_dirty = False

    print(f"🔎 Expanding clusters via GDELT ({date_str})")

    queries_used = 0
    unmapped = []

    for idx, cluster in enumerate(clusters):
        # Preserve the verified core exactly as-is.
        core_articles = cluster.get("articles", [])

        # Coverage starts with the verified core, then adds validated GDELT coverage.
        coverage_articles = list(core_articles)
        cluster["coverage_articles"] = coverage_articles

        existing_urls = {a.get("url") for a in coverage_articles if a.get("url")}
        domain_counts = {}

        for a in coverage_articles:
            d = domain_from_url(a.get("url", ""))
            domain_counts[d] = domain_counts.get(d, 0) + 1

        queries = build_queries_from_cluster(cluster)

        added = 0
        for q in queries:

            if queries_used >= MAX_QUERIES_PER_RUN:
                print("🛑 Reached MAX_QUERIES_PER_RUN; stopping early to avoid 429.")
                break
            try:
                ck = f"{start}|{end}|{q}"
                if ck in cache:
                    data = cache[ck]
                else:
                    data = gdelt_query(q, start, end)
                    cache[ck] = data
                    cache_dirty = True
                print(f"    query='{q}' → {len(data.get('articles', []))} hits")
                queries_used += 1
            except Exception as e:
                print(f"⚠️ GDELT query failed: {e}")
                msg = str(e)
                if "GDELT_429" in msg:
                    # back off harder on rate limit
                    time.sleep(12.0)
                    continue

                if "Read timed out" in msg:
                    time.sleep(6.0)
                    continue

                time.sleep(REQUEST_SLEEP_SECONDS)
                continue

            for raw in data.get("articles", []):
                art = normalize_gdelt_article(raw)
                if art["bias"] == "Unknown":
                    unmapped.append({
                        "url": art.get("url"),
                        "domain": domain_from_url(art.get("url", "")),
                        "title": art.get("title"),
                    })
                if not art:
                    continue

                url = art["url"]
                domain = domain_from_url(url)

                if url in existing_urls:
                    continue
                if domain_counts.get(domain, 0) >= MAX_PER_DOMAIN:
                    continue

                cluster["coverage_articles"].append(art)
                existing_urls.add(url)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                added += 1

                if len(cluster["coverage_articles"]) >= MAX_ARTICLES_PER_CLUSTER:
                    break

            time.sleep(REQUEST_SLEEP_SECONDS)
            if len(cluster["coverage_articles"]) >= MAX_ARTICLES_PER_CLUSTER:
                break

        print(f"  • Cluster {idx}: +{added} articles")
    
    if unmapped:
        try:
            if os.path.exists(UNMAPPED_BIAS_FILE):
                with open(UNMAPPED_BIAS_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.extend(unmapped)

            # de-dupe by domain
            seen = set()
            deduped = []
            for x in existing:
                d = x.get("domain")
                if d and d not in seen:
                    seen.add(d)
                    deduped.append(x)

            with open(UNMAPPED_BIAS_FILE, "w", encoding="utf-8") as f:
                json.dump(deduped, f, indent=2, ensure_ascii=False)

            print(f"🧭 Exported {len(deduped)} unmapped domains → {UNMAPPED_BIAS_FILE}")
        except Exception as e:
            print(f"⚠️ Failed to write {UNMAPPED_BIAS_FILE}: {e}")

    if cache_dirty:
        save_cache(cache)
        print(f"💾 Saved GDELT cache → {GDELT_CACHE_FILE}")
    save_json(output_file, grouped)
    print(f"✅ Wrote expanded clusters → {output_file}")

if __name__ == "__main__":
    main()
