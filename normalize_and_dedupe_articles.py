# normalize_and_dedupe_articles.py — normalize URLs and remove duplicates (additive)
import json, argparse, os
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

STRIP_QUERY_KEYS = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "gclid","fbclid","mc_cid","mc_eid","igshid","spm"
}

def normalize_url(u: str) -> str:
    if not u:
        return u
    try:
        p = urlparse(u.strip())
        # lowercase host; keep scheme/path as-is
        netloc = (p.netloc or "").lower()
        if netloc.startswith("m."):
            netloc = netloc[2:]  # strip "m." subdomain conservatively

        # prune tracking params (preserve order deterministically)
        q_pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k not in STRIP_QUERY_KEYS]
        query = urlencode(q_pairs, doseq=True)

        # trim trailing slash (but keep root "/")
        path = p.path.rstrip("/") or "/"

        norm = urlunparse((p.scheme or "https", netloc, path, "", query, ""))

        return norm
    except Exception:
        return u

def _get_pub_date(a: dict) -> str:
    d = (a.get("published_date") or "").strip()
    if d and len(d) >= 10:
        return d[:10]
    pa = (a.get("published_at") or "").strip()
    return pa[:10] if pa and len(pa) >= 10 else ""

def run(date_str: str):
    in_path  = f"articles_raw_{date_str}.json"
    out_path = f"articles_raw_normalized_{date_str}.json"
    if not os.path.exists(in_path):
        print(f"❌ Missing {in_path}")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # Time-box: keep only articles within date_str ± 1 day (fail-open if date missing)
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
        lo = target - timedelta(days=1)
        hi = target + timedelta(days=1)
    except Exception:
        lo = hi = None

    if lo and hi:
        boxed = []
        for a in articles:
            d = _get_pub_date(a)
            if not d:
                boxed.append(a)  # fail-open
                continue
            try:
                ad = datetime.strptime(d, "%Y-%m-%d").date()
                if lo <= ad <= hi:
                    boxed.append(a)
            except Exception:
                boxed.append(a)  # fail-open
        articles = boxed

    seen = {}
    deduped = []
    for a in articles:
        url = a.get("url") or ""
        norm = normalize_url(url)
        a["url_normalized"] = norm
        if norm not in seen:
            seen[norm] = True
            deduped.append(a)

    removed = len(articles) - len(deduped)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"🔧 Normalized {len(articles)} → {len(deduped)} (removed {removed} dups)")
    print(f"💾 Wrote {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str)
    args = ap.parse_args()
    date_str = args.date or datetime.today().strftime("%Y-%m-%d")
    run(date_str)
