# summarize_grouped_topics.py — summarize top clusters with restored bias & source info + summaries cache

import json
import openai
import os
from dotenv import load_dotenv
from datetime import datetime
import argparse
import tiktoken
import difflib, re

# ✅ NEW: summaries cache helpers (add summaries_cache.py next to this file)
from summaries_cache import (
    load_summ_cache, save_summ_cache,
    make_summ_key, get_cached_summary, put_cached_summary
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser()
parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
args = parser.parse_args()

date_str = args.date or datetime.today().strftime("%Y-%m-%d")
print(f"📅 Using input date: {date_str}")

INPUT_FILE = f"grouped_articles_final_{date_str}.json"
OUTPUT_FILE = f"topic_summaries_{date_str}.json"

MIN_ARTICLES = 6
MAX_ARTICLES_PER_CLUSTER = 10
MAX_CLUSTERS = 10
MAX_TOKENS = 7000
ENCODING = tiktoken.encoding_for_model("gpt-4")

# ✅ NEW: cache config (bump PROMPT_VERSION when you change the prompt format)
PROMPT_VERSION = "v1.0-2025-08-12"
SUMM_MODEL = "gpt-4"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    clusters = json.load(f)

valid_clusters = [c for c in clusters if len(c["articles"]) >= MIN_ARTICLES]
ranked = sorted(valid_clusters, key=lambda c: len(c["articles"]), reverse=True)
top_clusters = ranked[:MAX_CLUSTERS]

summaries = []

def compute_bias_distribution(articles):
    counts = {}
    total = 0
    for a in articles:
        bias_raw = a.get("bias") or "Unknown"
        # Canonicalize to match UI keys
        bias = bias_raw.title().replace("-", " ")
        if bias != "Unknown":
            counts[bias] = counts.get(bias, 0) + 1
            total += 1
    if total == 0:
        return {}
    # Integer percentages that sum ≈ 100 (simple rounding)
    return {k: round(v / total * 100) for k, v in counts.items()}

def format_article(a):
    title = (a.get("title") or "")[:200].strip()
    desc = (a.get("description") or "")[:300].strip()
    return f"{title}. {desc}"

def truncate_prompt(prompt, max_tokens):
    tokens = ENCODING.encode(prompt or "")
    if len(tokens) <= max_tokens:
        return prompt or ""
    return ENCODING.decode(tokens[:max_tokens])

def make_html_chips(articles):
    # Same colors as the pie chart
    color_map = {
        "Far Left": "#0B36B8",
        "Left": "#275BF5",
        "Center": "#894AB3",
        "Right": "#EB4040",
        "Far Right": "#C43D31",
        "Unknown": "#C6C6C6"
    }

    # Normalize bias and sort by bias order, then alphabetically by source
    bias_order = ["Far Left", "Left", "Center", "Right", "Far Right", "Unknown"]

    def bias_sort_key(article):
        bias = (article.get("bias") or "Unknown").title().replace("-", " ")
        if bias not in bias_order:
            bias = "Unknown"
        return (bias_order.index(bias), (article.get("source_name") or "").lower())

    sorted_articles = sorted(articles, key=bias_sort_key)

    chips = []
    for a in sorted_articles:
        url = a.get("url", "#")
        source = a.get("source_name") or url.split("//")[-1].split("/")[0]
        bias = (a.get("bias") or "Unknown").title().replace("-", " ")
        if bias not in color_map:
            bias = "Unknown"
        bias_color = color_map[bias]
        # Background at ~20% opacity via hex alpha
        bias_color_20 = bias_color + "33"
        chips.append(
            f'<a href="{url}" target="_blank" class="chip" '
            f'style="background-color: {bias_color_20}; border-color: {bias_color};">{source}</a>'
        )
    return " ".join(chips)

def summarize_cluster(articles):
    text_block = "\n\n".join([format_article(a) for a in articles])
    prompt = f"""
You are a neutral news assistant. Summarize the key theme across the following news articles. Provide:
1. A short, clear headline summarizing the topic
2. A factual summary in 3-4 sentences
3. Three bullet-point takeaways explaining why it matters (political, economic, legal, civil, etc)

Articles:
{text_block}

Respond in this format:
Headline: <headline>
Summary: <summary>
Takeaways:
- <point 1>
- <point 2>
- <point 3>
"""
    prompt = truncate_prompt(prompt.strip(), MAX_TOKENS)

    try:
        response = openai.ChatCompletion.create(
            model=SUMM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
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

def extract_body_and_takeaways(summary_text):
    parts = (summary_text or "").split("Takeaways:")
    summary = ""
    takeaways = []
    if len(parts) == 2:
        summary = parts[0].split("Summary:")[-1].strip().rstrip("Key")
        takeaways = [line.strip("- ").strip() for line in parts[1].strip().splitlines() if line.strip()]
    else:
        summary = summary_text or ""
    return summary.strip(), takeaways

# --- Safety valve: collapse near-duplicate topics after summarization (no tokens) ---
_STOPWORDS = {
    "a","an","the","and","or","but","to","of","for","in","on","at","over","under","after","before",
    "with","without","by","from","as","about","into","during","including","until","against","among",
    "between","through","because","so","since","due","has","have","had","is","was","are","were",
    "be","been","being","will","would","should","may","might","can","could"
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
    # overlap measured against the smaller set to be strict
    return len(s1 & s2) / max(1, min(len(s1), len(s2)))

def _seq_ratio(a: str, b: str):
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()

def _looks_like_duplicate(x: dict, y: dict, url_thresh: float, title_thresh: float, body_thresh: float):
    title_x = x.get("topic_title") or ""
    title_y = y.get("topic_title") or ""
    body_x  = (x.get("summary") or "")[:600]
    body_y  = (y.get("summary") or "")[:600]
    urls_x  = x.get("sources") or []
    urls_y  = y.get("sources") or []

    # URL overlap (exact URLs) — strong signal
    url_ov = _overlap_ratio(urls_x, urls_y)

    # Title similarity (character + token Jaccard) — catches near rephrasings
    t_ratio = _seq_ratio(title_x.lower(), title_y.lower())
    t_jacc  = _jaccard(_norm_title_tokens(title_x), _norm_title_tokens(title_y))

    # Body similarity — backstop if titles differ but summaries are essentially the same
    b_ratio = _seq_ratio(body_x.lower(), body_y.lower())

    return (url_ov >= url_thresh) and ((t_ratio >= title_thresh) or (t_jacc >= 0.70) or (b_ratio >= body_thresh))

def dedupe_topic_summaries(items, url_overlap=0.50, title_sim=0.86, body_sim=0.88):
    """
    Keeps the first (higher-ranked) item and merges later duplicates into it.
    - No OpenAI calls. Pure string/URL math.
    - Preserves existing order/ranking from clustering.
    """
    result = []
    for cand in items:
        merged = False
        for kept in result:
            if _looks_like_duplicate(kept, cand, url_overlap, title_sim, body_sim):
                # Merge sources & counts into the higher-ranked 'kept' card
                merged_sources = list({*(kept.get("sources") or []), *(cand.get("sources") or [])})
                kept["sources"] = merged_sources
                kept["num_sources"] = len(merged_sources)
                # We intentionally do NOT touch title/summary/takeaways/bias/html_chips of the winner.
                merged = True
                break
        if not merged:
            result.append(cand)
    # Respect your MAX_CLUSTERS cap
    try:
        return result[:MAX_CLUSTERS]
    except NameError:
        # If MAX_CLUSTERS is defined later, slice when calling instead.
        return result
# --- end safety valve ---

# load summaries cache once
summ_cache = load_summ_cache()
summ_cache_dirty = False

for idx, cluster in enumerate(top_clusters):
    print(f"🧠 Summarizing topic {idx + 1}/{len(top_clusters)} with {len(cluster['articles'])} articles")

    # Use the same selection cap you already have
    selected_articles = cluster["articles"][:MAX_ARTICLES_PER_CLUSTER]
    # Build cache key from the selected set of URLs (+ model, prompt version, cap)
    selected_urls = sorted([a.get("url") for a in selected_articles if a.get("url")])
    cache_key = make_summ_key(selected_urls, SUMM_MODEL, PROMPT_VERSION, MAX_ARTICLES_PER_CLUSTER)

    cached = get_cached_summary(summ_cache, cache_key)
    if cached:
        print(f"💾 Cache hit for topic {idx + 1} — reused summary")
        headline = cached["headline"]
        body = cached["summary"]
        takeaways = cached["takeaways"]
    else:
        summary_text = summarize_cluster(selected_articles)
        if not summary_text:
            continue
        headline = extract_headline(summary_text)
        body, takeaways = extract_body_and_takeaways(summary_text)
        put_cached_summary(summ_cache, cache_key, headline, body, takeaways)
        summ_cache_dirty = True

    # For visuals, prefer all cluster articles (not just selected)
    all_articles = cluster["articles"]

    # Bias distribution from cluster if present, else compute from articles
    bias_dist = cluster.get("bias_distribution") or compute_bias_distribution(all_articles)
    if not bias_dist:
        print(f"⚠️ Cluster {idx} has no bias_distribution field")

    summaries.append({
        "topic_title": headline,
        "summary": body,
        "takeaways": takeaways,
        "bias_distribution": bias_dist,
        "sources": [a.get("url") for a in all_articles if a.get("url")],
        "num_sources": len(all_articles),
        "html_chips": make_html_chips(all_articles)
    })

# ✅ Safety valve — de‑duplicate near‑identical topic cards before writing JSON
summaries = dedupe_topic_summaries(
    summaries,
    url_overlap=0.50,  # require ≥ 50% URL overlap
    title_sim=0.86,    # high title similarity
    body_sim=0.88      # high body similarity
)

# ✅ NEW: save cache if we wrote anything
if summ_cache_dirty:
    save_summ_cache(summ_cache)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)

print(f"✅ Saved top summaries to {OUTPUT_FILE}")
