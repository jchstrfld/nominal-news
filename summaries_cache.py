# summaries_cache.py — tiny JSON cache for cluster summaries
import json, os, hashlib, time
from typing import Dict, Any, Optional

CACHE_PATH = "summaries_cache.json"

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_summ_key(urls_sorted, model: str, prompt_version: str, max_articles: int) -> str:
    # key depends ONLY on which URLs are summarized + model + prompt version + sampling cap
    joined = "\n".join(urls_sorted)
    return f"{model}:{prompt_version}:max{max_articles}:{_sha256(joined)}"

def load_summ_cache(path: str = CACHE_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_summ_cache(cache: Dict[str, Any], path: str = CACHE_PATH) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def get_cached_summary(cache: Dict[str, Any], key: str) -> Optional[dict]:
    return cache.get(key)

def put_cached_summary(cache: Dict[str, Any], key: str, headline: str, body: str, takeaways: list) -> None:
    cache[key] = {
        "headline": headline,
        "summary": body,
        "takeaways": takeaways,
        "ts": int(time.time())
    }
