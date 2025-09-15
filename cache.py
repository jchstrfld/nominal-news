# cache.py — tiny JSON cache for embeddings (additive, safe)
import json, os, hashlib, time
from typing import Dict, Any, Optional, List

CACHE_PATH = "embeddings_cache.json"

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def key_for(text: str, model: str) -> str:
    # hash text content + model
    return f"{model}:{_sha256(text)}"

def load_cache(path: str = CACHE_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # corrupted? start fresh
        return {}

def save_cache(cache: Dict[str, Any], path: str = CACHE_PATH) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, path)

def get_cached_vector(cache: Dict[str, Any], key: str) -> Optional[List[float]]:
    item = cache.get(key)
    if not item:
        return None
    # structure: { "v": [floats], "model": "...", "ts": 1690000000 }
    return item.get("v")

def put_cached_vector(cache: Dict[str, Any], key: str, vector: List[float], model: str) -> None:
    cache[key] = {"v": vector, "model": model, "ts": int(time.time())}
