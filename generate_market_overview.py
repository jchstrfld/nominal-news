# generate_market_overview.py
# Constructed Market & Economy Overview (always-on, not cluster-based).
# Reads articles_with_bias_{date}.json and writes market_overview_{date}.json.
#
# Outputs:
# - Index moves (Nasdaq, S&P 500, Dow) as close-to-close % change for the date
# - Two short sentences:
#     1) "Markets are doing X today because of Y..."
#     2) "If you have any investments, they're doing Z today..."
#
# Uses:
# - Free Stooq daily data (no key) for index moves
# - One small OpenAI call (titles-only) to infer 1–2 main reasons (optional; fails open)

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
from datetime import datetime, UTC

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---- Config ----
# Stooq symbols (daily)
STOOQ_SYMBOLS = {
    "nasdaq": "^ndq",
    "sp500": "^spx",
    "dow": "^dji",
}

STOOQ_URL = "https://stooq.com/q/d/l/"
# docs-ish: https://stooq.com/q/d/l/?s=^spx&i=d  (CSV)

LOCAL_EMB_MODEL = os.getenv("NN_MARKET_EMB_MODEL", "all-MiniLM-L6-v2")
MARKET_MODEL = os.getenv("NN_MARKET_MODEL", "gpt-4o-mini")  # override via env if needed
MARKET_MODEL_FALLBACK = os.getenv("NN_MARKET_MODEL_FALLBACK", "gpt-4")

MAX_FINANCE_TITLES = int(os.getenv("NN_MARKET_MAX_TITLES", "18"))
MIN_FINANCE_TITLES = int(os.getenv("NN_MARKET_MIN_TITLES", "6"))

# Anchors for semantic filtering (not a blacklist; just vector targets)
FINANCE_ANCHORS = [
    "stock market",
    "S&P 500",
    "Nasdaq",
    "Dow Jones",
    "Treasury yields",
    "bond yields",
    "Federal Reserve",
    "interest rates",
    "inflation",
    "jobs report",
    "economic growth",
    "recession",
    "oil prices",
    "dollar strengthened",
]

# If you already have a curated set of finance domains, put it here (optional).
FINANCE_DOMAIN_HINTS = {
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "cnbc.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "investing.com",
    "economist.com",
    "forbes.com",
    "fortune.com",
}

# ---------- Helpers ----------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    return p.parse_args()

def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""

def _stooq_csv(symbol: str) -> str:
    # Example: https://stooq.com/q/d/l/?s=%5Espx&i=d
    return f"{STOOQ_URL}?s={requests.utils.quote(symbol)}&i=d"

def _fetch_stooq_series(symbol: str) -> List[Dict[str, str]]:
    r = requests.get(_stooq_csv(symbol), timeout=30)
    r.raise_for_status()
    text = (r.text or "").strip()
    # header: Date,Open,High,Low,Close,Volume
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip().lower() for h in lines[0].split(",")]
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) != len(header):
            continue
        row = dict(zip(header, parts))
        rows.append(row)
    return rows

def _closest_trading_pair(rows: List[Dict[str, str]], date_str: str) -> Optional[Tuple[float, float, str]]:
    """
    Return (prev_close, close, close_date_str) for the requested date.
    If requested date is non-trading day, use the latest close <= date_str.
    """
    # Parse rows into (date, close)
    parsed = []
    for r in rows:
        ds = r.get("date")
        cs = r.get("close")
        if not ds or not cs:
            continue
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
            c = float(cs)
            parsed.append((d, c))
        except Exception:
            continue
    if len(parsed) < 2:
        return None

    parsed.sort(key=lambda x: x[0])  # ascending
    target = datetime.strptime(date_str, "%Y-%m-%d").date()

    # Find last close <= target
    idx = None
    for i in range(len(parsed) - 1, -1, -1):
        if parsed[i][0] <= target:
            idx = i
            break
    if idx is None or idx == 0:
        return None

    prev_close = parsed[idx - 1][1]
    close = parsed[idx][1]
    close_date = parsed[idx][0].strftime("%Y-%m-%d")
    return prev_close, close, close_date

def _pct_change(prev_close: float, close: float) -> float:
    if prev_close == 0:
        return 0.0
    return (close - prev_close) / prev_close * 100.0

def _pick_finance_titles(articles: List[dict]) -> List[str]:
    # Title text candidates
    candidates = []
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title:
            continue
        url = a.get("url") or ""
        d = _domain(url)
        candidates.append((title, d))

    if not candidates:
        return []

    # Domain hint boost + embedding similarity
    model = SentenceTransformer(LOCAL_EMB_MODEL)

    anchor_vecs = model.encode(FINANCE_ANCHORS, normalize_embeddings=True)
    anchor_centroid = np.mean(anchor_vecs, axis=0, keepdims=True)
    # normalize
    denom = np.linalg.norm(anchor_centroid, axis=1, keepdims=True)
    anchor_centroid = anchor_centroid / np.maximum(denom, 1e-12)

    titles = [t for (t, _) in candidates]
    title_vecs = model.encode(titles, normalize_embeddings=True)

    sims = (title_vecs @ anchor_centroid.T).reshape(-1)

    scored = []
    for i, (t, d) in enumerate(candidates):
        score = float(sims[i])
        if d in FINANCE_DOMAIN_HINTS:
            score += 0.06  # small boost, not decisive
        scored.append((score, t))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [t for _, t in scored[:MAX_FINANCE_TITLES]]

    # Deduplicate while preserving order
    out = []
    seen = set()
    for t in picked:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t

def _summarize_reason_and_investment(date_str: str, index_summary: str, finance_titles: List[str]) -> Dict[str, str]:
    """
    One short OpenAI call to produce:
    - markets_because
    - investments_doing
    Fail-open if API missing/unavailable.
    """
    # Always-on: If no key, produce a deterministic fallback.
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "markets_because": f"Markets were mixed today; this briefing is based on index moves ({index_summary}).",
            "investments_doing": "If you have diversified investments, your performance likely tracked the broader indexes today."
        }

    titles_block = "\n".join(f"- {t}" for t in finance_titles[:MAX_FINANCE_TITLES])

    prompt = f"""You are writing a calm, plain-language Market & Economy Overview for Nominal News.
Date: {date_str}
Index moves (close-to-close): {index_summary}

Use ONLY the evidence in the headlines below. Do not add extra facts.
Headlines:
{titles_block}

Write valid JSON with exactly these keys:
- "markets_because": 1 short paragraph explaining what markets did today and the 1–2 main reasons why.
- "investments_doing": exactly 1 short, plain-English sentence explaining what this likely means for a typical person's investments.

Rules:
- Keep "markets_because" concise but it may be 1–2 sentences if needed.
- Keep "investments_doing" to ONE sentence only.
- "investments_doing" must be tight, calm, and broadly applicable.
- Do not give financial advice.
- Do not mention specific people unless necessary.
- Do not use hype words like plunge, chaos, panic, or soar.
- Return JSON only. No markdown fences.
"""

    messages = [
        {"role": "system", "content": "You are a careful financial explainer. No hype, no advice."},
        {"role": "user", "content": prompt},
    ]

    def _call(model_name: str) -> str:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
        )
        return resp["choices"][0]["message"]["content"]

    text = None
    for model_name in (MARKET_MODEL, MARKET_MODEL_FALLBACK):
        try:
            text = _call(model_name)
            break
        except Exception:
            continue
        
    text = _strip_code_fences(text) if text else text
    if not text:
        return {
            "markets_because": plain,
            "investments_doing": "If you have investments, they likely moved with the broader market today."
        }

    # Parse JSON safely (fail-open)
    try:
        obj = json.loads(text)
        mb = (obj.get("markets_because") or "").strip()
        inv = (obj.get("investments_doing") or "").strip()
        if mb and inv:
            return {"markets_because": mb, "investments_doing": inv}
    except Exception:
        pass

    # If the model didn't return JSON, coerce by splitting lines.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    plain = " ".join(lines).strip()
    if plain:
        return {
            "markets_because": plain,
            "investments_doing": "If you have any investments, they're doing about what the major indexes did today."
        }
    return {
        "markets_because": f"Markets were mixed today; this briefing is based on index moves ({index_summary}).",
        "investments_doing": "If you have any investments, they're doing about what the major indexes did today."
    }

def main():
    args = _parse_args()
    date_str = args.date or datetime.today().strftime("%Y-%m-%d")

    in_path = BASE_DIR / f"articles_with_bias_{date_str}.json"
    out_path = BASE_DIR / f"market_overview_{date_str}.json"

    if not in_path.exists():
        print(f"❌ Missing {in_path.name}. Run bias_labeler.py first.")
        raise SystemExit(1)

    articles = _read_json(in_path)
    if not isinstance(articles, list):
        print("❌ Unexpected input shape: expected a list of articles.")
        raise SystemExit(1)

    # Fetch index moves
    indexes = {}
    parts = []
    for key, symbol in STOOQ_SYMBOLS.items():
        try:
            rows = _fetch_stooq_series(symbol)
            pair = _closest_trading_pair(rows, date_str)
            if not pair:
                raise RuntimeError("No trading pair")
            prev_close, close, close_date = pair
            pct = _pct_change(prev_close, close)
            direction = "up" if pct > 0 else ("down" if pct < 0 else "flat")
            indexes[key] = {
                "symbol": symbol,
                "close_date": close_date,
                "pct_change": round(pct, 2),
                "direction": direction,
            }
            parts.append(f"{key.upper()} {indexes[key]['pct_change']}%")
        except Exception as e:
            indexes[key] = {
                "symbol": symbol,
                "close_date": None,
                "pct_change": None,
                "direction": "unknown",
                "error": str(e),
            }
            parts.append(f"{key.upper()} n/a")

    index_summary = ", ".join(parts)

    finance_titles = _pick_finance_titles(articles)
    # Always-on fallback even if finance titles are scarce
    if len(finance_titles) < MIN_FINANCE_TITLES:
        finance_titles = finance_titles[:MAX_FINANCE_TITLES]

    text_bits = _summarize_reason_and_investment(date_str, index_summary, finance_titles)

    out = {
        "date": date_str,
        "label": "Market & Economy Overview",
        "indexes": indexes,
        "markets_because": text_bits["markets_because"],
        "investments_doing": text_bits["investments_doing"],
        "evidence_titles": finance_titles[:MAX_FINANCE_TITLES],
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    _write_json(out_path, out)
    print(f"✅ Wrote market overview → {out_path.name}")

if __name__ == "__main__":
    main()
