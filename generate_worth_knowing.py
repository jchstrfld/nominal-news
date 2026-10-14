from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

try:
    import openai
except Exception:
    openai = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if openai is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = os.getenv("NN_WORTH_KNOWING_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
PROMPT_VERSION = "worth-knowing-v3-neutral-2026-09-26"
MAX_ITEMS = 4
MAX_CANDIDATES = 12
MAX_EVIDENCE_PER_CANDIDATE = 4
MAX_DESCRIPTION_CHARS = 320
CACHE_FILE = "worth_knowing_cache.json"
GPT4O_MINI_INPUT_USD_PER_MILLION = 0.15
GPT4O_MINI_OUTPUT_USD_PER_MILLION = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the additive Worth Knowing section from already-produced "
            "Nominal News ranking and article evidence."
        )
    )
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached model output")
    parser.add_argument("--dry-run", action="store_true", help="Print candidates without calling the model or writing output")
    return parser.parse_args()


def load_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def display_headline(value: Any) -> str:
    """Normalize Worth Knowing display headlines without altering source/ranking titles."""
    text = normalize_space(value)
    if not text.endswith("."):
        return text

    # Preserve terminal initialisms such as "U.S." while removing sentence-style
    # periods from ordinary display headlines.
    if re.search(r"(?:\b[A-Za-z]\.){2,}$", text):
        return text

    return text[:-1].rstrip()


def normalize_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", normalize_space(value).lower()))


def title_similarity(a: str, b: str) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0

    seq = difflib.SequenceMatcher(None, a_n, b_n).ratio()
    a_tokens = set(a_n.split())
    b_tokens = set(b_n.split())
    union = a_tokens | b_tokens
    jaccard = (len(a_tokens & b_tokens) / len(union)) if union else 0.0
    return (0.7 * seq) + (0.3 * jaccard)


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def display_source_name(url: str, raw_source: str = "") -> str:
    """Return a stable reader-facing source label without RSS/feed-name noise."""
    domain = domain_from_url(normalize_space(url))
    if domain:
        return domain
    return normalize_space(raw_source)


def source_name(article: Dict[str, Any]) -> str:
    source = article.get("source")
    if isinstance(source, dict):
        source = source.get("name") or source.get("id")
    return normalize_space(source) or domain_from_url(normalize_space(article.get("url")))


def clip_description(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", normalize_space(value))
    text = normalize_space(text)
    if len(text) <= MAX_DESCRIPTION_CHARS:
        return text
    return text[: MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"


def find_ranking_file(base_dir: Path, date_str: str) -> Optional[Path]:
    choices = [
        base_dir / f"gdelt_global_ranking_{date_str}.json",
        base_dir / f"gdelt_global_ranking_shadow_{date_str}.json",
    ]
    for path in choices:
        if path.exists():
            return path
    return None


def load_cluster_map(base_dir: Path, date_str: str) -> Dict[str, Dict[str, Any]]:
    path = base_dir / f"grouped_articles_filtered_{date_str}.json"
    data = load_json(path, [])
    if isinstance(data, dict):
        data = data.get("clusters") or []
    if not isinstance(data, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for cluster in data:
        if isinstance(cluster, dict) and cluster.get("topic"):
            out[str(cluster["topic"])] = cluster
    return out


def load_discovery_candidate_map(base_dir: Path, date_str: str) -> Dict[str, Dict[str, Any]]:
    path = base_dir / f"gdelt_discovery_candidates_{date_str}.json"
    payload = load_json(path, {})
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list):
        return {}
    return {
        str(c.get("candidate_id")): c
        for c in candidates
        if isinstance(c, dict) and c.get("candidate_id")
    }


def discovery_primary_id(row: Dict[str, Any]) -> str:
    evidence = row.get("gdelt_global_attention") or {}
    if not isinstance(evidence, dict):
        return ""
    return normalize_space(evidence.get("primary_candidate_id"))


def evidence_from_discovery(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    reps = candidate.get("representatives") or []
    if not isinstance(reps, list):
        return []

    out: List[Dict[str, Any]] = []
    seen_urls = set()
    for rep in reps:
        if not isinstance(rep, dict):
            continue
        url = normalize_space(rep.get("url"))
        title = normalize_space(rep.get("title"))
        if not url or not title or url in seen_urls:
            continue
        seen_urls.add(url)
        out.append(
            {
                "title": title,
                "description": "",
                "url": url,
                "source": normalize_space(rep.get("domain")) or domain_from_url(url),
                "origin": "gdelt_representative",
                "title_similarity": 1.0,
            }
        )
        if len(out) >= MAX_EVIDENCE_PER_CANDIDATE:
            break
    return out


def evidence_from_cluster(cluster: Dict[str, Any], canonical_title: str) -> List[Dict[str, Any]]:
    articles = cluster.get("articles") or []
    if not isinstance(articles, list):
        return []

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = normalize_space(article.get("title"))
        url = normalize_space(article.get("url") or article.get("url_normalized"))
        if not title or not url:
            continue
        scored.append((title_similarity(canonical_title, title), article))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    out: List[Dict[str, Any]] = []
    seen_urls = set()
    for score, article in scored:
        # Below this, the article is usually a neighboring story from a broad cluster.
        if score < 0.34:
            continue
        url = normalize_space(article.get("url") or article.get("url_normalized"))
        if url in seen_urls:
            continue
        seen_urls.add(url)
        out.append(
            {
                "title": normalize_space(article.get("title")),
                "description": clip_description(article.get("description") or article.get("content") or ""),
                "url": url,
                "source": source_name(article),
                "origin": "filtered_cluster",
                "title_similarity": round(score, 4),
            }
        )
        if len(out) >= MAX_EVIDENCE_PER_CANDIDATE:
            break
    return out


def merge_evidence(*groups: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_urls = set()
    for group in groups:
        for item in group:
            url = normalize_space(item.get("url"))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            out.append(item)
            if len(out) >= MAX_EVIDENCE_PER_CANDIDATE:
                return out
    return out


def build_candidates(
    ranking: Dict[str, Any],
    clusters: Dict[str, Dict[str, Any]],
    discovery: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ranked = ranking.get("all_purifier_approved_events_by_global_shadow_rank") or []
    if not isinstance(ranked, list):
        return [], []

    top_k = int(ranking.get("top_k", 10) or 10)
    main_rows = [r for r in ranked[:top_k] if isinstance(r, dict)]
    candidate_rows = [r for r in ranked[top_k : top_k + MAX_CANDIDATES] if isinstance(r, dict)]

    candidates: List[Dict[str, Any]] = []
    for row in candidate_rows:
        canonical_title = normalize_space(row.get("title"))
        topic = normalize_space(row.get("topic"))
        event_id = normalize_space(row.get("event_id"))
        if not canonical_title or not event_id:
            continue

        gdelt_id = discovery_primary_id(row)
        discovery_evidence = evidence_from_discovery(discovery.get(gdelt_id, {})) if gdelt_id else []
        cluster_evidence = evidence_from_cluster(clusters.get(topic, {}), canonical_title)
        evidence = merge_evidence(discovery_evidence, cluster_evidence)

        distinct_domains = {
            domain_from_url(normalize_space(e.get("url")))
            for e in evidence
            if domain_from_url(normalize_space(e.get("url")))
        }
        if len(evidence) < 2 or len(distinct_domains) < 2:
            continue

        candidates.append(
            {
                "candidate_id": event_id,
                "topic": topic,
                "rank": int(row.get("global_shadow_rank") or 0),
                "title": canonical_title,
                "source_type": normalize_space(row.get("source_type")),
                "global_score": float(row.get("global_shadow_score") or 0.0),
                "core_domain_count": int(row.get("core_domain_count") or 0),
                "related_domain_count": int(row.get("related_domain_count") or 0),
                "evidence": evidence,
            }
        )

    return main_rows, candidates


def compact_candidate_for_prompt(candidate: Dict[str, Any]) -> Dict[str, Any]:
    evidence = []
    for idx, item in enumerate(candidate.get("evidence") or [], start=1):
        evidence.append(
            {
                "source_index": idx,
                "title": item.get("title"),
                "description": item.get("description"),
                "source": item.get("source"),
            }
        )
    return {
        "candidate_id": candidate["candidate_id"],
        "rank": candidate["rank"],
        "canonical_title": candidate["title"],
        "evidence": evidence,
    }


def prompt_payload(main_rows: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "main_stories_already_shown": [normalize_space(r.get("title")) for r in main_rows],
        "candidates": [compact_candidate_for_prompt(c) for c in candidates],
    }


def make_prompt(payload: Dict[str, Any]) -> str:
    return f"""
You are selecting the small "Worth Knowing" section for Nominal News, a calm high-signal daily briefing.

The full-size main stories are already chosen and MUST NOT be duplicated. The candidates below have already passed Nominal News' discrete-event publication purifier and are ranked just below the main stories. Do not re-rank the main section.

Choose 0 to {MAX_ITEMS} candidate events that are useful for broad awareness even though they do not need a full story card.

Editorial standard:
- Prefer concrete developments with real public, scientific, health, consumer, travel, infrastructure, international, cultural, or similarly meaningful awareness value.
- Favor breadth across the day rather than several versions of the same subject.
- A famous person or political actor does not make an item important by itself. Apply the same concrete-impact standard to every subject.
- Avoid commentary, generic trends, pure horse-race/poll stories, celebrity minutiae, and items that substantially duplicate a main story.
- Do not force a quota. Return no items if nothing clearly improves the briefing.
- Use ONLY the supplied evidence. Do not use outside knowledge.

For every selected item:
- Keep the supplied candidate_id.
- Write exactly one neutral factual sentence, ideally 18-32 words, explaining the development.
- Keep the public sentence focused on WHAT HAPPENED. Do not append generalized significance, interpretation, trend language, or a "why it matters" clause unless that claim is explicitly stated in the selected evidence.
- Put editorial significance only in why_selected, which is internal and is not shown to readers.
- Preserve the evidence's level of certainty and attribution. Do not turn "clearing the way," "could," "may," "would," "expected," "proposed," or similar qualified language into a definite outcome.
- Preserve neutral attribution language. If the evidence says critics "said" something, do not replace that with a stronger or more adversarial verb such as "claimed" unless the evidence itself uses that wording.
- Do not add a date, number, person, place, cause, motive, outcome, or broader implication that is not explicitly supported by the evidence.
- If the candidate evidence contains titles but little or no description text, keep the sentence especially close to the supplied headline wording rather than filling in context.
- Pick source_index pointing to the single supplied evidence item that best supports the ENTIRE public sentence.

Return JSON only in this shape:
{{
  "items": [
    {{
      "candidate_id": "...",
      "summary": "...",
      "source_index": 1,
      "why_selected": "brief internal rationale"
    }}
  ]
}}

DATA:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def cache_key(payload: Dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"{MODEL}:{PROMPT_VERSION}:{digest}"


def parse_json_object(raw: str) -> Dict[str, Any]:
    raw = normalize_space(raw)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def call_model(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if openai is None:
        raise RuntimeError("The openai package is required for Worth Knowing generation.")
    if not getattr(openai, "api_key", None):
        raise RuntimeError("OPENAI_API_KEY is not available in the environment.")

    response = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        max_tokens=700,
        messages=[
            {"role": "system", "content": "Return valid JSON only. Be conservative and evidence-grounded."},
            {"role": "user", "content": make_prompt(payload)},
        ],
    )
    text = response["choices"][0]["message"]["content"]

    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    estimated_cost_usd = (
        (prompt_tokens / 1_000_000) * GPT4O_MINI_INPUT_USD_PER_MILLION
        + (completion_tokens / 1_000_000) * GPT4O_MINI_OUTPUT_USD_PER_MILLION
    )

    return parse_json_object(text), {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 8),
    }


def sentence_count(text: str) -> int:
    """Count prose sentences without splitting common abbreviations such as U.S."""
    value = normalize_space(text)
    if not value:
        return 0

    placeholder = "\uE000"

    # Protect initialisms such as U.S., U.N., U.K., D.C., etc.
    value = re.sub(
        r"\b(?:[A-Za-z]\.){2,}",
        lambda m: m.group(0).replace(".", placeholder),
        value,
    )

    # Protect a small set of ordinary prose abbreviations. These are grammar, not
    # editorial/topic rules, and prevent false multi-sentence rejection.
    value = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Sen|Rep|Gov|Gen|Lt|Col|Maj|Capt|Sgt)\.",
        lambda m: m.group(0).replace(".", placeholder),
        value,
        flags=re.I,
    )

    # Protect decimal points.
    value = re.sub(r"(?<=\d)\.(?=\d)", placeholder, value)

    pieces = [
        p
        for p in re.split(r'(?<=[.!?])[\"”’\']*\s+(?=[A-Z0-9\"“‘])', value)
        if p.strip()
    ]
    return len(pieces)


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", normalize_space(text)))


def validate_model_items(
    model_result: Dict[str, Any], candidates: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidate_map = {c["candidate_id"]: c for c in candidates}
    raw_items = model_result.get("items") or []
    if not isinstance(raw_items, list):
        raw_items = []

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_ids = set()

    for raw in raw_items[: MAX_ITEMS + 3]:
        if not isinstance(raw, dict):
            continue
        candidate_id = normalize_space(raw.get("candidate_id"))
        summary = normalize_space(raw.get("summary"))
        reason = normalize_space(raw.get("why_selected"))
        try:
            source_index = int(raw.get("source_index"))
        except Exception:
            source_index = 0

        candidate = candidate_map.get(candidate_id)
        reject_reason = ""
        if not candidate or candidate_id in seen_ids:
            reject_reason = "unknown_or_duplicate_candidate"
        elif not summary or len(summary.split()) > 42 or sentence_count(summary) != 1:
            reject_reason = "summary_shape_failed"
        elif source_index < 1 or source_index > len(candidate.get("evidence") or []):
            reject_reason = "invalid_source_index"
        else:
            source = candidate["evidence"][source_index - 1]
            grounding_text = " ".join(
                [
                    candidate.get("title", ""),
                    source.get("title", ""),
                    source.get("description", ""),
                ]
            )
            unsupported_numbers = numeric_tokens(summary) - numeric_tokens(grounding_text)
            if unsupported_numbers:
                reject_reason = "unsupported_numeric_detail"

        if reject_reason:
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "summary": summary,
                    "reason": reject_reason,
                }
            )
            continue

        source = candidate["evidence"][source_index - 1]
        accepted.append(
            {
                "event_id": candidate_id,
                "topic": candidate.get("topic"),
                "rank": candidate.get("rank"),
                "headline": display_headline(candidate.get("title")),
                "summary": summary,
                "source_url": source.get("url"),
                "source_name": display_source_name(
                    source.get("url", ""), source.get("source", "")
                ),
                "why_selected": reason,
                "source_index": source_index,
                "coverage_domain_count": max(
                    int(candidate.get("core_domain_count") or 0),
                    int(candidate.get("related_domain_count") or 0),
                ),
            }
        )
        seen_ids.add(candidate_id)
        if len(accepted) >= MAX_ITEMS:
            break

    # Keep presentation deterministic and aligned with the already-computed
    # global shadow ranking rather than depending on model output order.
    accepted.sort(key=lambda item: (int(item.get("rank") or 10**9), item.get("headline") or ""))

    return accepted, rejected


def run() -> int:
    args = parse_args()
    try:
        date_str = datetime.strptime(args.date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        print("❌ Invalid date. Use YYYY-MM-DD.")
        return 1

    base_dir = Path(__file__).resolve().parent
    ranking_file = find_ranking_file(base_dir, date_str)
    if ranking_file is None:
        print(f"❌ No GDELT global ranking report found for {date_str}.")
        return 1

    ranking = load_json(ranking_file, {})
    if not isinstance(ranking, dict) or ranking.get("status") != "OK":
        print(f"❌ Ranking report is not usable: {ranking_file.name}")
        return 1

    clusters = load_cluster_map(base_dir, date_str)
    discovery = load_discovery_candidate_map(base_dir, date_str)
    main_rows, candidates = build_candidates(ranking, clusters, discovery)

    print(f"🧭 Worth Knowing candidate pool for {date_str}: {len(candidates)}")
    for c in candidates:
        print(f"   #{c['rank']} {c['title']} ({len(c['evidence'])} evidence items)")

    if args.dry_run:
        print("🧪 Dry run only; no model call and no output file written.")
        return 0

    output_file = base_dir / f"worth_knowing_{date_str}.json"
    report_file = base_dir / f"worth_knowing_report_{date_str}.json"

    if not candidates:
        payload = {
            "schema_version": "1.0",
            "date": date_str,
            "items": [],
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
        }
        save_json(output_file, payload)
        save_json(
            report_file,
            {
                **payload,
                "candidate_count": 0,
                "ranking_file": ranking_file.name,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "reason": "No eligible below-the-main-section candidates were available.",
            },
        )
        print(f"✅ Wrote empty Worth Knowing section: {output_file.name}")
        return 0

    model_input = prompt_payload(main_rows, candidates)
    key = cache_key(model_input)
    cache_path = base_dir / CACHE_FILE
    cache = load_json(cache_path, {})
    if not isinstance(cache, dict):
        cache = {}

    cache_hit = False
    api_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    if not args.refresh and key in cache and isinstance(cache[key], dict):
        model_result = cache[key]
        cache_hit = True
        print("♻️ Reused cached Worth Knowing selection.")
    else:
        model_result, api_usage = call_model(model_input)
        cache[key] = model_result
        save_json(cache_path, cache)
        print("🤖 Used 1 model call for the entire Worth Knowing section.")

    items, rejected = validate_model_items(model_result, candidates)

    public_payload = {
        "schema_version": "1.0",
        "date": date_str,
        "items": [
            {
                "headline": item["headline"],
                "summary": item["summary"],
                "source_url": item["source_url"],
                "source_name": item["source_name"],
                "coverage_domain_count": item["coverage_domain_count"],
            }
            for item in items
        ],
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
    }
    save_json(output_file, public_payload)

    report_payload = {
        "schema_version": "1.0",
        "date": date_str,
        "ranking_file": ranking_file.name,
        "candidate_count": len(candidates),
        "main_story_count": len(main_rows),
        "selected_count": len(items),
        "cache_hit": cache_hit,
        "model_call_count": 0 if cache_hit else 1,
        "prompt_tokens": api_usage["prompt_tokens"],
        "completion_tokens": api_usage["completion_tokens"],
        "total_tokens": api_usage["total_tokens"],
        "estimated_cost_usd": api_usage["estimated_cost_usd"],
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "candidates": candidates,
        "model_result": model_result,
        "accepted_items": items,
        "rejected_model_items": rejected,
    }
    save_json(report_file, report_payload)

    print(f"✅ Worth Knowing items: {len(items)}")
    for item in items:
        print(f"   • {item['headline']}")
    print(f"💾 {output_file.name}")
    print(f"🧾 {report_file.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
