#!/usr/bin/env python3
"""
enrich_gdelt_discovery_articles.py

Shadow-only metadata enrichment for purifier-approved GDELT discovery articles.

- Reads grouped_articles_final_global_shadow_{date}.json by default.
- Fetches only GDELT-origin articles whose descriptions are blank/placeholder.
- Extracts public page metadata (description, site name, publication time, image).
- Never changes article titles or URLs.
- Writes a separate enriched shadow JSON.
- Uses a local cache to avoid repeated requests.
- Makes no OpenAI, NewsAPI, or GDELT API calls.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


USER_AGENT = (
    "NominalNews-Metadata-Enrichment/1.0 "
    "(bounded public-page metadata fetch; contact via project repository)"
)
DEFAULT_MAX_REQUESTS = 32
DEFAULT_MAX_BYTES = 1_500_000
DEFAULT_TIMEOUT = 12
MIN_DESCRIPTION_CHARS = 45
MAX_DESCRIPTION_CHARS = 900


class MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self._in_json_ld = False
        self._json_ld_parts: list[str] = []
        self.json_ld_blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {
            (k or "").lower(): (v or "")
            for k, v in attrs
        }

        if tag.lower() == "meta":
            key = (
                attrs_dict.get("property")
                or attrs_dict.get("name")
                or attrs_dict.get("itemprop")
                or ""
            ).strip().lower()
            value = attrs_dict.get("content", "").strip()
            if key and value and key not in self.meta:
                self.meta[key] = value

        if tag.lower() == "script":
            typ = attrs_dict.get("type", "").lower().strip()
            if "ld+json" in typ:
                self._in_json_ld = True
                self._json_ld_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self._json_ld_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._in_json_ld:
            block = "".join(self._json_ld_parts).strip()
            if block:
                self.json_ld_blocks.append(block)
            self._in_json_ld = False
            self._json_ld_parts = []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enrich purifier-approved GDELT articles from public page metadata."
    )
    p.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    p.add_argument("--input-file", default=None)
    p.add_argument("--output-file", default=None)
    p.add_argument(
        "--cache-file",
        default="gdelt_discovery_metadata_cache.json",
    )
    p.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
    )
    p.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
    )
    p.add_argument(
        "--cached-only",
        action="store_true",
        help="Use only cached metadata; make no page requests.",
    )
    return p.parse_args()


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def usable_description(value: Any, title: str = "") -> str:
    text = clean_text(value)
    if len(text) < MIN_DESCRIPTION_CHARS:
        return ""

    lowered = text.lower()
    bad_fragments = (
        "enable javascript",
        "browser is not supported",
        "accept cookies",
        "cookie policy",
        "access denied",
        "subscribe to continue",
        "sign in to continue",
        "please verify you are human",
    )
    if any(fragment in lowered for fragment in bad_fragments):
        return ""

    title_norm = clean_text(title).lower().strip(" .")
    text_norm = lowered.strip(" .")
    if title_norm and text_norm == title_norm:
        return ""

    if len(text) > MAX_DESCRIPTION_CHARS:
        text = text[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-") + "…"

    return text


def is_gdelt_article(article: dict) -> bool:
    return bool(
        article.get("origin") == "gdelt_discovery"
        or article.get("gdelt_candidate_id")
    )


def needs_enrichment(article: dict) -> bool:
    if not is_gdelt_article(article):
        return False
    return not usable_description(
        article.get("description"),
        article.get("title", ""),
    )


def nested_values(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nested_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_values(child)


def parse_json_ld(blocks: list[str], title: str) -> dict[str, str]:
    result: dict[str, str] = {}

    for block in blocks:
        candidates = [block]
        # Some sites wrap JSON-LD in comments.
        cleaned = re.sub(r"^\s*<!--|-->\s*$", "", block.strip())
        if cleaned != block:
            candidates.append(cleaned)

        data = None
        for candidate in candidates:
            try:
                data = json.loads(candidate)
                break
            except Exception:
                continue
        if data is None:
            continue

        for obj in nested_values(data):
            if not result.get("description"):
                desc = usable_description(obj.get("description"), title)
                if desc:
                    result["description"] = desc

            if not result.get("published_at"):
                published = clean_text(
                    obj.get("datePublished")
                    or obj.get("dateCreated")
                    or ""
                )
                if published:
                    result["published_at"] = published

            if not result.get("site_name"):
                publisher = obj.get("publisher")
                if isinstance(publisher, dict):
                    name = clean_text(publisher.get("name"))
                    if name:
                        result["site_name"] = name

            if not result.get("image_url"):
                image = obj.get("image")
                if isinstance(image, str):
                    result["image_url"] = image.strip()
                elif isinstance(image, dict):
                    image_url = image.get("url") or image.get("contentUrl")
                    if image_url:
                        result["image_url"] = str(image_url).strip()
                elif isinstance(image, list) and image:
                    first = image[0]
                    if isinstance(first, str):
                        result["image_url"] = first.strip()
                    elif isinstance(first, dict):
                        image_url = first.get("url") or first.get("contentUrl")
                        if image_url:
                            result["image_url"] = str(image_url).strip()

    return result


def extract_metadata(body: str, title: str) -> dict[str, str]:
    parser = MetadataParser()
    try:
        parser.feed(body)
    except Exception:
        pass

    meta = parser.meta
    result: dict[str, str] = {}

    description_keys = (
        "og:description",
        "twitter:description",
        "description",
        "dc.description",
        "sailthru.description",
    )
    for key in description_keys:
        desc = usable_description(meta.get(key), title)
        if desc:
            result["description"] = desc
            break

    site_name_keys = (
        "og:site_name",
        "application-name",
        "twitter:site",
    )
    for key in site_name_keys:
        site = clean_text(meta.get(key))
        if site:
            result["site_name"] = site.lstrip("@")
            break

    date_keys = (
        "article:published_time",
        "datepublished",
        "date",
        "dc.date",
        "parsely-pub-date",
    )
    for key in date_keys:
        value = clean_text(meta.get(key))
        if value:
            result["published_at"] = value
            break

    image_keys = (
        "og:image",
        "twitter:image",
        "twitter:image:src",
    )
    for key in image_keys:
        value = clean_text(meta.get(key))
        if value:
            result["image_url"] = value
            break

    ld = parse_json_ld(parser.json_ld_blocks, title)
    for key, value in ld.items():
        result.setdefault(key, value)

    return result


def response_text_limited(
    response: requests.Response,
    max_bytes: int,
) -> str:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        remaining = max_bytes - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_bytes:
            break

    raw = b"".join(chunks)
    encoding = response.encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace")
    except Exception:
        return raw.decode("utf-8", errors="replace")


def fetch_metadata(
    session: requests.Session,
    url: str,
    title: str,
    timeout: int,
    max_bytes: int,
) -> tuple[dict[str, str], str, bool]:
    try:
        with session.get(
            url,
            stream=True,
            timeout=(6, timeout),
            allow_redirects=True,
            headers={
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
            },
        ) as response:
            status_code = int(response.status_code)
            if status_code == 429 or status_code >= 500:
                return {}, f"http_{status_code}", False
            if status_code >= 400:
                return {}, f"http_{status_code}", True

            content_type = (response.headers.get("content-type") or "").lower()
            if content_type and "html" not in content_type:
                return {}, "not_html", True

            body = response_text_limited(response, max_bytes)
            metadata = extract_metadata(body, title)
            if metadata.get("description"):
                return metadata, "ok", True
            return metadata, "no_description", True
    except requests.Timeout:
        return {}, "timeout", False
    except requests.RequestException as exc:
        return {}, type(exc).__name__, False
    except Exception as exc:
        return {}, type(exc).__name__, False


def apply_metadata(article: dict, metadata: dict[str, str]) -> bool:
    changed = False
    title = article.get("title", "")

    description = usable_description(metadata.get("description"), title)
    if description and not usable_description(article.get("description"), title):
        article["description"] = description
        article["metadata_enriched"] = True
        changed = True

    published_at = clean_text(metadata.get("published_at"))
    if published_at and not article.get("published_at"):
        article["published_at"] = published_at
        if len(published_at) >= 10 and not article.get("published_date"):
            article["published_date"] = published_at[:10]
        changed = True

    site_name = clean_text(metadata.get("site_name"))
    current_source = clean_text(article.get("source"))
    host = (urlparse(article.get("url") or "").hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if site_name and (not current_source or current_source.lower() == host):
        article["source"] = site_name
        changed = True

    image_url = clean_text(metadata.get("image_url"))
    if image_url and not article.get("image_url"):
        article["image_url"] = image_url
        changed = True

    return changed


def main() -> int:
    args = parse_args()

    input_path = Path(
        args.input_file
        or f"grouped_articles_final_global_shadow_{args.date}.json"
    )
    output_path = Path(
        args.output_file
        or f"grouped_articles_final_global_enriched_shadow_{args.date}.json"
    )
    cache_path = Path(args.cache_file)

    if input_path.resolve() == output_path.resolve():
        print("❌ Refusing to overwrite the input file. Use a separate --output-file.")
        return 2
    if not input_path.exists():
        print(f"❌ Missing {input_path}")
        return 1

    raw = load_json(input_path, None)
    if isinstance(raw, dict) and isinstance(raw.get("clusters"), list):
        clusters = raw["clusters"]
        wrapper = deepcopy(raw)
    elif isinstance(raw, list):
        clusters = raw
        wrapper = None
    else:
        print(f"❌ Unsupported JSON structure in {input_path}")
        return 1

    output_clusters = deepcopy(clusters)
    cache = load_json(cache_path, {})
    if not isinstance(cache, dict):
        cache = {}

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    stats = {
        "eligible_articles": 0,
        "cache_hits": 0,
        "live_requests": 0,
        "descriptions_added": 0,
        "metadata_updates": 0,
        "failed_or_empty": 0,
        "request_cap_reached": False,
    }
    status_counts: dict[str, int] = {}

    # Avoid requesting the same URL twice when it appears in core + related lists.
    metadata_by_url: dict[str, dict[str, str]] = {}

    article_lists = []
    for cluster in output_clusters:
        for key in ("articles", "related_articles"):
            values = cluster.get(key)
            if isinstance(values, list):
                article_lists.append(values)

    for articles in article_lists:
        for article in articles:
            if not isinstance(article, dict) or not needs_enrichment(article):
                continue

            stats["eligible_articles"] += 1
            url = clean_text(article.get("url"))
            if not url:
                stats["failed_or_empty"] += 1
                continue

            metadata: dict[str, str] = {}
            cache_entry = cache.get(url)

            if url in metadata_by_url:
                metadata = metadata_by_url[url]
            elif isinstance(cache_entry, dict) and cache_entry.get("cacheable"):
                metadata = cache_entry.get("metadata") or {}
                metadata_by_url[url] = metadata
                stats["cache_hits"] += 1
            elif args.cached_only:
                metadata_by_url[url] = {}
                stats["failed_or_empty"] += 1
                continue
            elif stats["live_requests"] >= max(0, args.max_requests):
                stats["request_cap_reached"] = True
                metadata_by_url[url] = {}
                stats["failed_or_empty"] += 1
                continue
            else:
                stats["live_requests"] += 1
                metadata, status, cacheable = fetch_metadata(
                    session=session,
                    url=url,
                    title=clean_text(article.get("title")),
                    timeout=max(1, args.timeout),
                    max_bytes=max(50_000, args.max_bytes),
                )
                status_counts[status] = status_counts.get(status, 0) + 1
                metadata_by_url[url] = metadata

                if cacheable:
                    cache[url] = {
                        "status": status,
                        "cacheable": True,
                        "metadata": metadata,
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                    }

                # Small courtesy pause between different public sites.
                time.sleep(0.15)

            before_desc = bool(
                usable_description(article.get("description"), article.get("title", ""))
            )
            changed = apply_metadata(article, metadata)
            after_desc = bool(
                usable_description(article.get("description"), article.get("title", ""))
            )

            if changed:
                stats["metadata_updates"] += 1
            if not before_desc and after_desc:
                stats["descriptions_added"] += 1
            if not after_desc:
                stats["failed_or_empty"] += 1

    save_json(cache_path, cache)

    if wrapper is not None:
        wrapper["clusters"] = output_clusters
        wrapper["metadata_enrichment"] = {
            "shadow_only": True,
            "input_file": str(input_path),
            "statistics": stats,
            "status_counts": status_counts,
        }
        output_obj = wrapper
    else:
        output_obj = output_clusters

    save_json(output_path, output_obj)

    print(f"🧾 GDELT metadata enrichment: {stats['descriptions_added']} descriptions added")
    print(
        f"   {stats['live_requests']} live requests, "
        f"{stats['cache_hits']} cache hits, "
        f"{stats['failed_or_empty']} still without usable descriptions"
    )
    if stats["request_cap_reached"]:
        print(f"⚠️ Request cap reached at {args.max_requests}; remaining articles were left unchanged.")
    if status_counts:
        print(
            "   statuses: "
            + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
        )
    print(f"✅ Wrote enriched global shadow → {output_path}")
    print("🛡️ Input and production files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
