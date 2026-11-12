# main.py — serves the production newsletter by date

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, Query
from edition_intro import introduction_for_date, sign_off_for_date
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime
import json
import re
from typing import Optional, List, Dict, Any
from reader_security import ReaderSecurityHeaders, legacy_chips, safe_http_url, validate_edition_date

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(ReaderSecurityHeaders, template_path=BASE_DIR / "templates" / "index.html")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.filters.update(safe_http_url=safe_http_url, legacy_chips=legacy_chips)


@app.get("/static/style.css", include_in_schema=False)
async def stylesheet():
    # An explicit public asset route, never a mount of the local static tree.
    expected = BASE_DIR.resolve() / "static" / "style.css"
    if expected.resolve() != expected or not expected.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(expected, media_type="text/css")


def edition_path(prefix: str, date_str: str) -> Path:
    validate_edition_date(date_str)
    if prefix not in {"topic_summaries", "market_overview", "worth_knowing"}:
        raise ValueError("Unsupported edition file.")
    expected = BASE_DIR.resolve() / f"{prefix}_{date_str}.json"
    # Also reject aliases/symlinks to a different file, even inside BASE_DIR.
    if expected.resolve() != expected:
        raise HTTPException(status_code=404, detail="Edition not available")
    return expected


def list_summary_files() -> List[Path]:
    """Return all topic_summaries_YYYY-MM-DD.json files sorted by name."""
    return sorted((BASE_DIR).glob("topic_summaries_*.json"))


def load_market_overview(date_str: str) -> Optional[Dict[str, Any]]:
    """Load market_overview_YYYY-MM-DD.json if present."""
    path = edition_path("market_overview", date_str)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_worth_knowing(date_str: str) -> Optional[Dict[str, Any]]:
    """Load the additive Worth Knowing section for a given date, if present."""
    path = edition_path("worth_knowing", date_str)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            return None
        return payload
    except Exception:
        return None

def latest_summary_date() -> Optional[str]:
    """Find the newest date that we have a topic_summaries_*.json file for."""
    dates: List[str] = []
    for p in list_summary_files():
        m = re.search(r"topic_summaries_(\d{4}-\d{2}-\d{2})\.json$", p.name)
        if m:
            try:
                if edition_path("topic_summaries", m.group(1)).is_file():
                    dates.append(m.group(1))
            except (ValueError, HTTPException):
                continue
    return max(dates) if dates else None


def load_summaries(date: str) -> List[Dict[str, Any]]:
    """Load production summaries for a given YYYY-MM-DD date, or [] if not found."""
    path = edition_path("topic_summaries", date)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
async def homepage(
    request: Request,
    date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
):
    """
    Shows the newsletter page.
    - If ?date=YYYY-MM-DD is provided, we use that day.
    - If no date is provided, we auto-pick the latest available
      topic_summaries_*.json.
    - This reader never runs the pipeline, regardless of environment settings.
    """
    status_msg = ""
    try:
        requested_date = validate_edition_date(date) if date is not None else None
    except ValueError:
        raise HTTPException(status_code=422, detail="Use a valid date in YYYY-MM-DD format.") from None

    if not requested_date:
        # No date provided — pick the newest file on disk if available
        latest = latest_summary_date()
        requested_date = latest or datetime.now().strftime("%Y-%m-%d")
        if latest:
            status_msg = f"Loaded latest available summaries for {requested_date}."

    summaries = load_summaries(requested_date)

    if not summaries:
        status_msg = f"No published edition is available for {requested_date}."

    market_overview = load_market_overview(requested_date) if requested_date else None
    worth_knowing = load_worth_knowing(requested_date) if requested_date else None
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "summaries": summaries,
            "requested_date": requested_date,
            "edition_intro": introduction_for_date(requested_date),
            "edition_sign_off": sign_off_for_date(requested_date),
            "status_msg": status_msg,
            "market_overview": market_overview,
            "worth_knowing": worth_knowing,
        },
    )
