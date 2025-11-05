# main.py — serves the Jinja template and loads topic_summaries by date (no-token preview friendly)

from __future__ import annotations

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
import subprocess
import os
import json
import re
from typing import Optional, List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def list_summary_files() -> List[Path]:
    """Return all topic_summaries_YYYY-MM-DD.json files sorted by name."""
    return sorted((BASE_DIR).glob("topic_summaries_*.json"))


def latest_summary_date() -> Optional[str]:
    """Find the newest date that we have a topic_summaries_*.json file for."""
    dates: List[str] = []
    for p in list_summary_files():
        m = re.search(r"topic_summaries_(\d{4}-\d{2}-\d{2})\.json$", p.name)
        if m:
            dates.append(m.group(1))
    return max(dates) if dates else None


def load_summaries(date: str) -> List[Dict[str, Any]]:
    """Load summaries for a given YYYY-MM-DD date, or [] if not found."""
    path = BASE_DIR / f"topic_summaries_{date}.json"
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
    - If not, we auto-pick the latest available topic_summaries_*.json.
    - We DO NOT auto-run the pipeline by default (saves tokens).
      To allow auto-run if missing, set env var AUTO_RUN_PIPELINE=1.
    """
    status_msg = ""
    requested_date = date

    if not requested_date:
        # No date provided — pick the newest file on disk if available
        latest = latest_summary_date()
        requested_date = latest or datetime.now().strftime("%Y-%m-%d")
        if latest:
            status_msg = f"Loaded latest available summaries for {requested_date}."

    summaries = load_summaries(requested_date)

    if not summaries:
        # Optional: auto-run full pipeline (costs tokens!)
        if os.getenv("AUTO_RUN_PIPELINE", "0") == "1":
            try:
                subprocess.run(
                    ["python", str(BASE_DIR / "run_pipeline_by_date.py"), "--date", requested_date],
                    check=True,
                )
                summaries = load_summaries(requested_date)
                status_msg = status_msg or f"Generated summaries for {requested_date}."
            except Exception as e:
                status_msg = f"No summaries found for {requested_date}. Pipeline errored: {e}"
        else:
            status_msg = (
                f"No summaries found for {requested_date}. "
                f"Either pass ?date=YYYY-MM-DD with a day you already ran, "
                f"or set AUTO_RUN_PIPELINE=1 to generate on demand."
            )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "summaries": summaries,
            "requested_date": requested_date,
            "status_msg": status_msg,
        },
    )
