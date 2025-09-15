# main.py — supports dynamic topic summary loading by date

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
import os
from datetime import datetime
import subprocess

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index(request: Request, date: str = None):
    summaries = []

    # ✅ If no date is given, use today's date
    if not date:
        date = datetime.today().strftime("%Y-%m-%d")

    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        summary_file = f"topic_summaries_{date}.json"

        if not os.path.exists(summary_file):
            print(f"⏳ Running full pipeline for {date}...")
            subprocess.run(["python", "run_pipeline_by_date.py", "--date", date])

        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                summaries = json.load(f)
        else:
            print(f"⚠️ No summaries found after pipeline run for {date}")
    except Exception as e:
        print(f"❌ Date handling or pipeline error: {e}")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "summaries": summaries,
        "requested_date": date  # ✅ Ensures date field in the form is filled
    })
