# Nominal News

A daily news pipeline and local web reader. It groups related articles, summarizes
stories, adds Next Signal checkpoints, and selects context images.

## Install

Use Python 3.13. From the repository directory, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
Copy-Item .env.example .env
```

Only run the copy command if `.env` does not already exist. Edit `.env` locally
to supply `OPENAI_API_KEY`, `NEWS_API_KEY`, and `UNSPLASH_ACCESS_KEY`. Never commit
credentials or include them in screenshots, issue reports, or documentation.
The variable names above are configuration labels, not credentials.

The pipeline uses the Sentence Transformers model `all-MiniLM-L6-v2`; its first
use may download model files. The spaCy model supports image subject extraction.
Keep installed models and operational caches for later runs. On macOS/Linux, use
`.venv/bin/python` instead of the Windows executable path.

## Preview a saved issue

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`, or select a date with `/?date=YYYY-MM-DD`.
A fresh clone has no published issues: daily JSON files are excluded from Git.
Copy your own saved issue data locally or generate an issue. Opening the reader
does not run the daily pipeline. Keep the development server bound to localhost.

## Run the daily pipeline

Replace `YYYY-MM-DD` with the intended issue date. Inspect the plan first:

```powershell
.\.venv\Scripts\python.exe run_pipeline_by_date.py --date YYYY-MM-DD --dry-run
```

The dry run prints planned stages without executing them. To execute:

```powershell
.\.venv\Scripts\python.exe run_pipeline_by_date.py --date YYYY-MM-DD
```

The real run makes external requests and paid OpenAI calls. It uses global
discovery with the existing baseline fallback, then runs Next Signal. The final
image check runs before images are saved. Unresolved image checks withhold the
image and retain a review reason. These metadata checks do not guarantee ownership
or legal clearance.

Review the exit status, dated outputs, and rendered issue. Do not import pipeline
scripts just to inspect them: some execute work at import time. Worth Knowing has
a separate generator and is not part of the daily runner.

## Offline regression tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Tests use mocks and one fixed fixture under `tests/fixtures/`. They do not need
daily JSON files, local provider caches, credentials, or paid calls. Fixture dates
are historical regression examples, not runtime configuration.

## Local data and Git

Commit source code, templates, styles, setup files, `bias_overrides.json`, and
tests. Keep secrets, environments, model/provider caches, daily outputs, and
review evidence local. Removing a file from Git tracking keeps its local copy;
it does not erase earlier Git history.

Keep published issue files and image-rights evidence while retaining the issue.
Keep operational caches to avoid repeating paid work. After reviewing and backing
up an issue, intermediate article/grouping outputs can be considered for deletion
if you no longer need to reproduce that run. There is no automatic cleanup job.
Old local daily data is not required to run the tests.

## Optional Docker preview

The Dockerfile starts the reader, not a paid pipeline. Its build context excludes
secrets, caches, daily JSON, and Git history.

```sh
docker build -t nominal-news .
docker run --rm -p 127.0.0.1:8000:8000 nominal-news
```

Mount your saved issue JSON files read-only under `/app/` to view them. The image
does not include local issues or downloaded language models. Validate Docker on
the target host before deployment; this preview setup does not provide production
hosting or authentication.
