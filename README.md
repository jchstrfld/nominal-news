# Nominal News

A daily news pipeline and local web reader. It groups related articles, summarizes
stories, adds Next Signal checkpoints, and selects context images.

## Install

Use a security-patched Python 3.13 runtime (3.13.18 or later). Check the Python
release page when provisioning a host; a virtual environment does not update
the underlying interpreter. From the repository directory, in PowerShell:

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

The public reader has a separate, pinned dependency set in `requirements-reader.txt`.
Install only that file on a reader host. The full `requirements.txt` also uses
`constraints.txt` to preserve tested worker dependencies. These constraints were
validated on Windows; additional platform dependencies need auditing on the target
host. After installing, run `python -m pip check` and the offline tests below.

## Preview a saved issue

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`, or select a date with `/?date=YYYY-MM-DD`.
A fresh clone has no published issues: daily JSON files are excluded from Git.
Copy your own saved issue data locally or generate an issue. The reader accepts
valid calendar dates in `YYYY-MM-DD` format and never runs the pipeline, even if
`AUTO_RUN_PIPELINE` is set. Only `static/style.css` is a public static asset;
local review pages, caches and other private files are not served. Keep the
development server bound to localhost.

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

Story review is required on either ranking path. An incomplete review, budget
stop, or review error stops the pipeline without switching to baseline ranking
or running summaries and signals. Completed review decisions remain in the local
cache, and the existing edition is preserved. Discovery can still fall back
before review begins; later optional coverage-stage fallbacks reuse a completed
review rather than retrying it.

Set the story-review allowance with `--story-review-budget-usd`. It applies to
**this review attempt only**, separately from Next Signal and all other stages.
Without the flag, it uses the shell's `NN_EVENT_BUDGET_USD` value or $0.90. The
default may not finish a busy edition; the runner stops rather than increasing it.
Use an approved allowance and account for earlier spending. These stage limits
are not a whole-pipeline spending cap.

To resume a stopped story review, inspect its dated
`grouped_articles_final_YYYY-MM-DD.review.json` and spending records first. Then
replace `APPROVED_REMAINING_ALLOWANCE` below with the additional amount approved
for this attempt:

```powershell
.\.venv\Scripts\python.exe run_pipeline_by_date.py --date YYYY-MM-DD --resume-story-review --story-review-budget-usd APPROVED_REMAINING_ALLOWANCE
```

Add `--dry-run` to inspect this plan without making calls. Resuming skips fetching,
discovery and clustering, reuses saved reviews, and continues downstream only
after review finishes. The private operational receipt at
`outputs/story_review_run_YYYY-MM-DD.json` preserves the ranking path and checks
that inputs, review code and settings have not changed. A normal restart cannot
bypass a pending review. Keep this receipt and the review cache; inspect changed
or missing inputs rather than deleting state to force a retry. A resume always
requires an explicit additional allowance. Previous spending still counts toward
any agreed daily cap.

The daily runner uses source-bound checkpoint review for Next Signal, with an
edition-wide default limit of $0.25 and 40 reviewer requests. These limits apply
only to Next Signal, not to the full pipeline's cost. Set them explicitly with
`--next-signal-budget-usd` and `--next-signal-max-calls`. The tested reviewer response
allowance is 2,400 tokens. Limits include previous attempts for that edition;
restarting does not reset spending. A budget stop or incomplete review stops the
run rather than silently counting as a verified absence of a signal.

Next Signal is optional. It selects a complete source sentence of up to 40 words,
preserving the source's wording, timing, conditions, and attribution. A local
check rejects rewritten or pronoun-dependent sentences. The reviewer checks each
candidate event against its source evidence, including its connection to the
story and whether it is still upcoming. Date checks compare eligible alternatives;
uncertain events or timing stay blank. Month-only timing remains month-only.
This deliberately omits some useful checkpoints rather than inventing missing
context. It does not guarantee finding every upcoming event. The daily route
uses available article evidence and does not run generic external signal searches.

Exact review receipts and cumulative spending are saved locally in
`outputs/checkpoint_receipts_YYYY-MM-DD.json` and
`outputs/checkpoint_usage_YYYY-MM-DD.json`. Keep these files to avoid repeating
paid reviews. To resume just Next Signal after inspecting a budget stop, use the
saved summary and the same cluster file used by the successful summary stage:

```powershell
.\.venv\Scripts\python.exe generate_next_signals.py --date YYYY-MM-DD --summary-file topic_summaries_YYYY-MM-DD.json --cluster-file YOUR_SAVED_CLUSTER_FILE.json --checkpoint-review --checkpoint-live --checkpoint-budget-usd 0.25 --checkpoint-max-calls 40 --checkpoint-max-output-tokens 2400 --checkpoint-cache-file outputs/checkpoint_receipts_YYYY-MM-DD.json --checkpoint-ledger-file outputs/checkpoint_usage_YYYY-MM-DD.json --no-external
```

Use the approved cumulative limits; a higher limit is not a fresh allowance.
Failed or unfinished requests retain their reservations and require inspection
before retrying. Do not delete the spending ledger to bypass a stop.

After that check, the bundled OpenCV YuNet detector positions the existing 16:9
crop around a clear primary face. Detection runs locally, with no paid API calls.
Selected photos and credits stay unchanged. No clear face, ambiguous faces, or
download/detection failure keeps the centered crop. Position metadata is cached
under `data/image_framing/`; no photos are retained by this step. The small model
and its MIT license are included under `models/face_detection/`.

Review the exit status, dated outputs, and rendered issue. Do not import pipeline
scripts just to inspect them: some execute work at import time. Worth Knowing has
a separate generator and is not part of the daily runner.

Public article metadata and optional Next Signal page downloads use checked public
HTTP/HTTPS destinations on standard ports. Each redirect is checked before
connecting, with at most five redirects and no HTTPS downgrade. Connections use
the checked IP while retaining hostname-based TLS verification. These anonymous
fetches do not use environment proxies, stored credentials or cookies. Existing
content-size limits and editorial checks still apply. Install the pinned HTTP
dependencies before running; cloud workers also need outbound network restrictions.
Authenticated provider API calls are separate from this public-page transport.
NewsAPI and Unsplash requests accept only their fixed HTTPS endpoints, reject
redirects, disable environment proxies and stored credentials, and bound response
sizes. API keys belong in headers, never query strings. Provider failures are
logged without response bodies or request details.

## Offline regression tests

```powershell
.\.venv\Scripts\python.exe -B tests/run_offline.py
```

The test runner blocks external networking. Tests use mocks and fixed fixtures
under `tests/fixtures/`. They do not need
daily JSON files, local provider caches, credentials, or paid calls. Fixture dates
are historical regression examples, not runtime configuration. Fixtures retain
inputs and expected responses needed for regression checks; development journals,
spending records, and experiment logs remain private.

For a test-only environment, install `requirements-test.txt`; it does not download
language models. The proposed GitHub checks run these tests on Windows and Linux,
audit the pinned packages, analyze code with CodeQL, and build the reader container.
They run after the workflow is pushed, including a weekly scheduled check. They
do not receive provider secrets, run the pipeline, publish issues, or send emails.

## Operating securely

Keep the public site/reader separate from the private generator. Give the reader
only published edition files, mounted read-only, with no API keys, subscriber
records, operational caches, or pipeline scripts. The reader applies a restricted
script policy, anti-framing and MIME protections, and a no-referrer policy. It
still loads public image URLs; image hosts can receive visitors' network details.

The public CMS can be Squarespace. Run generation and email delivery separately;
never put provider keys in Squarespace code blocks, forms, or browser JavaScript.
Keep subscribers in the chosen email service rather than in edition files or Git.
Signup consent, unsubscribe/deletion handling, email authentication, analytics
consent, and privacy notices must be configured and tested before collecting emails.

Before exposure, configure HTTPS, allowed hostnames, trusted proxies, request-rate
and connection limits at the hosting edge. Restrict worker network access to public
destinations and approved services. Use separate, least-privilege credentials and
encrypted backups; test restoring published issues and required operational state.
Do not deploy by uploading the working project directory.

Review dependency/security alerts weekly and patch the host's Python and OS too.
Dependabot proposes updates; do not auto-merge them. Run regression tests and an
installed-environment vulnerability scan before release, for example using
`pip-audit` in the deployment environment. A clean scan covers known advisories,
not all possible flaws. Require successful CI checks before merging release changes.
Also scan the final container's operating-system packages; a clean Python-package
audit does not clear the base image. Review unresolved advisories and available
vendor patches before public deployment.

Keep language-model choices and weights fixed between reviewed releases. Check
provider deprecations regularly. Model or embedding-library changes require saved
positive/negative cases, multi-edition comparisons, and an approved budget for any
live checks. Do not update models automatically just because a new version exists.
Recheck the complete signup-to-delivery system whenever its services or permissions
change. Never include credentials or subscriber records in public bug reports.

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

The Dockerfile contains only the reader and its dependencies, running as a
non-root user, with package installers removed after building. Rebuild the image
to apply dependency updates. It excludes the pipeline, models, secrets, caches, daily JSON, and
Git history. Docker is optional; the same separation is required on other hosts.

```sh
docker build -t nominal-news .
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges -p 127.0.0.1:8000:8000 nominal-news
```

Mount individual published issue JSON files read-only under `/app/` to view them;
do not mount the entire project. The image
does not include local issues or downloaded language models. Validate Docker on
the target host before deployment; this preview setup does not provide production
hosting or authentication.
