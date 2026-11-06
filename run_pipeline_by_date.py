# run_pipeline_by_date.py — full pipeline runner with GDELT global coverage enabled by default

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence


PRE_MERGE_STAGES = [
    "fetch_articles_by_date.py",
    "normalize_and_dedupe_articles.py",
    "bias_labeler.py",
    "generate_market_overview.py",
    "cluster_articles_by_embedding.py",
]

POST_MERGE_PRE_FINAL_STAGES = [
    "filter_outlier_articles.py",
    "report_cluster_cohesion.py",
]

BASELINE_POST_FINAL_STAGES = [
    "expand_cluster_coverage_gdelt.py",
    "summarize_grouped_topics.py",
]

REVIEW_ENV_NAMES = (
    "NN_EVENT_MODEL", "NN_EVENT_MAX_CALLS", "NN_EVENT_CACHE_FILE",
    "NN_MIN_PUBLISH_ARTICLES", "NN_MIN_PUBLISH_DOMAINS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    gdelt_group = parser.add_mutually_exclusive_group()
    gdelt_group.add_argument(
        "--with-gdelt-discovery",
        dest="with_gdelt_discovery",
        action="store_true",
        help=(
            "Use the full bounded GDELT global path. This is now the default; "
            "the flag is retained for backward compatibility."
        ),
    )
    gdelt_group.add_argument(
        "--no-gdelt-discovery",
        dest="with_gdelt_discovery",
        action="store_false",
        help=(
            "Opt out of GDELT global discovery/ranking/receipts and run the "
            "baseline NewsAPI/RSS path only."
        ),
    )
    parser.set_defaults(with_gdelt_discovery=True)
    parser.add_argument(
        "--gdelt-discovery-cached-only",
        action="store_true",
        help=(
            "Use cached GSG files only and make no new GDELT discovery download "
            "requests. GDELT is enabled by default."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    parser.add_argument(
        "--story-review-budget-usd",
        help="Additional allowance for this story-review attempt only; defaults to NN_EVENT_BUDGET_USD or 0.90. Required explicitly when resuming. Not a whole-pipeline cap.",
    )
    parser.add_argument(
        "--resume-story-review", action="store_true",
        help="Resume a stopped story review using its unchanged saved inputs and ranking path; skip fetching and clustering.",
    )
    parser.add_argument(
        "--next-signal-budget-usd", default="0.25",
        help="Cumulative Next Signal cap per edition, including earlier attempts (default: 0.25 USD). Not a whole-pipeline cap.",
    )
    parser.add_argument(
        "--next-signal-max-calls", type=int, default=40,
        help="Cumulative Next Signal request limit per edition (default: 40). Exhaustion stops the run; it does not skip reviews.",
    )
    args = parser.parse_args()
    try:
        limit = Decimal(args.next_signal_budget_usd)
        valid_limit = limit.is_finite() and limit > 0
    except InvalidOperation:
        valid_limit = False
    if not valid_limit or args.next_signal_max_calls <= 0:
        parser.error("Next Signal requires a positive finite dollar cap and a positive call limit.")
    if args.resume_story_review and args.story_review_budget_usd is None:
        parser.error("Resuming requires --story-review-budget-usd with the approved remaining allowance.")
    if args.story_review_budget_usd is None:
        args.story_review_budget_usd = os.getenv("NN_EVENT_BUDGET_USD", "0.90")
    try:
        limit = Decimal(args.story_review_budget_usd)
        valid_limit = limit.is_finite() and limit > 0 and math.isfinite(float(limit)) and float(limit) > 0
    except (InvalidOperation, OverflowError):
        valid_limit = False
    if not valid_limit:
        parser.error("Story review requires a positive finite dollar allowance.")
    return args


def _display_command(command: Sequence[str], base_dir: Path) -> str:
    shown = []
    for part in command:
        text = str(part)
        try:
            path = Path(text)
            if path.is_absolute() and path.parent == base_dir:
                text = path.name
        except Exception:
            pass
        shown.append(text)
    return " ".join(shown)


def run_required(
    *,
    base_dir: Path,
    script: str,
    date_str: str,
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> int:
    script_path = base_dir / script
    if not script_path.exists():
        print(f"❌ Missing pipeline stage: {script}")
        print("🛑 Pipeline stopped.")
        return 1

    command = [
        sys.executable,
        str(script_path),
        "--date",
        date_str,
        *map(str, extra_args),
    ]
    print(f"➡️ Running: {script}")
    if dry_run:
        print(f"   DRY RUN: {_display_command(command, base_dir)}")
        if env_overrides:
            print(f"   Story-review allowance: ${env_overrides['NN_EVENT_BUDGET_USD']} for this attempt.")
        return 0

    options = {"env": {**os.environ, **env_overrides}} if env_overrides else {}
    result = subprocess.run(command, cwd=str(base_dir), **options)
    if result.returncode != 0:
        print(f"❌ Stage failed: {script} (exit code {result.returncode})")
        print("🛑 Pipeline stopped. Downstream stages were not run.")
        return result.returncode or 1
    return 0


def run_optional(
    *,
    base_dir: Path,
    script: str,
    date_str: str,
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
) -> bool:
    script_path = base_dir / script
    if not script_path.exists():
        print(f"⚠️ Optional GDELT stage missing: {script}")
        print("↪ Falling back to the unchanged baseline path.")
        return False

    command = [
        sys.executable,
        str(script_path),
        "--date",
        date_str,
        *map(str, extra_args),
    ]
    print(f"➡️ Running optional GDELT stage: {script}")
    if dry_run:
        print(f"   DRY RUN: {_display_command(command, base_dir)}")
        return True

    result = subprocess.run(command, cwd=str(base_dir))
    if result.returncode != 0:
        print(
            f"⚠️ Optional GDELT stage failed: {script} "
            f"(exit code {result.returncode})"
        )
        print("↪ Falling back to the unchanged baseline path.")
        return False
    return True


def manifest_is_usable(path: Path, augmented_path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return (
            manifest.get("status") == "OK"
            and int(manifest.get("seed_cluster_count", 0) or 0) > 0
            and Path(manifest.get("augmented_cluster_file", "")).name
            == augmented_path.name
            and augmented_path.exists()
        )
    except Exception:
        return False


def outputs_exist(*paths: Path) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def review_state_path(base_dir: Path, date_str: str) -> Path:
    return base_dir / "outputs" / f"story_review_run_{date_str}.json"


def review_context(base_dir: Path, date_str: str, global_mode: bool) -> dict:
    """Bind retries to the same evidence, ranking path, code and review settings."""
    names = [f"grouped_articles_filtered_{date_str}.json"]
    if global_mode:
        names.append(f"gdelt_discovery_candidates_{date_str}.json")
    for name in names:
        if not (base_dir / name).is_file():
            raise ValueError(f"Missing saved review input: {name}")
    names += ["final_cohesion_check.py", "audit_gdelt_global_discovery.py",
              "inject_gdelt_discovery_candidates.py", "bias_overrides.json"]
    return {
        "date": date_str, "global_mode": global_mode,
        "files": {name: hashlib.sha256((base_dir / name).read_bytes()).hexdigest()
                  if (base_dir / name).is_file() else None for name in names},
        "settings": {name: os.getenv(name) for name in REVIEW_ENV_NAMES},
    }


def save_review_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(state, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_story_review(*, base_dir: Path, date_str: str, global_mode: bool,
                     budget: str, extra_args: Sequence[str] = (), dry_run: bool = False) -> int:
    """An incomplete review is required work, never a discovery fallback."""
    path = review_state_path(base_dir, date_str)
    state = None
    if not dry_run:
        try:
            context = review_context(base_dir, date_str, global_mode)
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            attempts = state.get("attempts", []) if state.get("context") == context else []
            state = {"version": 1, "status": "pending", "context": context, "attempts": attempts}
            state["attempts"].append({"allowance_usd": budget})
            save_review_state(path, state)
        except (ValueError, OSError) as exc:
            print(f"❌ Cannot prepare story review: {exc}")
            return 1
    code = run_required(base_dir=base_dir, script="final_cohesion_check.py", date_str=date_str,
                        extra_args=extra_args, dry_run=dry_run,
                        env_overrides={"NN_EVENT_BUDGET_USD": budget})
    expected = [base_dir / f"grouped_articles_final_{date_str}.json"]
    if global_mode:
        expected += [base_dir / f"gdelt_global_ranking_{date_str}.json",
                     base_dir / f"grouped_articles_final_global_{date_str}.json"]
    if not code and not dry_run and not outputs_exist(*expected):
        print("❌ Story review did not produce all expected outputs; stopping.")
        code = 1
    if state is not None:
        state["status"] = "pending" if code else "complete"
        state["attempts"][-1]["exit_code"] = code
        save_review_state(path, state)
    if code:
        print("🛑 Story review stopped. No ranking fallback or downstream stages will run.")
        print(f"   Saved review state: {path.relative_to(base_dir)}")
        print("   Inspect the review status and spending before retrying. Resume with:")
        print(f"   python run_pipeline_by_date.py --date {date_str} --resume-story-review "
              "--story-review-budget-usd APPROVED_REMAINING_ALLOWANCE")
    return code


def run_baseline_post_final(
    *,
    base_dir: Path,
    date_str: str,
    local_final: Path,
    local_final_ready: bool,
    dry_run: bool,
    story_review_budget: str,
) -> int:
    """Run the original final -> expansion -> summary path."""
    if not local_final_ready:
        code = run_story_review(
            base_dir=base_dir,
            date_str=date_str,
            global_mode=False,
            budget=story_review_budget,
            dry_run=dry_run,
        )
        if code:
            return code

    for script in BASELINE_POST_FINAL_STAGES:
        code = run_required(
            base_dir=base_dir,
            script=script,
            date_str=date_str,
            dry_run=dry_run,
        )
        if code:
            return code
    return 0


def run() -> int:
    args = parse_args()

    try:
        date_obj = datetime.strptime(args.date, "%Y-%m-%d")
        date_str = date_obj.strftime("%Y-%m-%d")
    except Exception:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        return 1

    if args.gdelt_discovery_cached_only and not args.with_gdelt_discovery:
        print(
            "❌ --gdelt-discovery-cached-only cannot be combined with "
            "--no-gdelt-discovery"
        )
        return 1

    base_dir = Path(__file__).resolve().parent
    state_path = review_state_path(base_dir, date_str)
    resume_global = False
    try:
        saved = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        if args.resume_story_review:
            if not isinstance(saved, dict) or saved.get("version") != 1 or saved.get("status") != "pending":
                raise ValueError("No stopped story review to resume for this edition.")
            resume_global = saved["context"]["global_mode"]
            if not isinstance(resume_global, bool) or saved["context"] != review_context(base_dir, date_str, resume_global):
                raise ValueError("Saved review inputs, code or settings changed; inspect before starting another review.")
            if resume_global and not args.with_gdelt_discovery:
                raise ValueError("Cannot resume a global review as a baseline review.")
        elif saved is not None and (not isinstance(saved, dict) or saved.get("status") != "complete"):
            raise ValueError("An unfinished review exists. Use --resume-story-review with an approved remaining allowance.")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"❌ Cannot start story review: {exc}")
        return 1
    print(f"🚀 Running full pipeline for {date_str}...")
    print(f"🔎 Story review allowance: ${args.story_review_budget_usd} for this attempt only.")
    if args.resume_story_review:
        print("↪ Resuming saved story review; fetching, discovery and clustering are skipped.")
    if args.with_gdelt_discovery:
        print("🌍 End-to-end GDELT global path: ENABLED")
    else:
        print("🌍 End-to-end GDELT global path: disabled")
    print(
        "🧭 Next Signal: source-bound checkpoint review; "
        f"edition limits ${args.next_signal_budget_usd} / {args.next_signal_max_calls} requests "
        "(other pipeline stages billed separately)."
    )

    for script in ([] if args.resume_story_review else PRE_MERGE_STAGES):
        code = run_required(
            base_dir=base_dir,
            script=script,
            date_str=date_str,
            dry_run=args.dry_run,
        )
        if code:
            return code

    merge_extra: list[str] = []
    discovery_used = resume_global

    candidate_json = base_dir / f"gdelt_discovery_candidates_{date_str}.json"
    candidate_csv = base_dir / f"gdelt_discovery_candidates_{date_str}.csv"
    augmented_json = base_dir / f"clustered_articles_with_gdelt_{date_str}.json"
    manifest_json = base_dir / f"gdelt_discovery_injection_{date_str}.json"

    if args.with_gdelt_discovery and not args.resume_story_review:
        if not args.dry_run:
            # Never permit stale optional discovery data to enter a fresh run.
            for stale in (
                candidate_json,
                candidate_csv,
                augmented_json,
                manifest_json,
            ):
                stale.unlink(missing_ok=True)

        audit_args = [
            "--pipeline-mode",
            "--upstream-file",
            f"clustered_articles_{date_str}.json",
            "--normalized-file",
            f"articles_raw_normalized_{date_str}.json",
            "--output-json",
            candidate_json.name,
            "--output-csv",
            candidate_csv.name,
        ]
        if args.gdelt_discovery_cached_only:
            audit_args.append("--cached-only")

        audit_ok = run_optional(
            base_dir=base_dir,
            script="audit_gdelt_global_discovery.py",
            date_str=date_str,
            extra_args=audit_args,
            dry_run=args.dry_run,
        )

        inject_ok = False
        if audit_ok:
            inject_ok = run_optional(
                base_dir=base_dir,
                script="inject_gdelt_discovery_candidates.py",
                date_str=date_str,
                extra_args=[
                    "--input-file",
                    f"clustered_articles_{date_str}.json",
                    "--audit-file",
                    candidate_json.name,
                    "--output-file",
                    augmented_json.name,
                    "--manifest-file",
                    manifest_json.name,
                ],
                dry_run=args.dry_run,
            )

        if args.dry_run:
            print(
                "   DRY RUN: merge would use the augmented input only if "
                "both optional discovery stages succeeded and the manifest "
                "validated."
            )
        elif inject_ok and manifest_is_usable(manifest_json, augmented_json):
            merge_extra = ["--input-file", augmented_json.name]
            discovery_used = True
            print(
                f"✅ GDELT candidate feed validated; merge input set to "
                f"{augmented_json.name}"
            )
        else:
            print(
                "↪ GDELT candidate feed was not activated; merge will use "
                f"clustered_articles_{date_str}.json"
            )

    if not args.resume_story_review:
        code = run_required(
            base_dir=base_dir,
            script="merge_similar_clusters.py",
            date_str=date_str,
            extra_args=merge_extra,
            dry_run=args.dry_run,
        )
        if code:
            return code

    for script in ([] if args.resume_story_review else POST_MERGE_PRE_FINAL_STAGES):
        code = run_required(
            base_dir=base_dir,
            script=script,
            date_str=date_str,
            dry_run=args.dry_run,
        )
        if code:
            return code

    local_final = base_dir / f"grouped_articles_final_{date_str}.json"
    global_pipeline_used = False
    local_final_ready = False

    if discovery_used or (args.dry_run and args.with_gdelt_discovery and not args.resume_story_review):
        global_ranking = base_dir / f"gdelt_global_ranking_{date_str}.json"
        global_final = base_dir / f"grouped_articles_final_global_{date_str}.json"
        global_enriched = (
            base_dir / f"grouped_articles_final_global_enriched_{date_str}.json"
        )
        global_expanded = (
            base_dir / f"grouped_articles_final_global_expanded_{date_str}.json"
        )
        global_receipts = (
            base_dir / f"grouped_articles_final_global_receipts_{date_str}.json"
        )
        production_summary = base_dir / f"topic_summaries_{date_str}.json"

        print("🌐 Running post-purity global ranking and receipt path...")
        code = run_story_review(
            base_dir=base_dir,
            date_str=date_str,
            global_mode=True,
            budget=args.story_review_budget_usd,
            extra_args=[
                "--output-file",
                local_final.name,
                "--gdelt-ranking-shadow",
                "--gdelt-ranking-shadow-file",
                global_ranking.name,
                "--gdelt-audit-file",
                candidate_json.name,
                "--gdelt-global-output-file",
                global_final.name,
            ],
            dry_run=args.dry_run,
        )
        if code:
            return code
        local_final_ready = True
        global_ok = True
        if not args.dry_run:
            # Keep the existing edition and receipts intact until review succeeds.
            for stale in (global_enriched, global_expanded, global_receipts):
                stale.unlink(missing_ok=True)

        if global_ok:
            global_ok = run_optional(
                base_dir=base_dir,
                script="enrich_gdelt_discovery_articles.py",
                date_str=date_str,
                extra_args=[
                    "--input-file",
                    global_final.name,
                    "--output-file",
                    global_enriched.name,
                ],
                dry_run=args.dry_run,
            )
            if global_ok and not args.dry_run:
                global_ok = outputs_exist(global_enriched)

        if global_ok:
            global_ok = run_optional(
                base_dir=base_dir,
                script="expand_cluster_coverage_gdelt.py",
                date_str=date_str,
                extra_args=[
                    "--input-file",
                    global_enriched.name,
                    "--output-file",
                    global_expanded.name,
                ],
                dry_run=args.dry_run,
            )
            if global_ok and not args.dry_run:
                global_ok = outputs_exist(global_expanded)

        if global_ok:
            global_ok = run_optional(
                base_dir=base_dir,
                script="attach_gdelt_global_receipts.py",
                date_str=date_str,
                extra_args=[
                    "--input-file",
                    global_expanded.name,
                    "--candidate-file",
                    candidate_json.name,
                    "--ranking-file",
                    global_ranking.name,
                    "--output-file",
                    global_receipts.name,
                ],
                dry_run=args.dry_run,
            )
            if global_ok and not args.dry_run:
                global_ok = outputs_exist(global_receipts)

        if global_ok:
            # A summary/provider failure is not a discovery failure. Stop here
            # rather than repeat paid summarization through the baseline path.
            code = run_required(
                base_dir=base_dir,
                script="summarize_grouped_topics.py",
                date_str=date_str,
                extra_args=[
                    "--input-file",
                    global_receipts.name,
                    "--output-file",
                    production_summary.name,
                ],
                dry_run=args.dry_run,
            )
            if code:
                return code
            if not args.dry_run and not outputs_exist(production_summary):
                print("❌ Summarizer did not produce an edition; stopping without a retry.")
                return 1

        if global_ok:
            global_pipeline_used = True
            print(
                f"✅ Global path complete; production summary written to "
                f"{production_summary.name}"
            )
        else:
            print("⚠️ Global path incomplete; activating baseline fallback.")

    if not global_pipeline_used:
        code = run_baseline_post_final(
            base_dir=base_dir,
            date_str=date_str,
            local_final=local_final,
            local_final_ready=local_final_ready,
            dry_run=args.dry_run,
            story_review_budget=args.story_review_budget_usd,
        )
        if code:
            return code

    # Use the same evidence input as the summarizer on the path that succeeded.
    # Do not let leftover global artifacts leak into a baseline fallback.
    if global_pipeline_used:
        signal_clusters = global_receipts
    else:
        expanded = base_dir / f"grouped_articles_final_expanded_{date_str}.json"
        signal_clusters = (
            expanded if expanded.exists() and (
                not local_final.exists()
                or expanded.stat().st_mtime >= local_final.stat().st_mtime
            ) else local_final
        )
    code = run_required(
        base_dir=base_dir,
        script="generate_next_signals.py",
        date_str=date_str,
        extra_args=[
            "--summary-file", f"topic_summaries_{date_str}.json",
            "--cluster-file", signal_clusters.name,
            "--checkpoint-review",
            "--checkpoint-live",
            "--checkpoint-budget-usd", args.next_signal_budget_usd,
            "--checkpoint-max-calls", str(args.next_signal_max_calls),
            "--checkpoint-max-output-tokens", "2400",
            "--checkpoint-cache-file", f"outputs/checkpoint_receipts_{date_str}.json",
            "--checkpoint-ledger-file", f"outputs/checkpoint_usage_{date_str}.json",
            "--no-external",
        ],
        dry_run=args.dry_run,
    )
    if code:
        return code

    if args.dry_run:
        print(f"✅ Dry run complete for {date_str}; no stages were executed.")
    else:
        print(f"✅ Finished pipeline for {date_str}")
        if args.with_gdelt_discovery:
            if global_pipeline_used:
                result = "GLOBAL PATH ACTIVE"
            elif discovery_used:
                result = "BASELINE FALLBACK AFTER GDELT DISCOVERY"
            else:
                result = "BASELINE FALLBACK"
            print(f"🌍 GDELT pipeline result: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
