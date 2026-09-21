# run_pipeline_by_date.py — full pipeline runner with GDELT global coverage enabled by default

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
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
    return parser.parse_args()


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
        return 0

    result = subprocess.run(command, cwd=str(base_dir))
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


def run_baseline_post_final(
    *,
    base_dir: Path,
    date_str: str,
    local_final: Path,
    local_final_ready: bool,
    dry_run: bool,
) -> int:
    """Run the original final -> expansion -> summary path."""
    if not local_final_ready:
        code = run_required(
            base_dir=base_dir,
            script="final_cohesion_check.py",
            date_str=date_str,
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
    print(f"🚀 Running full pipeline for {date_str}...")
    if args.with_gdelt_discovery:
        print("🌍 End-to-end GDELT global path: ENABLED")
    else:
        print("🌍 End-to-end GDELT global path: disabled")

    for script in PRE_MERGE_STAGES:
        code = run_required(
            base_dir=base_dir,
            script=script,
            date_str=date_str,
            dry_run=args.dry_run,
        )
        if code:
            return code

    merge_extra: list[str] = []
    discovery_used = False

    candidate_json = base_dir / f"gdelt_discovery_candidates_{date_str}.json"
    candidate_csv = base_dir / f"gdelt_discovery_candidates_{date_str}.csv"
    augmented_json = base_dir / f"clustered_articles_with_gdelt_{date_str}.json"
    manifest_json = base_dir / f"gdelt_discovery_injection_{date_str}.json"

    if args.with_gdelt_discovery:
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

    code = run_required(
        base_dir=base_dir,
        script="merge_similar_clusters.py",
        date_str=date_str,
        extra_args=merge_extra,
        dry_run=args.dry_run,
    )
    if code:
        return code

    for script in POST_MERGE_PRE_FINAL_STAGES:
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

    if discovery_used or (args.dry_run and args.with_gdelt_discovery):
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

        if not args.dry_run:
            # Remove only optional/intermediate global artifacts. Do not remove
            # the existing production summary until a replacement is written.
            for stale in (
                global_ranking,
                global_final,
                global_enriched,
                global_expanded,
                global_receipts,
            ):
                stale.unlink(missing_ok=True)

        print("🌐 Running post-purity global ranking and receipt path...")
        global_ok = run_optional(
            base_dir=base_dir,
            script="final_cohesion_check.py",
            date_str=date_str,
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

        if global_ok:
            local_final_ready = args.dry_run or outputs_exist(local_final)
            global_ok = args.dry_run or outputs_exist(global_ranking, global_final)
            if not global_ok:
                print("⚠️ Global ranking stage did not produce all expected outputs.")

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
            global_ok = run_optional(
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
            if global_ok and not args.dry_run:
                global_ok = outputs_exist(production_summary)

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
