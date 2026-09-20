# run_pipeline_by_date.py — full pipeline runner with optional GDELT discovery

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

POST_MERGE_STAGES = [
    "filter_outlier_articles.py",
    "report_cluster_cohesion.py",
    "final_cohesion_check.py",
    "expand_cluster_coverage_gdelt.py",
    "summarize_grouped_topics.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    parser.add_argument(
        "--with-gdelt-discovery",
        action="store_true",
        help=(
            "Opt in to bounded, fail-open GDELT candidate discovery after "
            "baseline article clustering and before cluster merging."
        ),
    )
    parser.add_argument(
        "--gdelt-discovery-cached-only",
        action="store_true",
        help=(
            "With --with-gdelt-discovery, use cached GSG files only and make "
            "no new GDELT download requests."
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
        print("↪ Continuing with the unchanged NewsAPI/RSS path.")
        return False

    command = [
        sys.executable,
        str(script_path),
        "--date",
        date_str,
        *map(str, extra_args),
    ]
    print(f"➡️ Running optional stage: {script}")
    if dry_run:
        print(f"   DRY RUN: {_display_command(command, base_dir)}")
        return True

    result = subprocess.run(command, cwd=str(base_dir))
    if result.returncode != 0:
        print(
            f"⚠️ Optional stage failed: {script} "
            f"(exit code {result.returncode})"
        )
        print("↪ Continuing with the unchanged NewsAPI/RSS path.")
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
            "❌ --gdelt-discovery-cached-only requires "
            "--with-gdelt-discovery"
        )
        return 1

    base_dir = Path(__file__).resolve().parent
    print(f"🚀 Running full pipeline for {date_str}...")
    if args.with_gdelt_discovery:
        print("🌍 Optional pre-merge GDELT discovery: ENABLED")
    else:
        print("🌍 Optional pre-merge GDELT discovery: disabled")

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

    if args.with_gdelt_discovery:
        candidate_json = base_dir / f"gdelt_discovery_candidates_{date_str}.json"
        candidate_csv = base_dir / f"gdelt_discovery_candidates_{date_str}.csv"
        augmented_json = (
            base_dir / f"clustered_articles_with_gdelt_{date_str}.json"
        )
        manifest_json = (
            base_dir / f"gdelt_discovery_injection_{date_str}.json"
        )

        if not args.dry_run:
            # Never permit stale optional data to enter a fresh run.
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
                "both optional stages succeeded and the manifest validated."
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

    for script in POST_MERGE_STAGES:
        code = run_required(
            base_dir=base_dir,
            script=script,
            date_str=date_str,
            dry_run=args.dry_run,
        )
        if code:
            return code

    if args.dry_run:
        print(f"✅ Dry run complete for {date_str}; no stages were executed.")
    else:
        print(f"✅ Finished pipeline for {date_str}")
        if args.with_gdelt_discovery:
            print(
                "🌍 Pre-merge GDELT discovery result: "
                + ("ACTIVE" if discovery_used else "BASELINE FALLBACK")
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
