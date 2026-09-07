# run_pipeline_by_date.py — full pipeline runner with --date support

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

STAGE_SCRIPTS = [
    "fetch_articles_by_date.py",
    "normalize_and_dedupe_articles.py",
    "bias_labeler.py",
    "generate_market_overview.py",
    "cluster_articles_by_embedding.py",
    "merge_similar_clusters.py",
    "filter_outlier_articles.py",
    "report_cluster_cohesion.py",
    "final_cohesion_check.py",
    "expand_cluster_coverage_gdelt.py",
    "summarize_grouped_topics.py",
]

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    args = parser.parse_args()

    try:
        date_obj = datetime.strptime(args.date, "%Y-%m-%d")
        date_str = date_obj.strftime("%Y-%m-%d")
    except Exception:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        return 1

    base_dir = Path(__file__).resolve().parent
    print(f"🚀 Running full pipeline for {date_str}...")

    for script in STAGE_SCRIPTS:
        script_path = base_dir / script

        if not script_path.exists():
            print(f"❌ Missing pipeline stage: {script}")
            print("🛑 Pipeline stopped.")
            return 1

        print(f"➡️ Running: {script}")
        result = subprocess.run(
            [sys.executable, str(script_path), "--date", date_str],
            cwd=str(base_dir),
        )

        if result.returncode != 0:
            print(f"❌ Stage failed: {script} (exit code {result.returncode})")
            print("🛑 Pipeline stopped. Downstream stages were not run.")
            return result.returncode or 1

    print(f"✅ Finished pipeline for {date_str}")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
