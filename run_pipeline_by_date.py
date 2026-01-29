# run_pipeline_by_date.py — full pipeline runner with --date support

import os
import subprocess
from datetime import datetime
import argparse

STAGE_SCRIPTS = [
    "fetch_articles_by_date.py",
    "normalize_and_dedupe_articles.py",
    "bias_labeler.py",
    "cluster_articles_by_embedding.py",
    "merge_similar_clusters.py",
    "filter_outlier_articles.py",
    "report_cluster_cohesion.py",
    "final_cohesion_check.py",
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
        return

    print(f"🚀 Running full pipeline for {date_str}...")

    for script in STAGE_SCRIPTS:
        print(f"➡️ Running: {script}")
        subprocess.run(["python", script, "--date", date_str])

    print(f"✅ Finished pipeline for {date_str}")

if __name__ == "__main__":
    run()
