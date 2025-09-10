import os
import subprocess
import argparse

# Configuration
PIPELINE_SCRIPTS = {
    "selected1956": "build_selectedStats_from1956.py",
    "selected1980": "build_selectedStats_from1980.py",
    "all1956": "build_allStats_from1956.py",
    "all1980": "build_allStats_from1980.py"
}

# Pipeline groups
PIPELINE_GROUPS = {
    "all": list(PIPELINE_SCRIPTS.keys()),
    "allall": ["all1956", "all1980"],
    "allyear1980": ["selected1980", "all1980"],
    "allyear1956": ["selected1956", "all1956"],
    "allselected": ["selected1956", "selected1980"]
}

# Limits
MIN_YEAR = 1956
MAX_YEAR = 2025

if __name__ == "__main__":
    # Example usage from root:
    # python scripts_data_process/build_all_pipelines.py --pipelines all
    # python scripts_data_process/build_all_pipelines.py --pipelines allselected --start 1985 --end 2020
    # python scripts_data_process/build_all_pipelines.py --pipelines selected1956 all1980
    # python scripts_data_process/build_all_pipelines.py --pipelines all1956 --start 2000
    
    parser = argparse.ArgumentParser(description="Run one or more build pipelines.")
    parser.add_argument("--pipelines", nargs="+", default=["all"],
                        help="Pipelines to run (ex: all1956, selected1956, selected1980, all1980, allselected, allall, allyear1980, allyear1956, all)")
    parser.add_argument("--start", type=int, default=MIN_YEAR, help="Start year (default {MIN_YEAR})")
    parser.add_argument("--end", type=int, default=MAX_YEAR, help="End year (default {MAX_YEAR})")

    args = parser.parse_args()

    # Resolve pipelines to run
    pipelines_to_run = []
    for p in args.pipelines:
        if p in PIPELINE_GROUPS:
            pipelines_to_run.extend(PIPELINE_GROUPS[p])
        elif p in PIPELINE_SCRIPTS:
            pipelines_to_run.append(p)
        else:
            print(f"[ERROR] Unknown pipeline or group: '{p}', skipping.")

    # Deduplicate
    pipelines_to_run = list(dict.fromkeys(pipelines_to_run))

    # Run
    for pipeline in pipelines_to_run:
        print()
        script_name = PIPELINE_SCRIPTS[pipeline]

        # Determine correct year range for this pipeline
        pipeline_min_year = 1956 if "1956" in pipeline else 1980

        year_start = args.start
        year_end = args.end

        if year_start < pipeline_min_year:
            print(f"[WARN] {pipeline}: start {year_start} < {pipeline_min_year}, forcing to {pipeline_min_year}.")
            year_start = pipeline_min_year
        if year_end > MAX_YEAR:
            print(f"[WARN] {pipeline}: end {year_end} > {MAX_YEAR}, forcing to {MAX_YEAR}.")
            year_end = MAX_YEAR
        if year_start > year_end:
            print(f"[ERROR] {pipeline}: start {year_start} > end {year_end}, skipping.")
            continue

        # Build command
        cmd = f"python scripts_data_process/{script_name} --start {year_start} --end {year_end}"

        print(f"\n[INFO] Running pipeline: {pipeline} -> {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    print("\n[INFO] All selected pipelines finished.")