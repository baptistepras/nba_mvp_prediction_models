import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configuration
PIPELINE_ALIASES = {
    "selected1956": "selectedStats_from1956",
    "selected1980": "selectedStats_from1980",
    "all1956": "allStats_from1956",
    "all1980": "allStats_from1980"
}

# Pipeline groups
PIPELINE_GROUPS = {
    "all": list(PIPELINE_ALIASES.keys()),
    "allall": ["all1956", "all1980"],
    "allyear1980": ["selected1980", "all1980"],
    "allyear1956": ["selected1956", "all1956"],
    "allselected": ["selected1956", "selected1980"]
}

# Limits
MIN_YEAR = 1956
MAX_YEAR = 2025


def create_loso_splits_to_datasets(year_start: int, year_end: int, pipeline_dir: str) -> None:
    """
    Creates leave-one-season-out splits from processed pipeline data.
    For each year, creates a train/ and test/ folder:
    - train/ contains X, y_top1, y_top10 in compressed .npz format
    - test/ contains X, y_top1, y_top10 in compressed .npz format
    - Name.csv kept as CSV for debug

    Parameters:
        year_start (int): First year to include.
        year_end (int): Last year to include.
        pipeline_dir (str): Path to the processed pipeline directory (e.g. "allStats_from1956").

    Returns:
        None
    """
    print()
    print(f"[INFO] Creating leave-one-season-out splits for pipeline '{pipeline_dir}'")
    files_to_process = ["Data.csv", "Y_top1.csv", "Y_top10.csv", "Name.csv"]
    output_dir = os.path.join(datasets_dir, pipeline_dir)
    data_dir = os.path.join(processed_dir, pipeline_dir)

    os.makedirs(output_dir, exist_ok=True)

    for year in tqdm(range(year_start, year_end + 1), desc="Building LOSO splits", file=sys.stdout):
        try:
            year_str = str(year)
            year_output_dir = os.path.join(output_dir, year_str)
            train_dir = os.path.join(year_output_dir, "train")
            test_dir = os.path.join(year_output_dir, "test")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Initialize train content
            X_train_list = []
            y_top1_train_list = []
            y_top10_train_list = []
            names_train = []

            # Process all years for this fold
            for other_year in range(year_start, year_end + 1):
                other_year_str = str(other_year)
                year_folder = os.path.join(data_dir, other_year_str)

                # Check files exist
                file_paths = {file: os.path.join(year_folder, file) for file in files_to_process}
                if not all(os.path.isfile(path) for path in file_paths.values()):
                    print(f"[WARN] Missing files for year {other_year}, skipping")
                    continue

                # Load data
                df_X = pd.read_csv(file_paths["Data.csv"]).astype(np.float32)
                X = df_X.values
                y_top1 = pd.read_csv(file_paths["Y_top1.csv"]).values.squeeze().astype(np.int64)
                y_top10 = pd.read_csv(file_paths["Y_top10.csv"]).values.squeeze().astype(np.int64)
                df_names = pd.read_csv(file_paths["Name.csv"])

                if other_year == year:
                    # Save test split
                    np.savez_compressed(os.path.join(test_dir, "test.npz"),
                                        X=X, y_top1=y_top1, y_top10=y_top10)
                    df_names.to_csv(os.path.join(test_dir, "Name.csv"), index=False)
                else:
                    # Accumulate for train split
                    X_train_list.append(X)
                    y_top1_train_list.append(y_top1)
                    y_top10_train_list.append(y_top10)
                    names_train.append(df_names)

            # Save train split
            if X_train_list:
                X_train = np.concatenate(X_train_list, axis=0)
                y_top1_train = np.concatenate(y_top1_train_list, axis=0)
                y_top10_train = np.concatenate(y_top10_train_list, axis=0)
                df_names_train = pd.concat(names_train, axis=0, ignore_index=True)

                np.savez_compressed(os.path.join(train_dir, "train.npz"),
                                    X=X_train, y_top1=y_top1_train, y_top10=y_top10_train)
                df_names_train.to_csv(os.path.join(train_dir, "Name.csv"), index=False)

        except Exception as e:
            print(f"[WARN] Failed for year {year}: {e}, skipping")

    print(f"[DONE] Finished creating LOSO splits for pipeline '{pipeline_dir}'")


def check_loso_split_integrity(year_start: int, year_end: int, pipeline_dir: str) -> None:
    """
    Verifies integrity of LOSO splits:
    - X, y_top1, y_top10, Name must have matching number of rows
    - For both train/ and test/ of each year

    Parameters:
        year_start (int): First year to check
        year_end (int): Last year to check
        pipeline_dir (str): Name of the pipeline (folder inside datasets/)

    Returns:
        None
    """
    print()
    print(f"[INFO] Checking LOSO splits integrity for pipeline '{pipeline_dir}'")
    split_dir = os.path.join(datasets_dir, pipeline_dir)

    has_error = False

    for year in tqdm(range(year_start, year_end + 1), desc="Checking splits", file=sys.stdout):
        year_str = str(year)
        year_folder = os.path.join(split_dir, year_str)
        train_folder = os.path.join(year_folder, "train")
        test_folder = os.path.join(year_folder, "test")

        for split_name, split_folder in [("train", train_folder), ("test", test_folder)]:
            npz_path = os.path.join(split_folder, f"{split_name}.npz")
            name_path = os.path.join(split_folder, "Name.csv")

            if not (os.path.isfile(npz_path) and os.path.isfile(name_path)):
                print(f"[WARN] Missing files for year {year}, split '{split_name}', skipping")
                continue

            # Load data
            data = np.load(npz_path)
            X = data["X"]
            y_top1 = data["y_top1"]
            y_top10 = data["y_top10"]
            df_name = pd.read_csv(name_path)

            # Check consistency
            n_samples = X.shape[0]
            n_y_top1 = y_top1.shape[0]
            n_y_top10 = y_top10.shape[0]
            n_names = df_name.shape[0]

            if not (n_samples == n_y_top1 == n_y_top10 == n_names):
                print(f"[ERROR] Inconsistent sizes for year {year} split '{split_name}':")
                print(f"    X: {n_samples}, y_top1: {n_y_top1}, y_top10: {n_y_top10}, Name: {n_names}")
                has_error = True

    if not has_error:
        print(f"[DONE] All LOSO splits are consistent for pipeline '{pipeline_dir}'")


if __name__ == "__main__":
    # Example usage from root:
    # python scripts_data_process/build_splits.py --pipeline all
    # python scripts_data_process/build_splits.py --pipeline allselected --start 2000 --end 2020
    # python scripts_data_process/build_splits.py --pipeline selected1980 all1956 --start 1990

    # Set constants
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_dir = os.path.join(base_dir, "processed_data")
    datasets_dir = os.path.join(base_dir, "datasets")

    # Argparser
    parser = argparse.ArgumentParser(description="Create LOSO splits for one or more pipelines.")
    parser.add_argument("--pipelines", nargs="+", default=["all"],
                        help="Pipelines to run (ex: all1956, selected1956, selected1980, all1980, allselected, allall, allyear1980, allyear1956, all)")
    parser.add_argument("--start", type=int, default=MIN_YEAR, help=f"Start year (default {MIN_YEAR})")
    parser.add_argument("--end", type=int, default=MAX_YEAR, help=f"End year (default {MAX_YEAR})")

    args = parser.parse_args()

    # Resolve pipelines to run
    pipelines_to_run = []
    for p in args.pipelines:
        if p in PIPELINE_GROUPS:
            pipelines_to_run.extend(PIPELINE_GROUPS[p])
        elif p in PIPELINE_ALIASES:
            pipelines_to_run.append(p)
        else:
            print(f"[ERROR] Unknown pipeline or group: '{p}', skipping.")

    # Deduplicate
    pipelines_to_run = list(dict.fromkeys(pipelines_to_run))

    # Process each pipeline
    for pipeline_key in pipelines_to_run:
        print()
        pipeline_name = PIPELINE_ALIASES[pipeline_key]

        # Determine correct year range for this pipeline
        pipeline_min_year = 1956 if "1956" in pipeline_name else 1980

        year_start = args.start
        year_end = args.end

        if year_start < pipeline_min_year:
            print(f"[WARN] {pipeline_key}: start {year_start} < {pipeline_min_year}, forcing to {pipeline_min_year}.")
            year_start = pipeline_min_year
        if year_end > MAX_YEAR:
            print(f"[WARN] {pipeline_key}: end {year_end} > {MAX_YEAR}, forcing to {MAX_YEAR}.")
            year_end = MAX_YEAR
        if year_start > year_end:
            print(f"[ERROR] {pipeline_key}: start {year_start} > end {year_end}, skipping.")
            continue

        # Run
        print(f"[INFO] Creating LOSO splits for pipeline '{pipeline_name}' from {year_start} to {year_end}...")
        create_loso_splits_to_datasets(year_start, year_end, pipeline_name)

        print(f"[INFO] Checking LOSO splits for pipeline '{pipeline_name}' from {year_start} to {year_end}...")
        check_loso_split_integrity(year_start, year_end, pipeline_name)

    print("\n[INFO] All selected pipelines finished.")