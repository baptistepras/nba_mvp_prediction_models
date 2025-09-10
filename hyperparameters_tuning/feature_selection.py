import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from train_models import load_dataset, get_default_hyperparams, evaluate_model
from train_models import PIPELINE_ALIASES, PIPELINE_GROUPS, MODEL_CLASSES

# Constants

MIN_YEAR = 1956
MAX_YEAR = 2025


def compute_recall_at_1_avg(model_class, fixed_params, dataset_dir, start_year, end_year, feature_indices):
    """
    Trains and evaluates the model on each year using only selected features.
    Returns the average Recall@1 across all years.
    """
    recalls = []

    for year in range(start_year, end_year + 1):
        data = load_dataset(dataset_dir, year)
        X_train = data["X_train"][:, feature_indices]
        y_train = data["y_top1_train"]
        X_test = data["X_test"][:, feature_indices]
        y_test = data["y_top1_test"]
        y10_test = data["y_top10_test"]
        player_names = data["Name_test"]

        model = model_class(**fixed_params)
        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test, y10_test, player_names, top_ks=[1])
        recalls.append(results["top_1_hit"])

    return np.mean(recalls)


def custom_rfe_recall1(model_class, dataset_dir, pipeline_name, fixed_params,
                       output_dir, year_start, year_end, min_k, max_k):
    """
    Performs RFE using Recall@1 as custom scoring function.
    """
    print(f"[INFO] Running custom RFE (Recall@1) on dataset: {dataset_dir}")

    data = load_dataset(dataset_dir, year_start)
    num_features = data["X_train"].shape[1]
    best_score = -1
    best_mask = None
    scores = []

    for k in range(min_k, max_k + 1):
        print(f"[INFO] Evaluating {k} features...")
        combs = list(combinations(range(num_features), k))
        best_score_k = -1
        best_comb_k = None

        for comb in tqdm(combs, desc=f" Testing {k}-feature subsets", leave=False, file=sys.stdout):
            score = compute_recall_at_1_avg(model_class, fixed_params, dataset_dir,
                                            year_start, year_end, list(comb))
            if score > best_score_k:
                best_score_k = score
                best_comb_k = comb

        scores.append(best_score_k)

        if best_score_k > best_score:
            best_score = best_score_k
            best_mask = np.zeros(num_features, dtype=bool)
            best_mask[list(best_comb_k)] = True

    print(f"[DONE] Optimal number of features: {int(best_mask.sum())}")
    print(f"[DONE] Best Recall@1 score: {best_score:.4f}")

    # Load feature names from processed_data/<pipeline>/<year>/Data.csv
    processed_dir = os.path.join(BASE_DIR, "processed_data", pipeline_name, str(year_start))
    csv_path = os.path.join(processed_dir, "Data.csv")
    with open(csv_path, "r") as f:
        header = f.readline().strip()
    feature_names = np.array(header.split(","))

    selected_indices = np.where(best_mask)[0]
    selected_features = feature_names[selected_indices]

    print(f"[INFO] Selected feature indices: {selected_indices.tolist()}")
    print(f"[INFO] Selected feature names  : {selected_features.tolist()}")

    # Save selected mask
    np.save(os.path.join(output_dir, "rfe_support_mask.npy"), best_mask)
    np.save(os.path.join(output_dir, "rfe_selected_indices.npy"), selected_indices)
    with open(os.path.join(output_dir, "rfe_selected_features.txt"), "w") as f:
        for name in selected_features:
            f.write(name + "\n")

    # Plot
    plt.figure()
    plt.title("Custom RFE - Recall@1 vs Number of Selected Features")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross-validated Recall@1")
    k_values = list(range(min_k, max_k + 1))
    plt.plot(k_values, scores)
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "rfe_recall_plot.png")
    plt.savefig(plot_path)
    print(f"[DONE] Saved RFE recall plot to {plot_path}")


if __name__ == "__main__":
    # Example usage:
    # python hyperparameters_tuning/feature_selection.py --pipeline selected1980 --model logreg --min 10 --max 14

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", nargs="+", default=["all"],
                        help="Pipelines to run (e.g. selected1980, all1956, allselected, etc.)")
    parser.add_argument("--model", type=str, default="logreg",
                        help="Model to use (e.g. logreg)")
    parser.add_argument("--min", type=int, required=True,
                        help="Min number of features")
    parser.add_argument("--max", type=int, required=True,
                        help="Max number of features")
    args = parser.parse_args()

    model_class = MODEL_CLASSES[args.model]
    model_name = model_class.__name__

    # Resolve pipelines
    pipelines_to_run = []
    for p in args.pipeline:
        if p in PIPELINE_GROUPS:
            pipelines_to_run.extend(PIPELINE_GROUPS[p])
        elif p in PIPELINE_ALIASES:
            pipelines_to_run.append(p)
        else:
            print(f"[WARN] Unknown pipeline or group '{p}', skipping.")
    pipelines_to_run = list(dict.fromkeys(pipelines_to_run))

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    datasets_dir = os.path.join(base_dir, "datasets")
    output_base_dir = os.path.join(base_dir, "hyperparameters_tuning", "rfe_results")

    for pipeline_key in pipelines_to_run:
        pipeline_name = PIPELINE_ALIASES[pipeline_key]
        print(f"[INFO] Calculating RFE for pipeline: {pipeline_name}")
        dataset_dir = os.path.join(datasets_dir, pipeline_name)

        year_start = 1956 if "1956" in pipeline_name else 1980
        year_end = MAX_YEAR

        output_dir = os.path.join(output_base_dir, f"{model_name}_{pipeline_name}")
        os.makedirs(output_dir, exist_ok=True)

        fixed_params = get_default_hyperparams(model_class, pipeline_name)

        custom_rfe_recall1(model_class, dataset_dir, pipeline_name,
                           fixed_params, output_dir, year_start, year_end,
                           args.min, args.max)

    print()
    print("[DONE] All RFE runs completed.")