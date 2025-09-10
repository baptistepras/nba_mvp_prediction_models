import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from train_models import load_dataset, get_default_hyperparams, evaluate_model
from train_models import PIPELINE_ALIASES, PIPELINE_GROUPS, MODEL_CLASSES
from hyperparameters_tuning.feature_selection import compute_recall_at_1_avg

MIN_YEAR = 1956
MAX_YEAR = 2025


def greedy_forward_selection(model_class, dataset_dir, pipeline_name, fixed_params,
                              output_dir, year_start, year_end, patience=3):
    """
    Greedy forward selection using Recall@1 as the scoring metric with early stopping.
    """
    print(f"[INFO] Running greedy forward selection (Recall@1) on dataset: {dataset_dir}")

    data = load_dataset(dataset_dir, year_start)
    num_features = data["X_train"].shape[1]
    available_features = list(range(num_features))
    selected_features = []
    best_score = -1
    no_improve_count = 0
    recall_scores = []

    for i in range(1, num_features + 1):
        current_best_score = -1
        feature_to_add = None

        for feat in tqdm(available_features, desc=f"Testing additions (current={len(selected_features)})", leave=False):
            candidate = selected_features + [feat]
            score = compute_recall_at_1_avg(model_class, fixed_params, dataset_dir, year_start, year_end, candidate)
            if score > current_best_score or (score == current_best_score and feat < (feature_to_add or float('inf'))):
                current_best_score = score
                feature_to_add = feat

        if current_best_score > best_score:
            best_score = current_best_score
            best_combination = selected_features + [feature_to_add]
            no_improve_count = 0
        else:
            no_improve_count += 1

        recall_scores.append(current_best_score)
        selected_features.append(feature_to_add)
        available_features.remove(feature_to_add)

        print(f"[INFO] Added feature {feature_to_add} | Total: {len(selected_features)} | Recall@1: {current_best_score:.3f}")

        if no_improve_count >= patience:
            print(f"[EARLY STOP] No improvement for {patience} consecutive steps. Stopping.")
            break

    # Load feature names from processed_data/<pipeline>/<year>/Data.csv
    processed_dir = os.path.join(BASE_DIR, "processed_data", pipeline_name, str(year_start))
    csv_path = os.path.join(processed_dir, "Data.csv")
    with open(csv_path, "r") as f:
        header = f.readline().strip()
    feature_names = np.array(header.split(","))

    best_mask = np.zeros(num_features, dtype=bool)
    best_mask[best_combination] = True

    np.save(os.path.join(output_dir, "best_support_mask.npy"), best_mask)
    np.savetxt(os.path.join(output_dir, "best_k.txt"), [len(best_combination)], fmt='%d')
    np.savetxt(os.path.join(output_dir, "best_score.txt"), [best_score], fmt='%.4f')

    with open(os.path.join(output_dir, "best_features.txt"), "w") as f:
        for idx in best_combination:
            f.write(f"{feature_names[idx]}\n")

    print(f"[DONE] Optimal number of features: {len(best_combination)}")
    print(f"[DONE] Best combination: {[feature_names[idx] for idx in best_combination]}")
    print(f"[DONE] Best Recall@1 score: {best_score:.4f}")

    plt.figure()
    plt.title("Greedy Forward - Recall@1 vs Number of Features")
    plt.xlabel("Number of features selected")
    plt.ylabel("Recall@1")
    plt.plot(range(1, len(recall_scores) + 1), recall_scores)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "greedy_forward_recall_plot.png"))


if __name__ == "__main__":
    # Example usage:
    # python hyperparameters_tuning/greedy_forward_feature_selection.py --pipeline selected1980 --model logreg
    # python hyperparameters_tuning/greedy_forward_feature_selection.py --pipeline all --model logreg --patience 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", nargs="+", default=["all"], help="Pipelines to run")
    parser.add_argument("--model", type=str, default="logreg", help="Model to use")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default=3)")
    args = parser.parse_args()

    model_class = MODEL_CLASSES[args.model]
    model_name = model_class.__name__

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
    output_base_dir = os.path.join(base_dir, "hyperparameters_tuning", "greedy_rfe_results")

    for pipeline_key in pipelines_to_run:
        pipeline_name = PIPELINE_ALIASES[pipeline_key]
        print(f"[INFO] Running greedy forward for pipeline: {pipeline_name}")
        dataset_dir = os.path.join(datasets_dir, pipeline_name)

        year_start = 1956 if "1956" in pipeline_name else 1980
        year_end = MAX_YEAR

        output_dir = os.path.join(output_base_dir, f"forward_{model_name}_{pipeline_name}")
        os.makedirs(output_dir, exist_ok=True)

        fixed_params = get_default_hyperparams(model_class, pipeline_name)

        greedy_forward_selection(model_class, dataset_dir, pipeline_name, fixed_params,
                                 output_dir, year_start, year_end, patience=args.patience)

    print("\n[DONE] All greedy forward runs completed.")