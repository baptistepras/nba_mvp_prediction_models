import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from typing import List, Any
verbose = False
vprint = print if verbose else lambda *args, **kwargs: None

# Dir
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
base_dir = BASE_DIR

# Pipelines
PIPELINE_ALIASES = {
    "selected1956": "selectedStats_from1956",
    "selected1980": "selectedStats_from1980",
    "all1956": "allStats_from1956",
    "all1980": "allStats_from1980"
}

PIPELINE_GROUPS = {
    "all": list(PIPELINE_ALIASES.keys()),
    "allall": ["all1956", "all1980"],
    "allyear1980": ["selected1980", "all1980"],
    "allyear1956": ["selected1956", "all1956"],
    "allselected": ["selected1956", "selected1980"]
}

# Predefined selected feature indices (based on RFE/greedy results)
SELECTED_FEATURES = {
    "logreg": {
        "selectedStats_from1980": ["Team Overall", "AST", "PTS"], 
        "selectedStats_from1956": ["Team Overall", "FT%", "TRB", "AST", "PTS", "POS_SF"],
        "allStats_from1980": ["Team Overall", "G", "MP", "FG", "FG%", "2P", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", 
                              "STL", "BLK", "TOV", "PF", "PTS", "POS_C", "POS_PF", "POS_SF"],
        "allStats_from1956": ["Team Overall", "G", "FG", "FT", "FTA", "TRB", "AST", "PF", "POS_PF", "POS_SF", "POS_SG"]
    },
}

# Models
MODEL_CLASSES = {
    "logreg": LogisticRegression,
    "rf": RandomForestClassifier,
    "xgb": XGBClassifier,
    "gb": GradientBoostingClassifier,        
    "histgb": HistGradientBoostingClassifier,
    "lgbm": LGBMClassifier,
}

# Limits
MIN_YEAR = 1956
MAX_YEAR = 2025


def get_feature_names(pipeline_name: str, year: int) -> List[str]:
    """
    Loads feature names from processed_data CSV header.
    
    Parameters:
        pipeline_name (str): Name of the pipeline (e.g. "selectedStats_from1980").
        year (int): Year to use to locate the file.

    Returns:
        List[str]: List of feature names.
    """
    csv_path = os.path.join(base_dir, "processed_data", pipeline_name, str(year), "Data.csv")
    with open(csv_path, "r") as f:
        header = f.readline().strip()
    return header.split(",")


def load_dataset(dataset_dir: str, year: int) -> dict:
    """
    Loads train and test data for a given year.

    Parameters:
        dataset_dir (str): Path to the dataset folder (e.g. 'datasets/selectedStats_from1980/1980').
        year (str): The year to load.

    Returns:
        X (np.ndarray): Feature matrix.
        y_top1 (np.ndarray): Binary labels (MVP or not).
        y_top10 (np.ndarray): Ranking labels (MVP rank or -1).
        names (List[str]): List of player names (aligned with X and Y).
    """
    year_dir = os.path.join(dataset_dir, str(year))
    data = {}

    for split in ["train", "test"]:
        split_dir = os.path.join(year_dir, split)
        npz_path = os.path.join(split_dir, f"{split}.npz")
        name_path = os.path.join(split_dir, "Name.csv")

        # Load .npz compressed data
        npz_data = np.load(npz_path)
        data[f"X_{split}"] = npz_data["X"]
        data[f"y_top1_{split}"] = npz_data["y_top1"]
        data[f"y_top10_{split}"] = npz_data["y_top10"]

        # Load player names
        data[f"Name_{split}"] = pd.read_csv(name_path)["Name"].tolist()

    return data


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, y_10: np.ndarray,
                   player_names: List[str], top_ks: List[int]=[1, 3, 5, 10]) -> dict:
    """
    Evaluates model and prints top-k accuracy metrics

    Parameters:
        model (Any): Trained model.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Binary labels.
        y_10 (np.ndarray): Top 10 labels.
        player_names (List[str]): List of player names.
        top_ks (List[int]): List of K values for top-K accuracy.

    Returns:
        results (dict): Dictionary with evaluation metrics and predicted ranking.
    """
    # Predict probabilities for class 1 (MVP)
    probs = model.predict_proba(X)[:, 1]

    # Sort players by predicted probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_names = [player_names[i] for i in sorted_indices]
    sorted_true = y[sorted_indices]
    sorted_y10 = y_10[sorted_indices]

    # Metrics
    results = {}

    # Top-k
    for k in top_ks:
        # Recall@k: whether the true MVP is present in the top-k predictions (1 = yes, 0 = no)
        top_k_true = sorted_true[:k]
        hit = 1 in top_k_true
        results[f"top_{k}_hit"] = int(hit)
        vprint(f"[INFO] Top-{k} hit: {hit}")

        pred_top_k_names = sorted_names[:k]
        pred_top_k_real_ranks = sorted_y10[:k]

        # Precision@k vs real top-k: proportion of top-k predicted players who are ranked in the real top-k (according to available top-k annotations)
        pred_top_k_real_ranks = sorted_y10[:k]
        real_topk_mask = (y_10 >= 1) & (y_10 <= k)
        nb_real_topk = real_topk_mask.sum()
        n_in_real_topk = sum(1 for r in pred_top_k_real_ranks if 1 <= r <= k)
        denom_topk = min(k, nb_real_topk) if nb_real_topk > 0 else 1
        precision_at_k_exact = n_in_real_topk / denom_topk
        results[f"precision_at_{k}_topk"] = precision_at_k_exact
        vprint(f"[METRIC] Precision@{k} vs real top-{k}: {precision_at_k_exact:.3f}")

        # Precision@k vs real top-10: proportion of top-k predicted players who are ranked in the real top-10 (according to available top-10 annotations)
        real_top10_mask = (y_10 >= 1) & (y_10 <= 10)
        nb_real_top10 = real_top10_mask.sum()
        n_in_real_top10 = sum(1 for r in pred_top_k_real_ranks if 1 <= r <= 10)
        denom_top10 = min(k, nb_real_top10) if nb_real_top10 > 0 else 1
        precision_at_k_top10 = n_in_real_top10 / denom_top10
        results[f"precision_at_{k}_top10"] = precision_at_k_top10
        vprint(f"[METRIC] Precision@{k} vs real top-10: {precision_at_k_top10:.3f}")

        # Mean absolute rank error@k: distance between predicted rank table and real rank table
        abs_errors = []
        for pred_rank_idx, real_rank in enumerate(pred_top_k_real_ranks, 1):
            if real_rank == -1:
                assumed_rank = 11  # If not in real top-10, treat as 11th
            else:
                assumed_rank = real_rank
            abs_error = abs(pred_rank_idx - assumed_rank)
            abs_errors.append(abs_error)
        mean_abs_error = np.mean(abs_errors)
        results[f"mean_abs_rank_error_at_{k}"] = mean_abs_error
        vprint(f"[METRIC] Mean absolute rank error@{k}: {mean_abs_error:.3f}")
        vprint()

    # True MVP ranking
    if 1 in sorted_true:
        true_index = np.where(sorted_true == 1)[0][0]
        true_rank = true_index + 1
        results[f"true_index"] = true_rank
        vprint(f"[INFO] True MVP ranked at position: {true_rank}")
        vprint()
    else:
        print("[WARN] True MVP not found in test set.")

    # Top 10 table comparing real vs predicted
    vprint()
    vprint("[INFO] Top 10: Real Rank vs Predicted Rank")
    vprint(f"{'Real Rank':>10s} | {'Player':30s} | {'Pred Rank':>10s} | {'Prob':>8s}")

    for pred_rank, (name, prob, true_y10, true_label) in enumerate(zip(sorted_names[:10], probs[sorted_indices][:10], sorted_y10[:10], sorted_true[:10]), 1):
        real_rank_str = f"{true_y10}" if true_y10 != -1 else "-"
        label_str = "TRUE MVP" if true_label == 1 else ""
        vprint(f"{real_rank_str:>10s} | {name:30s} | {pred_rank:10d} | {prob:8.4f} {label_str}")

    # Save predicted ranking in results
    predicted_ranking = []
    for pred_rank, (name, prob) in enumerate(zip(sorted_names, probs[sorted_indices]), 1):
        predicted_ranking.append({
            "rank": pred_rank,
            "player": name,
            "prob": float(prob)
        })
    results["predicted_ranking"] = predicted_ranking

    return results


def train_and_evaluate(dataset_dir: str, year: int, output_model_dir: str, 
                       model_class: Any, model_name: str, fixed_params: dict,
                       selected_feature_names: List[str] = None,
                       pipeline_name: str = None) -> dict:
    """
    Trains a logistic regression model on a given year's split and evaluates it.

    Parameters:
        dataset_dir (str): Path to the dataset folder (e.g. 'datasets/selectedStats_from1980/1980').
        year (str): The year to load.
        model_type (str): Type of model to train ("logreg", "rf", "xgb").
        class_weight (str): Class weight strategy ("balanced" or "none").
        model_class (Type[Any]): Model class
        model_name (str): Model name
        fixed_params (dict): Dict of the default hyperparameters.

    Returns:
        results (dict): Dictionary with evaluation metrics and predicted ranking.
    """
    print()
    print(f"[INFO] Training model for year {year}...")
    data = load_dataset(dataset_dir, year)

    X_train = data["X_train"]
    y_train = data["y_top1_train"]
    X_test = data["X_test"]
    y_test = data["y_top1_test"]
    y10_test = data["y_top10_test"]
    player_names_test = data["Name_test"]

    if selected_feature_names is not None:
        all_feature_names = get_feature_names(pipeline_name, year)
        selected_indices = [i for i, name in enumerate(all_feature_names) if name in selected_feature_names]
        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]

    # Define model
    model = model_class(**fixed_params)

    # Fit
    model.fit(X_train, y_train)

    # Save model
    checkpoint_dir = os.path.join(output_model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"{model_name}_{year}.joblib")
    joblib.dump(model, model_path)
    print(f"[DONE] Model saved to {model_path}")

    # Evaluate
    results = evaluate_model(model, X_test, y_test, y10_test, player_names_test)

    return results


def summarize_global_results(df_results: pd.DataFrame, top_ks: List[int]=[1, 3, 5, 10]) -> dict:
    """
    Summarizes global evaluation metrics over all seasons.
    Displays results per metric (recall@k, precision@k real top-k, precision@k real top-10, mean abs error, true MVP).
    
    Parameters:
        df_results (pd.DataFrame): DataFrame containing results per season.
        top_ks (List[int]): List of top-K values to summarize.
    
    Returns:
        dict: Dictionary of aggregated mean metrics, to be saved or further analyzed.
    """
    print()
    print("[INFO] Aggregating global metrics over all years:\n")
    summary_means = {}

    # Recall@k
    print("[SUMMARY] Recall@k (Top-k hit rate):")
    for k in top_ks:
        col = f"top_{k}_hit"
        if col in df_results.columns:
            hit_rate = df_results[col].mean()
            total_hits = df_results[col].sum()
            summary_means[f"top_{k}_hit_rate"] = hit_rate
            print(f"Top-{k} hit rate: {hit_rate:.3f} ({int(total_hits)}/{len(df_results)})")
        else:
            print(f"[WARN] Column {col} not found in results.")
    print()

    # Precision@k vs real top-k
    print("[SUMMARY] Avg Precision@k vs real top-k:")
    for k in top_ks:
        col = f"precision_at_{k}_topk"
        if col in df_results.columns:
            avg_prec = df_results[col].mean()
            summary_means[f"precision_at_{k}_topk"] = avg_prec
            print(f"Precision@{k} vs real top-{k}: {avg_prec:.3f}")
        else:
            print(f"[WARN] Column {col} not found in results.")
    print()

    # Precision@k vs real top-10
    print("[SUMMARY] Avg Precision@k vs real top-10:")
    for k in top_ks:
        col = f"precision_at_{k}_top10"
        if col in df_results.columns:
            avg_prec = df_results[col].mean()
            summary_means[f"precision_at_{k}_topk"] = avg_prec
            print(f"Precision@{k} vs real top-10: {avg_prec:.3f}")
        else:
            print(f"[WARN] Column {col} not found in results.")
    print()

    # Mean Abs Rank Error@k
    print("[SUMMARY] Avg Mean Absolute Rank Error@k:")
    for k in top_ks:
        col = f"mean_abs_rank_error_at_{k}"
        if col in df_results.columns:
            avg_error = df_results[col].mean()
            summary_means[f"mean_abs_rank_error_at_{k}"] = avg_error
            print(f"Mean Abs Rank Error@{k}: {avg_error:.3f}")
        else:
            print(f"[WARN] Column {col} not found in results.")
    print()

    # True MVP rank stats
    if "true_index" in df_results.columns:
        true_ranks = df_results["true_index"].dropna().values
        if len(true_ranks) > 0:
            avg_rank = np.mean(true_ranks)
            min_rank = np.min(true_ranks)
            max_rank = np.max(true_ranks)
            num_correct = np.sum(true_ranks == 1)

            summary_means["true_mvp_avg_rank"] = avg_rank
            summary_means["true_mvp_min_rank"] = min_rank
            summary_means["true_mvp_max_rank"] = max_rank
            summary_means["true_mvp_correct_count"] = num_correct
            summary_means["true_mvp_total"] = len(true_ranks)

            print("[SUMMARY] True MVP rank statistics:")
            print(f"Average rank: {avg_rank:.2f}")
            print(f"Min rank    : {min_rank}")
            print(f"Max rank    : {max_rank}")
            print(f"Correct MVP predictions (rank=1): {num_correct}/{len(true_ranks)}")
        else:
            print("[WARN] No true_index available in results.")
    else:
        print("[WARN] 'true_index' column not found in results.")

    return summary_means


def get_default_hyperparams(model_class: Any, dataset: str) -> dict:
    """
    Returns the default hyperparameters for a given model class and dataset.

    Parameters:
        model_class (Any): The model class (e.g. LogisticRegression, RandomForestClassifier, etc.).
        dataset (str): Name of the dataset (e.g. "selectedStats_from1980").

    Returns:
        dict: Dictionary of default hyperparameters for this model and dataset.
              Returns an empty dict if no defaults are defined for this combination.
    """
    if model_class == LogisticRegression:
        if "1980" in dataset:
            return {
                "solver": "saga",
                "penalty": "l2",
                "class_weight": None,
                "max_iter": 50000,
                "C": 5.0
            }
        else:
            return {
                "solver": "saga",
                "penalty": "l2",
                "class_weight": None,
                "max_iter": 50000,
                "C": 1.0
            }
        
    elif model_class == RandomForestClassifier:
        return {
            "n_estimators": 100,
            "max_depth": None,
            "class_weight": None,
            "min_samples_leaf": 2,
            "random_state": 42
        }

    elif model_class == XGBClassifier:
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 4,
            "scale_pos_weight": 5,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42
        }

    elif model_class == GradientBoostingClassifier:
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
            "random_state": 42
        }

    elif model_class == HistGradientBoostingClassifier:
        return {
            "max_iter": 100,
            "learning_rate": 0.1,
            "max_depth": None,
            "l2_regularization": 1.0,
            "early_stopping": True,
            "random_state": 42
        }
    
    elif model_class == LGBMClassifier:
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": -1,
            "class_weight": "balanced",
            "random_state": 42
        }
    
    return {}
 

if __name__ == "__main__":
    # Example usage from root:
    # python train_models.py --pipeline all --model logreg
    # python train_models.py --pipeline allselected --model gb --start 1990 --end 2020
    # python train_models.py --pipeline selected1956 --model xgb --verbose
    # python train_models.py --pipeline all1956 --model rf --start 2000

    # Constants
    datasets_base_dir = os.path.join(base_dir, "datasets")
    models_base_dir = os.path.join(base_dir, "models")

    # Argument parser
    parser = argparse.ArgumentParser(description="Train MVP prediction models on LOSO splits.")

    parser.add_argument("--pipeline", nargs="+", default=["all"],
                        help="Pipelines to run (ex: all1956, selected1956, selected1980, all1980, allselected, allall, allyear1980, allyear1956, all)")
    parser.add_argument("--model", type=str, default="logreg",
                        help=f"Model to use ({', '.join(MODEL_CLASSES.keys())}), default: logreg")
    parser.add_argument("--start", type=int, default=MIN_YEAR,
                        help=f"Start year (default {MIN_YEAR})")
    parser.add_argument("--end", type=int, default=MAX_YEAR,
                        help=f"End year (default {MAX_YEAR})")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--full", action="store_true",
                    help="Use all features (ignores feature selection)")

    args = parser.parse_args()

    # Verbose
    verbose = args.verbose
    vprint = print if verbose else lambda *a, **kw: None

    # Model
    if args.model not in MODEL_CLASSES:
        print(f"[ERROR] Unknown model '{args.model}'. Available: {list(MODEL_CLASSES.keys())}")
        sys.exit(1)

    model_class = MODEL_CLASSES[args.model]
    model_name = model_class.__name__

    # Pipelines to run
    pipelines_to_run = []
    for p in args.pipeline:
        if p in PIPELINE_GROUPS:
            pipelines_to_run.extend(PIPELINE_GROUPS[p])
        elif p in PIPELINE_ALIASES:
            pipelines_to_run.append(p)
        else:
            print(f"[ERROR] Unknown pipeline or group: '{p}', skipping.")

    # Deduplicate
    pipelines_to_run = list(dict.fromkeys(pipelines_to_run))

    # Run
    for pipeline_key in pipelines_to_run:
        print()
        pipeline_name = PIPELINE_ALIASES[pipeline_key]

        # Determine year range
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

        # Paths
        dataset_dir = os.path.join(datasets_base_dir, pipeline_name)
        output_model_dir = os.path.join(models_base_dir, f"{model_name}_{pipeline_name}")

        os.makedirs(output_model_dir, exist_ok=True)

        # Get hyperparameters
        fixed_params = get_default_hyperparams(model_class, pipeline_name)

        print(f"[INFO] Running model '{model_name}' on pipeline '{pipeline_name}' from {year_start} to {year_end}...")
        print(f"[INFO] Using hyperparameters: {fixed_params}")

        # Get feature selection
        selected_feature_names = None
        if not args.full:
            selected_feature_names = SELECTED_FEATURES.get(args.model, {}).get(pipeline_name, None)
            if selected_feature_names is None:
                print(f"[WARN] No selected features found for model '{args.model}' and pipeline '{pipeline_name}', using full feature set.")

        # Run training
        global_results = []
        for year in range(year_start, year_end + 1):
            results = train_and_evaluate(dataset_dir, year, output_model_dir, model_class, model_name, fixed_params, selected_feature_names, pipeline_name)
            global_results.append({
                "year": year,
                **results
            })

        # Save results
        df_results = pd.DataFrame(global_results)
        results_path = os.path.join(output_model_dir, "summary_results.csv")
        df_results.to_csv(results_path, index=False)
        print()
        print(f"[DONE] Summary saved to {results_path}")

        summary_means = summarize_global_results(df_results)
        summary_df = pd.DataFrame([summary_means])
        mean_summary_path = os.path.join(output_model_dir, "mean_summary.csv")
        summary_df.to_csv(mean_summary_path, index=False)
        print(f"[DONE] Mean summary saved to {mean_summary_path}")

    print()
    print("[INFO] All selected pipelines finished.")


