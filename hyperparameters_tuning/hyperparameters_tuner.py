import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import List, Any, Type

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from train_models import load_dataset, evaluate_model, get_default_hyperparams, get_feature_names
from train_models import MODEL_CLASSES, SELECTED_FEATURES, base_dir, vprint

# Pipelines
pipelines = ["selectedStats_from1980", "selectedStats_from1956", "allStats_from1980", "allStats_from1956"]

# Combos
COMBO_PRESETS = {
    "solver+penalty": [
        {"solver": "lbfgs", "penalty": "l2"},
        {"solver": "lbfgs", "penalty": None},
        {"solver": "saga", "penalty": "l2"},
        {"solver": "saga", "penalty": "l1"},
        {"solver": "liblinear", "penalty": "l1"},
        {"solver": "liblinear", "penalty": "l2"}
    ],
}


def sweep_hyperparameter_on_pipeline(model_class: Type[Any],
                                     param_name: str,
                                     param_values: List[Any],
                                     dataset_name: str,
                                     top_ks: List[int]=[1, 3, 5, 10],
                                     year_start: int=None,
                                     year_end: int=None,
                                     fixed_params: dict=None,
                                     selected_feature_names: List[str] = None) -> pd.DataFrame:
    """
    Sweeps a given hyperparameter on a pipeline and compares metrics.
    Returns a DataFrame with metrics per hyperparam value.

    Parameters:
        model_class (Type[Any]): Model class (e.g. LogisticRegression)
        param_name (str): Name of hyperparameter to sweep (must be valid for model_class)
        param_values (List[Any]): List of values to test for the hyperparameter
        dataset_name (str): Pipeline/dataset name
        top_ks (List[int]): List of top-Ks to evaluate
        year_start (int): First year of LOSO (default auto-detect)
        year_end (int): Last year of LOSO (default auto-detect)
        fixed_params (dict): Dict of the default hyperparameters

    Returns:
        pd.DataFrame: Table of mean metrics per hyperparameter value
    """
    datasets_dir = os.path.join(base_dir, "datasets", dataset_name)

    # Infer years if not provided
    if year_start is None or year_end is None:
        if "1956" in dataset_name:
            year_start = 1956
        else:
            year_start = 1980
        year_end = 2025

    print()
    print(f"[INFO] Tuning '{param_name}' on pipeline '{dataset_name}' [{year_start}-{year_end}]\n")

    summary_table = {}

    for param_value in param_values:
        print()
        print(f"=== Testing {param_name} = {param_value} ===\n")
        
        # Prepare model with this hyperparam
        model_init_params = fixed_params.copy() if fixed_params else {}
        model_init_params[param_name] = param_value

        # Check param exists
        model = model_class()
        valid_params = model.get_params()
        if param_name not in valid_params:
            raise ValueError(f"[ERROR] Hyperparameter '{param_name}' is not valid for {model_class.__name__}. Available: {list(valid_params.keys())}")

        # Loop over seasons
        global_results = []
        for year in tqdm(range(year_start, year_end + 1), desc=f"{param_name}={param_value}", file=sys.stdout):
            # Load dataset
            data = load_dataset(datasets_dir, year)
            X_train = data["X_train"]
            y_train = data["y_top1_train"]
            X_test = data["X_test"]
            y_test = data["y_top1_test"]
            y10_test = data["y_top10_test"]
            player_names_test = data["Name_test"]

            if selected_feature_names is not None:
                all_feature_names = get_feature_names(dataset_name, year)
                selected_indices = [i for i, name in enumerate(all_feature_names) if name in selected_feature_names]
                X_train = X_train[:, selected_indices]
                X_test = X_test[:, selected_indices]

            # Train model
            model = model_class(**model_init_params)
            model.fit(X_train, y_train)

            # Evaluate
            results = evaluate_model(model, X_test, y_test, y10_test, player_names_test, top_ks=top_ks)
            global_results.append({
                "year": year,
                **results
            })

        # Aggregate
        df_results = pd.DataFrame(global_results)

        # Summarize → same logic as summarize_global_results
        summary_metrics = {}

        # Recall@k
        for k in top_ks:
            col = f"top_{k}_hit"
            if col in df_results.columns:
                hit_rate = df_results[col].mean()
                summary_metrics[f"Recall@{k}"] = hit_rate

        # Precision@k real top-k
        for k in top_ks:
            col = f"precision_at_{k}_topk"
            if col in df_results.columns:
                prec = df_results[col].mean()
                summary_metrics[f"Precision@{k} (top-k)"] = prec

        # Precision@k real top-10
        for k in top_ks:
            col = f"precision_at_{k}_top10"
            if col in df_results.columns:
                prec = df_results[col].mean()
                summary_metrics[f"Precision@{k} (top-10)"] = prec

        # Mean abs rank error@k
        for k in top_ks:
            col = f"mean_abs_rank_error_at_{k}"
            if col in df_results.columns:
                err = df_results[col].mean()
                summary_metrics[f"Mean Abs Rank Error@{k}"] = err

        # True MVP rank
        if "true_index" in df_results.columns:
            true_ranks = df_results["true_index"].dropna().values
            if len(true_ranks) > 0:
                avg_rank = np.mean(true_ranks)
                min_rank = np.min(true_ranks)
                max_rank = np.max(true_ranks)
                correct = np.sum(true_ranks == 1)

                summary_metrics["True MVP avg rank"] = avg_rank
                summary_metrics["True MVP min rank"] = min_rank
                summary_metrics["True MVP max rank"] = max_rank
                summary_metrics["True MVP correct (rank=1)"] = correct / len(true_ranks)

        # Store for this param_value
        summary_table[param_value] = summary_metrics

    # Final DataFrame
    df_summary = pd.DataFrame(summary_table).T
    df_summary.index.name = param_name

    return df_summary


def sweep_hyperparameter_combinations_on_pipeline(model_class: Type[Any],
                                                  param_combinations: List[dict],
                                                  dataset_name: str,
                                                  top_ks: List[int] = [1, 3, 5, 10],
                                                  year_start: int = None,
                                                  year_end: int = None,
                                                  fixed_params: dict = None,
                                                  selected_feature_names: List[str] = None) -> pd.DataFrame:
    """
    Sweeps combinations of hyperparameters on a pipeline and compares metrics.
    Returns a DataFrame with metrics per hyperparam combination.

    Parameters:
        model_class (Type[Any]): Model class
        param_combinations (List[dict]): List of hyperparameter combinations to test
        dataset_name (str): Pipeline/dataset name
        top_ks (List[int]): List of top-Ks to evaluate
        year_start (int): First year of LOSO (default auto-detect)
        year_end (int): Last year of LOSO (default auto-detect)
        fixed_params (dict): Dict of the fixed hyperparameters

    Returns:
        pd.DataFrame: Table of mean metrics per hyperparameter combination
    """
    datasets_dir = os.path.join(base_dir, "datasets", dataset_name)

    # Infer years if not provided
    if year_start is None or year_end is None:
        if "1956" in dataset_name:
            year_start = 1956
        else:
            year_start = 1980
        year_end = 2025

    print()
    print(f"[INFO] Tuning hyperparameter combinations on pipeline '{dataset_name}' [{year_start}-{year_end}]\n")

    summary_table = {}

    for combo in param_combinations:
        print()
        print(f"=== Testing combo: {combo} ===\n")

        # Prepare model with this combo + fixed params
        model_init_params = fixed_params.copy() if fixed_params else {}
        model_init_params.update(combo)

        # Check that all params are valid
        model = model_class()
        valid_params = model.get_params()
        for param in combo:
            if param not in valid_params:
                raise ValueError(f"[ERROR] Hyperparameter '{param}' is not valid for {model_class.__name__}. Available: {list(valid_params.keys())}")

        # Loop over seasons
        global_results = []
        for year in tqdm(range(year_start, year_end + 1), desc=f"combo={combo}", file=sys.stdout):
            # Load dataset
            data = load_dataset(datasets_dir, year)
            X_train = data["X_train"]
            y_train = data["y_top1_train"]
            X_test = data["X_test"]
            y_test = data["y_top1_test"]
            y10_test = data["y_top10_test"]
            player_names_test = data["Name_test"]

            if selected_feature_names is not None:
                all_feature_names = get_feature_names(dataset_name, year)
                selected_indices = [i for i, name in enumerate(all_feature_names) if name in selected_feature_names]
                X_train = X_train[:, selected_indices]
                X_test = X_test[:, selected_indices]

            # Train model
            model = model_class(**model_init_params)
            model.fit(X_train, y_train)

            # Evaluate
            results = evaluate_model(model, X_test, y_test, y10_test, player_names_test, top_ks=top_ks)
            global_results.append({
                "year": year,
                **results
            })

        # Aggregate
        df_results = pd.DataFrame(global_results)

        # Summarize → same logic as summarize_global_results
        summary_metrics = {}

        # Recall@k
        for k in top_ks:
            col = f"top_{k}_hit"
            if col in df_results.columns:
                hit_rate = df_results[col].mean()
                summary_metrics[f"Recall@{k}"] = hit_rate

        # Precision@k real top-k
        for k in top_ks:
            col = f"precision_at_{k}_topk"
            if col in df_results.columns:
                prec = df_results[col].mean()
                summary_metrics[f"Precision@{k} (top-k)"] = prec

        # Precision@k real top-10
        for k in top_ks:
            col = f"precision_at_{k}_top10"
            if col in df_results.columns:
                prec = df_results[col].mean()
                summary_metrics[f"Precision@{k} (top-10)"] = prec

        # Mean abs rank error@k
        for k in top_ks:
            col = f"mean_abs_rank_error_at_{k}"
            if col in df_results.columns:
                err = df_results[col].mean()
                summary_metrics[f"Mean Abs Rank Error@{k}"] = err

        # True MVP rank
        if "true_index" in df_results.columns:
            true_ranks = df_results["true_index"].dropna().values
            if len(true_ranks) > 0:
                avg_rank = np.mean(true_ranks)
                min_rank = np.min(true_ranks)
                max_rank = np.max(true_ranks)
                correct = np.sum(true_ranks == 1)

                summary_metrics["True MVP avg rank"] = avg_rank
                summary_metrics["True MVP min rank"] = min_rank
                summary_metrics["True MVP max rank"] = max_rank
                summary_metrics["True MVP correct (rank=1)"] = correct / len(true_ranks)

        # Store this combination
        combo_str = ", ".join([f"{k}={v}" for k, v in combo.items()])
        summary_table[combo_str] = summary_metrics

    # Final DataFrame
    df_summary = pd.DataFrame(summary_table).T
    df_summary.index.name = "hyperparam_combo"

    return df_summary


if __name__ == "__main__":
    # Example usage from root:
    # Sweep a single hyperparameter (example on C):
    # python hyperparameters_tuning/hyperparameters_tuner.py --model logreg --param C --values 0.01 0.1 1.0 5.0 10.0
    # python hyperparameters_tuning/hyperparameters_tuner.py --model logreg --param class_weight --values None balanced
    #
    # Sweep hyperparameter combinations:
    # python hyperparameters_tuning/hyperparameters_tuner.py --model logreg --param solver+penalty --combo 

    # Constants
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    styled_pd_path = os.path.join(base_dir, "hyperparameters_tuning", "results")
    os.makedirs(styled_pd_path, exist_ok=True)

    # Argument parser
    parser = argparse.ArgumentParser(description="Sweep hyperparameters on MVP prediction pipelines.")

    parser.add_argument("--model", type=str, required=True,
                        help=f"Model to use ({', '.join(MODEL_CLASSES.keys())})")
    parser.add_argument("--param", type=str, required=True,
                        help="Name of hyperparameter to sweep")
    parser.add_argument("--values", nargs="*",
                        help="List of values to test (ignored if --combo is used)")
    parser.add_argument("--combo", action="store_true",
                        help="Use combo mode with predefined combos")
    parser.add_argument("--full", action="store_true",
                    help="Use all features (ignores feature selection)")

    args = parser.parse_args()

    # Model
    if args.model not in MODEL_CLASSES:
        print(f"[ERROR] Unknown model '{args.model}'. Available: {list(MODEL_CLASSES.keys())}")
        sys.exit(1)

    model_class = MODEL_CLASSES[args.model]
    model_name = model_class.__name__

    # Param
    param = args.param

    # Values
    if args.combo:
        if param not in COMBO_PRESETS:
            print(f"[ERROR] No combo preset defined for param '{param}'. Available: {list(COMBO_PRESETS.keys())}")
            sys.exit(1)
        param_combinations = COMBO_PRESETS[param]
        print(f"[INFO] Using combo preset for '{param}': {param_combinations}")
    else:
        if args.values is None or len(args.values) == 0:
            print(f"[ERROR] You must provide --values when not using --combo.")
            sys.exit(1)
        # Try to convert values to float if possible
        param_values = []
        for v in args.values:
            try:
                param_values.append(float(v))
            except ValueError:
                if v.lower() == "none":
                    param_values.append(None)
                else:
                    param_values.append(v)
        print(f"[INFO] Using param values for '{param}': {param_values}")

    # HTML content init
    html_content = ""

    # Run
    for pipeline in pipelines:
        print()
        print()
        print(f"#########################")
        print(f"## Pipeline: {pipeline}")
        print(f"#########################\n")

        # Get fixed params
        fixed_params = get_default_hyperparams(model_class, pipeline)

        # Get feature selection
        selected_feature_names = None
        if not args.full:
            selected_feature_names = SELECTED_FEATURES.get(args.model, {}).get(pipeline, None)
            if selected_feature_names is None:
                print(f"[WARN] No selected features found for model '{args.model}' and pipeline '{pipeline}', using full feature set.")

        # Sweep
        if args.combo:
            df = sweep_hyperparameter_combinations_on_pipeline(model_class=model_class,
                                                               param_combinations=param_combinations,
                                                               dataset_name=pipeline,
                                                               fixed_params=fixed_params,
                                                               selected_feature_names=selected_feature_names)
        else:
            df = sweep_hyperparameter_on_pipeline(model_class=model_class,
                                                  param_name=param,
                                                  param_values=param_values,
                                                  dataset_name=pipeline,
                                                  fixed_params=fixed_params,
                                                  selected_feature_names=selected_feature_names)

        # Style
        df.index = df.index.map(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))

        # Build metric_targets
        metric_targets = {}
        for col in df.columns:
            if "Mean Abs Rank Error" in col or "avg rank" in col or "min rank" in col or "max rank" in col:
                metric_targets[col] = 0  # low is good
            else:
                metric_targets[col] = 1  # high is good

        styled_df = df.style.format(precision=3)
        for col in df.columns:
            target = metric_targets[col]
            if target == 1:
                styled_df = styled_df.background_gradient(subset=[col], cmap="RdYlGn", vmin=df[col].min(), vmax=df[col].max())
            else:
                styled_df = styled_df.background_gradient(subset=[col], cmap="RdYlGn_r", vmin=df[col].min(), vmax=df[col].max())

        # Append HTML
        html_content += f"<h2>Pipeline: {pipeline}</h2>\n"
        html_content += styled_df.to_html()

    # Save HTML
    html_filename = f"{model_name}_{param}.html"
    html_path = os.path.join(styled_pd_path, html_filename)

    with open(html_path, "w") as f:
        f.write(html_content)

    print()
    print(f"[DONE] All pipelines saved to: {html_path}")
