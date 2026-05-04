import warnings
import itertools
from typing import Dict, Any, List

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data_preprocessing import prepare_data
mlflow.set_tracking_uri("sqlite:///mlflow.db")
warnings.filterwarnings("ignore")
from skops.io import dump, get_untrusted_types
import tempfile
def get_model_trusted_types(model):
    with tempfile.NamedTemporaryFile(suffix=".skops", delete=False) as f:
        dump(model, f.name)
        return get_untrusted_types(file=f.name)




def grid_search_with_mlflow(
    X_train,
    y_train,
    X_test,
    y_test,
    param_grid,
    experiment_name="himalayan_hyperparam_search",
):
    mlflow.set_experiment(experiment_name)

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"Total combinations to try: {len(combinations)}")
    print(f"Parameter grid: {param_grid}")

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        run_name = "_".join(f"{k}={v}" for k, v in params.items())

        with mlflow.start_run(run_name=f"RF_{i+1:03d}"):
            print(f"[{i+1}/{len(combinations)}] {params}")

            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            mlflow.log_metric("cv_f1_mean", cv_mean)
            mlflow.log_metric("cv_f1_std", cv_std)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_pred, zero_division=0),
                "test_f1": f1_score(y_test, y_pred, zero_division=0),
                "test_auc": roc_auc_score(y_test, y_prob),
            }

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            print(f"CV F1: {cv_mean:.4f} (+/- {cv_std:.4f})")
            print(f"Test F1: {metrics['test_f1']:.4f}")
            trusted = get_model_trusted_types(model)
            mlflow.sklearn.log_model(model, name=f"RF_{i+1:03d}", serialization_format="skops", skops_trusted_types=trusted)
            results.append({
                "run": i + 1,
                **params,
                "cv_f1_mean": cv_mean,
                "cv_f1_std": cv_std,
                **metrics,
            })

    return pd.DataFrame(results)


def visualize_results(results_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1 = axes[0, 0]
    for depth in results_df["max_depth"].unique():
        subset = results_df[results_df["max_depth"] == depth]
        ax1.plot(
            subset["n_estimators"],
            subset["test_f1"],
            marker="o",
            label=f"depth={depth}",
        )
    ax1.set_xlabel("n_estimators")
    ax1.set_ylabel("Test F1 Score")
    ax1.set_title("F1 Score vs Number of Trees")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2 = axes[0, 1]
    for n_est in results_df["n_estimators"].unique():
        subset = results_df[results_df["n_estimators"] == n_est]
        ax2.plot(
            subset["max_depth"],
            subset["test_f1"],
            marker="s",
            label=f"trees={n_est}",
        )
    ax2.set_xlabel("max_depth")
    ax2.set_ylabel("Test F1 Score")
    ax2.set_title("F1 Score vs Max Depth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.scatter(results_df["cv_f1_mean"], results_df["test_f1"], alpha=0.7)
    ax3.plot([0.5, 1], [0.5, 1], "k--", alpha=0.3)  # diagonal line
    ax3.set_xlabel("CV F1 Score")
    ax3.set_ylabel("Test F1 Score")
    ax3.set_title("Cross-Validation vs Test Performance")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    pivot = results_df.pivot_table(
        values="test_f1",
        index="max_depth",
        columns="n_estimators",
        aggfunc="mean",
    )
    im = ax4.imshow(pivot.values, cmap="YlGn", aspect="auto")
    ax4.set_xticks(range(len(pivot.columns)))
    ax4.set_xticklabels(pivot.columns)
    ax4.set_yticks(range(len(pivot.index)))
    ax4.set_yticklabels(pivot.index)
    ax4.set_xlabel("n_estimators")
    ax4.set_ylabel("max_depth")
    ax4.set_title("F1 Score Heatmap")
    plt.colorbar(im, ax=ax4, label="F1 Score")

    plt.tight_layout()
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [5, 8, 10, 15, None],
        "min_samples_split": [2, 4, 5, 10],
    }

    print(f"Data: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Features: {len(feature_names)}")

    results_df = grid_search_with_mlflow(
        X_train, y_train, X_test, y_test, param_grid
    )

    top5 = results_df.nlargest(5, "test_f1")
    print("Top 5 configurations by Test F1:")
    print(top5[["n_estimators", "max_depth", "min_samples_split", "cv_f1_mean", "test_f1"]].to_string(index=False))

    best = results_df.loc[results_df["test_f1"].idxmax()]
    print(f"Best configuration:")
    print(f"> n_estimators: {best['n_estimators']}")
    print(f"> max_depth: {best['max_depth']}")
    print(f"> min_samples_split: {best['min_samples_split']}")
    print(f"> Test F1: {best['test_f1']:.4f}")

    fig = visualize_results(results_df)
    with mlflow.start_run(run_name="search_summary"):
        mlflow.log_param("total_combinations", len(results_df))
        mlflow.log_metric("best_test_f1", best["test_f1"])
        mlflow.log_param("best_n_estimators", best["n_estimators"])
        mlflow.log_param("best_max_depth", best["max_depth"])
        mlflow.log_figure(fig, "search_results.png")

    plt.close(fig)
    return results_df


if __name__ == "__main__":
    main()
