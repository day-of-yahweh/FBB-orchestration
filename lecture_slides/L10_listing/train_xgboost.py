import warnings
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
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


def evaluate_and_log(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        print(f"{name}: {value:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Failed", "Success"],
                yticklabels=["Failed", "Success"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices][::-1], color="steelblue")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([X_test.columns[i] for i in indices][::-1])
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance: {model_name}")
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

    return metrics


def run_xgboost_experiments():
    mlflow.set_experiment("himalayan_summit_prediction")

    X_train, X_test, y_train, y_test, feature_names = prepare_data()

    print("XGBoost Experiments")
    print("-" * 60)

    configs = [
        {
            "name": "XGBoost_baseline",
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "eval_metric": "logloss",
            },
        },
        {
            "name": "XGBoost_deep",
            "params": {
                "n_estimators": 150,
                "max_depth": 10,
                "learning_rate": 0.1,
                "min_child_weight": 3,
                "random_state": 42,
                "eval_metric": "logloss",
            },
        },
        {
            "name": "XGBoost_regularized",
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "eval_metric": "logloss",
            },
        },
        {
            "name": "XGBoost_subsampled",
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
            },
        },
    ]

    results = []
    for config in configs:
        with mlflow.start_run(run_name=config["name"]):
            print(f"Training {config['name']}...")
            for param, value in config["params"].items():
                mlflow.log_param(param, value)

            model = XGBClassifier(**config["params"])
            model.fit(X_train, y_train)
            metrics = evaluate_and_log(model, X_test, y_test, config["name"])
            mlflow.xgboost.log_model(model, name=config["name"])
            results.append({"model": config["name"], **metrics})
    return results


def run_lightgbm_experiments():
    mlflow.set_experiment("himalayan_summit_prediction")

    X_train, X_test, y_train, y_test, feature_names = prepare_data()

    print("LightGBM Experiments")
    print("-" * 60)

    configs = [
        {
            "name": "LightGBM_baseline",
            "params": {
                "n_estimators": 100,
                "max_depth": -1,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "random_state": 42,
                "verbose": -1,
            },
        },
        {
            "name": "LightGBM_more_leaves",
            "params": {
                "n_estimators": 150,
                "max_depth": 8,
                "learning_rate": 0.1,
                "num_leaves": 63,
                "random_state": 42,
                "verbose": -1,
            },
        },
        {
            "name": "LightGBM_regularized",
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
                "verbose": -1,
            },
        },
        {
            "name": "LightGBM_dart",
            "params": {
                "n_estimators": 150,
                "max_depth": 7,
                "learning_rate": 0.08,
                "num_leaves": 50,
                "boosting_type": "dart",
                "random_state": 42,
                "verbose": -1,
            },
        },
    ]

    results = []
    for config in configs:
        with mlflow.start_run(run_name=config["name"]):
            print(f"Training {config['name']}...")

            for param, value in config["params"].items():
                mlflow.log_param(param, value)

            model = LGBMClassifier(**config["params"])
            model.fit(X_train, y_train)
            trusted = get_model_trusted_types(model)

            metrics = evaluate_and_log(model, X_test, y_test, config["name"])
            mlflow.lightgbm.log_model(model, name="model", serialization_format="skops", skops_trusted_types=trusted)
            results.append({"model": config["name"], **metrics})

    return results


if __name__ == "__main__":
    xgb_results = run_xgboost_experiments()
    lgbm_results = run_lightgbm_experiments()

    print("ALL RESULTS SUMMARY")

    all_results = pd.DataFrame(xgb_results + lgbm_results)
    all_results = all_results.sort_values("f1_score", ascending=False)
    print(all_results.to_string(index=False))

    print(f"Best model: {all_results.iloc[0]['model']}")
    print("View results: mlflow ui")
