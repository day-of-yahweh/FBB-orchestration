import argparse
import warnings
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import prepare_data, prepare_scaled_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*cloudpickle.*")

from skops.io import dump, get_untrusted_types
import tempfile
def get_model_trusted_types(model):
    with tempfile.NamedTemporaryFile(suffix=".skops", delete=False) as f:
        dump(model, f.name)
        return get_untrusted_types(file=f.name)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    if auc is not None:
        metrics["auc_roc"] = auc

    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Failed", "Success"],
        yticklabels=["Failed", "Success"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15] #top 15
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(indices)),
        importances[indices][::-1],
        color="steelblue",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance: {model_name}")
    plt.tight_layout()
    return fig


def train_and_log_model(
    model,
    model_name,
    params,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", len(feature_names))

        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        trusted = get_model_trusted_types(model)
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")

        cm_fig = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        plt.close(cm_fig)

        fi_fig = plot_feature_importance(model, feature_names, model_name)
        if fi_fig is not None:
            mlflow.log_figure(fi_fig, "feature_importance.png")
            plt.close(fi_fig)

        mlflow.sklearn.log_model(model, name=model_name, serialization_format="skops", skops_trusted_types=trusted)
        report = classification_report(
            y_test, y_pred, target_names=["Failed", "Success"]
        )
        mlflow.log_text(report, "classification_report.txt")

        return metrics


def run_experiments(experiment_name: str = "himalayan_summit_prediction"):
    mlflow.set_experiment(experiment_name)

    # Prepare data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        test_size=0.2, random_state=75)
    X_train_scaled, X_test_scaled, _, _, _, scaler = prepare_scaled_data(
        test_size=0.2, random_state=75)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Success rate (train): {y_train.mean():.2%}")

    models = [
        {
            "model": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "name": "LogisticRegression_baseline",
            "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
            "scaled": True,
        },
        {
            "model": LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=42),
            "name": "LogisticRegression_regularized",
            "params": {"C": 0.1, "penalty": "l2", "max_iter": 1000},
            "scaled": True,
        },
        {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "name": "RandomForest_100trees",
            "params": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
            "scaled": False,
        },
        {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            ),
            "name": "RandomForest_tuned",
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
            "scaled": False,
        },
        {
            "model": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
            ),
            "name": "GradientBoosting_default",
            "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
            "scaled": False,
        },
        {
            "model": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            ),
            "name": "GradientBoosting_tuned",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.8,
            },
            "scaled": False,
        },
        {
            "model": ExtraTreesClassifier(
                n_estimators=150,
                max_depth=12,
                random_state=42,
            ),
            "name": "ExtraTrees",
            "params": {"n_estimators": 150, "max_depth": 12},
            "scaled": False,
        },
        {
            "model": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "name": "KNN_7neighbors",
            "params": {"n_neighbors": 7, "weights": "distance", "metric": "minkowski"},
            "scaled": True,
        },
        {
            "model": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
            "name": "SVM_RBF",
            "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            "scaled": True,
        },
        {
            "model": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=42,
            ),
            "name": "MLP_64_32",
            "params": {
                "hidden_layers": "(64, 32)",
                "activation": "relu",
                "max_iter": 500,
            },
            "scaled": True,
        },
    ]

    results = []

    for model_config in models:
        if model_config["scaled"]:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        metrics = train_and_log_model(
            model=model_config["model"],
            model_name=model_config["name"],
            params=model_config["params"],
            X_train=X_tr,
            X_test=X_te,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
        )
        results.append({"model": model_config["name"], **metrics})

    print("EXPERIMENT SUMMARY")
    print("-" * 60)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1_score", ascending=False)
    print(results_df.to_string(index=False))

    print(f"Best model by F1: {results_df.iloc[0]['model']}")
    print(f"MLflow experiment: {experiment_name}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with MLflow tracking")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="himalayan_summit_prediction",
        help="MLflow experiment name",
    )
    args = parser.parse_args()
    run_experiments(args.experiment_name)
