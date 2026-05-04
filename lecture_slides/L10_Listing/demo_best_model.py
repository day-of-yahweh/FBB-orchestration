import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import prepare_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def find_best_model():
    experiments = [
        "himalayan_summit_prediction",
        "himalayan_hyperparam_search"
    ]
    all_runs = []
    for exp_name in experiments:
        try:
            runs = mlflow.search_runs(experiment_names=[exp_name])
            if not runs.empty:
                runs["experiment"] = exp_name
                all_runs.append(runs)
        except Exception:
            continue

    if not all_runs:
        print("No runs found. Run training scripts first.")
        return None

    all_runs_df = pd.concat(all_runs, ignore_index=True)
    best_run = all_runs_df.sort_values("metrics.f1_score", ascending=False).iloc[0]

    print(f"Top 5 models by F1 score:")
    top5 = all_runs_df.nlargest(5, "metrics.f1_score")[
        ["tags.mlflow.runName", "metrics.f1_score", "metrics.accuracy", "experiment"]
    ]
    print(top5.to_string(index=False))

    print(f"Best model: {best_run['tags.mlflow.runName']}")
    print(f"Experiment: {best_run['experiment']}")
    print(f"F1 Score:   {best_run['metrics.f1_score']:.4f}")
    print(f"Run ID:     {best_run['run_id']}")
    return best_run


def load_and_predict(run_id, artifact_name):
    model_uri = f"runs:/{run_id}/{artifact_name}"
    if "XGBoost" in artifact_name:
        model = mlflow.xgboost.load_model(model_uri)
    elif "LightGBM" in artifact_name:
        model = mlflow.lightgbm.load_model(model_uri)
    else:
        model = mlflow.sklearn.load_model(model_uri)
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"{'Actual':<10} {'Predicted':<12} {'P(Success)':<12}")
    for i in range(min(10, len(y_test))):
        actual = "Success" if y_test.iloc[i] == 1 else "Failure"
        predicted = "Success" if y_pred[i] == 1 else "Failure"
        prob = y_prob[i]
        print(f"{actual:<10} {predicted:<12} {prob:.2%}")

    print(f"Test set performance:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred):.4f}")

    return model


def register_model(run_id, artifact_name, model_name="HimalayanSummitPredictor"):
    model_uri = f"runs:/{run_id}/{artifact_name}"
    try:
        result = mlflow.register_model(model_uri, model_name)
        print(f"Model registered successfully!")
        print(f"Name:    {result.name}")
        print(f"Version: {result.version}")
        return result
    except Exception as e:
        print(f"Registration note: {e}")
        print("Model Registry may require database backend for full functionality.")
        return None


def main():
    print("  MLflow Demo: Best Model Selection & Deployment")
    best_run = find_best_model()
    if best_run is None:
        return
    run_id = best_run["run_id"]
    artifact_name = best_run["tags.mlflow.runName"]
    model = load_and_predict(run_id, artifact_name)
    register_model(run_id, artifact_name)


if __name__ == "__main__":
    main()
