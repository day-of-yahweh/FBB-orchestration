import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

st.set_page_config(
    page_title="Himalayan Expeditions ML Dashboard",
    page_icon="",
    layout="wide"
)


def get_coords(peak_id):
    if peak_id in PEAK_COORDS:
        return PEAK_COORDS[peak_id][:2]
    lat = np.random.uniform(27.5, 29.0)
    lon = np.random.uniform(83.5, 88.5)
    return (lat, lon)


@st.cache_data
def load_raw_data():
    """Load raw expedition and peaks data."""
    data_path = Path("data")
    expeditions = pd.read_csv(data_path / "exped_tidy.csv", encoding="latin-1")
    peaks = pd.read_csv(data_path / "peaks_tidy.csv", encoding="latin-1")
    return expeditions, peaks


@st.cache_data
def engineer_features(expeditions, peaks):
    """Merge datasets and create ML features."""
    df = expeditions.merge(
        peaks[["PEAKID", "HEIGHTM", "HIMAL_FACTOR", "REGION_FACTOR", "TREKKING"]],
        on="PEAKID",
        how="left"
    )

    features = pd.DataFrame()
    features["target"] = df["SUCCESS1"].astype(int)
    features["season"] = df["SEASON"].fillna(1).astype(int)
    features["host"] = df["HOST"].fillna(1).astype(int)
    features["total_members"] = df["TOTMEMBERS"].fillna(0).astype(int)
    features["total_hired"] = df["TOTHIRED"].fillna(0).astype(int)
    features["hired_ratio"] = features["total_hired"] / (features["total_members"] + 1)
    features["o2_used"] = df["O2USED"].fillna(False).astype(int)
    features["o2_climb"] = df["O2CLIMB"].fillna(False).astype(int)
    features["o2_sleep"] = df["O2SLEEP"].fillna(False).astype(int)
    features["camps"] = df["CAMPS"].fillna(0).astype(int)
    features["rope_fixed"] = df["ROPE"].fillna(0).astype(int)
    features["height_m"] = df["HEIGHTM"].fillna(df["HEIGHTM"].median())
    features["height_scaled"] = features["height_m"] / 1000
    features["is_8000er"] = (features["height_m"] >= 8000).astype(int)
    features["is_trekking_peak"] = df["TREKKING"].fillna(False).astype(int)
    features["year"] = df["YEAR"].astype(int)
    features["total_days"] = df["TOTDAYS"].fillna(df["TOTDAYS"].median())
    features["summit_days"] = df["SMTDAYS"].fillna(df["SMTDAYS"].median())
    features["standard_route"] = df["STDRTE"].fillna(False).astype(int)
    features["commercial_route"] = df["COMRTE"].fillna(False).astype(int)
    features["solo"] = (features["total_members"] == 1).astype(int)
    features["small_team"] = ((features["total_members"] >= 2) & (features["total_members"] <= 5)).astype(int)
    features["large_team"] = (features["total_members"] > 10).astype(int)

    features["peak_id"] = df["PEAKID"]
    features["season_name"] = df["SEASON_FACTOR"]
    features["host_name"] = df["HOST_FACTOR"]

    return features.dropna(subset=["target"])


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Data Engineering", "Exploratory Analysis", "Peak Map", "ML Training", "MLOps Dashboard", "Model Inference"]
)

expeditions_raw, peaks_raw = load_raw_data()
features_df = engineer_features(expeditions_raw, peaks_raw)

if page == "Data Engineering":
    st.title("🔧 Data Engineering")
    st.markdown("### Raw Data Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Expeditions Dataset")
        st.write(f"Shape: {expeditions_raw.shape}")
        st.dataframe(expeditions_raw.head(10), height=300)

    with col2:
        st.subheader("Peaks Dataset")
        st.write(f"Shape: {peaks_raw.shape}")
        st.dataframe(peaks_raw.head(10), height=300)

    st.markdown("---")
    st.markdown("### Feature Engineering Pipeline")

    st.code("""
# Merge expeditions with peak info
df = expeditions.merge(peaks[["PEAKID", "HEIGHTM", ...]], on="PEAKID")

# Create features
features["hired_ratio"] = total_hired / (total_members + 1)
features["is_8000er"] = (height_m >= 8000).astype(int)
features["height_scaled"] = height_m / 1000  # Convert to km
    """, language="python")

    st.markdown("### Engineered Features")
    st.write(f"Shape: {features_df.shape}")
    st.dataframe(features_df.head(20), height=400)

    st.markdown("### Feature Statistics")
    st.dataframe(features_df.describe())

    st.markdown("### Missing Values")
    missing = features_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.bar_chart(missing)
    else:
        st.success("No missing values in engineered features!")


elif page == "Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    col1, col2, col3, col4 = st.columns(4)
    success_rate = features_df["target"].mean()
    col1.metric("Total Expeditions", len(features_df))
    col2.metric("Success Rate", f"{success_rate:.1%}")
    col3.metric("8000m+ Peaks", features_df["is_8000er"].sum())
    col4.metric("Avg Team Size", f"{features_df['total_members'].mean():.1f}")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Success Factors", "Peak Analysis", "Team Composition", "Time Trends"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Success by Season")
            season_success = features_df.groupby("season_name")["target"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            season_success.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_ylabel("Success Rate")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_ylim(0, 1)
            for i, v in enumerate(season_success):
                ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
            st.pyplot(fig)

        with col2:
            st.subheader("Success by Oxygen Usage")
            o2_success = features_df.groupby("o2_used")["target"].mean()
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(["No O2", "With O2"], o2_success.values, color=["coral", "steelblue"])
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1)
            for bar, v in zip(bars, o2_success.values):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.0%}", ha="center")
            st.pyplot(fig)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Success Rate vs Peak Height")
            fig, ax = plt.subplots(figsize=(8, 5))
            height_bins = pd.cut(features_df["height_m"], bins=[5000, 6000, 7000, 8000, 9000])
            height_success = features_df.groupby(height_bins)["target"].mean()
            height_success.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_ylabel("Success Rate")
            ax.set_xlabel("Peak Height (m)")
            ax.set_xticklabels(["5-6k", "6-7k", "7-8k", "8k+"], rotation=0)
            st.pyplot(fig)

        with col2:
            st.subheader("Top 10 Most Attempted Peaks")
            peak_counts = features_df["peak_id"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            peak_counts.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel("Number of Expeditions")
            st.pyplot(fig)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Team Size Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            features_df["total_members"].clip(upper=30).hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
            ax.set_xlabel("Team Size")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.subheader("Success by Team Size Category")
            team_cats = ["solo", "small_team", "large_team"]
            team_success = [features_df[features_df[cat] == 1]["target"].mean() for cat in team_cats]
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(["Solo", "Small (2-5)", "Large (10+)"], team_success, color="steelblue")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1)
            for bar, v in zip(bars, team_success):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.0%}", ha="center")
            st.pyplot(fig)

    with tab4:
        st.subheader("Expeditions Over Time")
        yearly = features_df.groupby("year").agg(
            expeditions=("target", "count"),
            success_rate=("target", "mean")
        )

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()

        ax1.bar(yearly.index, yearly["expeditions"], color="steelblue", alpha=0.7, label="Expeditions")
        ax2.plot(yearly.index, yearly["success_rate"], color="coral", marker="o", linewidth=2, label="Success Rate")

        ax1.set_xlabel("Year")
        ax1.set_ylabel("Number of Expeditions", color="steelblue")
        ax2.set_ylabel("Success Rate", color="coral")
        ax2.set_ylim(0, 1)

        st.pyplot(fig)

elif page == "Peak Map":
    st.title("Himalayan Peaks Map")
    st.markdown("### Success/Failure Distribution by Peak Location")

    # Known peak coordinates (major peaks)
    PEAK_COORDS = {
        "EVER": (27.9881, 86.9250, "Everest"),
        "KANG": (27.7025, 88.1475, "Kangchenjunga"),
        "LHOT": (27.9617, 86.9330, "Lhotse"),
        "MAKA": (27.8897, 87.0886, "Makalu"),
        "CHOY": (27.9397, 86.6600, "Cho Oyu"),
        "DHAU": (28.6967, 83.4875, "Dhaulagiri"),
        "MANA": (28.5497, 84.5597, "Manaslu"),
        "NANG": (28.5961, 84.6269, "Nanga Parbat"),
        "ANNA": (28.5961, 83.8203, "Annapurna"),
        "AMAD": (27.8614, 86.8614, "Ama Dablam"),
        "PUMU": (28.0167, 85.7667, "Pumori"),
        "NUPS": (27.9667, 86.8833, "Nuptse"),
        "BARU": (27.7000, 86.7833, "Baruntse"),
        "HIML": (27.9167, 86.7333, "Himlung"),
        "TILI": (28.6833, 83.8000, "Tilicho"),
    }

    peak_stats = features_df.groupby("peak_id").agg(
        total=("target", "count"),
        successes=("target", "sum")
    ).reset_index()
    peak_stats["failures"] = peak_stats["total"] - peak_stats["successes"]
    peak_stats["success_rate"] = peak_stats["successes"] / peak_stats["total"]
    peak_stats = peak_stats.merge(
        peaks_raw[["PEAKID", "PKNAME", "HEIGHTM"]],
        left_on="peak_id",
        right_on="PEAKID",
        how="left"
    )

    np.random.seed(42)
    def get_coords(peak_id):
        if peak_id in PEAK_COORDS:
            return PEAK_COORDS[peak_id][:2]
        lat = np.random.uniform(27.5, 29.0)
        lon = np.random.uniform(83.5, 88.5)
        return (lat, lon)

    peak_stats["lat"] = peak_stats["peak_id"].apply(lambda x: get_coords(x)[0])
    peak_stats["lon"] = peak_stats["peak_id"].apply(lambda x: get_coords(x)[1])

    # Filter to peaks with enough expeditions
    min_expeditions = st.slider("Minimum expeditions to show", 1, 20, 3)
    filtered_peaks = peak_stats[peak_stats["total"] >= min_expeditions]

    st.write(f"Showing {len(filtered_peaks)} peaks with ≥{min_expeditions} expeditions")

    import plotly.express as px
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=28.0, lon=86.0),
            zoom=5
        ),
        showlegend=True,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    for _, peak in filtered_peaks.iterrows():
        success_pct = peak["success_rate"] * 100
        fail_pct = 100 - success_pct
        size = min(50, 10 + peak["total"] * 2)
        fig.add_trace(go.Scattermapbox(
            lat=[peak["lat"]],
            lon=[peak["lon"]],
            mode="markers",
            marker=dict(
                size=size * (peak["success_rate"]),
                color="green",
                opacity=0.7
            ),
            name=f"{peak['PKNAME']} - Success",
            hovertemplate=(
                f"<b>{peak['PKNAME']}</b><br>"
                f"Height: {peak['HEIGHTM']:.0f}m<br>"
                f"Total: {peak['total']} expeditions<br>"
                f"Success: {peak['successes']} ({success_pct:.0f}%)<br>"
                f"Failed: {peak['failures']} ({fail_pct:.0f}%)<br>"
                "<extra></extra>"
            ),
            showlegend=False
        ))
        fig.add_trace(go.Scattermapbox(
            lat=[peak["lat"] + 0.02],
            lon=[peak["lon"] + 0.02],
            mode="markers",
            marker=dict(
                size=size * (1 - peak["success_rate"]),
                color="red",
                opacity=0.7
            ),
            name=f"{peak['PKNAME']} - Failed",
            showlegend=False
        ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Peak Statistics")
    col1, col2 = st.columns([2, 1])
    with col1:
        top_peaks = filtered_peaks.nlargest(10, "total")
        fig2, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for idx, (_, peak) in enumerate(top_peaks.iterrows()):
            if idx >= 10:
                break
            ax = axes[idx]
            sizes = [peak["successes"], peak["failures"]]
            colors = ["#2ecc71", "#e74c3c"]
            ax.pie(sizes, colors=colors, autopct="%1.0f%%",
                  startangle=90, textprops={"fontsize": 8})
            ax.set_title(f"{peak['PKNAME']}\n({peak['total']} exp)", fontsize=9)
        plt.suptitle("Top 10 Peaks by Expedition Count", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)

    with col2:
        st.markdown("**Legend**")
        st.markdown("🟢 Success")
        st.markdown("🔴 Failure")
        st.markdown("")
        st.markdown("**Summary**")
        st.metric("Total Peaks", len(filtered_peaks))
        st.metric("Total Expeditions", filtered_peaks["total"].sum())
        st.metric("Avg Success Rate", f"{filtered_peaks['success_rate'].mean():.1%}")


elif page == "ML Training":
    st.title("Machine Learning Training")

    st.markdown("### Configure Training")

    col1, col2, col3 = st.columns(3)

    with col1:
        model_type = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )

    with col2:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

    with col3:
        random_state = st.number_input("Random State", 0, 100, 42)

    st.markdown("### Model Hyperparameters")

    if model_type == "Logistic Regression":
        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
        with col2:
            max_iter = st.slider("Max Iterations", 100, 2000, 1000)
        model_params = {"C": C, "max_iter": max_iter, "random_state": random_state}

    elif model_type == "Random Forest":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("Number of Trees", 50, 300, 100)
        with col2:
            max_depth = st.slider("Max Depth", 3, 20, 10)
        with col3:
            min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": random_state
        }

    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("Number of Trees", 50, 300, 100)
        with col2:
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        with col3:
            max_depth = st.slider("Max Depth", 2, 10, 3)
        model_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": random_state
        }

    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            feature_cols = [col for col in features_df.columns
                          if col not in ["target", "peak_id", "season_name", "host_name"]]
            X = features_df[feature_cols].fillna(0)
            y = features_df["target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            if model_type == "Logistic Regression":
                model = LogisticRegression(**model_params)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                if model_type == "Random Forest":
                    model = RandomForestClassifier(**model_params)
                else:
                    model = GradientBoostingClassifier(**model_params)
                X_train_scaled, X_test_scaled = X_train, X_test


            mlflow.set_experiment("streamlit_training")
            with mlflow.start_run(run_name=f"{model_type.replace(' ', '_')}"):
                for param, value in model_params.items():
                    mlflow.log_param(param, value)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("test_size", test_size)
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "auc_roc": roc_auc_score(y_test, y_prob)
                }

                for name, value in metrics.items():
                    mlflow.log_metric(name, value)
                mlflow.sklearn.log_model(model, name=model_type.replace(" ", "_"))
                run_id = mlflow.active_run().info.run_id
            st.success(f"Model trained and logged to MLflow! Run ID: {run_id[:8]}...")

            st.markdown("### Results")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            col2.metric("Precision", f"{metrics['precision']:.3f}")
            col3.metric("Recall", f"{metrics['recall']:.3f}")
            col4.metric("F1 Score", f"{metrics['f1_score']:.3f}")
            col5.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                           xticklabels=["Failure", "Success"],
                           yticklabels=["Failure", "Success"], ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with col2:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, color="steelblue", linewidth=2,
                       label=f"AUC = {metrics['auc_roc']:.3f}")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()
                st.pyplot(fig)

            if hasattr(model, "feature_importances_"):
                st.subheader("Feature Importance")
                importance = pd.Series(model.feature_importances_, index=feature_cols)
                importance = importance.sort_values(ascending=True).tail(15)
                fig, ax = plt.subplots(figsize=(10, 6))
                importance.plot(kind="barh", ax=ax, color="steelblue")
                ax.set_xlabel("Importance")
                st.pyplot(fig)


elif page == "MLOps Dashboard":
    st.title("MLOps Dashboard")

    st.markdown("### MLflow Experiment Runs")

    # Get experiments
    experiments = mlflow.search_experiments()
    exp_names = [exp.name for exp in experiments if not exp.name.startswith("Default")]

    if not exp_names:
        st.warning("No experiments found. Train some models first!")
    else:
        selected_exp = st.selectbox("Select Experiment", exp_names)

        runs = mlflow.search_runs(experiment_names=[selected_exp])

        if runs.empty:
            st.warning("No runs in this experiment.")
        else:
            st.write(f"Found {len(runs)} runs")

            # Display runs table
            display_cols = ["tags.mlflow.runName", "metrics.f1_score", "metrics.accuracy",
                          "metrics.auc_roc", "start_time"]
            display_cols = [c for c in display_cols if c in runs.columns]

            st.dataframe(
                runs[display_cols].sort_values("metrics.f1_score", ascending=False),
                height=300
            )

            # Compare top models
            st.markdown("### Model Comparison")

            if "metrics.f1_score" in runs.columns:
                top_runs = runs.nlargest(5, "metrics.f1_score")

                fig, ax = plt.subplots(figsize=(10, 5))
                metrics_to_plot = ["metrics.accuracy", "metrics.precision",
                                  "metrics.recall", "metrics.f1_score"]
                metrics_to_plot = [m for m in metrics_to_plot if m in top_runs.columns]

                x = np.arange(len(top_runs))
                width = 0.2

                for i, metric in enumerate(metrics_to_plot):
                    ax.bar(x + i*width, top_runs[metric], width,
                          label=metric.replace("metrics.", ""))

                ax.set_xticks(x + width * 1.5)
                ax.set_xticklabels(top_runs["tags.mlflow.runName"].values, rotation=45, ha="right")
                ax.legend()
                ax.set_ylim(0, 1)
                ax.set_ylabel("Score")
                st.pyplot(fig)

            # Best model info
            st.markdown("### Best Model")
            best_run = runs.loc[runs["metrics.f1_score"].idxmax()]

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Run Info**")
                st.write(f"- Name: {best_run.get('tags.mlflow.runName', 'N/A')}")
                st.write(f"- Run ID: {best_run['run_id'][:8]}...")
                st.write(f"- Model Type: {best_run.get('params.model_type', 'N/A')}")

            with col2:
                st.write("**Metrics**")
                st.write(f"- F1 Score: {best_run.get('metrics.f1_score', 0):.4f}")
                st.write(f"- Accuracy: {best_run.get('metrics.accuracy', 0):.4f}")
                st.write(f"- AUC-ROC: {best_run.get('metrics.auc_roc', 0):.4f}")


elif page == "Model Inference":
    st.title("Model Inference")
    st.markdown("### Configure Expedition Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Team")
        total_members = st.slider("Team Members", 1, 30, 8)
        total_hired = st.slider("Hired Personnel", 0, 30, 5)

    with col2:
        st.subheader("Logistics")
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
        season_map = {"Spring": 1, "Summer": 2, "Autumn": 3, "Winter": 4}
        o2_used = st.checkbox("Using Oxygen", value=True)
        camps = st.slider("Number of Camps", 0, 6, 4)

    with col3:
        st.subheader("Peak")
        height_m = st.slider("Peak Height (m)", 5000, 8849, 8000)
        commercial = st.checkbox("Commercial Route", value=True)
        standard = st.checkbox("Standard Route", value=True)

    # Prepare input
    input_data = {
        "season": season_map[season],
        "host": 1,
        "total_members": total_members,
        "total_hired": total_hired,
        "hired_ratio": total_hired / (total_members + 1),
        "o2_used": int(o2_used),
        "o2_climb": int(o2_used),
        "o2_sleep": int(o2_used),
        "camps": camps,
        "rope_fixed": 0,
        "height_m": height_m,
        "height_scaled": height_m / 1000,
        "is_8000er": int(height_m >= 8000),
        "is_trekking_peak": 0,
        "year": 2024,
        "total_days": 30,
        "summit_days": 20,
        "standard_route": int(standard),
        "commercial_route": int(commercial),
        "solo": int(total_members == 1),
        "small_team": int(2 <= total_members <= 5),
        "large_team": int(total_members > 10)
    }

    # Load model and predict
    if st.button("Predict Success", type="primary"):
        # Try to load best model from MLflow
        runs = mlflow.search_runs(experiment_names=["streamlit_training"])
        if runs.empty:
            st.error("No trained models found. Go to ML Training page first!")
        else:
            best_run = runs.loc[runs["metrics.f1_score"].idxmax()]
            run_id = best_run["run_id"]
            model_name = best_run.get("tags.mlflow.runName", "model")
            # Load from run artifacts
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")

                # Predict
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]

            st.markdown("---")
            st.markdown("### Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.success(f"## ✅ SUCCESS LIKELY")
                else:
                    st.error(f"## ❌ FAILURE LIKELY")
            with col2:
                st.metric("Success Probability", f"{probability[1]:.1%}")
                st.metric("Failure Probability", f"{probability[0]:.1%}")


st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard demonstrates workflows "
    "using the Himalayan Expeditions dataset (2020-2024)."
)
