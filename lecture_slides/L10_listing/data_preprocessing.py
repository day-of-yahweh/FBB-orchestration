import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


def load_and_merge_data(data_dir: str = "data") -> pd.DataFrame:
    """Load expedition and peaks data, merge them."""
    data_path = Path(data_dir)

    expeditions = pd.read_csv(data_path / "exped_tidy.csv", encoding="latin-1")
    peaks = pd.read_csv(data_path / "peaks_tidy.csv", encoding="latin-1")

    df = expeditions.merge(
        peaks[["PEAKID", "HEIGHTM", "HIMAL_FACTOR", "REGION_FACTOR", "TREKKING"]],
        on="PEAKID",
        how="left"
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for modeling."""
    features = pd.DataFrame()

    # Target: success on primary route
    features["target"] = df["SUCCESS1"].astype(int)

    # Season encoding (Spring=1, Summer=2, Autumn=3, Winter=4)
    features["season"] = df["SEASON"].fillna(1).astype(int)

    # Host country (Nepal=1, China=2)
    features["host"] = df["HOST"].fillna(1).astype(int)

    # Team composition
    features["total_members"] = df["TOTMEMBERS"].fillna(0).astype(int)
    features["total_hired"] = df["TOTHIRED"].fillna(0).astype(int)
    features["hired_ratio"] = features["total_hired"] / (features["total_members"] + 1)

    # Oxygen usage
    features["o2_used"] = df["O2USED"].fillna(False).astype(int)
    features["o2_climb"] = df["O2CLIMB"].fillna(False).astype(int)
    features["o2_sleep"] = df["O2SLEEP"].fillna(False).astype(int)

    # Logistics
    features["camps"] = df["CAMPS"].fillna(0).astype(int)
    features["rope_fixed"] = df["ROPE"].fillna(0).astype(int)

    # Peak characteristics
    features["height_m"] = df["HEIGHTM"].fillna(df["HEIGHTM"].median())
    features["height_scaled"] = features["height_m"] / 1000  # km for easier interpretation
    features["is_8000er"] = (features["height_m"] >= 8000).astype(int)
    features["is_trekking_peak"] = df["TREKKING"].fillna(False).astype(int)

    # Year for temporal patterns
    features["year"] = df["YEAR"].astype(int)

    # Expedition duration (if available)
    features["total_days"] = df["TOTDAYS"].fillna(df["TOTDAYS"].median())
    features["summit_days"] = df["SMTDAYS"].fillna(df["SMTDAYS"].median())

    # Commercial route indicator
    features["standard_route"] = df["STDRTE"].fillna(False).astype(int)
    features["commercial_route"] = df["COMRTE"].fillna(False).astype(int)

    # Team size categories
    features["solo"] = (features["total_members"] == 1).astype(int)
    features["small_team"] = ((features["total_members"] >= 2) & (features["total_members"] <= 5)).astype(int)
    features["large_team"] = (features["total_members"] > 10).astype(int)

    return features


def prepare_data(test_size: float = 0.2, random_state: int = 42):
    """Load, process and split data for modeling."""
    df = load_and_merge_data()
    features = create_features(df)

    # Remove rows with missing target
    features = features.dropna(subset=["target"])

    # Define feature columns (exclude target)
    feature_cols = [col for col in features.columns if col != "target"]

    X = features[feature_cols].fillna(0)
    y = features["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols


def prepare_scaled_data(test_size: float = 0.2, random_state: int = 42):
    """Prepare data with standard scaling for algorithms that need it."""
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(test_size, random_state)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, cols = prepare_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(cols)}")
    print(f"Target distribution (train): {y_train.value_counts().to_dict()}")
    print(f"\nFeature columns:\n{cols}")
