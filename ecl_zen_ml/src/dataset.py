from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_BASE_COLUMNS = ["sample_id", "concentration_ng_ml"]


def load_csv_dataset(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        print(f"CSV file not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read CSV: {exc}")
        return pd.DataFrame()


def clean_dataset(df: pd.DataFrame, feature_columns: list[str], output_dir: Path) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    missing = [column for column in REQUIRED_BASE_COLUMNS + feature_columns if column not in df.columns]
    if missing:
        print(f"Missing required columns: {', '.join(missing)}")
        return pd.DataFrame()
    cleaned = df.copy()
    cleaned["concentration_ng_ml"] = pd.to_numeric(cleaned["concentration_ng_ml"], errors="coerce")
    for column in feature_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["concentration_ng_ml"] + feature_columns)
    cleaned = cleaned[cleaned["concentration_ng_ml"] > 0].copy()
    cleaned["class_label"] = cleaned["concentration_ng_ml"].map(lambda value: f"{value:g}")
    cleaned["log10_concentration"] = np.log10(cleaned["concentration_ng_ml"])
    dropped = before - len(cleaned)
    if dropped:
        print(f"Dropped {dropped} invalid rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_dir / "cleaned_features.csv", index=False)
    print(f"Cleaned dataset contains {len(cleaned)} samples")
    return cleaned


def feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return df[feature_columns].to_numpy(dtype=float)


def classification_labels(df: pd.DataFrame) -> np.ndarray:
    return df["class_label"].to_numpy()


def regression_labels(df: pd.DataFrame, target_log: bool = True) -> np.ndarray:
    column = "log10_concentration" if target_log else "concentration_ng_ml"
    return df[column].to_numpy(dtype=float)
