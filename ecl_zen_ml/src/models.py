from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import save_json


def usable_folds(sample_count: int, requested_folds: int) -> int:
    return max(2, min(requested_folds, sample_count))


def classification_cv(y: np.ndarray, folds: int, random_state: int):
    unique, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    if len(unique) > 1 and min_count >= folds:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state), "StratifiedKFold"
    print("Warning: some classes have too few samples for stratification; using KFold")
    return KFold(n_splits=folds, shuffle=True, random_state=random_state), "KFold"


def run_knn_classification(x: np.ndarray, y: np.ndarray, sample_ids: np.ndarray, output_dir: Path, folds: int, random_state: int, neighbors: list[int], weights: list[str]) -> dict:
    if len(y) < 2 or len(np.unique(y)) < 2:
        print("Skipping KNN: at least two classes and two samples are required")
        return {}
    folds = usable_folds(len(y), folds)
    cv, cv_name = classification_cv(y, folds, random_state)
    best_score = -np.inf
    best_params = {}
    best_predictions = None
    for k in neighbors:
        if k > len(y):
            continue
        for weight in weights:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k, weights=weight, metric="euclidean")),
            ])
            predictions = cross_val_predict(model, x, y, cv=cv)
            score = accuracy_score(y, predictions)
            if score > best_score:
                best_score = score
                best_params = {"n_neighbors": k, "weights": weight}
                best_predictions = predictions
    if best_predictions is None:
        print("Skipping KNN: no valid neighbor setting")
        return {}
    labels = sorted(np.unique(y), key=lambda item: float(item))
    report = classification_report(y, best_predictions, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, best_predictions, labels=labels)
    metrics = {
        "author": "jiang xuyue",
        "cv": cv_name,
        "folds": folds,
        "best_params": best_params,
        "accuracy": accuracy_score(y, best_predictions),
        "macro_f1": f1_score(y, best_predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, best_predictions, average="weighted", zero_division=0),
        "classification_report": report,
        "labels": labels,
        "confusion_matrix": matrix,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / "knn_metrics.json")
    pd.DataFrame({"sample_id": sample_ids, "true_label": y, "predicted_label": best_predictions}).to_csv(output_dir / "knn_predictions.csv", index=False)
    return metrics


def gpr_kernels() -> list:
    return [
        ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
        ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2)),
        ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2)),
    ]


def run_gpr_regression(x: np.ndarray, y: np.ndarray, concentrations: np.ndarray, sample_ids: np.ndarray, output_dir: Path, folds: int, random_state: int, target_log: bool) -> dict:
    if len(y) < 2:
        print("Skipping GPR: at least two samples are required")
        return {}
    folds = usable_folds(len(y), folds)
    cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    best_rmse = np.inf
    best_predictions = None
    best_kernel_name = ""
    for kernel in gpr_kernels():
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_state, n_restarts_optimizer=1)),
        ])
        predictions = cross_val_predict(model, x, y, cv=cv)
        rmse = float(np.sqrt(np.mean((y - predictions) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_predictions = predictions
            best_kernel_name = str(kernel)
    if best_predictions is None:
        return {}
    true_concentration = concentrations.astype(float)
    predicted_concentration = np.power(10.0, best_predictions) if target_log else best_predictions
    metrics = {
        "author": "jiang xuyue",
        "folds": folds,
        "target": "log10_concentration" if target_log else "concentration_ng_ml",
        "best_kernel": best_kernel_name,
        "MAE": mean_absolute_error(y, best_predictions),
        "RMSE": best_rmse,
        "R2": r2_score(y, best_predictions) if len(y) > 1 else float("nan"),
        "median_absolute_error": median_absolute_error(y, best_predictions),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / "gpr_metrics.json")
    pd.DataFrame({
        "sample_id": sample_ids,
        "true_target": y,
        "predicted_target": best_predictions,
        "true_concentration_ng_ml": true_concentration,
        "predicted_concentration_ng_ml": predicted_concentration,
    }).to_csv(output_dir / "gpr_predictions.csv", index=False)
    return metrics
