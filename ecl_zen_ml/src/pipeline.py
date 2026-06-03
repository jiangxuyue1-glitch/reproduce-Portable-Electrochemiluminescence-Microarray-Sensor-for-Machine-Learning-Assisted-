from pathlib import Path

import pandas as pd

from .config import AppConfig
from .dataset import classification_labels, clean_dataset, feature_matrix, load_csv_dataset, regression_labels
from .features import extract_features_from_image_dataset
from .models import run_gpr_regression, run_knn_classification
from .plots import calibration_analysis, plot_confusion_matrix, plot_example_images, plot_figure5_summary, plot_gpr_predictions, plot_rgb


def run_pipeline(mode: str, input_path: Path, config: AppConfig) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "images":
        print("Extracting image features")
        raw = extract_features_from_image_dataset(input_path, output_dir, config.roi)
    elif mode == "csv":
        print("Loading CSV features")
        raw = load_csv_dataset(input_path)
    else:
        print(f"Invalid mode: {mode}")
        return
    cleaned = clean_dataset(raw, config.model.feature_columns, output_dir)
    if cleaned.empty:
        print("No usable samples found")
        return
    x = feature_matrix(cleaned, config.model.feature_columns)
    sample_ids = cleaned["sample_id"].astype(str).to_numpy()
    print("Running KNN classification")
    knn_metrics = run_knn_classification(
        x,
        classification_labels(cleaned),
        sample_ids,
        output_dir,
        config.model.folds,
        config.model.random_state,
        config.model.knn_neighbors,
        config.model.knn_weights,
    )
    print("Running GPR regression")
    run_gpr_regression(
        x,
        regression_labels(cleaned, config.model.target_log),
        cleaned["concentration_ng_ml"].to_numpy(dtype=float),
        sample_ids,
        output_dir,
        config.model.folds,
        config.model.random_state,
        config.model.target_log,
    )
    print("Running calibration analysis")
    calibration_analysis(cleaned, output_dir)
    if config.make_plots:
        print("Generating plots")
        if knn_metrics:
            plot_confusion_matrix(pd.DataFrame(knn_metrics["confusion_matrix"]).to_numpy(), knn_metrics["labels"], output_dir)
        predictions_path = output_dir / "gpr_predictions.csv"
        if predictions_path.exists():
            plot_gpr_predictions(pd.read_csv(predictions_path), output_dir)
        plot_rgb(cleaned, output_dir)
        plot_example_images(cleaned, output_dir)
        plot_figure5_summary(cleaned, output_dir)
    print(f"Done. Outputs saved in {output_dir}")
