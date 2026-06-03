from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .features import read_rgb_image
from .metrics import save_json


def save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], output_dir: Path) -> None:
    if matrix.size == 0:
        return
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted concentration")
    plt.ylabel("True concentration")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, int(matrix[row, col]), ha="center", va="center")
    save_current(output_dir / "knn_confusion_matrix.png")


def plot_gpr_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    if predictions.empty:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(predictions["true_target"], predictions["predicted_target"], alpha=0.8)
    low = min(predictions["true_target"].min(), predictions["predicted_target"].min())
    high = max(predictions["true_target"].max(), predictions["predicted_target"].max())
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.xlabel("True log10 concentration")
    plt.ylabel("Predicted log10 concentration")
    save_current(output_dir / "gpr_predicted_vs_true_log.png")
    plt.figure(figsize=(6, 5))
    plt.scatter(predictions["true_concentration_ng_ml"], predictions["predicted_concentration_ng_ml"], alpha=0.8)
    low = min(predictions["true_concentration_ng_ml"].min(), predictions["predicted_concentration_ng_ml"].min())
    high = max(predictions["true_concentration_ng_ml"].max(), predictions["predicted_concentration_ng_ml"].max())
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("True concentration ng/mL")
    plt.ylabel("Predicted concentration ng/mL")
    save_current(output_dir / "gpr_predicted_vs_true_concentration.png")
    true_values = predictions["true_concentration_ng_ml"].to_numpy(dtype=float)
    predicted_values = predictions["predicted_concentration_ng_ml"].to_numpy(dtype=float)
    means = (true_values + predicted_values) / 2
    differences = predicted_values - true_values
    mean_diff = differences.mean()
    sd_diff = differences.std(ddof=1) if len(differences) > 1 else 0.0
    plt.figure(figsize=(6, 5))
    plt.scatter(means, differences, alpha=0.8)
    plt.axhline(mean_diff, color="black", linewidth=1)
    plt.axhline(mean_diff + 1.96 * sd_diff, color="gray", linestyle="--")
    plt.axhline(mean_diff - 1.96 * sd_diff, color="gray", linestyle="--")
    plt.xscale("log")
    plt.xlabel("Mean concentration ng/mL")
    plt.ylabel("Predicted - true concentration")
    save_current(output_dir / "bland_altman_plot.png")


def calibration_analysis(df: pd.DataFrame, output_dir: Path) -> dict:
    if df.empty or "intensity_mean" not in df.columns:
        return {}
    x = df[["log10_concentration"]].to_numpy(dtype=float)
    y = df["intensity_mean"].to_numpy(dtype=float)
    if len(df) < 2:
        return {}
    model = LinearRegression().fit(x, y)
    predicted = model.predict(x)
    slope = float(model.coef_[0])
    metrics = {
        "author": "jiang xuyue",
        "slope": slope,
        "intercept": float(model.intercept_),
        "R2": float(r2_score(y, predicted)),
        "relationship": "signal-off" if slope < 0 else "signal-on",
    }
    save_json(metrics, output_dir / "calibration_metrics.json")
    ordered = df.sort_values("log10_concentration")
    plt.figure(figsize=(6, 5))
    plt.scatter(df["log10_concentration"], df["intensity_mean"], alpha=0.8)
    line_x = ordered[["log10_concentration"]].to_numpy(dtype=float)
    plt.plot(line_x.ravel(), model.predict(line_x), color="black", linewidth=1)
    plt.xlabel("log10 concentration ng/mL")
    plt.ylabel("Intensity mean")
    save_current(output_dir / "calibration_curve.png")
    return metrics


def plot_rgb(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    grouped = df.groupby("concentration_ng_ml")[["R_mean", "G_mean", "B_mean"]].mean().reset_index().sort_values("concentration_ng_ml")
    colors = {"R_mean": "red", "G_mean": "green", "B_mean": "blue"}
    plt.figure(figsize=(7, 5))
    for column, color in colors.items():
        plt.plot(grouped["concentration_ng_ml"], grouped[column], marker="o", label=column, color=color)
    plt.xscale("log")
    plt.xlabel("Concentration ng/mL")
    plt.ylabel("Mean channel value")
    plt.legend()
    save_current(output_dir / "rgb_vs_concentration.png")
    plt.figure(figsize=(7, 5))
    for column, color in colors.items():
        plt.plot(np.log10(grouped["concentration_ng_ml"]), grouped[column], marker="o", label=column, color=color)
    plt.xlabel("log10 concentration ng/mL")
    plt.ylabel("Mean channel value")
    plt.legend()
    save_current(output_dir / "rgb_vs_log_concentration.png")
    ordered = df.sort_values(["concentration_ng_ml", "sample_id"])
    rgb_values = ordered[["R_mean", "G_mean", "B_mean"]].to_numpy(dtype=float)
    plt.figure(figsize=(5, max(4, len(ordered) * 0.18)))
    plt.imshow(rgb_values, aspect="auto", cmap="viridis")
    plt.xticks(range(3), ["R", "G", "B"])
    plt.yticks(range(len(ordered)), ordered["sample_id"].astype(str), fontsize=6)
    plt.colorbar(label="Mean value")
    save_current(output_dir / "rgb_heatmap.png")
    fig, axes = plt.subplots(1, 3, figsize=(9, max(4, len(ordered) * 0.18)), sharey=True)
    for axis, column, title in zip(axes, ["R_mean", "G_mean", "B_mean"], ["R", "G", "B"]):
        axis.imshow(ordered[[column]].to_numpy(dtype=float), aspect="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xticks([])
    axes[0].set_yticks(range(len(ordered)))
    axes[0].set_yticklabels(ordered["sample_id"].astype(str), fontsize=6)
    save_current(output_dir / "rgb_channel_heatmaps.png")


def plot_example_images(df: pd.DataFrame, output_dir: Path) -> bool:
    if df.empty or "image_path" not in df.columns:
        return False
    rows = df.dropna(subset=["image_path"]).sort_values("concentration_ng_ml").groupby("concentration_ng_ml").head(1)
    rows = rows[rows["image_path"].map(lambda value: Path(str(value)).exists())]
    if rows.empty:
        return False
    count = len(rows)
    cols = min(4, count)
    rows_count = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 2.4, rows_count * 2.4))
    axes_array = np.array(axes).reshape(-1)
    for axis, (_, row) in zip(axes_array, rows.iterrows()):
        image = read_rgb_image(Path(str(row["image_path"])))
        if image is not None:
            axis.imshow(image)
        axis.set_title(f'{row["concentration_ng_ml"]:g} ng/mL')
        axis.axis("off")
    for axis in axes_array[len(rows):]:
        axis.axis("off")
    save_current(output_dir / "example_ecl_images.png")
    return True


def plot_figure5_summary(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(grid[0, 0])
    if "image_path" in df.columns:
        candidates = df.dropna(subset=["image_path"]).sort_values("concentration_ng_ml")
        image_row = next((row for _, row in candidates.iterrows() if Path(str(row["image_path"])).exists()), None)
        if image_row is not None:
            image = read_rgb_image(Path(str(image_row["image_path"])))
            if image is not None:
                ax_a.imshow(image)
                ax_a.set_title("Panel A: representative ECL image")
            else:
                ax_a.text(0.5, 0.5, "Panel A skipped", ha="center")
        else:
            ax_a.text(0.5, 0.5, "Panel A skipped", ha="center")
    else:
        ax_a.text(0.5, 0.5, "Panel A skipped", ha="center")
    ax_a.axis("off")
    ax_b = fig.add_subplot(grid[0, 1])
    ordered = df.sort_values(["concentration_ng_ml", "sample_id"])
    ax_b.imshow(ordered[["R_mean", "G_mean", "B_mean"]].to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax_b.set_title("Panel B: RGB heatmap")
    ax_b.set_xticks(range(3), ["R", "G", "B"])
    ax_c = fig.add_subplot(grid[1, 0])
    grouped = df.groupby("concentration_ng_ml")[["R_mean", "G_mean", "B_mean"]].mean().reset_index().sort_values("concentration_ng_ml")
    for column, color in [("R_mean", "red"), ("G_mean", "green"), ("B_mean", "blue")]:
        ax_c.plot(grouped["concentration_ng_ml"], grouped[column], marker="o", label=column, color=color)
    ax_c.set_xscale("log")
    ax_c.set_title("Panel C: RGB values vs concentration")
    ax_c.set_xlabel("Concentration ng/mL")
    ax_c.legend()
    ax_d = fig.add_subplot(grid[1, 1])
    ax_d.axis("off")
    ax_d.set_title("Panel D: workflow")
    workflow = "Smartphone image\nRGB extraction\nKNN classification\nGPR prediction\nMetrics and plots"
    ax_d.text(0.5, 0.5, workflow, ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "black"})
    save_current(output_dir / "figure5_style_summary.png")
