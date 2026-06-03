from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import RoiConfig

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_rgb_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def center_crop_mask(shape: tuple[int, int], ratio: float) -> np.ndarray:
    height, width = shape
    crop_height = max(1, int(height * ratio))
    crop_width = max(1, int(width * ratio))
    top = max(0, (height - crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    mask = np.zeros((height, width), dtype=bool)
    mask[top:top + crop_height, left:left + crop_width] = True
    return mask


def auto_roi_mask(rgb_image: np.ndarray, config: RoiConfig) -> np.ndarray | None:
    intensity = rgb_image.mean(axis=2).astype(np.float32)
    threshold = np.percentile(intensity, config.threshold_percentile)
    binary = (intensity >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= config.min_spot_area]
    if not valid_contours:
        return None
    largest = max(valid_contours, key=cv2.contourArea)
    mask = np.zeros(intensity.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    return mask.astype(bool)


def background_mask(roi_mask: np.ndarray, margin: int) -> np.ndarray:
    kernel_size = max(3, margin * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(roi_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return np.logical_and(dilated, ~roi_mask)


def extract_image_features(image_path: Path, config: RoiConfig) -> dict[str, float | str] | None:
    rgb_image = read_rgb_image(image_path)
    if rgb_image is None or rgb_image.size == 0:
        print(f"Skipping unreadable image: {image_path}")
        return None
    roi_mask = auto_roi_mask(rgb_image, config) if config.use_auto_roi else None
    if roi_mask is None or int(roi_mask.sum()) < config.min_spot_area:
        roi_mask = center_crop_mask(rgb_image.shape[:2], config.center_crop_ratio)
    roi_pixels = rgb_image[roi_mask]
    if roi_pixels.size == 0:
        print(f"Skipping empty ROI: {image_path}")
        return None
    intensity = roi_pixels.mean(axis=1)
    bg_mask = background_mask(roi_mask, config.background_margin)
    bg_pixels = rgb_image[bg_mask]
    background_mean = float(bg_pixels.mean()) if bg_pixels.size else float(np.mean(rgb_image[~roi_mask])) if np.any(~roi_mask) else 0.0
    intensity_mean = float(intensity.mean())
    signal_to_background_ratio = float(intensity_mean / background_mean) if background_mean > 0 else float("nan")
    channel_mean = roi_pixels.mean(axis=0)
    channel_std = roi_pixels.std(axis=0)
    return {
        "sample_id": image_path.stem,
        "image_path": str(image_path),
        "R_mean": float(channel_mean[0]),
        "G_mean": float(channel_mean[1]),
        "B_mean": float(channel_mean[2]),
        "R_std": float(channel_std[0]),
        "G_std": float(channel_std[1]),
        "B_std": float(channel_std[2]),
        "intensity_mean": intensity_mean,
        "intensity_std": float(intensity.std()),
        "spot_area": int(roi_mask.sum()),
        "background_mean": background_mean,
        "signal_to_background_ratio": signal_to_background_ratio,
    }


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def extract_features_from_image_dataset(input_dir: Path, output_dir: Path, config: RoiConfig) -> pd.DataFrame:
    rows = []
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Image folder not found: {input_dir}")
        return pd.DataFrame()
    for concentration_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        try:
            concentration = float(concentration_dir.name)
        except ValueError:
            print(f"Skipping folder with invalid concentration: {concentration_dir.name}")
            continue
        if concentration <= 0:
            print(f"Skipping non-positive concentration folder: {concentration_dir.name}")
            continue
        for image_path in image_files(concentration_dir):
            features = extract_image_features(image_path, config)
            if features:
                features["concentration_ng_ml"] = concentration
                rows.append(features)
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "extracted_features.csv", index=False)
    print(f"Extracted features from {len(df)} images")
    return df
