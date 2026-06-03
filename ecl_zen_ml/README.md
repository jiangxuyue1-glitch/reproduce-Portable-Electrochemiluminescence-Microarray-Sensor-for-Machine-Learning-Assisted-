# ECL ZEN ML Workflow Reproduction

Author: jiang xuyue

This project reproduces the computational workflow from **Portable Electrochemiluminescence Microarray Sensor for Machine Learning-Assisted Quantitative Analysis of Zearalenone**. It focuses only on image feature extraction, concentration classification, regression, calibration analysis, and plot generation.

The wet-lab ECL sensor construction, microarray fabrication, chemistry, smartphone acquisition protocol, and hidden preprocessing details from the paper are not reproduced. Exact paper results cannot be guaranteed without the original image dataset and preprocessing settings. Users should tune ROI parameters and model parameters for their own dataset.

## Reproduced workflow

1. Load smartphone ECL images or pre-extracted feature CSV files.
2. Extract RGB and intensity features from valid luminescent regions.
3. Classify concentration classes with KNN and 3-fold cross-validation.
4. Predict continuous ZEN concentration with Gaussian Process Regression and 3-fold cross-validation.
5. Fit a calibration curve using intensity mean versus log10 concentration.
6. Save metrics, predictions, RGB plots, calibration plots, confusion matrix, predicted-vs-true plots, Bland-Altman plot, and a Figure 5 style summary.

## Not reproduced

- Wet-lab ECL sensor construction.
- Sample preparation and toxin incubation.
- Microarray hardware fabrication.
- Smartphone optical setup.
- Original private dataset and hidden preprocessing details.

## Installation

```bash
cd ecl_zen_ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd ecl_zen_ml
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Image dataset format

Place images in concentration-named folders. Folder names are parsed as ZEN concentrations in ng/mL.

```text
data/images/
0.0001/sample_001.png
0.0005/sample_001.png
0.001/sample_001.png
0.005/sample_001.png
0.01/sample_001.png
0.05/sample_001.png
0.1/sample_001.png
0.5/sample_001.png
1/sample_001.png
10/sample_001.png
50/sample_001.png
100/sample_001.png
```

Supported image extensions are PNG, JPG, JPEG, TIF, TIFF, and BMP.

## CSV dataset format

Required columns:

```text
sample_id, concentration_ng_ml, R_mean, G_mean, B_mean, R_std, G_std, B_std, intensity_mean, intensity_std
```

Optional columns:

```text
spot_area, background_mean, signal_to_background_ratio, image_path
```

Rows with missing required values, non-numeric features, or non-positive concentrations are dropped.

## Run image mode

```bash
python -m src.main --mode images --input data/images --output outputs
```

## Run CSV mode

```bash
python -m src.main --mode csv --input data/features.csv --output outputs
```

## Useful options

```bash
python -m src.main --mode images --input data/images --output outputs --roi center
python -m src.main --mode csv --input data/features.csv --feature-columns R_mean,G_mean,B_mean,intensity_mean --folds 3
python -m src.main --mode images --input data/images --no-plots
```

ROI settings can be changed with:

- `--roi auto`
- `--roi center`
- `--threshold-percentile`
- `--min-spot-area`
- `--center-crop-ratio`
- `--background-margin`

Model settings can be changed with:

- `--feature-columns`
- `--target-log` or `--no-target-log`
- `--random-state`
- `--folds`

## Output files

All outputs are saved in the selected output folder.

| File | Description |
| --- | --- |
| `extracted_features.csv` | Image-derived RGB, intensity, ROI, and background features |
| `cleaned_features.csv` | Valid rows used by the workflow |
| `knn_metrics.json` | KNN accuracy, F1 scores, per-class metrics, labels, and confusion matrix |
| `knn_predictions.csv` | Cross-validation KNN predictions |
| `knn_confusion_matrix.png` | KNN confusion matrix |
| `gpr_metrics.json` | GPR MAE, RMSE, R2, median absolute error, and selected kernel |
| `gpr_predictions.csv` | Cross-validation GPR predictions in target and concentration scales |
| `gpr_predicted_vs_true_log.png` | Predicted versus true log10 concentration |
| `gpr_predicted_vs_true_concentration.png` | Predicted versus true concentration |
| `bland_altman_plot.png` | Bland-Altman concentration agreement plot |
| `calibration_metrics.json` | Linear calibration slope, intercept, R2, and signal-on or signal-off relationship |
| `calibration_curve.png` | Intensity mean versus log10 concentration calibration curve |
| `rgb_vs_concentration.png` | RGB channel means versus concentration |
| `rgb_vs_log_concentration.png` | RGB channel means versus log10 concentration |
| `rgb_heatmap.png` | RGB heatmap with samples as rows |
| `rgb_channel_heatmaps.png` | Separate R, G, and B channel heatmaps |
| `example_ecl_images.png` | Representative image panel when image paths are available |
| `figure5_style_summary.png` | Figure 5 style computational workflow summary |

## Limitations

Auto ROI detection uses a percentile threshold and largest valid contour. If images have uneven backgrounds, reflections, multiple spots, or very dim luminescence, center crop ROI may be more stable. The paper reports a signal-off relationship, but this project fits the supplied data and reports the observed slope without forcing a negative calibration trend.
