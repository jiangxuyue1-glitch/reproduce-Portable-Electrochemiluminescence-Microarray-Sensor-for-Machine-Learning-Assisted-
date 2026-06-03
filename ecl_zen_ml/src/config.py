from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RoiConfig:
    threshold_percentile: float = 92.0
    min_spot_area: int = 25
    center_crop_ratio: float = 0.45
    use_auto_roi: bool = True
    background_margin: int = 12


@dataclass
class ModelConfig:
    feature_columns: list[str] = field(default_factory=lambda: [
        "R_mean",
        "G_mean",
        "B_mean",
        "R_std",
        "G_std",
        "B_std",
        "intensity_mean",
        "intensity_std",
    ])
    folds: int = 3
    random_state: int = 42
    target_log: bool = True
    knn_neighbors: list[int] = field(default_factory=lambda: [1, 3, 5, 7, 9])
    knn_weights: list[str] = field(default_factory=lambda: ["uniform", "distance"])


@dataclass
class AppConfig:
    output_dir: Path = Path("outputs")
    roi: RoiConfig = field(default_factory=RoiConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    make_plots: bool = True
