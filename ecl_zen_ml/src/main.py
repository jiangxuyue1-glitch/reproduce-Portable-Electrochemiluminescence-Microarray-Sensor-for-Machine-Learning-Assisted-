import argparse
from pathlib import Path

from .config import AppConfig, ModelConfig, RoiConfig
from .pipeline import run_pipeline


def parse_feature_columns(value: str | None) -> list[str] | None:
    if not value:
        return None
    columns = [column.strip() for column in value.split(",") if column.strip()]
    return columns or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the ECL ZEN machine learning workflow")
    parser.add_argument("--mode", choices=["images", "csv"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--feature-columns")
    parser.add_argument("--target-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--roi", choices=["auto", "center"], default="auto")
    parser.add_argument("--threshold-percentile", type=float, default=92.0)
    parser.add_argument("--min-spot-area", type=int, default=25)
    parser.add_argument("--center-crop-ratio", type=float, default=0.45)
    parser.add_argument("--background-margin", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    feature_columns = parse_feature_columns(args.feature_columns)
    model = ModelConfig(
        feature_columns=feature_columns or ModelConfig().feature_columns,
        folds=args.folds,
        random_state=args.random_state,
        target_log=args.target_log,
    )
    roi = RoiConfig(
        threshold_percentile=args.threshold_percentile,
        min_spot_area=args.min_spot_area,
        center_crop_ratio=args.center_crop_ratio,
        use_auto_roi=args.roi == "auto",
        background_margin=args.background_margin,
    )
    config = AppConfig(output_dir=Path(args.output), roi=roi, model=model, make_plots=not args.no_plots)
    run_pipeline(args.mode, Path(args.input), config)


if __name__ == "__main__":
    main()
