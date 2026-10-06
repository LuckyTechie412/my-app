"""Fit the selected car price model on all available cleaned rows."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from model_utils import TARGET, build_pipeline, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "Cars_data.csv"
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "model_selection.json",
    )
    args = parser.parse_args()

    if not args.selection_file.exists():
        raise FileNotFoundError(
            f"Model selection file not found: {args.selection_file}. Run compare_models.py first."
        )
    selection = json.loads(args.selection_file.read_text(encoding="utf-8"))
    features, target, rows_removed = load_dataset(args.data)
    model = build_pipeline(
        features, selection["selected_model"], selection["selected_params"]
    )
    model.fit(features, target)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "car_price_model.joblib"
    metadata_path = args.artifacts_dir / "model_metadata.json"
    joblib.dump(model, model_path, compress=3)
    metadata = {
        "model_file": model_path.name,
        "target": TARGET,
        "selected_model": selection["selected_model"],
        "selected_params": selection["selected_params"],
        "selection_metric": selection["selection_metric"],
        "holdout_metrics": selection["holdout_metrics"],
        "training_rows": len(features),
        "rows_removed_during_cleaning": rows_removed,
        "feature_columns": features.columns.tolist(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Fitted on {len(features):,} cleaned rows.")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()