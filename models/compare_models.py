"""Compare car price regressors on a reproducible hold-out split."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from model_utils import MODEL_CANDIDATES, RANDOM_STATE, TARGET, build_pipeline, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "Cars_data.csv"
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    args = parser.parse_args()

    features, target, rows_removed = load_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    results = []
    for candidate in MODEL_CANDIDATES:
        model = build_pipeline(features, candidate["name"], candidate["params"])
        model.fit(x_train, y_train)
        predictions = np.maximum(model.predict(x_test), 0)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        results.append(
            {
                "model": candidate["name"],
                "params": candidate["params"],
                "rmse": rmse,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "r2": float(r2_score(y_test, predictions)),
            }
        )

    results.sort(key=lambda result: result["rmse"])
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.artifacts_dir / "model_comparison.csv"
    selection_path = args.artifacts_dir / "model_selection.json"
    pd.DataFrame(results).to_csv(results_path, index=False)

    winner = results[0]
    selection = {
        "selected_model": winner["model"],
        "selected_params": winner["params"],
        "selection_metric": "holdout_rmse",
        "holdout_metrics": {key: winner[key] for key in ("rmse", "mae", "r2")},
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "target": TARGET,
        "training_rows_after_deduplication": len(features),
        "exact_duplicate_or_missing_target_rows_removed": rows_removed,
    }
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")

    print(pd.DataFrame(results).to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"\nSelected: {winner['model']} {winner['params']}")
    print("The hold-out split selected the winner; its score is not an independent final test estimate.")
    print(f"Comparison saved to: {results_path}")
    print(f"Selection saved to: {selection_path}")


if __name__ == "__main__":
    main()