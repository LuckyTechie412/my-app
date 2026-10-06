"""Train the selected ExtraTrees model without a preprocessing pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main() -> None:
    folder = Path(__file__).resolve().parent
    data_path = folder.parent / "data" / "Cars_data.csv"
    output_path = folder / "artifacts" / "car_price_model_simple.joblib"

    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()
    data = data.dropna(subset=["MSRP"]).drop_duplicates()
    features = pd.get_dummies(data.drop(columns="MSRP")).fillna(0)
    target = data["MSRP"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, np.log1p(y_train))
    predictions = np.maximum(np.expm1(model.predict(x_test)), 0)

    joblib.dump(model, output_path, compress=3)
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):,.2f}")
    print(f"Test MAE: {mean_absolute_error(y_test, predictions):,.2f}")
    print(f"Test R²: {r2_score(y_test, predictions):.3f}")
    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()