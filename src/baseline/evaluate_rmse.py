import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMN = "pickup_datetime"
TARGET_COLUMN = "base_passenger_fare"

DEFAULT_PREDICTION_FILES = {
    "Simple Linear Model": "submission_simple_linear_model.csv",
    "LightGBM": "submission_LightGBM.csv",
    "XGBoost": "submission_XGBoost.csv",
    "Random Forest": "submission_Random_Forest.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline submissions with RMSE.")
    parser.add_argument("--truth", default="data/processed/sample_submission.csv")
    parser.add_argument("--results-dir", default="results/baseline")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def load_submission(path):
    df = pd.read_csv(path)
    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    return df[[ID_COLUMN, TARGET_COLUMN]]


def calculate_rmse(y_true, y_pred):
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid_mask.any():
        raise ValueError("No valid rows available for RMSE calculation")
    score = np.sqrt(np.mean((y_pred[valid_mask] - y_true[valid_mask]) ** 2))
    return float(score), int(valid_mask.sum()), int((~valid_mask).sum())


def evaluate_file(truth, prediction_path, model_name):
    prediction = load_submission(prediction_path)

    if len(prediction) != len(truth):
        raise ValueError(
            f"{prediction_path} has {len(prediction)} rows, "
            f"but truth has {len(truth)} rows"
        )

    if not prediction[ID_COLUMN].equals(truth[ID_COLUMN]):
        raise ValueError(f"{prediction_path} does not match truth row order")

    score, evaluated_rows, skipped_rows = calculate_rmse(
        truth[TARGET_COLUMN].to_numpy(dtype=float),
        prediction[TARGET_COLUMN].to_numpy(dtype=float),
    )
    return {
        "model": model_name,
        "file": prediction_path.name,
        "rows": len(prediction),
        "evaluated_rows": evaluated_rows,
        "skipped_rows": skipped_rows,
        "rmse": score,
    }


def main():
    args = parse_args()
    truth = load_submission(Path(args.truth))
    results_dir = Path(args.results_dir)

    records = []
    for model_name, file_name in DEFAULT_PREDICTION_FILES.items():
        records.append(evaluate_file(truth, results_dir / file_name, model_name))

    result = pd.DataFrame(records).sort_values("rmse", ascending=True)
    result.insert(0, "rank", range(1, len(result) + 1))

    print(result.to_string(index=False, formatters={"rmse": "{:.6f}".format}))

    if args.output:
        result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
