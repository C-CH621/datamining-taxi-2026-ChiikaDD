"""Small regression evaluation helpers for submission-style CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMN = "pickup_datetime"
TARGET_COLUMN = "base_passenger_fare"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Return RMSE, MAE, R2 and row counts, skipping non-finite pairs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        raise ValueError("No finite prediction/target pairs to evaluate.")

    yt = y_true[valid]
    yp = y_pred[valid]
    err = yp - yt
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "rows": int(len(y_true)),
        "evaluated_rows": int(valid.sum()),
        "skipped_rows": int((~valid).sum()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def load_target_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {ID_COLUMN, TARGET_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df[[ID_COLUMN, TARGET_COLUMN]]


def evaluate_submission(truth_path: str | Path, prediction_path: str | Path) -> dict[str, float | int]:
    truth = load_target_csv(truth_path)
    pred = load_target_csv(prediction_path)
    if len(truth) != len(pred):
        raise ValueError(f"Row count mismatch: truth={len(truth)}, prediction={len(pred)}")
    if not truth[ID_COLUMN].equals(pred[ID_COLUMN]):
        raise ValueError("Prediction rows are not aligned with truth rows.")
    return regression_metrics(
        truth[TARGET_COLUMN].to_numpy(dtype=float),
        pred[TARGET_COLUMN].to_numpy(dtype=float),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a submission CSV against sample_submission.csv.")
    parser.add_argument("--truth", required=True, help="CSV with ground-truth target values.")
    parser.add_argument("--prediction", required=True, help="Submission/prediction CSV to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_submission(args.truth, args.prediction)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
