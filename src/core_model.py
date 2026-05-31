"""Core advanced model for taxi fare prediction.

The baseline scripts in ``src/baseline`` focus on one model per file and write a
submission directly.  This module keeps the same prediction-format input
(``train.csv``, ``test.csv``, ``sample_submission.csv``), adds a validation
split, writes reproducible metrics, and uses a stronger gradient-boosting model
from scikit-learn so it can run with the dependencies already declared by the
project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from src.baseline.model_utils import (
        add_datetime_features,
        add_haversine_distance,
        clean_train_data,
        resolve_input_files,
        write_submission,
    )
except ModuleNotFoundError:  # Allows ``python src/core_model.py`` from repo root.
    from baseline.model_utils import (  # type: ignore
        add_datetime_features,
        add_haversine_distance,
        clean_train_data,
        resolve_input_files,
        write_submission,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "new-york-city-taxi-fare-prediction"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "core_model" / "submission_core_model.csv"
DEFAULT_METRICS_OUTPUT = PROJECT_ROOT / "results" / "core_model" / "core_model_metrics.json"

TAXI_FARE_FEATURE_COLUMNS = {
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
}
TARGET_CANDIDATES = ("fare_amount", "base_passenger_fare", "total_amount")
ID_CANDIDATES = ("key", "pickup_datetime", "trip_id", "id")
DATETIME_HINTS = ("datetime", "date", "time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the advanced HistGradientBoosting core model."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing train.csv, test.csv, and optionally sample_submission.csv.",
    )
    parser.add_argument("--train", type=Path, help="Override path to train.csv.")
    parser.add_argument("--test", type=Path, help="Override path to test.csv.")
    parser.add_argument(
        "--sample-submission",
        type=Path,
        help="Override path to sample_submission.csv for id/target column names.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--nrows", type=int, default=1_000_000)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--target-column",
        help="Target column. If omitted, one of fare_amount/base_passenger_fare/total_amount is used.",
    )
    parser.add_argument(
        "--id-column",
        help="Identifier column used in submission. If omitted, sample_submission or common id names are used.",
    )
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    return parser.parse_args()


def _read_prediction_format(
    input_dir: Path,
    train_path: Path | None,
    test_path: Path | None,
    sample_submission_path: Path | None,
    nrows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    files = resolve_input_files(input_dir)
    resolved_train = train_path or files["train"]
    resolved_test = test_path or files["test"]
    resolved_sample = sample_submission_path or files["sample_submission"]

    if not resolved_train.exists():
        raise FileNotFoundError(f"Training file not found: {resolved_train}")
    if not resolved_test.exists():
        raise FileNotFoundError(f"Test file not found: {resolved_test}")

    train = pd.read_csv(resolved_train, nrows=nrows)
    test = pd.read_csv(resolved_test)
    return train, test, resolved_sample if resolved_sample.exists() else None


def _first_present(candidates: Iterable[str], columns: Iterable[str]) -> str | None:
    column_set = set(columns)
    return next((candidate for candidate in candidates if candidate in column_set), None)


def infer_target_column(
    train: pd.DataFrame,
    test: pd.DataFrame,
    requested_target: str | None,
    sample_submission_path: Path | None,
) -> str:
    if requested_target:
        if requested_target not in train.columns:
            raise ValueError(f"Target column not found in train data: {requested_target}")
        return requested_target

    target = _first_present(TARGET_CANDIDATES, train.columns)
    if target:
        return target

    if sample_submission_path is not None:
        sample_columns = list(pd.read_csv(sample_submission_path, nrows=0).columns)
        if len(sample_columns) >= 2 and sample_columns[1] in train.columns:
            return sample_columns[1]

    train_only_columns = [column for column in train.columns if column not in test.columns]
    numeric_train_only = [
        column
        for column in train_only_columns
        if pd.api.types.is_numeric_dtype(train[column])
    ]
    if len(numeric_train_only) == 1:
        return numeric_train_only[0]

    raise ValueError(
        "Could not infer target column. Pass --target-column explicitly."
    )


def infer_id_column(
    test: pd.DataFrame,
    requested_id: str | None,
    sample_submission_path: Path | None,
) -> str:
    if requested_id:
        if requested_id not in test.columns:
            raise ValueError(f"ID column not found in test data: {requested_id}")
        return requested_id

    if sample_submission_path is not None:
        sample_columns = list(pd.read_csv(sample_submission_path, nrows=0).columns)
        if sample_columns and sample_columns[0] in test.columns:
            return sample_columns[0]

    id_column = _first_present(ID_CANDIDATES, test.columns)
    if id_column:
        return id_column

    return test.columns[0]


def _is_kaggle_taxi_fare(train: pd.DataFrame, target_column: str) -> bool:
    return target_column == "fare_amount" and TAXI_FARE_FEATURE_COLUMNS.issubset(
        train.columns
    )


def prepare_kaggle_taxi_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = train.copy()
    test = test.copy()

    train = add_haversine_distance(train)
    test = add_haversine_distance(test)
    train = add_datetime_features(train)
    test = add_datetime_features(test)
    train = clean_train_data(train)

    test_keys = test["key"].copy()
    train = train.drop(["key", "pickup_datetime"], axis=1)
    test = test.drop(["key", "pickup_datetime"], axis=1)

    x_train = train.drop("fare_amount", axis=1)
    y_train = train["fare_amount"]
    return x_train, y_train, test, test_keys


def add_generic_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in list(df.columns):
        lowered = column.lower()
        if not any(hint in lowered for hint in DATETIME_HINTS):
            continue
        converted = pd.to_datetime(df[column], errors="coerce")
        if converted.notna().mean() < 0.5:
            continue
        df[f"{column}_year"] = converted.dt.year
        df[f"{column}_month"] = converted.dt.month
        df[f"{column}_day"] = converted.dt.day
        df[f"{column}_dayofweek"] = converted.dt.dayofweek
        df[f"{column}_hour"] = converted.dt.hour
        df[f"{column}_minute"] = converted.dt.minute
        df = df.drop(columns=[column])
    return df


def prepare_generic_tabular_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = train.copy()
    test = test.copy()
    test_keys = test[id_column].copy()

    train = train.dropna(subset=[target_column])
    y_train = pd.to_numeric(train[target_column], errors="coerce")
    valid_target = y_train.notna()
    train = train.loc[valid_target].drop(columns=[target_column])
    y_train = y_train.loc[valid_target]

    drop_columns = [column for column in (id_column,) if column in train.columns]
    train = train.drop(columns=drop_columns)
    test = test.drop(columns=drop_columns)

    train = add_generic_datetime_features(train)
    test = add_generic_datetime_features(test)

    train, test = train.align(test, join="outer", axis=1)
    return train, y_train, test, test_keys


def build_core_model(
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    l2_regularization: float,
    random_state: int,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=random_state,
    )


def build_generic_pipeline(
    x_train: pd.DataFrame,
    model: HistGradientBoostingRegressor,
) -> Pipeline:
    numeric_features = [
        column for column in x_train.columns if pd.api.types.is_numeric_dtype(x_train[column])
    ]
    categorical_features = [
        column for column in x_train.columns if column not in numeric_features
    ]

    try:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=20,
            sparse_output=False,
        )
    except TypeError:  # scikit-learn < 1.2
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=20,
            sparse=False,
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", categorical_encoder),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate_predictions(y_true: pd.Series | np.ndarray, predictions: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_submission(keys: pd.Series, predictions: np.ndarray, target_column: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if target_column == "fare_amount":
        write_submission(keys, predictions, output)
        return
    submission = pd.DataFrame({keys.name or "id": keys, target_column: predictions})
    submission.to_csv(output, index=False)


def run_core_model(args: argparse.Namespace) -> dict:
    train, test, sample_submission_path = _read_prediction_format(
        input_dir=args.input_dir,
        train_path=args.train,
        test_path=args.test,
        sample_submission_path=args.sample_submission,
        nrows=args.nrows,
    )
    target_column = infer_target_column(
        train=train,
        test=test,
        requested_target=args.target_column,
        sample_submission_path=sample_submission_path,
    )
    id_column = infer_id_column(
        test=test,
        requested_id=args.id_column,
        sample_submission_path=sample_submission_path,
    )

    model = build_core_model(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        random_state=args.random_state,
    )

    if _is_kaggle_taxi_fare(train, target_column):
        x_train, y_train, x_test, test_keys = prepare_kaggle_taxi_features(train, test)
        estimator = model
        feature_mode = "kaggle_taxi_fare"
    else:
        x_train, y_train, x_test, test_keys = prepare_generic_tabular_features(
            train=train,
            test=test,
            target_column=target_column,
            id_column=id_column,
        )
        estimator = build_generic_pipeline(x_train, model)
        feature_mode = "generic_tabular"

    if not 0 < args.valid_size < 1:
        raise ValueError("--valid-size must be between 0 and 1")

    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=args.valid_size,
        random_state=args.random_state,
    )
    estimator.fit(x_fit, y_fit)
    valid_predictions = estimator.predict(x_valid)
    metrics = evaluate_predictions(y_valid, valid_predictions)

    estimator.fit(x_train, y_train)
    test_predictions = np.maximum(estimator.predict(x_test), 0)
    save_submission(test_keys, test_predictions, target_column, args.output)

    metrics.update(
        {
            "model": "HistGradientBoostingRegressor",
            "feature_mode": feature_mode,
            "target_column": target_column,
            "id_column": id_column,
            "train_rows_after_cleaning": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "validation_size": float(args.valid_size),
            "random_state": int(args.random_state),
            "submission_path": str(args.output),
        }
    )
    save_metrics(metrics, args.metrics_output)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_core_model(args)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
