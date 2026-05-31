"""Core advanced model for taxi fare prediction.

The baseline scripts in ``src/baseline`` focus on one model per file and write a
submission directly.  This module keeps the same prediction-format input
(``train.csv``, ``test.csv``, ``sample_submission.csv``), adds a validation
split, writes reproducible metrics, and uses LightGBM with dataset-specific
features for NYC FHVHV fare prediction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".matplotlib"))
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
FHVHV_CATEGORICAL_COLUMNS = (
    "hvfhs_license_num",
    "dispatching_base_num",
    "originating_base_num",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
)
FHVHV_DATETIME_COLUMNS = (
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
)
FHVHV_FLAG_COLUMNS = (
    "flag_fare_negative",
    "flag_driver_pay_negative",
    "flag_trip_miles_negative",
    "flag_trip_time_negative",
)
FHVHV_DROP_DATETIME_COLUMNS = (
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
)


def resolve_input_files(input_dir: Path) -> dict[str, Path]:
    input_dir = Path(input_dir)
    return {
        "train": input_dir / "train.csv",
        "test": input_dir / "test.csv",
        "sample_submission": input_dir / "sample_submission.csv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the advanced LightGBM core model."
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
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=30)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--subsample-freq", type=int, default=1)
    parser.add_argument("--colsample-bytree", type=float, default=0.95)
    parser.add_argument("--reg-alpha", type=float, default=0.5)
    parser.add_argument("--reg-lambda", type=float, default=0.05)
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="LightGBM device type. Use gpu only when LightGBM was built with GPU support.",
    )
    parser.add_argument("--early-stopping-rounds", type=int, default=0)
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
    try:
        from src.baseline_kaggle.model_utils import (
            add_datetime_features,
            add_haversine_distance,
            clean_train_data,
        )
    except ModuleNotFoundError:  # Allows ``python src/core_model.py`` from repo root.
        from baseline_kaggle.model_utils import (  # type: ignore
            add_datetime_features,
            add_haversine_distance,
            clean_train_data,
        )

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


def add_fhvhv_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed_datetimes: dict[str, pd.Series] = {}
    for column in FHVHV_DATETIME_COLUMNS:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        parsed_datetimes[column] = parsed
        df[f"{column}_hour"] = parsed.dt.hour
        df[f"{column}_day"] = parsed.dt.day
        df[f"{column}_dayofweek"] = parsed.dt.dayofweek
        df[f"{column}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype("float")

    if {"pickup_datetime", "request_datetime"}.issubset(parsed_datetimes):
        df["request_to_pickup_seconds"] = (
            parsed_datetimes["pickup_datetime"] - parsed_datetimes["request_datetime"]
        ).dt.total_seconds()
    if {"on_scene_datetime", "request_datetime"}.issubset(parsed_datetimes):
        df["request_to_scene_seconds"] = (
            parsed_datetimes["on_scene_datetime"] - parsed_datetimes["request_datetime"]
        ).dt.total_seconds()
    if {"dropoff_datetime", "pickup_datetime"}.issubset(parsed_datetimes):
        df["pickup_to_dropoff_seconds"] = (
            parsed_datetimes["dropoff_datetime"] - parsed_datetimes["pickup_datetime"]
        ).dt.total_seconds()

    if {"PULocationID", "DOLocationID"}.issubset(df.columns):
        df["same_zone"] = (df["PULocationID"] == df["DOLocationID"]).astype("float")
        df["route"] = (
            df["PULocationID"].astype("Int64").astype(str)
            + "_"
            + df["DOLocationID"].astype("Int64").astype(str)
        )

    if {"trip_miles", "trip_time"}.issubset(df.columns):
        duration_hours = df["trip_time"].replace(0, np.nan) / 3600
        df["computed_speed_mph"] = df["trip_miles"] / duration_hours
        df["log_trip_miles"] = np.log1p(df["trip_miles"].clip(lower=0))
        df["log_trip_time"] = np.log1p(df["trip_time"].clip(lower=0))

    fee_columns = [
        column
        for column in (
            "tolls",
            "bcf",
            "sales_tax",
            "congestion_surcharge",
            "airport_fee",
            "tips",
            "cbd_congestion_fee",
        )
        if column in df.columns
    ]
    if fee_columns:
        df["known_extra_fees"] = df[fee_columns].sum(axis=1)
    if {"driver_pay", "trip_miles"}.issubset(df.columns):
        df["driver_pay_per_mile"] = df["driver_pay"] / df["trip_miles"].replace(0, np.nan)
    if {"driver_pay", "trip_time"}.issubset(df.columns):
        df["driver_pay_per_minute"] = df["driver_pay"] / (
            df["trip_time"].replace(0, np.nan) / 60
        )

    return df.drop(columns=[c for c in FHVHV_DATETIME_COLUMNS if c in df.columns])


def add_baseline_fhvhv_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed_datetimes: dict[str, pd.Series] = {}
    for column in FHVHV_DATETIME_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_datetime(df[column], errors="coerce")
        parsed_datetimes[column] = values
        df[f"{column}_hour"] = values.dt.hour
        df[f"{column}_dayofweek"] = values.dt.dayofweek
        df[f"{column}_day"] = values.dt.day
        df[f"{column}_month"] = values.dt.month

    if {"pickup_datetime", "request_datetime"}.issubset(parsed_datetimes):
        df["request_to_pickup_seconds"] = (
            parsed_datetimes["pickup_datetime"] - parsed_datetimes["request_datetime"]
        ).dt.total_seconds()
    if {"on_scene_datetime", "request_datetime"}.issubset(parsed_datetimes):
        df["request_to_scene_seconds"] = (
            parsed_datetimes["on_scene_datetime"] - parsed_datetimes["request_datetime"]
        ).dt.total_seconds()

    return df.drop(columns=[c for c in FHVHV_DROP_DATETIME_COLUMNS if c in df.columns])


def encode_baseline_fhvhv_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["__is_train"] = True
    test["__is_train"] = False

    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = add_baseline_fhvhv_datetime_features(combined)

    categorical_columns = [
        column for column in FHVHV_CATEGORICAL_COLUMNS if column in combined.columns
    ]
    for column in categorical_columns:
        combined[column] = combined[column].fillna("missing").astype(str)

    combined = pd.get_dummies(combined, columns=categorical_columns, dummy_na=False)
    bool_columns = combined.select_dtypes(include=["bool"]).columns
    combined[bool_columns] = combined[bool_columns].astype("int8")

    feature_columns = [
        column for column in combined.columns if column not in {"__is_train"}
    ]
    combined[feature_columns] = combined[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    combined[feature_columns] = combined[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    combined[feature_columns] = combined[feature_columns].fillna(0)

    train_mask = combined["__is_train"].astype(bool)
    x_train = combined.loc[train_mask, feature_columns]
    x_test = combined.loc[~train_mask, feature_columns]
    return x_train, x_test


def _is_fhvhv_prediction_format(train: pd.DataFrame, target_column: str) -> bool:
    required_columns = {
        "hvfhs_license_num",
        "pickup_datetime",
        "dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_miles",
        "trip_time",
    }
    return target_column == "base_passenger_fare" and required_columns.issubset(
        train.columns
    )


def finalize_lightgbm_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train, test = train.align(test, join="outer", axis=1)
    categorical_features = [
        column
        for column in train.columns
        if pd.api.types.is_object_dtype(train[column])
        or pd.api.types.is_string_dtype(train[column])
        or isinstance(train[column].dtype, pd.CategoricalDtype)
    ]

    for column in categorical_features:
        combined = pd.concat([train[column], test[column]], axis=0).astype("string")
        categories = pd.Index(combined.dropna().unique())
        train[column] = pd.Categorical(train[column].astype("string"), categories=categories)
        test[column] = pd.Categorical(test[column].astype("string"), categories=categories)

    for column in train.columns.difference(categorical_features):
        train[column] = pd.to_numeric(train[column], errors="coerce")
        test[column] = pd.to_numeric(test[column], errors="coerce")

    return train, test, categorical_features


def prepare_generic_tabular_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
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

    train, test, categorical_features = finalize_lightgbm_features(train, test)
    return train, y_train, test, test_keys, categorical_features


def prepare_fhvhv_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    train = train.copy()
    test = test.copy()
    test_keys = test[id_column].copy()

    train = train.dropna(subset=[target_column])
    y_train = pd.to_numeric(train[target_column], errors="coerce")
    valid_target = y_train.notna()
    train = train.loc[valid_target].drop(columns=[target_column])
    y_train = y_train.loc[valid_target]

    for column in FHVHV_FLAG_COLUMNS:
        if column in train.columns:
            train = train[~train[column].fillna(False)]
            y_train = y_train.loc[train.index]

    if target_column in test.columns:
        test = test.drop(columns=[target_column])

    x_train, x_test = encode_baseline_fhvhv_features(train, test)
    return x_train, y_train, x_test, test_keys, []


def build_core_model(
    max_iter: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    min_child_samples: int,
    subsample: float,
    subsample_freq: int,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    device: str,
    random_state: int,
) -> object:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lightgbm is required for the core model. Install dependencies with "
            "`pip install -r requirements.txt` in the environment used to run this script."
        ) from exc

    return LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=max_iter,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        device_type=device,
        n_jobs=-1,
        verbose=-1,
        random_state=random_state,
    )


def clone_lightgbm_model(estimator: object) -> object:
    try:
        from sklearn.base import clone
    except ModuleNotFoundError:
        return build_core_model(**estimator.get_params())  # type: ignore[attr-defined]
    return clone(estimator)


def fit_lightgbm_model(
    estimator: object,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_features: list[str],
    early_stopping_rounds: int,
) -> None:
    from lightgbm import early_stopping, log_evaluation

    callbacks = []
    if early_stopping_rounds > 0:
        callbacks.append(early_stopping(early_stopping_rounds, verbose=False))
    callbacks.append(log_evaluation(period=100))

    estimator.fit(
        x_fit,
        y_fit,
        eval_set=[(x_valid, y_valid)],
        eval_metric="rmse",
        categorical_feature=categorical_features or "auto",
        callbacks=callbacks,
    )


def fit_lightgbm_final_model(
    estimator: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str],
) -> None:
    estimator.fit(
        x_train,
        y_train,
        categorical_feature=categorical_features or "auto",
    )


def build_native_lightgbm_params(args: argparse.Namespace) -> dict:
    return {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": args.subsample_freq,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "device_type": args.device,
        "verbose": -1,
        "num_threads": -1,
    }


def fit_native_lightgbm(
    params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    num_boost_round: int,
    x_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | np.ndarray | None = None,
    early_stopping_rounds: int = 0,
) -> object:
    import lightgbm as lgbm

    train_set = lgbm.Dataset(x_train, label=y_train)
    valid_sets = None
    callbacks = []
    if x_valid is not None and y_valid is not None:
        valid_sets = [lgbm.Dataset(x_valid, label=y_valid, reference=train_set)]
        callbacks.append(lgbm.log_evaluation(period=100))
        if early_stopping_rounds > 0:
            callbacks.append(lgbm.early_stopping(early_stopping_rounds, verbose=False))

    return lgbm.train(
        params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def evaluate_predictions(y_true: pd.Series | np.ndarray, predictions: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate_sample_submission(
    sample_submission_path: Path | None,
    keys: pd.Series,
    predictions: np.ndarray,
    id_column: str,
    target_column: str,
) -> dict:
    if sample_submission_path is None:
        return {"sample_submission_evaluated": False}

    sample = pd.read_csv(sample_submission_path)
    required_columns = {id_column, target_column}
    if not required_columns.issubset(sample.columns):
        return {
            "sample_submission_evaluated": False,
            "sample_submission_reason": (
                f"missing columns: {sorted(required_columns - set(sample.columns))}"
            ),
        }

    if len(sample) != len(predictions):
        return {
            "sample_submission_evaluated": False,
            "sample_submission_reason": (
                f"row count mismatch: sample={len(sample)}, predictions={len(predictions)}"
            ),
        }

    prediction_ids = pd.Series(keys).reset_index(drop=True)
    sample_ids = sample[id_column].reset_index(drop=True)
    if not prediction_ids.equals(sample_ids):
        return {
            "sample_submission_evaluated": False,
            "sample_submission_reason": "sample_submission id order does not match predictions",
        }

    y_true = pd.to_numeric(sample[target_column], errors="coerce").to_numpy(dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid_mask.any():
        return {
            "sample_submission_evaluated": False,
            "sample_submission_reason": "no finite rows available",
        }

    sample_metrics = evaluate_predictions(y_true[valid_mask], y_pred[valid_mask])
    return {
        "sample_submission_evaluated": True,
        "sample_submission_path": str(sample_submission_path),
        "sample_submission_rows": int(len(sample)),
        "sample_submission_evaluated_rows": int(valid_mask.sum()),
        "sample_submission_skipped_rows": int((~valid_mask).sum()),
        "sample_submission_rmse": sample_metrics["rmse"],
        "sample_submission_mae": sample_metrics["mae"],
        "sample_submission_r2": sample_metrics["r2"],
    }


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_submission(keys: pd.Series, predictions: np.ndarray, target_column: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
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
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=args.subsample_freq,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        device=args.device,
        random_state=args.random_state,
    )

    if _is_kaggle_taxi_fare(train, target_column):
        x_train, y_train, x_test, test_keys = prepare_kaggle_taxi_features(train, test)
        estimator = model
        categorical_features = []
        feature_mode = "kaggle_taxi_fare"
    elif _is_fhvhv_prediction_format(train, target_column):
        x_train, y_train, x_test, test_keys, categorical_features = prepare_fhvhv_features(
            train=train,
            test=test,
            target_column=target_column,
            id_column=id_column,
        )
        estimator = model
        feature_mode = "fhvhv_baseline_aligned"
    else:
        x_train, y_train, x_test, test_keys, categorical_features = prepare_generic_tabular_features(
            train=train,
            test=test,
            target_column=target_column,
            id_column=id_column,
        )
        estimator = model
        feature_mode = "generic_tabular"

    if not 0 < args.valid_size < 1:
        raise ValueError("--valid-size must be between 0 and 1")

    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=args.valid_size,
        random_state=args.random_state,
    )
    if feature_mode == "fhvhv_baseline_aligned":
        native_params = build_native_lightgbm_params(args)
        validation_model = fit_native_lightgbm(
            params=native_params,
            x_train=x_fit,
            y_train=y_fit,
            num_boost_round=args.max_iter,
            x_valid=x_valid,
            y_valid=y_valid,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        valid_predictions = validation_model.predict(
            x_valid,
            num_iteration=validation_model.best_iteration or args.max_iter,
        )
        final_model = fit_native_lightgbm(
            params=native_params,
            x_train=x_train,
            y_train=y_train,
            num_boost_round=args.max_iter,
        )
        test_predictions = np.maximum(final_model.predict(x_test), 0)
        best_iteration = validation_model.best_iteration or args.max_iter
    else:
        fit_lightgbm_model(
            estimator=estimator,
            x_fit=x_fit,
            y_fit=y_fit,
            x_valid=x_valid,
            y_valid=y_valid,
            categorical_features=categorical_features,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        valid_predictions = estimator.predict(x_valid)
        final_estimator = clone_lightgbm_model(estimator)
        fit_lightgbm_final_model(
            estimator=final_estimator,
            x_train=x_train,
            y_train=y_train,
            categorical_features=categorical_features,
        )
        test_predictions = np.maximum(final_estimator.predict(x_test), 0)
        best_iteration = getattr(estimator, "best_iteration_", None) or args.max_iter
    metrics = evaluate_predictions(y_valid, valid_predictions)
    save_submission(test_keys, test_predictions, target_column, args.output)
    sample_metrics = evaluate_sample_submission(
        sample_submission_path=sample_submission_path,
        keys=test_keys,
        predictions=test_predictions,
        id_column=id_column,
        target_column=target_column,
    )

    metrics.update(
        {
            "model": "LGBMRegressor",
            "device": args.device,
            "feature_mode": feature_mode,
            "target_column": target_column,
            "id_column": id_column,
            "categorical_features": categorical_features,
            "best_iteration": best_iteration,
            "final_model_train_rows": int(len(x_train)),
            "train_rows_after_cleaning": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "validation_size": float(args.valid_size),
            "random_state": int(args.random_state),
            "submission_path": str(args.output),
        }
    )
    metrics.update(sample_metrics)
    save_metrics(metrics, args.metrics_output)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_core_model(args)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
