from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "base_passenger_fare"
ID_COLUMN = "pickup_datetime"

DATETIME_COLUMNS = [
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
]

CATEGORICAL_COLUMNS = [
    "hvfhs_license_num",
    "dispatching_base_num",
    "originating_base_num",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
]

DROP_COLUMNS = [
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
]


def resolve_input_files(input_dir):
    input_dir = Path(input_dir)
    return {
        "train": input_dir / "train.csv",
        "test": input_dir / "test.csv",
        "sample_submission": input_dir / "sample_submission.csv",
    }


def load_train_test(input_dir, nrows=1_000_000):
    files = resolve_input_files(input_dir)
    train = pd.read_csv(files["train"], nrows=nrows)
    test = pd.read_csv(files["test"])
    return train, test, files["sample_submission"]


def add_datetime_features(df):
    df = df.copy()
    for column in DATETIME_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_datetime(df[column], errors="coerce")
        df[f"{column}_hour"] = values.dt.hour
        df[f"{column}_dayofweek"] = values.dt.dayofweek
        df[f"{column}_day"] = values.dt.day
        df[f"{column}_month"] = values.dt.month

    if {"pickup_datetime", "request_datetime"}.issubset(df.columns):
        pickup = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        request = pd.to_datetime(df["request_datetime"], errors="coerce")
        df["request_to_pickup_seconds"] = (pickup - request).dt.total_seconds()

    if {"on_scene_datetime", "request_datetime"}.issubset(df.columns):
        on_scene = pd.to_datetime(df["on_scene_datetime"], errors="coerce")
        request = pd.to_datetime(df["request_datetime"], errors="coerce")
        df["request_to_scene_seconds"] = (on_scene - request).dt.total_seconds()

    return df


def clean_train_data(train):
    train = train.copy()
    train = train.dropna(subset=[TARGET_COLUMN])
    train = train[train[TARGET_COLUMN] >= 0]

    quality_columns = [
        "flag_fare_negative",
        "flag_driver_pay_negative",
        "flag_trip_miles_negative",
        "flag_trip_time_negative",
    ]
    for column in quality_columns:
        if column in train.columns:
            train = train[~train[column].fillna(False)]

    if "trip_miles_capped" in train.columns:
        train = train[train["trip_miles_capped"] >= 0]
    if "trip_time_capped" in train.columns:
        train = train[train["trip_time_capped"] >= 0]

    return train


def encode_features(train, test):
    train = train.copy()
    test = test.copy()

    train["__is_train"] = True
    test["__is_train"] = False

    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = add_datetime_features(combined)
    combined = combined.drop(columns=[c for c in DROP_COLUMNS if c in combined.columns])

    categorical = [c for c in CATEGORICAL_COLUMNS if c in combined.columns]
    for column in categorical:
        combined[column] = combined[column].fillna("missing").astype(str)

    combined = pd.get_dummies(combined, columns=categorical, dummy_na=False)

    bool_columns = combined.select_dtypes(include=["bool"]).columns
    combined[bool_columns] = combined[bool_columns].astype(int)

    feature_columns = [
        column
        for column in combined.columns
        if column not in {TARGET_COLUMN, "__is_train"}
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


def prepare_model_data(train, test):
    train = clean_train_data(train)
    test_ids = test[ID_COLUMN].copy()
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)

    x_train, x_test = encode_features(
        train.drop(columns=[TARGET_COLUMN]),
        test,
    )
    return x_train, y_train, x_test, test_ids


def write_submission(ids, predictions, output):
    submission = pd.DataFrame(
        {
            ID_COLUMN: ids,
            TARGET_COLUMN: np.maximum(predictions, 0),
        }
    )
    submission.to_csv(output, index=False)
