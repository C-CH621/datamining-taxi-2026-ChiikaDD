from pathlib import Path

import numpy as np
import pandas as pd


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


def add_haversine_distance(df):
    radius_km = 6371
    pickup_lat = np.radians(df["pickup_latitude"])
    dropoff_lat = np.radians(df["dropoff_latitude"])
    lat_delta = np.radians(df["dropoff_latitude"] - df["pickup_latitude"])
    lon_delta = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])

    a = (
        np.sin(lat_delta / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(lon_delta / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["H_Distance"] = radius_km * c
    return df


def add_datetime_features(df):
    df["key"] = pd.to_datetime(df["key"])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Date"] = df["pickup_datetime"].dt.day
    df["Day of Week"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour
    return df


def clean_train_data(train):
    train = train.dropna(how="any", axis=0)
    train = train[train["fare_amount"] >= 0]
    train = train[train["passenger_count"] != 208]
    train = train[train["pickup_latitude"].between(-90, 90)]
    train = train[train["dropoff_latitude"].between(-90, 90)]
    train = train[train["pickup_longitude"].between(-180, 180)]
    train = train[train["dropoff_longitude"].between(-180, 180)]

    train = train.drop(
        train[
            (train["pickup_latitude"] == 0)
            & (train["pickup_longitude"] == 0)
            & (train["dropoff_latitude"] != 0)
            & (train["dropoff_longitude"] != 0)
            & (train["fare_amount"] == 0)
        ].index
    )
    train = train.drop(
        train[
            (train["pickup_latitude"] != 0)
            & (train["pickup_longitude"] != 0)
            & (train["dropoff_latitude"] == 0)
            & (train["dropoff_longitude"] == 0)
            & (train["fare_amount"] == 0)
        ].index
    )

    high_distance = (train["H_Distance"] > 200) & (train["fare_amount"] != 0)
    train.loc[high_distance, "H_Distance"] = (
        train.loc[high_distance, "fare_amount"] - 2.50
    ) / 1.56

    train = train.drop(
        train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].index
    )

    rush_hour_invalid = (
        (train["Hour"] >= 6)
        & (train["Hour"] <= 20)
        & (train["Day of Week"] >= 1)
        & (train["Day of Week"] <= 5)
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 2.5)
    )
    train = train.drop(train[rush_hour_invalid].index)

    zero_fare = (train["H_Distance"] != 0) & (train["fare_amount"] == 0)
    train.loc[zero_fare, "fare_amount"] = (
        train.loc[zero_fare, "H_Distance"] * 1.56 + 2.50
    )

    missing_distance = (train["H_Distance"] == 0) & (train["fare_amount"] > 3.0)
    train.loc[missing_distance, "H_Distance"] = (
        train.loc[missing_distance, "fare_amount"] - 2.50
    ) / 1.56

    return train


def prepare_tree_model_data(train, test):
    train = train.copy()
    test = test.copy()

    train = add_haversine_distance(train)
    test = add_haversine_distance(test)
    train = add_datetime_features(train)
    test = add_datetime_features(test)
    train = clean_train_data(train)

    train = train.drop(["key", "pickup_datetime"], axis=1)
    test_keys = test["key"].copy()
    test = test.drop(["key", "pickup_datetime"], axis=1)

    x_train = train.drop("fare_amount", axis=1)
    y_train = train["fare_amount"].values
    return x_train, y_train, test, test_keys


def write_submission(keys, predictions, output):
    submission = pd.DataFrame({"key": keys, "fare_amount": predictions})
    submission.to_csv(output, index=False)
