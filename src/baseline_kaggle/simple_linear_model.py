import argparse

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Train simple linear baseline.")
    parser.add_argument("--input-dir", default="../input")
    parser.add_argument("--output", default="submission_simple-linear-model.csv")
    parser.add_argument("--nrows", type=int, default=10_000_000)
    return parser.parse_args()


def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    return df


def get_input_matrix(df):
    return np.column_stack(
        (df["abs_diff_longitude"], df["abs_diff_latitude"], np.ones(len(df)))
    )


def main():
    args = parse_args()
    train_path = f"{args.input_dir}/train.csv"
    test_path = f"{args.input_dir}/test.csv"

    train = pd.read_csv(train_path, nrows=args.nrows)
    train = add_travel_vector_features(train)
    train = train.dropna(how="any", axis=0)
    train = train[
        (train["abs_diff_longitude"] < 5.0)
        & (train["abs_diff_latitude"] < 5.0)
    ]

    train_x = get_input_matrix(train)
    train_y = np.array(train["fare_amount"])
    weights, _, _, _ = np.linalg.lstsq(train_x, train_y, rcond=None)

    test = pd.read_csv(test_path)
    test = add_travel_vector_features(test)
    test_x = get_input_matrix(test)
    predictions = np.matmul(test_x, weights).round(decimals=2)

    submission = pd.DataFrame(
        {"key": test["key"], "fare_amount": predictions},
        columns=["key", "fare_amount"],
    )
    submission.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
