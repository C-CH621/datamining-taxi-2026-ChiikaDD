import argparse

import numpy as np

from model_utils import load_train_test, prepare_model_data, write_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Train simple linear baseline.")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output", default="submission_simple_linear_model.csv")
    parser.add_argument("--nrows", type=int, default=1_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    train, test, _ = load_train_test(args.input_dir, nrows=args.nrows)
    x_train, y_train, x_test, test_ids = prepare_model_data(train, test)

    train_matrix = np.column_stack([x_train.to_numpy(dtype=float), np.ones(len(x_train))])
    test_matrix = np.column_stack([x_test.to_numpy(dtype=float), np.ones(len(x_test))])
    weights, _, _, _ = np.linalg.lstsq(train_matrix, y_train, rcond=None)
    predictions = test_matrix @ weights

    write_submission(test_ids, predictions, args.output)


if __name__ == "__main__":
    main()
