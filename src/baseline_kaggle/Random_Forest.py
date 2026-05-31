import argparse

from sklearn.ensemble import RandomForestRegressor

from model_utils import load_train_test, prepare_tree_model_data, write_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Forest baseline.")
    parser.add_argument("--input-dir", default="../input")
    parser.add_argument("--output", default="submission_Random_Forest.csv")
    parser.add_argument("--nrows", type=int, default=1_000_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    train, test, _ = load_train_test(args.input_dir, nrows=args.nrows)
    x_train, y_train, x_test, test_keys = prepare_tree_model_data(train, test)

    model = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    write_submission(test_keys, predictions, args.output)


if __name__ == "__main__":
    main()
