import argparse

import xgboost as xgb

from model_utils import load_train_test, prepare_tree_model_data, write_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost baseline.")
    parser.add_argument("--input-dir", default="../input")
    parser.add_argument("--output", default="submission_XGBoost.csv")
    parser.add_argument("--nrows", type=int, default=1_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    train, test, _ = load_train_test(args.input_dir, nrows=args.nrows)
    x_train, y_train, x_test, test_keys = prepare_tree_model_data(train, test)

    params = {
        "max_depth": 7,
        "eta": 1,
        "verbosity": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
    }

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)
    model = xgb.train(params, dtrain, num_boost_round=50)
    predictions = model.predict(dtest)

    write_submission(test_keys, predictions, args.output)


if __name__ == "__main__":
    main()
