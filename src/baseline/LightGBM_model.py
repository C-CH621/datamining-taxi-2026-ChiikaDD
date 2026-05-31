import argparse

import lightgbm as lgbm

from model_utils import load_train_test, prepare_tree_model_data, write_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM baseline.")
    parser.add_argument("--input-dir", default="../input")
    parser.add_argument("--output", default="submission_LightGBM.csv")
    parser.add_argument("--nrows", type=int, default=1_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    train, test, _ = load_train_test(args.input_dir, nrows=args.nrows)
    x_train, y_train, x_test, test_keys = prepare_tree_model_data(train, test)

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "nthread": -1,
        "verbose": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "max_depth": -1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 1,
        "reg_lambda": 0.001,
        "metric": "rmse",
        "min_split_gain": 0.5,
        "min_child_weight": 1,
        "min_child_samples": 10,
        "scale_pos_weight": 1,
    }

    train_set = lgbm.Dataset(x_train, y_train)
    model = lgbm.train(params, train_set=train_set, num_boost_round=300)
    predictions = model.predict(x_test)

    write_submission(test_keys, predictions, args.output)


if __name__ == "__main__":
    main()
