"""Generate LightGBM feature importance and SHAP interpretability plots.

Responds to the proposal's "model interpretability" commitment.
Produces:
  - fig_imp_gain.png      : LightGBM gain-based importance (top 20)
  - fig_imp_split.png     : LightGBM split-based importance (top 20)
  - fig_shap_summary.png  : SHAP beeswarm summary (top 20)
  - fig_shap_bar.png      : SHAP mean |SHAP| bar (top 20)

Usage:
    python src/plot_feature_importance.py --nrows 100000
    python src/plot_feature_importance.py --nrows 50000 --input-dir data/processed/... --max-display 30
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from core_model import encode_baseline_fhvhv_features

TARGET = "base_passenger_fare"
FLAG_COLS = [
    "flag_fare_negative",
    "flag_driver_pay_negative",
    "flag_trip_miles_negative",
    "flag_trip_time_negative",
]

DEFAULT_INPUT_DIR = Path("data/processed/fhvhv_tripdata_2026-03_prediction_format")
DEFAULT_OUTPUT_DIR = Path("results/figures")
DEFAULT_NROWS = 100_000
DEFAULT_MAX_DISPLAY = 20

LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "num_threads": 16,
    "learning_rate": 0.04,
    "num_leaves": 63,
    "max_depth": 7,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.5,
    "reg_lambda": 0.2,
    "min_child_samples": 20,
}
NUM_BOOST_ROUND = 160

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LightGBM feature importance and SHAP plots."
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=DEFAULT_NROWS,
        help=f"Number of training rows to load (default: {DEFAULT_NROWS}).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing train.csv, test.csv, sample_submission.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=DEFAULT_MAX_DISPLAY,
        help="Number of top features to display (default: 20).",
    )
    parser.add_argument(
        "--shap-sample",
        type=int,
        default=2000,
        help="Number of background samples for SHAP (default: 2000). "
        "Smaller = faster but noisier estimates.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def load_and_prepare(
    input_dir: Path, nrows: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    """Load data and apply the same cleaning pipeline as generate_final.py."""
    train = pd.read_csv(input_dir / "train.csv", nrows=nrows)
    test = pd.read_csv(input_dir / "test.csv")
    sample = pd.read_csv(input_dir / "sample_submission.csv")

    train = train.dropna(subset=[TARGET])
    y_train = pd.to_numeric(train[TARGET], errors="coerce")
    v = y_train.notna()
    train = train.loc[v]
    y_train = y_train.loc[v]

    for col in FLAG_COLS:
        if col in train.columns:
            m = ~train[col].fillna(False)
            train = train[m]
            y_train = y_train.loc[m.index[m.values]]

    y_true = pd.to_numeric(sample[TARGET], errors="coerce").to_numpy(dtype=float)
    return train, y_train, test, y_true


def plot_gain_importance(
    model, feature_names: list[str], max_display: int, output_dir: Path
) -> None:
    """LightGBM gain-based feature importance (top N)."""
    importance_gain = model.feature_importance(importance_type="gain")
    idx = np.argsort(importance_gain)[-max_display:]
    names = [feature_names[i] for i in idx]
    values = importance_gain[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    ax.barh(range(len(names)), values, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Gain Importance")
    ax.set_title("LightGBM Feature Importance (Gain) — Top {}".format(max_display))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    path = output_dir / "fig_imp_gain.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_split_importance(
    model, feature_names: list[str], max_display: int, output_dir: Path
) -> None:
    """LightGBM split-based feature importance (top N)."""
    importance_split = model.feature_importance(importance_type="split")
    idx = np.argsort(importance_split)[-max_display:]
    names = [feature_names[i] for i in idx]
    values = importance_split[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(names)))
    ax.barh(range(len(names)), values, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Split Count")
    ax.set_title("LightGBM Feature Importance (Split) — Top {}".format(max_display))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    path = output_dir / "fig_imp_split.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_shap_bar(
    shap_values, feature_names: list[str], max_display: int, output_dir: Path
) -> None:
    """SHAP mean |SHAP| bar plot."""
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(
        shap_values,
        plot_type="bar",
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    fig.tight_layout()
    path = output_dir / "fig_shap_bar.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_shap_summary(
    shap_values,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    max_display: int,
    output_dir: Path,
) -> None:
    """SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    fig.tight_layout()
    path = output_dir / "fig_shap_summary.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data -----------------------------------------------------------
    print(f"Loading data (nrows={args.nrows}) ...")
    train, y_train, test, y_true = load_and_prepare(args.input_dir, args.nrows)
    print(f"  Train rows: {len(train)}, Test rows: {len(test)}")

    # ---- Encode features -----------------------------------------------------
    x_tr, x_te = encode_baseline_fhvhv_features(
        train.drop(columns=[TARGET]), test
    )
    feature_names = list(x_tr.columns)
    print(f"  Feature dims: {len(feature_names)}")

    # ---- Train LightGBM ------------------------------------------------------
    print("Training LightGBM ...")
    import lightgbm as lgbm

    t0 = time.time()
    model = lgbm.train(
        LGB_PARAMS,
        train_set=lgbm.Dataset(x_tr, label=y_train),
        num_boost_round=NUM_BOOST_ROUND,
    )
    print(f"  Trained in {time.time() - t0:.1f}s")

    # ---- LightGBM built-in importance plots ----------------------------------
    print("\nGenerating LightGBM importance plots ...")
    plot_gain_importance(model, feature_names, args.max_display, args.output_dir)
    plot_split_importance(model, feature_names, args.max_display, args.output_dir)

    # ---- SHAP analysis -------------------------------------------------------
    print("\nComputing SHAP values ...")
    rng = np.random.default_rng(args.random_state)
    sample_idx = rng.choice(len(x_tr), size=min(args.shap_sample, len(x_tr)), replace=False)
    X_sample = x_tr.iloc[sample_idx].reset_index(drop=True)

    # Create TreeExplainer directly from the LightGBM booster
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    print(f"  SHAP computed in {time.time() - t0:.1f}s")

    print("Generating SHAP plots ...")
    plot_shap_bar(shap_vals, feature_names, args.max_display, args.output_dir)
    plot_shap_summary(shap_vals, X_sample, feature_names, args.max_display, args.output_dir)

    print(f"\nAll feature importance figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
