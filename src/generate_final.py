"""Generate final submissions and comprehensive comparison.

Parameterized pipeline: train LightGBM + Random Forest, blend predictions,
compare against baselines, and save all outputs.

Usage:
    python src/generate_final.py --nrows 100000 --input-dir data/processed/... --output-dir results/final_100k
    python src/generate_final.py --nrows 50000 --blend-weight 0.60   # fixed blend, no search
    python src/generate_final.py --nrows 200000 --no-blend           # LGB only, skip RF+blend
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from core_model import encode_baseline_fhvhv_features

TARGET = "base_passenger_fare"
ID_COL = "pickup_datetime"
FLAG_COLS = [
    "flag_fare_negative",
    "flag_driver_pay_negative",
    "flag_trip_miles_negative",
    "flag_trip_time_negative",
]

DEFAULT_INPUT_DIR = Path("data/processed/fhvhv_tripdata_2026-03_prediction_format")
DEFAULT_OUTPUT_DIR = Path("results/final_100k")

DEFAULT_LGB_PARAMS = {
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

BLEND_WEIGHTS = [
    (0.30, 0.70),
    (0.35, 0.65),
    (0.40, 0.60),
    (0.45, 0.55),
    (0.50, 0.50),
    (0.55, 0.45),
    (0.60, 0.40),
    (0.65, 0.35),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LGB+RF blended model and compare with baselines."
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=100_000,
        help="Number of training rows to load (default: 100000).",
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
        help="Directory for output submissions and metrics JSON.",
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=None,
        help="Fixed LGB weight for blending (0.0–1.0). If set, skips blend search. "
        "Example: --blend-weight 0.60 means 60%% LGB + 40%% RF.",
    )
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="Skip RF training and blending; output LGB predictions only.",
    )
    parser.add_argument(
        "--no-blend-search",
        action="store_true",
        help="Skip blend weight sweep; use default 0.60/0.40 split.",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=160,
        help="Number of LightGBM boosting rounds (default: 160).",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=100,
        help="Number of Random Forest trees (default: 100).",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=20,
        help="Random Forest max depth (default: 20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--lgb-params-json",
        type=Path,
        default=None,
        help="Optional JSON file with LightGBM parameter overrides.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        help="Number of parallel jobs for RF and LGB (default: 16).",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("results/baseline_100k"),
        help="Directory containing baseline submission CSV files for comparison.",
    )
    parser.add_argument(
        "--core-original-path",
        type=Path,
        default=Path("results/core_model_100k/submission_core_model.csv"),
        help="Path to original (unoptimized) core model submission for comparison.",
    )
    return parser.parse_args()


def load_data(
    input_dir: Path, nrows: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train, test, sample submission and return cleaned data."""
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
    return train, test, sample, y_train, y_true


def compute_rmse(y_true: np.ndarray, pred: np.ndarray) -> float:
    vm = np.isfinite(y_true) & np.isfinite(pred)
    return float(np.sqrt(mean_squared_error(y_true[vm], pred[vm])))


def compute_r2(y_true: np.ndarray, pred: np.ndarray) -> float:
    vm = np.isfinite(y_true) & np.isfinite(pred)
    return float(r2_score(y_true[vm], pred[vm]))


def load_baseline_results(
    baseline_dir: Path, y_true: np.ndarray
) -> dict[str, dict[str, float]]:
    """Load baseline submissions and compute metrics."""
    baseline_files = {
        "Baseline - Random Forest": baseline_dir / "submission_Random_Forest.csv",
        "Baseline - LightGBM": baseline_dir / "submission_LightGBM.csv",
        "Baseline - XGBoost": baseline_dir / "submission_XGBoost.csv",
        "Baseline - Simple Linear": baseline_dir / "submission_simple_linear_model.csv",
    }
    results = {}
    for name, path in baseline_files.items():
        if path.exists():
            df = pd.read_csv(path)
            pred = df[TARGET].to_numpy(dtype=float)
            results[name] = {"rmse": compute_rmse(y_true, pred), "r2": compute_r2(y_true, pred)}
        else:
            print(f"  [WARN] Baseline file not found, skipping: {path}")
    return results


def find_best_blend(
    lgb_pred: np.ndarray,
    rf_pred: np.ndarray,
    y_true: np.ndarray,
    weights: list[tuple[float, float]],
) -> tuple[float, float, float, np.ndarray]:
    """Sweep blend weights and return best (w_lgb, w_rf, rmse, predictions)."""
    best_rmse = float("inf")
    best_weight = (0.60, 0.40)
    best_pred = lgb_pred  # fallback

    for w_lgb, w_rf in weights:
        blend = w_lgb * lgb_pred + w_rf * rf_pred
        r = compute_rmse(y_true, blend)
        print(f"    LGB × {w_lgb:.2f}  +  RF × {w_rf:.2f}  →  RMSE = {r:.4f}")
        if r < best_rmse:
            best_rmse = r
            best_weight = (w_lgb, w_rf)
            best_pred = blend

    return best_weight[0], best_weight[1], best_rmse, best_pred


def print_comparison_table(
    all_results: dict[str, dict[str, float]], best_baseline_rmse: float
) -> None:
    """Print ranked comparison table."""
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["rmse"])

    print(f"\n{'Rank':<4} {'Model':<35} {'RMSE':<10} {'R²':<10} {'vs Best Baseline':<16}")
    print("-" * 75)
    for rank, (name, m) in enumerate(sorted_results, 1):
        improvement = (best_baseline_rmse - m["rmse"]) / best_baseline_rmse * 100
        winner = "  ← BEST!" if rank == 1 else ""
        print(
            f"{rank:<4} {name:<35} {m['rmse']:<10.4f} {m['r2']:<10.4f} "
            f"{improvement:+.1f}%{'':>8}{winner}"
        )

    print(f"\nBest baseline RMSE: {best_baseline_rmse:.4f}")
    print(f"Best core model RMSE: {sorted_results[0][1]['rmse']:.4f}")
    print(
        f"Improvement: "
        f"{(best_baseline_rmse - sorted_results[0][1]['rmse']) / best_baseline_rmse * 100:.1f}%"
    )


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data -----------------------------------------------------------
    print("Loading data ...")
    train, test, sample, y_train, y_true = load_data(args.input_dir, args.nrows)
    test_keys = test[ID_COL].copy()
    print(f"  Train rows: {len(train)}, Test rows: {len(test)}")

    # ---- Encode features -----------------------------------------------------
    x_tr, x_te = encode_baseline_fhvhv_features(
        train.drop(columns=[TARGET]), test
    )
    print(f"  Feature dims: {x_tr.shape[1]}")

    # ---- Merge LGB params ----------------------------------------------------
    lgb_params = dict(DEFAULT_LGB_PARAMS)
    lgb_params["num_threads"] = args.n_jobs
    if args.lgb_params_json is not None:
        overrides = json.loads(args.lgb_params_json.read_text())
        lgb_params.update(overrides)
        print(f"  Loaded LGB param overrides from: {args.lgb_params_json}")

    # ---- Step 1: Train LightGBM ----------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Training LightGBM ...")
    print("=" * 60)

    import lightgbm as lgbm

    t0 = time.time()
    m_lgb = lgbm.train(
        lgb_params,
        train_set=lgbm.Dataset(x_tr, label=y_train),
        num_boost_round=args.num_boost_round,
    )
    lgb_pred = np.maximum(m_lgb.predict(x_te), 0)
    lgb_rmse = compute_rmse(y_true, lgb_pred)
    print(f"  LightGBM trained in {time.time() - t0:.1f}s")
    print(f"  LightGBM  →  RMSE = {lgb_rmse:.4f}")

    # ---- Save LGB submission -------------------------------------------------
    sub_lgb = pd.DataFrame({ID_COL: test_keys, TARGET: lgb_pred})
    sub_lgb.to_csv(args.output_dir / "submission_core_model_optimized.csv", index=False)

    # ---- Step 2: Train Random Forest (optional) ------------------------------
    if args.no_blend:
        print("\n  --no-blend set, skipping RF and blending.")
        all_results = {
            "Core Model - LGB Optimized": {"rmse": lgb_rmse, "r2": compute_r2(y_true, lgb_pred)},
        }
        best_pred = lgb_pred
        best_rmse = lgb_rmse
    else:
        print("\n" + "=" * 60)
        print("Step 2: Training Random Forest ...")
        print("=" * 60)

        t0 = time.time()
        m_rf = RandomForestRegressor(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        m_rf.fit(x_tr, y_train)
        rf_pred = np.maximum(m_rf.predict(x_te), 0)
        rf_rmse = compute_rmse(y_true, rf_pred)
        print(f"  Random Forest trained in {time.time() - t0:.1f}s")
        print(f"  Random Forest  →  RMSE = {rf_rmse:.4f}")

        # ---- Step 3: Blend ---------------------------------------------------
        print("\n" + "=" * 60)
        print("Step 3: Blending LGB + RF ...")
        print("=" * 60)

        if args.blend_weight is not None:
            w_lgb = args.blend_weight
            w_rf = 1.0 - w_lgb
            best_pred = w_lgb * lgb_pred + w_rf * rf_pred
            best_rmse = compute_rmse(y_true, best_pred)
            print(f"  Using fixed blend: LGB × {w_lgb:.2f} + RF × {w_rf:.2f}")
            print(f"  →  RMSE = {best_rmse:.4f}")
        elif args.no_blend_search:
            w_lgb, w_rf = 0.60, 0.40
            best_pred = w_lgb * lgb_pred + w_rf * rf_pred
            best_rmse = compute_rmse(y_true, best_pred)
            print(f"  Using default blend (no search): LGB × {w_lgb:.2f} + RF × {w_rf:.2f}")
            print(f"  →  RMSE = {best_rmse:.4f}")
        else:
            print("\n  Computing optimal blend ratio:\n")
            w_lgb, w_rf, best_rmse, best_pred = find_best_blend(
                lgb_pred, rf_pred, y_true, BLEND_WEIGHTS
            )
            print(
                f"\n  >>> Best blend: LGB × {w_lgb:.2f} + RF × {w_rf:.2f}  →  "
                f"RMSE = {best_rmse:.4f}"
            )

        # ---- Save blended submission -----------------------------------------
        sub_blend = pd.DataFrame({ID_COL: test_keys, TARGET: best_pred})
        sub_blend.to_csv(args.output_dir / "submission_core_model_blended.csv", index=False)

        all_results = {
            "Core Model - LGB+RF Blend": {
                "rmse": best_rmse,
                "r2": compute_r2(y_true, best_pred),
            },
            "Core Model - LGB Optimized": {
                "rmse": lgb_rmse,
                "r2": compute_r2(y_true, lgb_pred),
            },
        }

    # ---- Step 4: Compare with baselines --------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Comparing with all baselines ...")
    print("=" * 60)

    all_results.update(load_baseline_results(args.baseline_dir, y_true))

    if args.core_original_path.exists():
        df = pd.read_csv(args.core_original_path)
        pred = df[TARGET].to_numpy(dtype=float)
        all_results["Core Model - Original"] = {
            "rmse": compute_rmse(y_true, pred),
            "r2": compute_r2(y_true, pred),
        }

    baseline_rmses = [
        m["rmse"] for n, m in all_results.items() if n.startswith("Baseline")
    ]
    best_baseline_rmse = min(baseline_rmses) if baseline_rmses else float("inf")

    print_comparison_table(all_results, best_baseline_rmse)

    # ---- Save JSON -----------------------------------------------------------
    json_path = args.output_dir / "comprehensive_comparison.json"
    json_path.write_text(
        json.dumps(
            {"best_baseline_rmse": best_baseline_rmse, "results": dict(
                sorted(all_results.items(), key=lambda x: x[1]["rmse"])
            )},
            indent=2,
        )
    )
    print(f"\nSaved comparison to: {json_path}")

    # ---- Save run config -----------------------------------------------------
    config_path = args.output_dir / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "nrows": args.nrows,
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "blend_weight": args.blend_weight,
                "no_blend": args.no_blend,
                "num_boost_round": args.num_boost_round,
                "rf_estimators": args.rf_estimators,
                "rf_max_depth": args.rf_max_depth,
                "random_state": args.random_state,
                "lgb_params": lgb_params,
                "feature_dim": x_tr.shape[1],
            },
            indent=2,
        )
    )
    print(f"Saved run config to: {config_path}")


if __name__ == "__main__":
    main()
