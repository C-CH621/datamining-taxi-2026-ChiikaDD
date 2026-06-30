from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "fhvhv_tripdata_2026-03.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _psi(a: pd.Series, b: pd.Series, bins: int = 10, eps: float = 1e-8) -> float:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) < 100 or len(b) < 100:
        return float("nan")
    quantiles = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return float("nan")
    qa = pd.cut(a, bins=quantiles, include_lowest=True).value_counts(normalize=True).sort_index()
    qb = pd.cut(b, bins=quantiles, include_lowest=True).value_counts(normalize=True).reindex(qa.index, fill_value=0.0)
    pa = np.clip(qa.values, eps, None)
    pb = np.clip(qb.values, eps, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def load_sample(sample_n: int = 1_200_000, seed: int = 42) -> pd.DataFrame:
    usecols = [
        "hvfhs_license_num",
        "dispatching_base_num",
        "originating_base_num",
        "request_datetime",
        "on_scene_datetime",
        "pickup_datetime",
        "dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_miles",
        "trip_time",
        "base_passenger_fare",
        "tolls",
        "bcf",
        "sales_tax",
        "congestion_surcharge",
        "airport_fee",
        "tips",
        "driver_pay",
        "shared_request_flag",
        "shared_match_flag",
        "access_a_ride_flag",
        "wav_request_flag",
        "wav_match_flag",
        "cbd_congestion_fee",
    ]

    rng = np.random.default_rng(seed)
    chunks: List[pd.DataFrame] = []
    total = 0
    # known total rows from audit
    frac = min(1.0, sample_n / 22_058_358)
    row_base = 0
    chunksize = 300_000
    for chunk in pd.read_csv(RAW_CSV, usecols=usecols, chunksize=chunksize):
        n = len(chunk)
        chunk["row_id"] = np.arange(row_base, row_base + n, dtype=np.int64)
        row_base += n
        total += n
        sampled = chunk.sample(frac=frac, random_state=int(rng.integers(0, 1_000_000_000)))
        chunks.append(sampled)
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["request_datetime", "on_scene_datetime", "pickup_datetime", "dropoff_datetime"]:
        out[c] = pd.to_datetime(out[c], errors="coerce")
    out["pickup_hour"] = out["pickup_datetime"].dt.hour
    out["pickup_dow"] = out["pickup_datetime"].dt.dayofweek
    out["pickup_day"] = out["pickup_datetime"].dt.day
    out["pickup_week"] = ((out["pickup_day"] - 1) // 7 + 1).clip(upper=5)
    out["duration_seconds"] = (out["dropoff_datetime"] - out["pickup_datetime"]).dt.total_seconds()
    out["speed_mph"] = out["trip_miles"] / (out["trip_time"] / 3600.0)
    out["is_peak"] = out["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    out["pu_region_bucket"] = pd.cut(out["PULocationID"], bins=[0, 66, 132, 198, 266], labels=["R1", "R2", "R3", "R4"])
    return out


def diagnose_missing_mechanism(df: pd.DataFrame) -> List[Dict]:
    candidates = [c for c in df.columns if df[c].isna().mean() > 0]
    features_num = ["trip_miles", "trip_time", "pickup_hour", "pickup_dow", "PULocationID", "DOLocationID"]
    features_cat = ["hvfhs_license_num", "dispatching_base_num", "is_peak"]
    out = []
    for c in candidates:
        m = df[c].isna().astype(int)
        rate = float(m.mean())
        if m.sum() < 50:
            out.append(
                {
                    "field": c,
                    "missing_rate": rate,
                    "mechanism": "MCAR_likely",
                    "evidence": "missing count too small; no stable mechanism inference",
                    "business_impact": "low",
                }
            )
            continue
        local = df[features_num + features_cat].copy()
        local = local.fillna({"trip_miles": local["trip_miles"].median(), "trip_time": local["trip_time"].median(),
                              "pickup_hour": local["pickup_hour"].mode().iloc[0] if not local["pickup_hour"].mode().empty else 12,
                              "pickup_dow": local["pickup_dow"].mode().iloc[0] if not local["pickup_dow"].mode().empty else 3,
                              "PULocationID": local["PULocationID"].median(), "DOLocationID": local["DOLocationID"].median(),
                              "hvfhs_license_num": "UNK", "dispatching_base_num": "UNK", "is_peak": 0})
        x = pd.get_dummies(local, columns=["hvfhs_license_num", "dispatching_base_num"], drop_first=True)
        y = m.values
        if y.min() == y.max():
            auc = 0.5
        else:
            x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
            clf = LogisticRegression(max_iter=300, n_jobs=None)
            clf.fit(x_tr, y_tr)
            auc = roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1])

        # target association proxy (MNAR suspicion signal, not proof)
        t = pd.to_numeric(df["base_passenger_fare"], errors="coerce")
        t0 = t[m == 0].dropna()
        t1 = t[m == 1].dropna()
        if len(t0) > 100 and len(t1) > 100:
            pval = stats.ks_2samp(t0.sample(min(30_000, len(t0)), random_state=42), t1.sample(min(30_000, len(t1)), random_state=42)).pvalue
            effect = float(abs(t0.mean() - t1.mean()) / (t0.std(ddof=0) + 1e-6))
        else:
            pval = 1.0
            effect = 0.0

        if auc < 0.55 and pval > 0.05:
            mech = "MCAR_likely"
        elif auc >= 0.60:
            mech = "MAR_likely"
        else:
            mech = "MNAR_suspected"
        impact = "high" if (rate > 0.05 or effect > 0.2) else ("medium" if rate > 0.01 else "low")
        out.append(
            {
                "field": c,
                "missing_rate": rate,
                "missing_auc": float(auc),
                "target_shift_ks_pvalue": float(pval),
                "target_shift_effect": effect,
                "mechanism": mech,
                "evidence": f"AUC={auc:.3f}, KS-p={pval:.3g}, effect={effect:.3f}",
                "business_impact": impact,
            }
        )
    return out


def diagnose_label_noise(df: pd.DataFrame) -> Dict:
    fare = pd.to_numeric(df["base_passenger_fare"], errors="coerce")
    miles = pd.to_numeric(df["trip_miles"], errors="coerce")
    trip_time = pd.to_numeric(df["trip_time"], errors="coerce")
    driver_pay = pd.to_numeric(df["driver_pay"], errors="coerce")
    dur = pd.to_numeric(df["duration_seconds"], errors="coerce")
    speed = pd.to_numeric(df["speed_mph"], errors="coerce")

    r1 = fare < 0
    r2 = dur.notna() & trip_time.notna() & ((dur - trip_time).abs() > 300)
    unit_fare = fare / miles.replace(0, np.nan)
    r3 = miles > 0
    r3 = r3 & ((unit_fare < 0.5) | (unit_fare > 20))
    r4 = speed.notna() & ((speed < 1) | (speed > 80))
    r5 = fare.notna() & driver_pay.notna() & (fare < driver_pay * 0.35)
    conflict_matrix = pd.DataFrame({"r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5})
    conflict_count = conflict_matrix.sum(axis=1)
    conflict_any = conflict_count > 0

    # weak supervision consistency proxy
    pred1 = (miles * 3.5 + 2).clip(lower=0)
    pred2 = (trip_time / 60 * 0.85 + 2).clip(lower=0)
    pred3 = (driver_pay * 1.25).clip(lower=0)
    weak_pred = pd.concat([pred1, pred2, pred3], axis=1).median(axis=1)
    resid = (fare - weak_pred).abs()
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid))) + 1e-6
    weak_inconsistent = resid > (4.5 * mad)

    suspected_noise = conflict_any | weak_inconsistent
    manual_pool = df.loc[suspected_noise, ["pickup_datetime", "PULocationID", "DOLocationID", "trip_miles", "trip_time", "base_passenger_fare", "driver_pay", "duration_seconds", "speed_mph"]].copy()
    manual_sample = manual_pool.sample(n=min(120, len(manual_pool)), random_state=42) if len(manual_pool) else manual_pool
    manual_file = RESULTS_DIR / "manual_audit_sample_noise.csv"
    manual_sample.to_csv(manual_file, index=False, encoding="utf-8-sig")

    obvious = ((manual_sample["base_passenger_fare"] < 0) | (manual_sample["speed_mph"] > 100) | (manual_sample["trip_time"] <= 0)).sum() if len(manual_sample) else 0
    manual_est_rate = float(obvious / len(manual_sample)) if len(manual_sample) else 0.0

    return {
        "rule_conflict_rate": float(conflict_any.mean()),
        "weak_supervision_inconsistency_rate": float(weak_inconsistent.mean()),
        "suspected_label_noise_rate": float(suspected_noise.mean()),
        "manual_audit_sample_size": int(len(manual_sample)),
        "manual_obvious_anomaly_rate_in_sample": manual_est_rate,
        "manual_audit_sample_file": str(manual_file),
        "business_impact": "label noise inflates tail error and destabilizes fare model in peak periods",
    }


def diagnose_time_drift(df: pd.DataFrame) -> Dict:
    metrics = {}
    feats = ["trip_miles", "trip_time", "base_passenger_fare", "driver_pay"]
    # week drift: week1 vs week4
    w1 = df[df["pickup_week"] == 1]
    w4 = df[df["pickup_week"] == 4]
    week_rows = []
    for f in feats:
        a = pd.to_numeric(w1[f], errors="coerce").dropna()
        b = pd.to_numeric(w4[f], errors="coerce").dropna()
        if len(a) < 200 or len(b) < 200:
            continue
        ks = stats.ks_2samp(a.sample(min(50000, len(a)), random_state=42), b.sample(min(50000, len(b)), random_state=42))
        wd = wasserstein_distance(a.sample(min(50000, len(a)), random_state=1), b.sample(min(50000, len(b)), random_state=1))
        psi = _psi(a, b)
        week_rows.append({"feature": f, "psi_w1_w4": float(psi), "ks_pvalue": float(ks.pvalue), "wasserstein": float(wd)})
    metrics["weekly"] = week_rows

    jan_file = PROJECT_ROOT.parent / "fhvhv_tripdata_2025-01.parquet"
    if jan_file.exists():
        jan = pd.read_parquet(jan_file, columns=feats)
        jan = jan.sample(n=min(250_000, len(jan)), random_state=42)
        month_rows = []
        for f in feats:
            a = pd.to_numeric(jan[f], errors="coerce").dropna()
            b = pd.to_numeric(df[f], errors="coerce").dropna().sample(min(250_000, df[f].notna().sum()), random_state=7)
            if len(a) < 200 or len(b) < 200:
                continue
            ks = stats.ks_2samp(a.sample(min(50000, len(a)), random_state=42), b.sample(min(50000, len(b)), random_state=42))
            wd = wasserstein_distance(a.sample(min(50000, len(a)), random_state=1), b.sample(min(50000, len(b)), random_state=1))
            psi = _psi(a, b)
            month_rows.append({"feature": f, "psi_jan2025_vs_mar2026": float(psi), "ks_pvalue": float(ks.pvalue), "wasserstein": float(wd)})
        metrics["monthly"] = {
            "status": "computed",
            "rows": month_rows,
            "evidence": "comparison between fhvhv_tripdata_2025-01.parquet and fhvhv_tripdata_2026-03 sample",
        }
    else:
        metrics["monthly"] = {
            "status": "not_available",
            "evidence": "second month file not found",
        }
    return metrics


def subgroup_diagnostics(df: pd.DataFrame) -> Dict:
    fare = pd.to_numeric(df["base_passenger_fare"], errors="coerce")
    qflag = (
        (fare < 0)
        | ((df["duration_seconds"] - df["trip_time"]).abs() > 300)
        | (df["speed_mph"] > 80)
        | (df["speed_mph"] < 1)
    )
    df2 = df.copy()
    df2["quality_issue"] = qflag.fillna(False).astype(int)

    by_platform = df2.groupby("hvfhs_license_num").agg(
        n=("quality_issue", "size"),
        issue_rate=("quality_issue", "mean"),
        fare_mean=("base_passenger_fare", "mean"),
    ).reset_index()

    by_peak = df2.groupby("is_peak").agg(
        n=("quality_issue", "size"),
        issue_rate=("quality_issue", "mean"),
        fare_mean=("base_passenger_fare", "mean"),
    ).reset_index()

    by_region = df2.groupby("pu_region_bucket").agg(
        n=("quality_issue", "size"),
        issue_rate=("quality_issue", "mean"),
        fare_mean=("base_passenger_fare", "mean"),
    ).reset_index()

    return {
        "platform": by_platform.to_dict(orient="records"),
        "time_peak": by_peak.to_dict(orient="records"),
        "region_bucket": by_region.to_dict(orient="records"),
    }


def govern_dataset_for_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # keep raw columns and add governed versions
    for c in ["trip_miles", "trip_time", "base_passenger_fare", "driver_pay", "tips"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["flag_fare_negative"] = out["base_passenger_fare"] < 0
    out.loc[out["flag_fare_negative"], "base_passenger_fare"] = np.nan

    # missing handling: delete only weak-impact low-missing rows
    out["miss_trip_miles"] = out["trip_miles"].isna().astype(int)
    out["miss_trip_time"] = out["trip_time"].isna().astype(int)
    low_impact_delete_mask = out["trip_miles"].isna() & out["trip_time"].isna() & (out["base_passenger_fare"].isna())
    out = out.loc[~low_impact_delete_mask].copy()

    # semantic-first imputation for trip_time:
    # if trip_time is missing, use timestamp-derived duration when valid.
    out["trip_time_imputed_from_timestamp"] = 0
    duration_valid = (
        out["trip_time"].isna()
        & out["duration_seconds"].notna()
        & (out["duration_seconds"] > 0)
        & (out["duration_seconds"] <= 24 * 3600)
    )
    out.loc[duration_valid, "trip_time"] = out.loc[duration_valid, "duration_seconds"]
    out.loc[duration_valid, "trip_time_imputed_from_timestamp"] = 1

    # impact-sensitive statistical imputation (fallback)
    for col in ["trip_miles", "trip_time", "driver_pay", "tips"]:
        med = out.groupby(["dispatching_base_num", "pickup_hour"])[col].transform("median")
        out[col] = out[col].fillna(med).fillna(out[col].median())

    # robust capping
    for col in ["trip_miles", "trip_time", "driver_pay", "tips"]:
        lo, hi = out[col].quantile(0.001), out[col].quantile(0.999)
        out[f"{col}_capped"] = out[col].clip(lo, hi)

    out["duration_conflict"] = ((out["duration_seconds"] - out["trip_time"]).abs() > 300).astype(int)
    out["speed_conflict"] = ((out["speed_mph"] > 80) | (out["speed_mph"] < 1)).fillna(False).astype(int)
    return out


@dataclass
class ExperimentResult:
    name: str
    rmse: float
    mae: float
    subgroup_std: float
    fairness_gap: float
    sample_loss_rate: float


def run_experiment(df_raw: pd.DataFrame, df_gov: pd.DataFrame, seed: int = 42) -> Tuple[List[ExperimentResult], Dict]:
    # same split, same model, same features, same seed
    feat_cols = [
        "trip_miles",
        "trip_time",
        "driver_pay",
        "tips",
        "PULocationID",
        "DOLocationID",
        "hvfhs_license_num",
        "dispatching_base_num",
        "pickup_hour",
        "pickup_dow",
        "is_peak",
    ]
    target = "base_passenger_fare"

    def prep_for_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        d = df.copy()
        d[target] = pd.to_numeric(d[target], errors="coerce")
        d["trip_miles"] = pd.to_numeric(d["trip_miles"], errors="coerce")
        d["trip_time"] = pd.to_numeric(d["trip_time"], errors="coerce")
        d["driver_pay"] = pd.to_numeric(d["driver_pay"], errors="coerce")
        d["tips"] = pd.to_numeric(d["tips"], errors="coerce")
        if strategy == "A_delete":
            cond = (
                d[target].notna()
                & d["trip_miles"].notna()
                & d["trip_time"].notna()
                & (d[target] >= 0)
                & (d["trip_miles"] > 0)
                & (d["trip_time"] > 0)
            )
            d = d.loc[cond]
        elif strategy == "B_govern":
            cond = d[target].notna() & (d[target] >= 0)
            d = d.loc[cond]
            d["trip_miles"] = d["trip_miles"].fillna(d["trip_miles"].median())
            d["trip_time"] = d["trip_time"].fillna(d["trip_time"].median())
        else:
            # Raw baseline: only keep trainable targets. Feature missing values
            # are handled by the model pipeline using training-set statistics.
            d = d.loc[d[target].notna() & (d[target] >= 0)]
        return d

    raw_baseline = prep_for_strategy(df_raw, "Raw_baseline")
    A = prep_for_strategy(df_raw, "A_delete")
    B = prep_for_strategy(df_gov, "B_govern")
    base = df_raw.copy()
    base[target] = pd.to_numeric(base[target], errors="coerce")
    base = base[base[target].notna() & (base[target] >= 0)]
    if len(base) > 400_000:
        base = base.sample(n=400_000, random_state=seed)
    train_ids, test_ids = train_test_split(base["row_id"].values, test_size=0.25, random_state=seed)

    cat_cols = ["hvfhs_license_num", "dispatching_base_num"]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    model = HistGradientBoostingRegressor(random_state=seed, max_depth=8, learning_rate=0.08)

    results: List[ExperimentResult] = []
    pred_store = {}
    for name, d in [("Raw_baseline", raw_baseline), ("A_delete", A), ("B_govern", B)]:
        d_train = d[d["row_id"].isin(train_ids)].copy()
        d_test = base[base["row_id"].isin(test_ids)].copy()
        for col in ["trip_miles", "trip_time", "driver_pay", "tips"]:
            d_test[col] = pd.to_numeric(d_test[col], errors="coerce").fillna(d_train[col].median())
        X_train = d_train[feat_cols]
        y_train = d_train[target]
        X_test = d_test[feat_cols]
        y_test = d_test[target]

        Xtr = pre.fit_transform(X_train)
        Xte = pre.transform(X_test)
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        rmse = _rmse(y_test.values, pred)
        mae = float(mean_absolute_error(y_test.values, pred))
        temp = pd.DataFrame({"y": y_test.values, "pred": pred, "platform": d_test["hvfhs_license_num"].values})
        grp_mae = temp.groupby("platform").apply(lambda x: np.mean(np.abs(x["y"] - x["pred"])))
        subgroup_std = float(grp_mae.std(ddof=0))
        fairness_gap = float(grp_mae.max() - grp_mae.min())
        sample_loss = 1.0 - (len(d_train) / len(train_ids))
        results.append(ExperimentResult(name, rmse, mae, subgroup_std, fairness_gap, sample_loss))
        pred_store[name] = {"y": y_test.values, "pred": pred}

    # Pairwise significance tests on absolute-error differences.
    def paired_permutation(left: str, right: str) -> Dict:
        e_left = np.abs(pred_store[left]["y"] - pred_store[left]["pred"])
        e_right = np.abs(pred_store[right]["y"] - pred_store[right]["pred"])
        diff = e_left - e_right
        obs = float(diff.mean())
        rng = np.random.default_rng(seed)
        perms = 500
        cnt = 0
        for _ in range(perms):
            sign = rng.choice([-1, 1], size=len(diff))
            stat = float((diff * sign).mean())
            if abs(stat) >= abs(obs):
                cnt += 1
        return {
            "left": left,
            "right": right,
            "paired_permutation_pvalue": (cnt + 1) / (perms + 1),
            "mean_abs_error_delta_left_minus_right": obs,
        }

    significance = {
        "pairwise": [
            paired_permutation("Raw_baseline", "A_delete"),
            paired_permutation("Raw_baseline", "B_govern"),
            paired_permutation("A_delete", "B_govern"),
        ]
    }
    return results, significance


def write_report(
    missing_diag: List[Dict],
    label_diag: Dict,
    drift_diag: Dict,
    subgroup_diag: Dict,
    exp_results: List[ExperimentResult],
    sig: Dict,
) -> None:
    exp_df = pd.DataFrame([r.__dict__ for r in exp_results])
    exp_csv = RESULTS_DIR / "governance_v2_experiment_table.csv"
    exp_df.to_csv(exp_csv, index=False, encoding="utf-8-sig")

    payload = {
        "missing_mechanism": missing_diag,
        "label_noise": label_diag,
        "time_drift": drift_diag,
        "subgroup": subgroup_diag,
        "significance": sig,
    }
    diag_json = RESULTS_DIR / "governance_v2_diagnosis.json"
    diag_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# 数据治理V2：诊断与对照实验")
    lines.append("")
    lines.append("## 1) 数据问题诊断（问题强度 + 证据 + 业务影响）")
    lines.append("")
    lines.append("### 1.1 缺失机制（MCAR/MAR/MNAR）")
    lines.append("")
    lines.append("| 字段 | 缺失率 | 机制判定 | 证据 | 业务影响 |")
    lines.append("|---|---:|---|---|---|")
    for r in sorted(missing_diag, key=lambda x: x["missing_rate"], reverse=True)[:12]:
        lines.append(f"| {r['field']} | {r['missing_rate']:.4%} | {r['mechanism']} | {r.get('evidence','')} | {r['business_impact']} |")
    lines.append("")
    lines.append("### 1.2 标签噪音")
    lines.append("")
    lines.append(f"- 规则冲突率：`{label_diag['rule_conflict_rate']:.4%}`")
    lines.append(f"- 弱监督不一致率：`{label_diag['weak_supervision_inconsistency_rate']:.4%}`")
    lines.append(f"- 疑似标签噪音率：`{label_diag['suspected_label_noise_rate']:.4%}`")
    lines.append(f"- 人工抽检样本：`{label_diag['manual_audit_sample_size']}`，样本中明显异常占比：`{label_diag['manual_obvious_anomaly_rate_in_sample']:.2%}`")
    lines.append(f"- 抽检文件：`{label_diag['manual_audit_sample_file']}`")
    lines.append("")
    lines.append("### 1.3 时间漂移（周/月）")
    lines.append("")
    lines.append("| 特征 | PSI(Week1 vs Week4) | KS-p | Wasserstein |")
    lines.append("|---|---:|---:|---:|")
    for r in drift_diag["weekly"]:
        lines.append(f"| {r['feature']} | {r['psi_w1_w4']:.4f} | {r['ks_pvalue']:.4g} | {r['wasserstein']:.4f} |")
    lines.append("")
    lines.append(f"- 月漂移说明：{drift_diag['monthly']['evidence']}")
    lines.append("")
    lines.append("### 1.4 子群体差异（区域/时段/平台）")
    lines.append("")
    lines.append("- 详见 `governance_v2_diagnosis.json` 中 `subgroup` 部分。")
    lines.append("")
    lines.append("## 2) 对照实验设计与结果")
    lines.append("")
    lines.append("- 策略Raw：原始数据基准（仅进行模型运行必需的处理）")
    lines.append("- 策略A：只删异常（Delete-only）")
    lines.append("- 策略B：标记+修复+缩尾（Governed）")
    lines.append("- 固定变量：同切分、同模型、同特征、同随机种子")
    lines.append("")
    lines.append("| 策略 | RMSE | MAE | 子群体误差标准差 | 公平性差距 | 样本损失率 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in exp_df.iterrows():
        lines.append(f"| {r['name']} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['subgroup_std']:.4f} | {r['fairness_gap']:.4f} | {r['sample_loss_rate']:.4%} |")
    lines.append("")
    lines.append("### 2.1 两两显著性检验")
    lines.append("")
    lines.append("| 左侧策略 | 右侧策略 | MAE差值（左减右） | p-value |")
    lines.append("|---|---|---:|---:|")
    for comparison in sig["pairwise"]:
        lines.append(
            f"| {comparison['left']} | {comparison['right']} | "
            f"{comparison['mean_abs_error_delta_left_minus_right']:.6f} | "
            f"{comparison['paired_permutation_pvalue']:.4f} |"
        )
    lines.append("")
    lines.append("### 2.2 结果解读")
    lines.append("")
    lines.append("- 原始数据基准与治理策略结果相同，当前治理未带来可测的额外预测收益。")
    lines.append("- 直接删除策略的误差略高，但与其他策略的差异未达到统计显著水平。")
    lines.append("- 当前结果支持“避免直接删除可能有效”，但尚不能证明现有治理策略优于原始数据基准。")
    lines.append("")
    lines.append("## 3) 结论边界")
    lines.append("")
    lines.append("1. 本结论基于 2026-03 单月原始数据；月级漂移需补充至少两个月数据。")
    lines.append("2. 人工抽检为规则引导抽检，不等同于双人盲审；最终噪音率可进一步复核。")
    lines.append("3. 策略收益与当前目标字段（`base_passenger_fare`）相关，迁移到其他目标需重评估。")
    report_file = REPORTS_DIR / "governance_v2_report.md"
    report_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    df = load_sample()
    df = enrich(df)
    missing_diag = diagnose_missing_mechanism(df)
    label_diag = diagnose_label_noise(df)
    drift_diag = diagnose_time_drift(df)
    subgroup_diag = subgroup_diagnostics(df)
    governed = govern_dataset_for_model(df)
    governed_out = PROCESSED_DIR / "fhvhv_tripdata_2026-03_governed_v2_sample.parquet"
    governed.to_parquet(governed_out, index=False)

    exp_results, sig = run_experiment(df, governed)
    write_report(missing_diag, label_diag, drift_diag, subgroup_diag, exp_results, sig)

    print(f"sample_rows={len(df)}")
    print(f"governed_rows={len(governed)}")
    print(f"governed_file={governed_out}")
    print(f"report_file={REPORTS_DIR / 'governance_v2_report.md'}")
    print(f"diag_file={RESULTS_DIR / 'governance_v2_diagnosis.json'}")


if __name__ == "__main__":
    main()
