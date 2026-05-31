from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class AuditResult:
    total_rows: int
    metrics: Dict[str, float]
    counts: Dict[str, int]


def _psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))


def audit_raw_data(
    raw_csv_path: str | Path,
    chunksize: int = 300_000,
) -> AuditResult:
    """Run a reproducible raw-data audit for FHVHV CSV."""
    usecols = [
        "dispatching_base_num",
        "pickup_datetime",
        "dropoff_datetime",
        "on_scene_datetime",
        "trip_miles",
        "trip_time",
        "base_passenger_fare",
        "driver_pay",
    ]

    total_rows = 0
    on_scene_missing = 0
    neg_fare = 0
    neg_driver_pay = 0
    duration_inconsistent = 0
    speed_outlier = 0

    first_half_hours = np.zeros(24, dtype=np.float64)
    second_half_hours = np.zeros(24, dtype=np.float64)
    dispatch_counter: Counter[str] = Counter()

    reader = pd.read_csv(raw_csv_path, usecols=usecols, chunksize=chunksize)
    for chunk in reader:
        total_rows += len(chunk)
        on_scene_missing += int(chunk["on_scene_datetime"].isna().sum())

        fare = pd.to_numeric(chunk["base_passenger_fare"], errors="coerce")
        pay = pd.to_numeric(chunk["driver_pay"], errors="coerce")
        miles = pd.to_numeric(chunk["trip_miles"], errors="coerce")
        sec = pd.to_numeric(chunk["trip_time"], errors="coerce")

        neg_fare += int((fare < 0).sum())
        neg_driver_pay += int((pay < 0).sum())

        pickup_dt = pd.to_datetime(chunk["pickup_datetime"], errors="coerce")
        dropoff_dt = pd.to_datetime(chunk["dropoff_datetime"], errors="coerce")

        derived_sec = (dropoff_dt - pickup_dt).dt.total_seconds()
        inconsistent = (
            derived_sec.notna()
            & sec.notna()
            & ((derived_sec - sec).abs() > 300)
        )
        duration_inconsistent += int(inconsistent.sum())

        speed = miles / (sec / 3600.0)
        speed_bad = (
            miles.notna()
            & sec.notna()
            & (miles > 0)
            & (sec > 0)
            & ((speed > 80) | (speed < 1))
        )
        speed_outlier += int(speed_bad.sum())

        day = pickup_dt.dt.day
        hour = pickup_dt.dt.hour
        first_mask = day.between(1, 15, inclusive="both") & hour.notna()
        second_mask = day.between(16, 31, inclusive="both") & hour.notna()

        if first_mask.any():
            h = hour[first_mask].astype(int).to_numpy()
            first_half_hours += np.bincount(h, minlength=24)
        if second_mask.any():
            h = hour[second_mask].astype(int).to_numpy()
            second_half_hours += np.bincount(h, minlength=24)

        dispatch_counter.update(chunk["dispatching_base_num"].astype(str).tolist())

    dispatch_freq = np.array(list(dispatch_counter.values()), dtype=np.int64)
    long_tail_entities = int((dispatch_freq < 3).sum())
    long_tail_ratio = float(long_tail_entities / len(dispatch_freq)) if len(dispatch_freq) else 0.0

    p = first_half_hours / first_half_hours.sum() if first_half_hours.sum() > 0 else np.ones(24) / 24
    q = second_half_hours / second_half_hours.sum() if second_half_hours.sum() > 0 else np.ones(24) / 24
    psi_hour = _psi(p, q)

    metrics = {
        "on_scene_missing_rate": on_scene_missing / total_rows,
        "duration_inconsistent_rate": duration_inconsistent / total_rows,
        "speed_outlier_rate": speed_outlier / total_rows,
        "negative_base_fare_rate": neg_fare / total_rows,
        "negative_driver_pay_rate": neg_driver_pay / total_rows,
        "dispatching_base_long_tail_ratio_freq_lt3": long_tail_ratio,
        "pickup_hour_psi_firsthalf_vs_secondhalf": psi_hour,
    }
    counts = {
        "on_scene_missing": on_scene_missing,
        "duration_inconsistent": duration_inconsistent,
        "speed_outlier": speed_outlier,
        "negative_base_fare": neg_fare,
        "negative_driver_pay": neg_driver_pay,
        "dispatching_base_total_entities": len(dispatch_freq),
        "dispatching_base_entities_freq_lt3": long_tail_entities,
    }
    return AuditResult(total_rows=total_rows, metrics=metrics, counts=counts)


def save_audit_result(result: AuditResult, out_json_path: str | Path) -> None:
    payload = {
        "total_rows": result.total_rows,
        "metrics": result.metrics,
        "counts": result.counts,
    }
    Path(out_json_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_temporal_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal consistency flags without dropping rows."""
    out = df.copy()
    pickup = pd.to_datetime(out["pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(out["dropoff_datetime"], errors="coerce")
    trip_time = pd.to_numeric(out["trip_time"], errors="coerce")
    derived_sec = (dropoff - pickup).dt.total_seconds()
    out["derived_duration_seconds"] = derived_sec
    out["flag_duration_inconsistent"] = (
        derived_sec.notna() & trip_time.notna() & ((derived_sec - trip_time).abs() > 300)
    )
    return out


def govern_numeric_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """Apply semantic constraints by nulling invalid values and keeping flags."""
    out = df.copy()
    nonnegative_cols = [
        "base_passenger_fare",
        "tolls",
        "bcf",
        "sales_tax",
        "congestion_surcharge",
        "airport_fee",
        "tips",
        "driver_pay",
        "cbd_congestion_fee",
    ]
    for col in nonnegative_cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        bad = out[col] < 0
        out[f"flag_{col}_negative"] = bad.fillna(False)
        out.loc[bad, col] = np.nan
    return out


def derive_and_cap_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Create speed and robust capped columns for modeling stability."""
    out = df.copy()
    out["trip_miles"] = pd.to_numeric(out["trip_miles"], errors="coerce")
    out["trip_time"] = pd.to_numeric(out["trip_time"], errors="coerce")
    speed = out["trip_miles"] / (out["trip_time"] / 3600.0)
    out["derived_speed_mph"] = speed
    out["flag_speed_outlier"] = (
        out["trip_miles"].notna()
        & out["trip_time"].notna()
        & (out["trip_miles"] > 0)
        & (out["trip_time"] > 0)
        & ((speed > 80) | (speed < 1))
    )
    for col in ["trip_miles", "trip_time", "base_passenger_fare", "driver_pay", "tips"]:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        lo, hi = series.quantile(0.001), series.quantile(0.999)
        out[f"{col}_capped"] = series.clip(lo, hi)
    return out


def build_quality_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Create row-level quality score and tier from all flag_* columns."""
    out = df.copy()
    flags = [c for c in out.columns if c.startswith("flag_")]
    if not flags:
        out["quality_issue_count"] = 0
        out["quality_tier"] = "A_clean"
        return out
    out["quality_issue_count"] = out[flags].astype(bool).sum(axis=1)
    out["quality_tier"] = pd.cut(
        out["quality_issue_count"],
        bins=[-1, 0, 2, 5, 10_000],
        labels=["A_clean", "B_minor", "C_moderate", "D_heavy"],
    ).astype(str)
    return out


def select_model_ready_view(df: pd.DataFrame) -> pd.DataFrame:
    """Return a training-friendly view while preserving most rows."""
    out = df.copy()
    if "flag_duration_inconsistent" in out.columns:
        out = out.loc[~out["flag_duration_inconsistent"]].copy()
    return out


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data" / "raw" / "fhvhv_tripdata_2026-03.csv"
    out_json = project_root / "results" / "raw_audit_2026_03.json"
    result = audit_raw_data(raw_csv_path=raw_csv)
    save_audit_result(result, out_json)
    print(f"rows={result.total_rows}")
    print(f"audit_json={out_json}")
