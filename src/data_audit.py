from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

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

