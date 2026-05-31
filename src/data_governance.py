from __future__ import annotations

import numpy as np
import pandas as pd


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

