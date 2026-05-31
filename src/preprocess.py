from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_audit import audit_raw_data, save_audit_result
from data_governance import (
    build_quality_tier,
    derive_and_cap_kinematics,
    govern_numeric_semantics,
    select_model_ready_view,
    validate_temporal_consistency,
)


def run_pipeline(
    raw_csv: str | Path,
    audit_json: str | Path,
    model_ready_parquet: str | Path | None = None,
    sample_rows: int = 200_000,
) -> None:
    """Lightweight entrypoint: run audit, then governance demo pipeline."""
    result = audit_raw_data(raw_csv_path=raw_csv)
    save_audit_result(result, audit_json)

    # Governance demo on a sample to keep this entrypoint lightweight.
    df = pd.read_csv(raw_csv, nrows=sample_rows)
    df = validate_temporal_consistency(df)
    df = govern_numeric_semantics(df)
    df = derive_and_cap_kinematics(df)
    df = build_quality_tier(df)
    df = select_model_ready_view(df)

    if model_ready_parquet is not None:
        Path(model_ready_parquet).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(model_ready_parquet, index=False)

    print(f"rows={result.total_rows}")
    print(f"audit_json={audit_json}")
    if model_ready_parquet is not None:
        print(f"model_ready_sample={model_ready_parquet}")
        print(f"model_ready_rows={len(df)}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data" / "raw" / "fhvhv_tripdata_2026-03.csv"
    out_json = project_root / "results" / "raw_audit_2026_03.json"
    out_model_ready = project_root / "data" / "processed" / "model_ready_sample.parquet"
    run_pipeline(raw_csv=raw_csv, audit_json=out_json, model_ready_parquet=out_model_ready)

