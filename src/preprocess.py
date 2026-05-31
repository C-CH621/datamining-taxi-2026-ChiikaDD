"""Pipeline entrypoint: orchestrate data_audit and governance_v2."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

try:
    from .data_audit import audit_raw_data, save_audit_result
    from . import governance_v2
except ImportError:
    from data_audit import audit_raw_data, save_audit_result
    import governance_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "raw" / "fhvhv_tripdata_2026-03.csv"
DEFAULT_AUDIT_JSON = PROJECT_ROOT / "results" / "raw_audit_2026_03.json"
DEFAULT_SPLIT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fhvhv_tripdata_2026-03_governed_v2_full.csv"
)
DEFAULT_SPLIT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fhvhv_tripdata_2026-03_prediction_format"
)


def run_audit(raw_csv: Path, audit_json: Path, chunksize: int) -> None:
    result = audit_raw_data(raw_csv_path=raw_csv, chunksize=chunksize)
    save_audit_result(result, audit_json)
    print(f"rows={result.total_rows}")
    print(f"audit_json={audit_json}")


def run_pipeline(raw_csv: Path, audit_json: Path, chunksize: int) -> None:
    """Run full pipeline: audit then governance_v2."""
    run_audit(raw_csv=raw_csv, audit_json=audit_json, chunksize=chunksize)
    governance_v2.main()


def _write_csv_part(df: pd.DataFrame, path: Path, write_header: bool) -> None:
    df.to_csv(path, mode="w" if write_header else "a", header=write_header, index=False)


def _sample_test_row_numbers(
    input_path: Path,
    test_size: int,
    chunksize: int,
    random_seed: int,
) -> tuple[set[int], int]:
    rng = random.Random(random_seed)
    reservoir: list[int] = []
    rows_seen = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize), start=1):
        for row_number in range(rows_seen, rows_seen + len(chunk)):
            if len(reservoir) < test_size:
                reservoir.append(row_number)
                continue
            replacement_idx = rng.randint(0, row_number)
            if replacement_idx < test_size:
                reservoir[replacement_idx] = row_number
        rows_seen += len(chunk)
        print(f"sample_pass chunk={chunk_idx} rows_seen={rows_seen}")

    if rows_seen < test_size:
        raise ValueError(f"Input has only {rows_seen} rows, fewer than test size {test_size}")
    return set(reservoir), rows_seen


def split_random_test(
    input_path: Path,
    output_dir: Path,
    test_size: int,
    chunksize: int,
    random_seed: int,
    target_column: str,
    id_column: str,
) -> dict:
    if test_size <= 0:
        raise ValueError("--test-size must be positive")
    if chunksize <= test_size:
        raise ValueError("--chunksize must be larger than --test-size")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    submission_path = output_dir / "sample_submission.csv"
    for path in (train_path, test_path, submission_path):
        if path.exists():
            path.unlink()

    test_row_numbers, total_rows = _sample_test_row_numbers(
        input_path=input_path,
        test_size=test_size,
        chunksize=chunksize,
        random_seed=random_seed,
    )

    train_rows = 0
    test_rows = 0
    train_header = True
    test_header = True
    submission_header = True
    rows_seen = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize), start=1):
        if target_column not in chunk.columns:
            raise ValueError(f"Target column not found: {target_column}")
        if id_column not in chunk.columns:
            raise ValueError(f"ID column not found: {id_column}")

        chunk_row_numbers = range(rows_seen, rows_seen + len(chunk))
        test_mask = pd.Series([rn in test_row_numbers for rn in chunk_row_numbers], index=chunk.index)

        train_part = chunk.loc[~test_mask]
        if not train_part.empty:
            _write_csv_part(train_part, train_path, train_header)
            train_header = False
            train_rows += len(train_part)

        test_with_target = chunk.loc[test_mask]
        if not test_with_target.empty:
            test_features = test_with_target.drop(columns=[target_column])
            sample_submission = test_with_target[[id_column, target_column]]
            _write_csv_part(test_features, test_path, test_header)
            _write_csv_part(sample_submission, submission_path, submission_header)
            test_header = False
            submission_header = False
            test_rows += len(test_with_target)

        rows_seen += len(chunk)
        print(f"write_pass chunk={chunk_idx} rows_seen={rows_seen} train_rows={train_rows} test_rows={test_rows}")

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "sample_submission_path": str(submission_path),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "submission_rows": test_rows,
        "total_rows": total_rows,
        "random_seed": random_seed,
        "target_column": target_column,
        "id_column": id_column,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    audit_parser = subparsers.add_parser("audit", help="Run only raw data audit.")
    audit_parser.add_argument("--input", type=Path, default=DEFAULT_RAW_CSV)
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT_JSON)
    audit_parser.add_argument("--chunksize", type=int, default=300_000)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run audit + governance_v2.")
    pipeline_parser.add_argument("--input", type=Path, default=DEFAULT_RAW_CSV)
    pipeline_parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT_JSON)
    pipeline_parser.add_argument("--chunksize", type=int, default=300_000)

    split_parser = subparsers.add_parser("split", help="Split governed CSV into train/test/submission.")
    split_parser.add_argument("--input", type=Path, default=DEFAULT_SPLIT_INPUT)
    split_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_OUTPUT_DIR)
    split_parser.add_argument("--test-size", type=int, default=10_000)
    split_parser.add_argument("--chunksize", type=int, default=500_000)
    split_parser.add_argument("--random-seed", type=int, default=42)
    split_parser.add_argument("--target-column", default="base_passenger_fare")
    split_parser.add_argument("--id-column", default="pickup_datetime")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "audit":
        run_audit(raw_csv=args.input, audit_json=args.output, chunksize=args.chunksize)
        return

    if args.command in (None, "pipeline"):
        run_pipeline(raw_csv=args.input, audit_json=args.output, chunksize=args.chunksize)
        return

    if args.command == "split":
        result = split_random_test(
            input_path=args.input,
            output_dir=args.output_dir,
            test_size=args.test_size,
            chunksize=args.chunksize,
            random_seed=args.random_seed,
            target_column=args.target_column,
            id_column=args.id_column,
        )
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

