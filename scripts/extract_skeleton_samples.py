#!/usr/bin/env python
"""Extract ADT skeleton joints aligned to existing gaze sample timestamps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.providers import resolve_sequence_path  # noqa: E402
from adt_sandbox.skeleton_features import (  # noqa: E402
    default_skeleton_samples_csv_path,
    default_skeleton_summary_json_path,
    extract_skeleton_samples_at_timestamps,
    summarize_skeleton_samples,
    write_json,
    write_skeleton_samples_csv,
)
from adt_sandbox.results import find_sequence_file  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        type=Path,
        help="Sequence id resolved under ADT_DATA_ROOT, or an explicit sequence directory.",
    )
    parser.add_argument(
        "--input-gaze-csv",
        type=Path,
        default=None,
        help=(
            "Existing gaze_samples.csv whose timestamps define the output rows. "
            "Default: outputs/reports/sequences/<sequence>/gaze/gaze_samples.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory for `<sequence>_skeleton_samples.csv`.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional explicit output CSV path.",
    )
    parser.add_argument(
        "--max-dt-ms",
        type=float,
        default=50.0,
        help="Maximum allowed nearest-skeleton timestamp offset in milliseconds. Default: 50.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = resolve_sequence_path(args.sequence)
    sequence_name = sequence_dir.name
    input_gaze_csv = args.input_gaze_csv or find_sequence_file(
        REPO_ROOT / "outputs" / "reports",
        sequence_name,
        "gaze",
        "gaze_samples.csv",
    )
    output_csv = args.output_csv or default_skeleton_samples_csv_path(
        sequence_name,
        output_dir=args.output_dir,
    )
    summary_json = default_skeleton_summary_json_path(output_csv)

    gaze_samples = read_samples_csv(input_gaze_csv)
    timestamps_ns = [sample.query_timestamp_ns for sample in gaze_samples]
    samples, metadata = extract_skeleton_samples_at_timestamps(
        sequence_dir,
        timestamps_ns,
        max_dt_ns=int(args.max_dt_ms * 1e6) if args.max_dt_ms is not None else None,
    )
    write_skeleton_samples_csv(output_csv, samples, metadata)
    summary = summarize_skeleton_samples(samples, metadata)
    summary.update(
        {
            "sequence_name": sequence_name,
            "input_sequence_dir": str(sequence_dir),
            "input_gaze_csv": str(input_gaze_csv),
            "output_csv": str(output_csv),
            "max_dt_ms": args.max_dt_ms,
            "coordinate_frame": "ADT Scene/world frame, meters",
            "source_file": "Skeleton_T.json",
        }
    )
    write_json(summary_json, summary)
    print(
        f"{sequence_name}: samples={summary['sample_count']} "
        f"valid={summary['valid_count']} valid_ratio={summary['valid_ratio']:.3f}"
    )
    print(f"csv: {output_csv}")
    print(f"json: {summary_json}")


if __name__ == "__main__":
    main()
