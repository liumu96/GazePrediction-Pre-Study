#!/usr/bin/env python
"""Batch extract ADT skeleton joints aligned to existing gaze sample timestamps."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

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

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence_names",
        nargs="*",
        help="Optional sequence ids. If omitted, process all *_gaze_samples.csv files in --reports-dir.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory containing existing `<sequence>_gaze_samples.csv` files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default is the same as --reports-dir.",
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
    reports_dir = args.reports_dir
    output_dir = args.output_dir or reports_dir
    sequence_names = (
        list(args.sequence_names)
        if args.sequence_names
        else discover_sequence_names(reports_dir)
    )

    batch_rows: list[dict[str, Any]] = []
    for index, sequence_name in enumerate(sequence_names, start=1):
        sequence_dir = resolve_sequence_path(sequence_name)
        gaze_csv = reports_dir / f"{sequence_name}_gaze_samples.csv"
        output_csv = default_skeleton_samples_csv_path(
            sequence_name,
            output_dir=output_dir,
        )
        summary_json = default_skeleton_summary_json_path(output_csv)

        gaze_samples = read_samples_csv(gaze_csv)
        samples, metadata = extract_skeleton_samples_at_timestamps(
            sequence_dir,
            [sample.query_timestamp_ns for sample in gaze_samples],
            max_dt_ns=int(args.max_dt_ms * 1e6) if args.max_dt_ms is not None else None,
        )
        write_skeleton_samples_csv(output_csv, samples, metadata)
        summary = summarize_skeleton_samples(samples, metadata)
        summary.update(
            {
                "sequence_name": sequence_name,
                "input_sequence_dir": str(sequence_dir),
                "input_gaze_csv": str(gaze_csv),
                "output_csv": str(output_csv),
                "max_dt_ms": args.max_dt_ms,
                "coordinate_frame": "ADT Scene/world frame, meters",
                "source_file": "Skeleton_T.json",
            }
        )
        write_json(summary_json, summary)
        batch_rows.append(sequence_batch_row(summary, summary_json))
        print(
            f"[{index}/{len(sequence_names)}] {sequence_name}: "
            f"samples={summary['sample_count']} valid_ratio={summary['valid_ratio']:.3f} "
            f"dt_p50_ns={summary['abs_dt_ns']['p50']:.0f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_csv = output_dir / "batch_skeleton_samples_summary.csv"
    batch_json = output_dir / "batch_skeleton_samples_report.json"
    write_batch_csv(batch_csv, batch_rows)
    write_json(
        batch_json,
        {
            "sequence_count": len(batch_rows),
            "mean_valid_ratio": (
                sum(float(row["valid_ratio"]) for row in batch_rows) / len(batch_rows)
                if batch_rows
                else None
            ),
            "rows": batch_rows,
        },
    )
    print(f"sequences: {len(batch_rows)}")
    print(f"batch_csv: {batch_csv}")
    print(f"batch_json: {batch_json}")


def discover_sequence_names(reports_dir: Path) -> list[str]:
    names = [
        path.stem[: -len("_gaze_samples")]
        for path in sorted(reports_dir.glob("*_gaze_samples.csv"))
    ]
    if not names:
        raise ValueError(f"No *_gaze_samples.csv files found in: {reports_dir}")
    return names


def sequence_batch_row(summary: dict[str, Any], summary_json: Path) -> dict[str, Any]:
    return {
        "sequence_name": summary["sequence_name"],
        "sample_count": summary["sample_count"],
        "valid_count": summary["valid_count"],
        "valid_ratio": summary["valid_ratio"],
        "joint_count": summary["joint_count"],
        "marker_count": summary["marker_count"],
        "abs_dt_p50_ns": summary["abs_dt_ns"]["p50"],
        "abs_dt_max_ns": summary["abs_dt_ns"]["max"],
        "output_csv": summary["output_csv"],
        "summary_json": str(summary_json),
    }


def write_batch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
