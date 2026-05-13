#!/usr/bin/env python
"""Batch extract Scene-frame object boxes for ADT sequences."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv, resolve_data_root  # noqa: E402
from adt_sandbox.providers import resolve_sequence_path  # noqa: E402
from adt_sandbox.scene_features import (  # noqa: E402
    default_scene_object_boxes_csv_path,
    default_scene_object_boxes_summary_json_path,
    extract_scene_object_box_rows,
    summarize_scene_object_box_rows,
    write_json,
    write_scene_object_boxes_csv,
)
from adt_sandbox.results import batch_dir  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence_names",
        nargs="*",
        help="Optional sequence ids or paths. If omitted, discover sequences under ADT_DATA_ROOT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory for per-sequence scene object box CSVs and batch summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dirs = (
        [resolve_sequence_path(name) for name in args.sequence_names]
        if args.sequence_names
        else discover_sequence_dirs()
    )
    if not sequence_dirs:
        raise ValueError("No ADT sequences found for scene object extraction")

    batch_rows: list[dict[str, Any]] = []
    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        sequence_dir = sequence_dir.resolve()
        sequence_name = sequence_dir.name
        output_csv = default_scene_object_boxes_csv_path(
            sequence_name,
            output_dir=args.output_dir,
        )
        summary_json = default_scene_object_boxes_summary_json_path(output_csv)
        rows = extract_scene_object_box_rows(sequence_dir)
        write_scene_object_boxes_csv(output_csv, rows)
        summary = summarize_scene_object_box_rows(rows)
        summary.update(
            {
                "sequence_name": sequence_name,
                "input_sequence_dir": str(sequence_dir),
                "output_csv": str(output_csv),
            }
        )
        write_json(summary_json, summary)
        batch_rows.append(sequence_batch_row(summary, summary_json))
        print(
            f"[{index}/{len(sequence_dirs)}] {sequence_name}: "
            f"rows={summary['row_count']} objects={summary['unique_object_count']} "
            f"timestamps={summary['unique_timestamp_count']}"
        )

    batch_output_dir = batch_dir(args.output_dir)
    batch_csv = batch_output_dir / "batch_scene_object_boxes_summary.csv"
    batch_json = batch_output_dir / "batch_scene_object_boxes_report.json"
    write_batch_csv(batch_csv, batch_rows)
    write_json(
        batch_json,
        {
            "sequence_count": len(batch_rows),
            "total_rows": sum(int(row["row_count"]) for row in batch_rows),
            "total_unique_objects_per_sequence_sum": sum(
                int(row["unique_object_count"]) for row in batch_rows
            ),
            "rows": batch_rows,
        },
    )
    print(f"sequences: {len(batch_rows)}")
    print(f"batch_csv: {batch_csv}")
    print(f"batch_json: {batch_json}")


def discover_sequence_dirs() -> list[Path]:
    data_root = resolve_data_root()
    if data_root is None:
        raise ValueError(
            "ADT_DATA_ROOT is not set and no sequence names were provided"
        )
    return sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and (path / "scene_objects.csv").exists()
    )


def sequence_batch_row(summary: dict[str, Any], summary_json: Path) -> dict[str, Any]:
    return {
        "sequence_name": summary["sequence_name"],
        "row_count": summary["row_count"],
        "unique_object_count": summary["unique_object_count"],
        "unique_timestamp_count": summary["unique_timestamp_count"],
        "static_row_count": summary["static_row_count"],
        "static_object_count": summary["object_motion_type_counts"].get("static", 0),
        "dynamic_object_count": summary["object_motion_type_counts"].get("dynamic", 0),
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
