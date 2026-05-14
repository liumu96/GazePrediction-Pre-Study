#!/usr/bin/env python
"""Batch compute Scene-frame gaze/object ray hits for ADT sequences."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.results import (  # noqa: E402
    batch_dir,
    discover_sequence_names,
    find_sequence_file,
)
from adt_sandbox.scene_gaze_object_hits import (  # noqa: E402
    compute_gaze_object_hit_rows,
    default_gaze_object_hits_csv_path,
    default_gaze_object_hits_summary_json_path,
    summarize_gaze_object_hits,
    write_gaze_object_hits_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence_names",
        nargs="*",
        help="Optional sequence ids. If omitted, discover sequences with gaze_samples.csv.",
    )
    parser.add_argument(
        "--reports-dir",
        "--output-dir",
        dest="reports_dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Organized reports root containing gaze and scene object box CSVs.",
    )
    parser.add_argument(
        "--max-dynamic-dt-ms",
        type=float,
        default=20.0,
        help="Maximum allowed dynamic object timestamp mismatch in milliseconds.",
    )
    parser.add_argument(
        "--min-hit-distance-m",
        type=float,
        default=0.05,
        help="Ignore ray intersections closer than this distance from the gaze origin.",
    )
    parser.add_argument(
        "--max-hit-distance-m",
        type=float,
        default=20.0,
        help="Ignore ray intersections farther than this distance. Use <=0 to disable.",
    )
    parser.add_argument(
        "--exclude-categories",
        default="shelter",
        help="Comma-separated object categories to ignore. Default excludes the room envelope.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_names = args.sequence_names or discover_sequence_names(
        args.reports_dir,
        "gaze",
        "gaze_samples.csv",
    )
    if not sequence_names:
        raise ValueError(f"No gaze samples found under {args.reports_dir}")

    max_hit_distance_m = args.max_hit_distance_m if args.max_hit_distance_m > 0 else None
    exclude_categories = _split_csv_list(args.exclude_categories)
    batch_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, sequence_name in enumerate(sequence_names, start=1):
        try:
            gaze_csv = find_sequence_file(
                args.reports_dir,
                sequence_name,
                "gaze",
                "gaze_samples.csv",
            )
            object_boxes_csv = find_sequence_file(
                args.reports_dir,
                sequence_name,
                "scene",
                "scene_object_boxes.csv",
            )
            output_csv = default_gaze_object_hits_csv_path(
                sequence_name,
                output_dir=args.reports_dir,
            )
            summary_json = default_gaze_object_hits_summary_json_path(output_csv)
            gaze_samples = read_samples_csv(gaze_csv)
            rows = compute_gaze_object_hit_rows(
                sequence_name,
                gaze_samples,
                object_boxes_csv,
                max_dynamic_dt_ns=int(args.max_dynamic_dt_ms * 1e6),
                min_hit_distance_m=args.min_hit_distance_m,
                max_hit_distance_m=max_hit_distance_m,
                exclude_categories=exclude_categories,
            )
            write_gaze_object_hits_csv(output_csv, rows)
            summary = summarize_gaze_object_hits(rows)
            summary.update(
                {
                    "sequence_name": sequence_name,
                    "gaze_csv": str(gaze_csv),
                    "object_boxes_csv": str(object_boxes_csv),
                    "output_csv": str(output_csv),
                    "parameters": {
                        "max_dynamic_dt_ms": args.max_dynamic_dt_ms,
                        "min_hit_distance_m": args.min_hit_distance_m,
                        "max_hit_distance_m": max_hit_distance_m,
                        "exclude_categories": exclude_categories,
                    },
                }
            )
            write_json(summary_json, summary)
            batch_rows.append(_batch_row(summary, summary_json))
            print(
                f"[{index}/{len(sequence_names)}] {sequence_name}: "
                f"ray_hit_ratio={summary['object_hit_ratio']:.3f} "
                f"point_inside_ratio={summary['gaze_point_inside_any_box_ratio']:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - keep batch runs informative.
            failures.append({"sequence_name": sequence_name, "error": str(exc)})
            print(f"[{index}/{len(sequence_names)}] {sequence_name}: FAILED {exc}")

    output_batch_dir = batch_dir(args.reports_dir)
    batch_csv = output_batch_dir / "batch_gaze_object_hits_summary.csv"
    batch_json = output_batch_dir / "batch_gaze_object_hits_report.json"
    _write_batch_csv(batch_csv, batch_rows)
    report = {
        "sequence_count": len(batch_rows),
        "failure_count": len(failures),
        "failures": failures,
        "mean_object_hit_ratio": _mean([row["object_hit_ratio"] for row in batch_rows]),
        "mean_gaze_point_inside_any_box_ratio": _mean(
            [row["gaze_point_inside_any_box_ratio"] for row in batch_rows]
        ),
        "rows": batch_rows,
    }
    write_json(batch_json, report)
    print(f"sequences: {len(batch_rows)}")
    print(f"failures: {len(failures)}")
    print(f"batch_csv: {batch_csv}")
    print(f"batch_json: {batch_json}")


def _batch_row(summary: dict[str, Any], summary_json: Path) -> dict[str, Any]:
    return {
        "sequence_name": summary["sequence_name"],
        "sample_count": summary["sample_count"],
        "valid_ray_count": summary["valid_ray_count"],
        "object_hit_count": summary["object_hit_count"],
        "object_hit_ratio": summary["object_hit_ratio"],
        "gaze_point_available_count": summary["gaze_point_available_count"],
        "gaze_point_inside_any_box_count": summary["gaze_point_inside_any_box_count"],
        "gaze_point_inside_any_box_ratio": summary["gaze_point_inside_any_box_ratio"],
        "gaze_point_inside_hit_box_count": summary["gaze_point_inside_hit_box_count"],
        "mean_candidate_box_count": summary["mean_candidate_box_count"],
        "output_csv": summary["output_csv"],
        "summary_json": str(summary_json),
    }


def _write_batch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _split_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
