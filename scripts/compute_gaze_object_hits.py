#!/usr/bin/env python
"""Compute Scene-frame gaze ray hits against ADT object boxes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.results import find_sequence_file  # noqa: E402
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
    parser.add_argument("sequence", help="Sequence id in the organized reports directory.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Organized reports root containing gaze and scene object box CSVs.",
    )
    parser.add_argument(
        "--gaze-csv",
        type=Path,
        default=None,
        help="Optional explicit gaze_samples.csv path.",
    )
    parser.add_argument(
        "--object-boxes-csv",
        type=Path,
        default=None,
        help="Optional explicit scene_object_boxes.csv path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional explicit gaze_object_hits.csv path.",
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
    sequence_name = args.sequence
    gaze_csv = args.gaze_csv or find_sequence_file(
        args.reports_dir,
        sequence_name,
        "gaze",
        "gaze_samples.csv",
    )
    object_boxes_csv = args.object_boxes_csv or find_sequence_file(
        args.reports_dir,
        sequence_name,
        "scene",
        "scene_object_boxes.csv",
    )
    output_csv = args.output_csv or default_gaze_object_hits_csv_path(
        sequence_name,
        output_dir=args.reports_dir,
    )
    summary_json = default_gaze_object_hits_summary_json_path(output_csv)
    max_hit_distance_m = args.max_hit_distance_m if args.max_hit_distance_m > 0 else None
    exclude_categories = _split_csv_list(args.exclude_categories)

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
            "coordinate_frame": "ADT Scene/world frame",
            "method": {
                "ray": "first intersection between gaze_origin_scene + t * gaze_dir_scene_unit and oriented object cuboids",
                "dynamic_objects": "nearest object-box timestamp within max_dynamic_dt_ms",
                "depth_point_check": "separate check using gaze_point_scene_xyz inside object cuboids",
            },
            "parameters": {
                "max_dynamic_dt_ms": args.max_dynamic_dt_ms,
                "min_hit_distance_m": args.min_hit_distance_m,
                "max_hit_distance_m": max_hit_distance_m,
                "exclude_categories": exclude_categories,
            },
        }
    )
    write_json(summary_json, summary)
    print(
        f"{sequence_name}: samples={summary['sample_count']} "
        f"ray_hit_ratio={summary['object_hit_ratio']:.3f} "
        f"point_inside_ratio={summary['gaze_point_inside_any_box_ratio']:.3f}"
    )
    print(f"csv: {output_csv}")
    print(f"json: {summary_json}")


def _split_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
