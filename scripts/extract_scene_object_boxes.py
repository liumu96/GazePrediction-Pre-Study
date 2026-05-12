#!/usr/bin/env python
"""Extract Scene-frame object boxes from ADT object pose and 3D box files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.providers import resolve_sequence_path  # noqa: E402
from adt_sandbox.scene_features import (  # noqa: E402
    default_scene_object_boxes_csv_path,
    default_scene_object_boxes_summary_json_path,
    extract_scene_object_box_rows,
    summarize_scene_object_box_rows,
    write_json,
    write_scene_object_boxes_csv,
)

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        type=Path,
        help="Sequence id resolved under ADT_DATA_ROOT, or an explicit sequence directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory for `<sequence>_scene_object_boxes.csv`.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional explicit output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = resolve_sequence_path(args.sequence)
    sequence_name = sequence_dir.name
    output_csv = args.output_csv or default_scene_object_boxes_csv_path(
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
            "coordinate_frame": {
                "scene_corners": "ADT Scene/world frame, meters",
                "local_aabb": "object local frame, meters",
                "pose": "T_scene_object from scene_objects.csv q_wo/t_wo fields",
            },
            "source_files": {
                "instances": "instances.json",
                "object_pose": "scene_objects.csv",
                "object_3d_box": "3d_bounding_box.csv",
            },
        }
    )
    write_json(summary_json, summary)
    print(
        f"{sequence_name}: rows={summary['row_count']} "
        f"objects={summary['unique_object_count']} timestamps={summary['unique_timestamp_count']}"
    )
    print(f"csv: {output_csv}")
    print(f"json: {summary_json}")


if __name__ == "__main__":
    main()
