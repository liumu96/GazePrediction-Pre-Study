#!/usr/bin/env python
"""Detect scene-direction gaze events from existing gaze sample CSV files.

This script does not reopen the ADT provider. It reads previously extracted
`*_gaze_samples.csv` files and writes a scene/world-direction event layer:

- `<sequence>_scene_gaze_event_features.csv`
- `<sequence>_scene_gaze_frame_labels.csv`
- `<sequence>_scene_gaze_event_segments.csv`
- `<sequence>_scene_gaze_event_summary.json`

The default rule is conservative: a frame is a fixation candidate only when
Scene-frame gaze velocity and centered Scene-frame dispersion are both below
threshold, and short fixation runs are demoted to transition.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.scene_gaze_events import (  # noqa: E402
    compute_scene_gaze_event_features,
    default_scene_gaze_event_features_csv_path,
    default_scene_gaze_event_segments_csv_path,
    default_scene_gaze_event_summary_json_path,
    default_scene_gaze_frame_labels_csv_path,
    label_scene_gaze_events,
    summarize_scene_gaze_events,
    write_scene_gaze_event_features_csv,
    write_scene_gaze_event_segments_csv,
    write_scene_gaze_frame_labels_csv,
    write_summary_json,
)
from adt_sandbox.results import batch_dir, discover_sequence_names as discover_feature_sequence_names, find_sequence_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence_names",
        nargs="*",
        help="Optional sequence names. If omitted, process all *_gaze_samples.csv in --reports-dir.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory containing per-sequence gaze sample CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for scene event outputs. Default is the same as --reports-dir.",
    )
    parser.add_argument(
        "--dispersion-window-frames",
        type=int,
        default=5,
        help="Centered Scene-gaze dispersion window in frames. Default is 5.",
    )
    parser.add_argument(
        "--velocity-threshold-deg-s",
        type=float,
        default=40.0,
        help="Scene angular velocity threshold for fixation frames. Default is 40 deg/s.",
    )
    parser.add_argument(
        "--dispersion-threshold-deg",
        type=float,
        default=2.5,
        help="Scene angular dispersion threshold for fixation frames. Default is 2.5 deg.",
    )
    parser.add_argument(
        "--min-fixation-duration-ms",
        type=float,
        default=133.0,
        help="Minimum final fixation segment duration. Default is 133 ms.",
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
        gaze_csv = find_sequence_file(
            reports_dir,
            sequence_name,
            "gaze",
            "gaze_samples.csv",
        )
        gaze_samples = read_samples_csv(gaze_csv)
        feature_rows = compute_scene_gaze_event_features(
            gaze_samples,
            dispersion_window_frames=args.dispersion_window_frames,
        )
        frame_labels, segments = label_scene_gaze_events(
            feature_rows,
            velocity_threshold_deg_s=args.velocity_threshold_deg_s,
            dispersion_threshold_deg=args.dispersion_threshold_deg,
            min_fixation_duration_ms=args.min_fixation_duration_ms,
        )

        feature_csv = default_scene_gaze_event_features_csv_path(
            sequence_name,
            output_dir=output_dir,
        )
        frame_label_csv = default_scene_gaze_frame_labels_csv_path(
            sequence_name,
            output_dir=output_dir,
        )
        segment_csv = default_scene_gaze_event_segments_csv_path(
            sequence_name,
            output_dir=output_dir,
        )
        summary_json = default_scene_gaze_event_summary_json_path(
            sequence_name,
            output_dir=output_dir,
        )
        write_scene_gaze_event_features_csv(feature_csv, feature_rows)
        write_scene_gaze_frame_labels_csv(frame_label_csv, frame_labels)
        write_scene_gaze_event_segments_csv(segment_csv, segments)

        summary = summarize_scene_gaze_events(feature_rows, frame_labels, segments)
        summary.update(
            {
                "sequence_name": sequence_name,
                "input_gaze_csv": str(gaze_csv),
                "output_feature_csv": str(feature_csv),
                "output_frame_label_csv": str(frame_label_csv),
                "output_segment_csv": str(segment_csv),
                "dispersion_window_frames": args.dispersion_window_frames,
                "velocity_threshold_deg_s": args.velocity_threshold_deg_s,
                "dispersion_threshold_deg": args.dispersion_threshold_deg,
                "min_fixation_duration_ms": args.min_fixation_duration_ms,
                "method": {
                    "event_space": "Scene-frame unit gaze direction",
                    "fixation_rule": (
                        "scene_velocity_deg_s <= velocity_threshold_deg_s and "
                        "scene_window_dispersion_deg <= dispersion_threshold_deg"
                    ),
                    "duration_rule": "short fixation runs are demoted to transition",
                    "event_labels": ["fixation", "transition", "invalid"],
                },
            }
        )
        write_summary_json(summary_json, summary)

        label_counts = summary["frame_label_counts"]
        batch_rows.append(
            {
                "sequence_name": sequence_name,
                "sample_count": summary["sample_count"],
                "scene_gaze_valid_ratio": summary["scene_gaze_valid_ratio"],
                "fixation_frame_count": label_counts.get("fixation", 0),
                "transition_frame_count": label_counts.get("transition", 0),
                "invalid_frame_count": label_counts.get("invalid", 0),
                "fixation_count": summary["fixation_count"],
                "scene_velocity_p95_deg_s": summary["scene_velocity_deg_s"]["p95"],
                "scene_dispersion_p95_deg": summary["scene_window_dispersion_deg"]["p95"],
                "fixation_duration_p50_ms": summary["fixation_duration_ms"]["p50"],
                "fixation_duration_p95_ms": summary["fixation_duration_ms"]["p95"],
                "frame_label_csv": str(frame_label_csv),
                "segment_csv": str(segment_csv),
                "summary_json": str(summary_json),
            }
        )
        print(
            f"[{index}/{len(sequence_names)}] {sequence_name}: "
            f"samples={summary['sample_count']} "
            f"valid={summary['scene_gaze_valid_ratio']:.3f} "
            f"fix_frames={label_counts.get('fixation', 0)} "
            f"fixations={summary['fixation_count']}"
        )

    batch_csv = batch_dir(output_dir) / "batch_scene_gaze_event_summary.csv"
    write_batch_csv(batch_csv, batch_rows)
    print(f"batch_csv: {batch_csv}")


def discover_sequence_names(reports_dir: Path) -> list[str]:
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {reports_dir}")
    names = discover_feature_sequence_names(
        reports_dir,
        "gaze",
        "gaze_samples.csv",
    )
    if not names:
        raise ValueError(f"No gaze sample CSV files found in: {reports_dir}")
    return names


def write_batch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No batch rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
