#!/usr/bin/env python
"""Copy a flat ADT reports directory into the organized sequence-first layout.

This is intentionally non-destructive: it copies files and never deletes the
source directory. By default it runs in dry-run mode. Pass `--execute` to copy.

Example:
    python scripts/organize_flat_reports.py \
      /mnt/d/SparseGaze/ADT-Gaze \
      /mnt/d/SparseGaze/ADT-Gaze-structured \
      --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


SUFFIX_RULES: list[tuple[str, str, str]] = [
    ("_scene_gaze_event_features.csv", "events", "scene_gaze_event_features.csv"),
    ("_scene_gaze_event_segments.csv", "events", "scene_gaze_event_segments.csv"),
    ("_scene_gaze_event_summary.json", "events", "scene_gaze_event_summary.json"),
    ("_scene_gaze_frame_labels.csv", "events", "scene_gaze_frame_labels.csv"),
    ("_scene_head_gaze_analysis_rows.csv", "analysis", "scene_head_gaze_analysis_rows.csv"),
    (
        "_scene_head_gaze_analysis_summary.json",
        "analysis",
        "scene_head_gaze_analysis_summary.json",
    ),
    ("_scene_object_boxes_summary.json", "scene", "scene_object_boxes_summary.json"),
    ("_scene_object_boxes.csv", "scene", "scene_object_boxes.csv"),
    ("_sparsegaze_head_utility_lead_lag.csv", "analysis", "sparsegaze_head_utility_lead_lag.csv"),
    (
        "_sparsegaze_head_utility_summary.csv",
        "analysis",
        "sparsegaze_head_utility_summary.csv",
    ),
    ("_head_gaze_analysis_summary.json", "analysis", "head_gaze_analysis_summary.json"),
    ("_head_gaze_analysis_rows.csv", "analysis", "head_gaze_analysis_rows.csv"),
    ("_gaze_dynamics_summary.json", "dynamics", "gaze_dynamics_summary.json"),
    ("_gaze_dynamics.csv", "dynamics", "gaze_dynamics.csv"),
    ("_event_features_summary.json", "dynamics", "event_features_summary.json"),
    ("_event_features.csv", "dynamics", "event_features.csv"),
    ("_fixation_frame_labels_combined.csv", "events_legacy", "fixation_frame_labels_combined.csv"),
    ("_fixation_frame_labels_idt.csv", "events_legacy", "fixation_frame_labels_idt.csv"),
    ("_fixation_frame_labels_ivt.csv", "events_legacy", "fixation_frame_labels_ivt.csv"),
    ("_fixation_segments_combined.csv", "events_legacy", "fixation_segments_combined.csv"),
    ("_fixation_segments_idt.csv", "events_legacy", "fixation_segments_idt.csv"),
    ("_fixation_segments_ivt.csv", "events_legacy", "fixation_segments_ivt.csv"),
    ("_fixation_summary_combined.json", "events_legacy", "fixation_summary_combined.json"),
    ("_fixation_summary_idt.json", "events_legacy", "fixation_summary_idt.json"),
    ("_fixation_summary_ivt.json", "events_legacy", "fixation_summary_ivt.json"),
    ("_gaze_events_combined.csv", "events_legacy", "gaze_events_combined.csv"),
    ("_gaze_events_idt.csv", "events_legacy", "gaze_events_idt.csv"),
    ("_gaze_events_ivt.csv", "events_legacy", "gaze_events_ivt.csv"),
    ("_gaze_events_summary.json", "events_legacy", "gaze_events_summary.json"),
    ("_gaze_events.csv", "events_legacy", "gaze_events.csv"),
    ("_frame_event_labels.csv", "events_legacy", "frame_event_labels.csv"),
    ("_skeleton_summary.json", "skeleton", "skeleton_summary.json"),
    ("_skeleton_samples.csv", "skeleton", "skeleton_samples.csv"),
    ("_head_summary.json", "head", "head_summary.json"),
    ("_head_samples.csv", "head", "head_samples.csv"),
    ("_gaze_summary.json", "gaze", "gaze_summary.json"),
    ("_gaze_samples.csv", "gaze", "gaze_samples.csv"),
]


@dataclass(frozen=True)
class CopyPlanRow:
    source: str
    destination: str
    kind: str
    sequence_name: str
    layer: str
    action: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Old flat reports directory.")
    parser.add_argument("output_dir", type=Path, help="New organized reports directory.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually copy files. Without this flag the script only prints a dry-run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    rows = build_copy_plan(input_dir, output_dir, overwrite=args.overwrite)

    action_counts: dict[str, int] = {}
    layer_counts: dict[str, int] = {}
    for row in rows:
        action_counts[row.action] = action_counts.get(row.action, 0) + 1
        layer_counts[row.layer] = layer_counts.get(row.layer, 0) + 1

    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"mode: {'execute' if args.execute else 'dry-run'}")
    print(f"files: {len(rows)}")
    print(f"actions: {dict(sorted(action_counts.items()))}")
    print(f"layers: {dict(sorted(layer_counts.items()))}")

    if args.execute:
        execute_copy_plan(rows)

    manifest_csv = output_dir / "organization_manifest.csv"
    manifest_json = output_dir / "organization_manifest.json"
    if args.execute:
        write_manifest_csv(manifest_csv, rows)
        write_manifest_json(manifest_json, rows, input_dir, output_dir)
        print(f"manifest_csv: {manifest_csv}")
        print(f"manifest_json: {manifest_json}")
    else:
        for row in rows[:20]:
            print(f"{row.action}: {row.source} -> {row.destination}")
        if len(rows) > 20:
            print(f"... {len(rows) - 20} more rows")


def build_copy_plan(input_dir: Path, output_dir: Path, overwrite: bool) -> list[CopyPlanRow]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Expected input directory: {input_dir}")

    rows: list[CopyPlanRow] = []
    for source in sorted(path for path in input_dir.iterdir() if path.is_file()):
        sequence_name, layer, destination_name, kind = classify_file(source.name)
        if kind == "batch":
            destination = output_dir / "batch" / destination_name
        elif sequence_name:
            destination = output_dir / "sequences" / sequence_name / layer / destination_name
        else:
            destination = output_dir / "misc" / destination_name

        action = "copy"
        if destination.exists() and not overwrite:
            action = "skip_exists"
        elif destination.exists() and overwrite:
            action = "overwrite"

        rows.append(
            CopyPlanRow(
                source=str(source),
                destination=str(destination),
                kind=kind,
                sequence_name=sequence_name,
                layer=layer,
                action=action,
            )
        )
    return rows


def classify_file(filename: str) -> tuple[str, str, str, str]:
    if (
        filename.startswith("batch_")
        or filename.startswith("gaze_quality_report.")
        or filename.startswith("fixation_policy_comparison.")
    ):
        return "", "batch", filename, "batch"

    for suffix, layer, destination_name in SUFFIX_RULES:
        if filename.endswith(suffix):
            sequence_name = filename[: -len(suffix)]
            return sequence_name, layer, destination_name, "sequence"

    return "", "misc", filename, "misc"


def execute_copy_plan(rows: list[CopyPlanRow]) -> None:
    for row in rows:
        if row.action == "skip_exists":
            continue
        source = Path(row.source)
        destination = Path(row.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_manifest_csv(path: Path, rows: list[CopyPlanRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_manifest_json(
    path: Path,
    rows: list[CopyPlanRow],
    input_dir: Path,
    output_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "file_count": len(rows),
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
