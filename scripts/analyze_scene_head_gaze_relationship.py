#!/usr/bin/env python
"""Analyze Scene-level head-gaze relationships from existing CSV exports.

This script joins:

- `<sequence>_gaze_samples.csv`
- `<sequence>_head_samples.csv`
- `<sequence>_scene_gaze_event_features.csv`
- `<sequence>_scene_gaze_frame_labels.csv`

It keeps Scene-world gaze dynamics and CPF-local gaze dynamics side by side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.head import read_head_samples_csv  # noqa: E402
from adt_sandbox.scene_gaze_events import (  # noqa: E402
    read_scene_gaze_event_features_csv,
    read_scene_gaze_frame_labels_csv,
)
from adt_sandbox.scene_head_gaze_analysis import (  # noqa: E402
    build_scene_head_gaze_analysis_rows,
    default_scene_head_gaze_analysis_rows_csv_path,
    default_scene_head_gaze_analysis_summary_json_path,
    summarize_scene_head_gaze_analysis_rows,
    write_batch_csv,
    write_scene_head_gaze_analysis_rows_csv,
    write_summary_json,
)
from adt_sandbox.results import batch_dir, discover_sequence_names as discover_feature_sequence_names, find_sequence_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence_names",
        nargs="*",
        help="Optional sequence names. If omitted, process all paired CSVs in --reports-dir.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory containing gaze/head/scene-event CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for analysis CSV/JSON outputs. Default is the same as --reports-dir.",
    )
    parser.add_argument(
        "--dispersion-window-frames",
        type=int,
        default=5,
        help="Centered dispersion window used when rebuilding CPF-local dynamics. Default is 5.",
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
        head_csv = find_sequence_file(
            reports_dir,
            sequence_name,
            "head",
            "head_samples.csv",
        )
        scene_features_csv = find_sequence_file(
            reports_dir,
            sequence_name,
            "events",
            "scene_gaze_event_features.csv",
        )
        scene_labels_csv = find_sequence_file(
            reports_dir,
            sequence_name,
            "events",
            "scene_gaze_frame_labels.csv",
        )

        rows = build_scene_head_gaze_analysis_rows(
            read_samples_csv(gaze_csv),
            read_head_samples_csv(head_csv),
            read_scene_gaze_event_features_csv(scene_features_csv),
            read_scene_gaze_frame_labels_csv(scene_labels_csv),
            dispersion_window_frames=args.dispersion_window_frames,
        )
        output_csv = default_scene_head_gaze_analysis_rows_csv_path(
            sequence_name,
            output_dir=output_dir,
        )
        summary_json = default_scene_head_gaze_analysis_summary_json_path(output_csv)
        write_scene_head_gaze_analysis_rows_csv(output_csv, rows)

        summary = summarize_scene_head_gaze_analysis_rows(rows)
        summary.update(
            {
                "sequence_name": sequence_name,
                "input_gaze_csv": str(gaze_csv),
                "input_head_csv": str(head_csv),
                "input_scene_features_csv": str(scene_features_csv),
                "input_scene_labels_csv": str(scene_labels_csv),
                "output_csv": str(output_csv),
                "dispersion_window_frames": args.dispersion_window_frames,
                "method": {
                    "scene_dynamics": "scene gaze angular velocity / dispersion",
                    "cpf_dynamics": "eye-in-head local gaze velocity",
                    "head_dynamics": "relative head rotation / translation",
                    "event_conditioning": "scene-direction fixation / transition labels",
                },
            }
        )
        write_summary_json(summary_json, summary)
        batch_rows.append(sequence_batch_row(summary))
        print(
            f"[{index}/{len(sequence_names)}] {sequence_name}: "
            f"valid={summary['analysis_valid_ratio']:.3f} "
            f"fix_frac={format_optional(batch_rows[-1]['scene_fixation_fraction'])} "
            f"corr_scene_vel_head_rot="
            f"{format_optional(batch_rows[-1]['corr_scene_velocity_vs_head_rotation_speed'])}"
        )

    batch_output_dir = batch_dir(output_dir)
    batch_csv = batch_output_dir / "batch_scene_head_gaze_analysis_summary.csv"
    write_batch_csv(batch_csv, batch_rows)
    batch_json = batch_output_dir / "batch_scene_head_gaze_analysis_report.json"
    write_summary_json(batch_json, summarize_batch_rows(batch_rows))
    print(f"sequences: {len(batch_rows)}")
    print(f"batch_csv: {batch_csv}")
    print(f"batch_json: {batch_json}")


def discover_sequence_names(reports_dir: Path) -> list[str]:
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {reports_dir}")
    name_sets = [
        set(
            discover_feature_sequence_names(
                reports_dir,
                "gaze",
                "gaze_samples.csv",
            )
        ),
        set(
            discover_feature_sequence_names(
                reports_dir,
                "head",
                "head_samples.csv",
            )
        ),
        set(
            discover_feature_sequence_names(
                reports_dir,
                "events",
                "scene_gaze_event_features.csv",
            )
        ),
        set(
            discover_feature_sequence_names(
                reports_dir,
                "events",
                "scene_gaze_frame_labels.csv",
            )
        ),
    ]
    names = sorted(set.intersection(*name_sets))
    if not names:
        raise ValueError(f"No complete scene head-gaze input sets found in: {reports_dir}")
    return names


def sequence_batch_row(summary: dict[str, Any]) -> dict[str, Any]:
    fixation_group = summary["event_groups"]["fixation"]
    transition_group = summary["event_groups"]["transition"]
    high_head_group = summary["head_rotation_speed_groups"]["groups"]["high"]
    low_head_group = summary["head_rotation_speed_groups"]["groups"]["low"]
    mid_head_group = summary["head_rotation_speed_groups"]["groups"]["mid"]
    return {
        "sequence_name": summary["sequence_name"],
        "sample_count": summary["sample_count"],
        "analysis_valid_ratio": summary["analysis_valid_ratio"],
        "scene_fixation_fraction": summary["scene_event_label_fractions"].get(
            "fixation",
            0.0,
        ),
        "scene_transition_fraction": summary["scene_event_label_fractions"].get(
            "transition",
            0.0,
        ),
        "corr_scene_velocity_vs_head_rotation_speed": summary["correlations"][
            "scene_velocity_vs_head_rotation_speed"
        ],
        "corr_scene_velocity_vs_head_translation_speed": summary["correlations"][
            "scene_velocity_vs_head_translation_speed"
        ],
        "corr_cpf_velocity_vs_head_rotation_speed": summary["correlations"][
            "cpf_velocity_vs_head_rotation_speed"
        ],
        "corr_cpf_velocity_vs_scene_velocity": summary["correlations"][
            "cpf_velocity_vs_scene_velocity"
        ],
        "mean_scene_velocity_fixation_deg_s": fixation_group["scene_velocity_deg_s"][
            "mean"
        ],
        "mean_scene_velocity_transition_deg_s": transition_group[
            "scene_velocity_deg_s"
        ]["mean"],
        "mean_cpf_velocity_fixation_deg_s": fixation_group[
            "cpf_local_velocity_deg_s"
        ]["mean"],
        "mean_cpf_velocity_transition_deg_s": transition_group[
            "cpf_local_velocity_deg_s"
        ]["mean"],
        "mean_head_rotation_fixation_deg_s": fixation_group[
            "head_rotation_speed_deg_s"
        ]["mean"],
        "mean_head_rotation_transition_deg_s": transition_group[
            "head_rotation_speed_deg_s"
        ]["mean"],
        "high_head_fixation_fraction": high_head_group["fixation_fraction"],
        "mid_head_fixation_fraction": mid_head_group["fixation_fraction"],
        "low_head_fixation_fraction": low_head_group["fixation_fraction"],
        "low_head_mean_scene_velocity_deg_s": low_head_group["scene_velocity_deg_s"][
            "mean"
        ],
        "mid_head_mean_scene_velocity_deg_s": mid_head_group["scene_velocity_deg_s"][
            "mean"
        ],
        "high_head_mean_scene_velocity_deg_s": high_head_group["scene_velocity_deg_s"][
            "mean"
        ],
        "low_head_mean_cpf_velocity_deg_s": low_head_group[
            "cpf_local_velocity_deg_s"
        ]["mean"],
        "mid_head_mean_cpf_velocity_deg_s": mid_head_group["cpf_local_velocity_deg_s"][
            "mean"
        ],
        "high_head_mean_cpf_velocity_deg_s": high_head_group[
            "cpf_local_velocity_deg_s"
        ]["mean"],
        "output_csv": summary["output_csv"],
    }


def summarize_batch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sequence_count": len(rows),
        "analysis_valid_ratio": describe_column(rows, "analysis_valid_ratio"),
        "scene_fixation_fraction": describe_column(rows, "scene_fixation_fraction"),
        "correlations": {
            "scene_velocity_vs_head_rotation_speed": describe_column(
                rows,
                "corr_scene_velocity_vs_head_rotation_speed",
            ),
            "cpf_velocity_vs_head_rotation_speed": describe_column(
                rows,
                "corr_cpf_velocity_vs_head_rotation_speed",
            ),
            "cpf_velocity_vs_scene_velocity": describe_column(
                rows,
                "corr_cpf_velocity_vs_scene_velocity",
            ),
        },
    }


def describe_column(rows: list[dict[str, Any]], name: str) -> dict[str, float | int | None]:
    values = [
        float(row[name])
        for row in rows
        if row.get(name) is not None and np_isfinite(row[name])
    ]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def np_isfinite(value: Any) -> bool:
    import numpy as np

    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def format_optional(value: float | None) -> str:
    return "nan" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
