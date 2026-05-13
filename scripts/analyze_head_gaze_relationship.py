#!/usr/bin/env python
"""Analyze head-gaze relationships from existing gaze/head CSV exports.

This script does not reopen the ADT provider. It reads previously extracted:
- `<sequence>_gaze_samples.csv`
- `<sequence>_head_samples.csv`

Outputs:
- `<sequence>_head_gaze_analysis_rows.csv`
- `<sequence>_head_gaze_analysis_summary.json`
- `batch_head_gaze_analysis_summary.csv`
- `batch_head_gaze_analysis_report.json`

zh-CN:
这一步的目标只收敛在 GT head-gaze relationship 上，而不是模型设定分析：
- Scene 几何关系：head 朝向和 gaze 朝向的夹角
- Local 动态关系：gaze 速度 / 角变化方向和 head motion 的关系
- 辅助时序关系：当前 head motion 和下一帧 gaze change 的 lagged relation

Important:
This analysis requires the refactored `head_samples.csv` schema that includes
absolute Scene axes plus relative rotation fields. Older head CSV exports are
rejected on purpose; re-run head extraction first.
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
from adt_sandbox.head_gaze_analysis import (  # noqa: E402
    build_head_gaze_analysis_rows,
    default_head_gaze_analysis_rows_csv_path,
    default_head_gaze_analysis_summary_json_path,
    summarize_batch_head_gaze_analysis,
    summarize_head_gaze_analysis_rows,
    write_batch_csv,
    write_head_gaze_analysis_rows_csv,
    write_summary_json,
)
from adt_sandbox.results import batch_dir, discover_sequence_names as discover_feature_sequence_names, find_sequence_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence_names",
        nargs="*",
        help="Optional sequence names. If omitted, process all paired gaze/head CSVs in --reports-dir.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory containing per-sequence gaze/head CSV files.",
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
        help="Centered dispersion window used when rebuilding local gaze dynamics features. Default is 5.",
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
        gaze_samples = read_samples_csv(gaze_csv)
        head_samples = read_head_samples_csv(head_csv)

        analysis_rows = build_head_gaze_analysis_rows(
            gaze_samples,
            head_samples,
            dispersion_window_frames=args.dispersion_window_frames,
        )
        output_csv = default_head_gaze_analysis_rows_csv_path(
            sequence_name,
            output_dir=output_dir,
        )
        summary_json = default_head_gaze_analysis_summary_json_path(output_csv)
        write_head_gaze_analysis_rows_csv(output_csv, analysis_rows)

        summary = summarize_head_gaze_analysis_rows(analysis_rows)
        summary.update(
            {
                "sequence_name": sequence_name,
                "input_gaze_csv": str(gaze_csv),
                "input_head_csv": str(head_csv),
                "output_csv": str(output_csv),
                "dispersion_window_frames": args.dispersion_window_frames,
                "method": {
                    "geometry": "scene-frame gaze/head direction relation",
                    "dynamics": "local gaze deltas, velocity, and direction vs head motion",
                    "directional_analysis": (
                        "signed yaw/pitch deltas vs signed relative head rotation "
                        "vector components in the previous head frame"
                    ),
                    "auxiliary_temporal_proxy": "current head motion vs next-step gaze change lagged correlation",
                    "head_representation": "scene absolute pose + relative head motion",
                    "gaze_representation": "CPF local dynamics + scene direction relation",
                },
            }
        )
        write_summary_json(summary_json, summary)

        batch_rows.append(sequence_batch_row(summary))
        print(
            f"[{index}/{len(sequence_names)}] {sequence_name}: "
            f"valid={summary['dynamics_input_valid_ratio']:.3f} "
            f"gaze_head_angle_p50={summary['geometry']['gaze_head_angle_deg']['p50']:.3f} "
            f"corr_next_vel_head_rot="
            f"{format_optional(batch_rows[-1]['corr_next_local_velocity_vs_current_head_rotation_speed'])}"
        )

    batch_output_dir = batch_dir(output_dir)
    batch_csv = batch_output_dir / "batch_head_gaze_analysis_summary.csv"
    batch_json = batch_output_dir / "batch_head_gaze_analysis_report.json"
    write_batch_csv(batch_csv, batch_rows)
    write_summary_json(batch_json, summarize_batch_head_gaze_analysis(batch_rows))
    print(f"sequences: {len(batch_rows)}")
    print(f"batch_csv: {batch_csv}")
    print(f"batch_json: {batch_json}")


def discover_sequence_names(reports_dir: Path) -> list[str]:
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {reports_dir}")
    gaze_names = set(
        discover_feature_sequence_names(
            reports_dir,
            "gaze",
            "gaze_samples.csv",
        )
    )
    head_names = set(
        discover_feature_sequence_names(
            reports_dir,
            "head",
            "head_samples.csv",
        )
    )
    names = sorted(gaze_names & head_names)
    if not names:
        raise ValueError(
            f"No paired gaze/head sample CSV files found in: {reports_dir}"
        )
    return names


def sequence_batch_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence_name": summary["sequence_name"],
        "sample_count": summary["sample_count"],
        "dynamics_input_valid_ratio": summary["dynamics_input_valid_ratio"],
        "median_gaze_head_angle_deg": summary["geometry"]["gaze_head_angle_deg"]["p50"],
        "mean_local_velocity_deg_s": summary["dynamics"]["local_velocity_deg_s"]["mean"],
        "mean_head_rotation_speed_deg_s": summary["dynamics"]["head_rotation_speed_deg_s"]["mean"],
        "mean_head_translation_speed_m_s": summary["dynamics"]["head_translation_speed_m_s"]["mean"],
        "corr_current_local_velocity_vs_head_rotation_speed": summary["correlations"][
            "current_local_velocity_vs_head_rotation_speed"
        ],
        "corr_current_local_velocity_vs_head_translation_speed": summary["correlations"][
            "current_local_velocity_vs_head_translation_speed"
        ],
        "corr_signed_delta_yaw_vs_head_rotvec_y": summary["correlations"][
            "signed_delta_yaw_vs_head_rotvec_y"
        ],
        "corr_signed_delta_pitch_vs_head_rotvec_x": summary["correlations"][
            "signed_delta_pitch_vs_head_rotvec_x"
        ],
        "corr_abs_delta_yaw_vs_abs_head_rotvec_y": summary["correlations"][
            "abs_delta_yaw_vs_abs_head_rotvec_y"
        ],
        "corr_abs_delta_pitch_vs_abs_head_rotvec_x": summary["correlations"][
            "abs_delta_pitch_vs_abs_head_rotvec_x"
        ],
        "mean_gaze_head_motion_alignment_2d": summary["directional_alignment"][
            "gaze_head_motion_alignment_2d"
        ]["mean"],
        "gaze_head_motion_aligned_fraction": summary["directional_alignment"][
            "aligned_fraction"
        ],
        "gaze_head_motion_opposed_fraction": summary["directional_alignment"][
            "opposed_fraction"
        ],
        "gaze_head_motion_weak_or_orthogonal_fraction": summary["directional_alignment"][
            "weak_or_orthogonal_fraction"
        ],
        "corr_next_local_velocity_vs_current_head_rotation_speed": summary["correlations"][
            "next_local_velocity_vs_current_head_rotation_speed"
        ],
        "corr_next_local_velocity_vs_current_head_translation_speed": summary["correlations"][
            "next_local_velocity_vs_current_head_translation_speed"
        ],
        "output_csv": summary["output_csv"],
    }


def format_optional(value: float | None) -> str:
    return "nan" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
