#!/usr/bin/env python
"""Extract head-proxy features aligned to an existing ADT gaze CSV.

This script reads the query timestamps from an existing `gaze_samples.csv`,
queries the nearest ADT pose on those timestamps, and writes a `head_samples.csv`
plus a lightweight `head_summary.json`.

zh-CN:
第一版 event analysis 不从 skeleton 提 head，而是直接用 device/CPF pose 作为
head proxy。这样 head 和当前 gaze 的 CPF / Scene 链天然对齐。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.head import (  # noqa: E402
    add_temporal_head_context,
    default_head_csv_path,
    default_head_summary_json_path,
    extract_head_sample,
    summarize_head_samples,
    write_head_samples_csv,
    write_head_summary_json,
)
from adt_sandbox.providers import create_adt_providers  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        type=Path,
        help="ADT sequence id resolved under ADT_DATA_ROOT, or an absolute sequence path.",
    )
    parser.add_argument(
        "--input-gaze-csv",
        type=Path,
        default=None,
        help="Input gaze CSV path. Defaults to outputs/reports/<sequence>_gaze_samples.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output head CSV path. Defaults to outputs/reports/<sequence>_head_samples.csv.",
    )
    return parser.parse_args()


def default_gaze_csv_path(sequence: Path) -> Path:
    return REPO_ROOT / "outputs" / "reports" / f"{sequence.name}_gaze_samples.csv"


def main() -> None:
    args = parse_args()
    gaze_csv = args.input_gaze_csv or default_gaze_csv_path(args.sequence)
    gaze_samples = read_samples_csv(gaze_csv)

    providers = create_adt_providers(args.sequence, skeleton_flag=True)
    output_csv = args.output_csv or default_head_csv_path(providers.sequence_path.name)
    summary_json = default_head_summary_json_path(output_csv)

    head_samples = [
        extract_head_sample(providers.gt_provider, sample.query_timestamp_ns)
        for sample in gaze_samples
    ]
    head_samples = add_temporal_head_context(head_samples)
    write_head_samples_csv(output_csv, head_samples)

    summary = summarize_head_samples(head_samples)
    summary.update(
        {
            "sequence_name": providers.sequence_path.name,
            "sequence_path": str(providers.sequence_path),
            "provider_mode": providers.provider_mode,
            "input_gaze_csv": str(gaze_csv),
            "output_csv": str(output_csv),
            "head_proxy_source": "device_pose_cpf",
            "field_coordinate_frames": {
                "query_timestamp_ns": "device_time_ns",
                "pose_dt_ns": "device_time_ns_delta",
                "head_origin_scene_x_m": "adt_scene_frame_m",
                "head_origin_scene_y_m": "adt_scene_frame_m",
                "head_origin_scene_z_m": "adt_scene_frame_m",
                "head_right_scene_unit_x": "adt_scene_frame_unit_direction",
                "head_right_scene_unit_y": "adt_scene_frame_unit_direction",
                "head_right_scene_unit_z": "adt_scene_frame_unit_direction",
                "head_up_scene_unit_x": "adt_scene_frame_unit_direction",
                "head_up_scene_unit_y": "adt_scene_frame_unit_direction",
                "head_up_scene_unit_z": "adt_scene_frame_unit_direction",
                "head_forward_scene_unit_x": "adt_scene_frame_unit_direction",
                "head_forward_scene_unit_y": "adt_scene_frame_unit_direction",
                "head_forward_scene_unit_z": "adt_scene_frame_unit_direction",
                "head_rot_scene_r00": "rotation_matrix_scene_from_cpf",
                "head_rot_scene_r01": "rotation_matrix_scene_from_cpf",
                "head_rot_scene_r02": "rotation_matrix_scene_from_cpf",
                "head_rot_scene_r10": "rotation_matrix_scene_from_cpf",
                "head_rot_scene_r11": "rotation_matrix_scene_from_cpf",
                "head_rot_scene_r12": "rotation_matrix_scene_from_cpf",
                "head_rot_scene_r20": "rotation_matrix_scene_from_cpf",
                "head_rot_scene_r21": "rotation_matrix_scene_from_cpf",
                "head_rot_scene_r22": "rotation_matrix_scene_from_cpf",
                "translation_scene_dx_m": "adt_scene_frame_m",
                "translation_scene_dy_m": "adt_scene_frame_m",
                "translation_scene_dz_m": "adt_scene_frame_m",
                "translation_prev_head_dx_m": "previous_head_frame_m",
                "translation_prev_head_dy_m": "previous_head_frame_m",
                "translation_prev_head_dz_m": "previous_head_frame_m",
                "origin_step_m": "adt_scene_frame_m",
                "head_translation_speed_m_s": "adt_scene_frame_m_per_s",
                "relative_rot_prev_to_cur_r00": "rotation_matrix_previous_head_to_current_head",
                "relative_rot_prev_to_cur_r01": "rotation_matrix_previous_head_to_current_head",
                "relative_rot_prev_to_cur_r02": "rotation_matrix_previous_head_to_current_head",
                "relative_rot_prev_to_cur_r10": "rotation_matrix_previous_head_to_current_head",
                "relative_rot_prev_to_cur_r11": "rotation_matrix_previous_head_to_current_head",
                "relative_rot_prev_to_cur_r12": "rotation_matrix_previous_head_to_current_head",
                "relative_rot_prev_to_cur_r20": "rotation_matrix_previous_head_to_current_head",
                "relative_rot_prev_to_cur_r21": "rotation_matrix_previous_head_to_current_head",
                "relative_rot_prev_to_cur_r22": "rotation_matrix_previous_head_to_current_head",
                "head_forward_angle_step_deg": "scene_frame_angle_deg",
                "head_rotation_angle_step_deg": "relative_rotation_angle_deg",
                "head_rotation_speed_deg_s": "relative_rotation_angle_deg_per_s",
            },
            "field_definitions": {
                "head_proxy_source": (
                    "Device pose plus CPF calibration, used as the first head-motion "
                    "proxy before introducing skeleton-based head signals."
                ),
                "head_origin_scene_xyz": "Scene-frame CPF origin.",
                "head_right_scene_unit_xyz": "Scene-frame CPF +X unit axis.",
                "head_up_scene_unit_xyz": "Scene-frame CPF +Y unit axis.",
                "head_forward_scene_unit_xyz": (
                    "Scene-frame unit forward direction obtained by transforming "
                    "the CPF +Z axis."
                ),
                "head_rot_scene_rij": (
                    "Entries of the Scene-from-CPF rotation matrix. Columns correspond "
                    "to CPF right / up / forward axes expressed in Scene frame."
                ),
                "translation_scene_dxyz_m": (
                    "Frame-to-frame translation in Scene frame between consecutive valid "
                    "head samples."
                ),
                "translation_prev_head_dxyz_m": (
                    "Frame-to-frame translation expressed in the previous valid head frame."
                ),
                "relative_rot_prev_to_cur_rij": (
                    "Rotation matrix from the previous valid head frame to the current head frame."
                ),
                "head_rotation_angle_step_deg": (
                    "Total relative rotation magnitude between consecutive valid head samples."
                ),
            },
        }
    )
    write_head_summary_json(summary_json, summary)
    print_summary(output_csv, summary_json, summary)


def print_summary(output_csv: Path, summary_json: Path, summary: dict[str, Any]) -> None:
    print(f"sequence: {summary['sequence_name']}")
    print(f"sequence_path: {summary['sequence_path']}")
    print(f"provider_mode: {summary['provider_mode']}")
    print(f"head_proxy_source: {summary['head_proxy_source']}")
    print(
        "samples: "
        f"{summary['sample_count']} "
        f"pose_valid={summary['pose_valid_count']} "
        f"temporal_context={summary['temporal_context_count']}"
    )
    print(
        "selected_timestamps_ns: "
        f"{summary['query_timestamp_start_ns']}..{summary['query_timestamp_end_ns']} "
        f"duration_s={summary['duration_s']:.3f}"
    )
    if summary["validation_note_counts"]:
        print(f"validation_note_counts: {summary['validation_note_counts']}")
    print(f"input_gaze_csv: {summary['input_gaze_csv']}")
    print(f"csv: {output_csv}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
