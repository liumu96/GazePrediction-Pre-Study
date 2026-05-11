"""Head-gaze relationship analysis built on extracted gaze/head feature layers.

This module is intentionally downstream of `gaze.py`, `head.py`, and
`gaze_dynamics.py`. It does not reopen the ADT provider. Instead, it consumes the
already extracted feature tables and produces:

- per-frame joined analysis rows
- per-sequence summaries
- batch-level aggregate reports

The goal is not to define a final event detector. The goal is to quantify what
information head carries relative to gaze, and which representations are most
useful for later SparseGaze-style modeling.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gaze_dynamics import compute_gaze_dynamics_features, describe_optional_numbers
from .gaze import GazeSample
from .head import HeadSample, relative_rotation_prev_to_cur_matrix


@dataclass(frozen=True)
class HeadGazeAnalysisRow:
    """One per-frame joined row for head-gaze relationship analysis."""

    query_timestamp_ns: int
    frame_index: int
    dynamics_input_valid: bool
    gaze_head_angle_deg: float | None
    local_angle_step_deg: float | None
    local_velocity_deg_s: float | None
    delta_yaw_rad: float | None
    delta_pitch_rad: float | None
    abs_delta_yaw_rad: float | None
    abs_delta_pitch_rad: float | None
    next_local_velocity_deg_s: float | None
    next_abs_delta_yaw_rad: float | None
    next_abs_delta_pitch_rad: float | None
    head_rotvec_prev_head_x_rad: float | None
    head_rotvec_prev_head_y_rad: float | None
    head_rotvec_prev_head_z_rad: float | None
    head_rotvec_prev_head_x_deg_s: float | None
    head_rotvec_prev_head_y_deg_s: float | None
    head_rotvec_prev_head_z_deg_s: float | None
    gaze_head_motion_alignment_2d: float | None
    head_translation_speed_m_s: float | None
    head_forward_angle_step_deg: float | None
    head_rotation_angle_step_deg: float | None
    head_rotation_speed_deg_s: float | None
    origin_step_m: float | None
    translation_scene_dx_m: float | None
    translation_scene_dy_m: float | None
    translation_scene_dz_m: float | None
    translation_prev_head_dx_m: float | None
    translation_prev_head_dy_m: float | None
    translation_prev_head_dz_m: float | None
    gaze_dir_scene_unit_x: float | None
    gaze_dir_scene_unit_y: float | None
    gaze_dir_scene_unit_z: float | None
    head_forward_scene_unit_x: float | None
    head_forward_scene_unit_y: float | None
    head_forward_scene_unit_z: float | None
    pose_quality_score: float | None
    gaze_validation_notes: str
    head_validation_notes: str

    def as_csv_row(self) -> dict[str, Any]:
        return asdict(self)


def build_head_gaze_analysis_rows(
    gaze_samples: Sequence[GazeSample],
    head_samples: Sequence[HeadSample],
    dispersion_window_frames: int = 5,
) -> list[HeadGazeAnalysisRow]:
    """Build a per-frame joined table for head-gaze analysis.

    zh-CN:
    这一步不直接做 event detection，而是把后续分析真正关心的量拼成一张表：
    - Scene frame 几何关系：`gaze_head_angle_deg`
    - Local gaze dynamics：`delta_yaw/pitch`、`local_velocity_deg_s`
    - Head dynamics：相对旋转、相对平移、speed
    """

    if len(gaze_samples) != len(head_samples):
        raise ValueError("gaze_samples and head_samples must have the same length")
    if not gaze_samples:
        raise ValueError("No samples provided")
    require_full_head_feature_schema(head_samples)

    feature_rows = compute_gaze_dynamics_features(
        gaze_samples,
        head_samples,
        dispersion_window_frames=dispersion_window_frames,
    )

    rows: list[HeadGazeAnalysisRow] = []
    for index, (feature_row, gaze_sample, head_sample) in enumerate(
        zip(feature_rows, gaze_samples, head_samples)
    ):
        next_feature = feature_rows[index + 1] if index + 1 < len(feature_rows) else None
        head_rotvec = rotation_vector_from_matrix(
            relative_rotation_prev_to_cur_matrix(head_sample)
        )
        head_rotvec_deg_s = signed_angular_velocity_deg_s(
            head_rotvec,
            feature_row.dt_s,
        )
        gaze_head_motion_alignment_2d = angular_plane_alignment(
            delta_yaw=feature_row.delta_yaw_rad,
            delta_pitch=feature_row.delta_pitch_rad,
            head_rotvec=head_rotvec,
        )
        rows.append(
            HeadGazeAnalysisRow(
                query_timestamp_ns=feature_row.query_timestamp_ns,
                frame_index=feature_row.frame_index,
                dynamics_input_valid=feature_row.dynamics_input_valid,
                gaze_head_angle_deg=feature_row.gaze_head_angle_deg,
                local_angle_step_deg=feature_row.local_angle_step_deg,
                local_velocity_deg_s=feature_row.local_velocity_deg_s,
                delta_yaw_rad=feature_row.delta_yaw_rad,
                delta_pitch_rad=feature_row.delta_pitch_rad,
                abs_delta_yaw_rad=_abs_or_none(feature_row.delta_yaw_rad),
                abs_delta_pitch_rad=_abs_or_none(feature_row.delta_pitch_rad),
                next_local_velocity_deg_s=(
                    next_feature.local_velocity_deg_s if next_feature is not None else None
                ),
                next_abs_delta_yaw_rad=(
                    _abs_or_none(next_feature.delta_yaw_rad) if next_feature is not None else None
                ),
                next_abs_delta_pitch_rad=(
                    _abs_or_none(next_feature.delta_pitch_rad)
                    if next_feature is not None
                    else None
                ),
                head_rotvec_prev_head_x_rad=_vector_component_or_none(head_rotvec, 0),
                head_rotvec_prev_head_y_rad=_vector_component_or_none(head_rotvec, 1),
                head_rotvec_prev_head_z_rad=_vector_component_or_none(head_rotvec, 2),
                head_rotvec_prev_head_x_deg_s=_vector_component_or_none(
                    head_rotvec_deg_s,
                    0,
                ),
                head_rotvec_prev_head_y_deg_s=_vector_component_or_none(
                    head_rotvec_deg_s,
                    1,
                ),
                head_rotvec_prev_head_z_deg_s=_vector_component_or_none(
                    head_rotvec_deg_s,
                    2,
                ),
                gaze_head_motion_alignment_2d=gaze_head_motion_alignment_2d,
                head_translation_speed_m_s=feature_row.head_translation_speed_m_s,
                head_forward_angle_step_deg=feature_row.head_forward_angle_step_deg,
                head_rotation_angle_step_deg=head_sample.head_rotation_angle_step_deg,
                head_rotation_speed_deg_s=head_sample.head_rotation_speed_deg_s,
                origin_step_m=feature_row.origin_step_m,
                translation_scene_dx_m=head_sample.translation_scene_dx_m,
                translation_scene_dy_m=head_sample.translation_scene_dy_m,
                translation_scene_dz_m=head_sample.translation_scene_dz_m,
                translation_prev_head_dx_m=head_sample.translation_prev_head_dx_m,
                translation_prev_head_dy_m=head_sample.translation_prev_head_dy_m,
                translation_prev_head_dz_m=head_sample.translation_prev_head_dz_m,
                gaze_dir_scene_unit_x=gaze_sample.gaze_dir_scene_unit_x,
                gaze_dir_scene_unit_y=gaze_sample.gaze_dir_scene_unit_y,
                gaze_dir_scene_unit_z=gaze_sample.gaze_dir_scene_unit_z,
                head_forward_scene_unit_x=head_sample.head_forward_scene_unit_x,
                head_forward_scene_unit_y=head_sample.head_forward_scene_unit_y,
                head_forward_scene_unit_z=head_sample.head_forward_scene_unit_z,
                pose_quality_score=feature_row.pose_quality_score,
                gaze_validation_notes=feature_row.gaze_validation_notes,
                head_validation_notes=feature_row.head_validation_notes,
            )
        )
    return rows


def summarize_head_gaze_analysis_rows(
    rows: Sequence[HeadGazeAnalysisRow],
) -> dict[str, Any]:
    """Summarize one sequence of head-gaze analysis rows."""

    if not rows:
        raise ValueError("No analysis rows to summarize")

    valid_rows = [row for row in rows if row.dynamics_input_valid]

    head_rotation_values = [
        row.head_rotation_speed_deg_s
        for row in valid_rows
        if row.head_rotation_speed_deg_s is not None
    ]
    head_rotation_strata = summarize_rotation_strata(valid_rows, head_rotation_values)

    return {
        "sample_count": len(rows),
        "dynamics_input_valid_count": len(valid_rows),
        "dynamics_input_valid_ratio": len(valid_rows) / len(rows),
        "geometry": {
            "gaze_head_angle_deg": describe_optional_numbers(
                [row.gaze_head_angle_deg for row in valid_rows]
            ),
        },
        "dynamics": {
            "local_angle_step_deg": describe_optional_numbers(
                [row.local_angle_step_deg for row in valid_rows]
            ),
            "local_velocity_deg_s": describe_optional_numbers(
                [row.local_velocity_deg_s for row in valid_rows]
            ),
            "head_translation_speed_m_s": describe_optional_numbers(
                [row.head_translation_speed_m_s for row in valid_rows]
            ),
            "head_forward_angle_step_deg": describe_optional_numbers(
                [row.head_forward_angle_step_deg for row in valid_rows]
            ),
            "head_rotation_angle_step_deg": describe_optional_numbers(
                [row.head_rotation_angle_step_deg for row in valid_rows]
            ),
            "head_rotation_speed_deg_s": describe_optional_numbers(
                [row.head_rotation_speed_deg_s for row in valid_rows]
            ),
            "head_rotvec_prev_head_x_deg_s": describe_optional_numbers(
                [row.head_rotvec_prev_head_x_deg_s for row in valid_rows]
            ),
            "head_rotvec_prev_head_y_deg_s": describe_optional_numbers(
                [row.head_rotvec_prev_head_y_deg_s for row in valid_rows]
            ),
            "head_rotvec_prev_head_z_deg_s": describe_optional_numbers(
                [row.head_rotvec_prev_head_z_deg_s for row in valid_rows]
            ),
            "gaze_head_motion_alignment_2d": describe_optional_numbers(
                [row.gaze_head_motion_alignment_2d for row in valid_rows]
            ),
        },
        "correlations": {
            "current_local_velocity_vs_head_rotation_speed": pearson_corr(
                [row.local_velocity_deg_s for row in valid_rows],
                [row.head_rotation_speed_deg_s for row in valid_rows],
            ),
            "current_local_velocity_vs_head_translation_speed": pearson_corr(
                [row.local_velocity_deg_s for row in valid_rows],
                [row.head_translation_speed_m_s for row in valid_rows],
            ),
            "current_abs_delta_yaw_vs_head_rotation_speed": pearson_corr(
                [row.abs_delta_yaw_rad for row in valid_rows],
                [row.head_rotation_speed_deg_s for row in valid_rows],
            ),
            "current_abs_delta_pitch_vs_head_rotation_speed": pearson_corr(
                [row.abs_delta_pitch_rad for row in valid_rows],
                [row.head_rotation_speed_deg_s for row in valid_rows],
            ),
            "signed_delta_yaw_vs_head_rotvec_y": pearson_corr(
                [row.delta_yaw_rad for row in valid_rows],
                [row.head_rotvec_prev_head_y_rad for row in valid_rows],
            ),
            "signed_delta_pitch_vs_head_rotvec_x": pearson_corr(
                [row.delta_pitch_rad for row in valid_rows],
                [row.head_rotvec_prev_head_x_rad for row in valid_rows],
            ),
            "abs_delta_yaw_vs_abs_head_rotvec_y": pearson_corr(
                [row.abs_delta_yaw_rad for row in valid_rows],
                [_abs_or_none(row.head_rotvec_prev_head_y_rad) for row in valid_rows],
            ),
            "abs_delta_pitch_vs_abs_head_rotvec_x": pearson_corr(
                [row.abs_delta_pitch_rad for row in valid_rows],
                [_abs_or_none(row.head_rotvec_prev_head_x_rad) for row in valid_rows],
            ),
            "next_local_velocity_vs_current_head_rotation_speed": pearson_corr(
                [row.next_local_velocity_deg_s for row in valid_rows],
                [row.head_rotation_speed_deg_s for row in valid_rows],
            ),
            "next_local_velocity_vs_current_head_translation_speed": pearson_corr(
                [row.next_local_velocity_deg_s for row in valid_rows],
                [row.head_translation_speed_m_s for row in valid_rows],
            ),
            "next_abs_delta_yaw_vs_current_head_rotation_speed": pearson_corr(
                [row.next_abs_delta_yaw_rad for row in valid_rows],
                [row.head_rotation_speed_deg_s for row in valid_rows],
            ),
            "next_abs_delta_pitch_vs_current_head_rotation_speed": pearson_corr(
                [row.next_abs_delta_pitch_rad for row in valid_rows],
                [row.head_rotation_speed_deg_s for row in valid_rows],
            ),
        },
        "directional_alignment": summarize_directional_alignment(valid_rows),
        "head_rotation_speed_strata": head_rotation_strata,
    }


def summarize_batch_head_gaze_analysis(
    sequence_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize batch-level sequence rows from the CLI script."""

    if not sequence_rows:
        raise ValueError("No batch rows to summarize")

    def metric(name: str) -> list[float | None]:
        return [row.get(name) for row in sequence_rows]

    next_corr_rows = [
        row
        for row in sequence_rows
        if row.get("corr_next_local_velocity_vs_current_head_rotation_speed") is not None
    ]

    return {
        "sequence_count": len(sequence_rows),
        "dynamics_input_valid_ratio": describe_optional_numbers(metric("dynamics_input_valid_ratio")),
        "median_gaze_head_angle_deg": describe_optional_numbers(metric("median_gaze_head_angle_deg")),
        "correlations": {
            "current_local_velocity_vs_head_rotation_speed": describe_optional_numbers(
                metric("corr_current_local_velocity_vs_head_rotation_speed")
            ),
            "current_local_velocity_vs_head_translation_speed": describe_optional_numbers(
                metric("corr_current_local_velocity_vs_head_translation_speed")
            ),
            "next_local_velocity_vs_current_head_rotation_speed": describe_optional_numbers(
                metric("corr_next_local_velocity_vs_current_head_rotation_speed")
            ),
            "next_local_velocity_vs_current_head_translation_speed": describe_optional_numbers(
                metric("corr_next_local_velocity_vs_current_head_translation_speed")
            ),
            "signed_delta_yaw_vs_head_rotvec_y": describe_optional_numbers(
                metric("corr_signed_delta_yaw_vs_head_rotvec_y")
            ),
            "signed_delta_pitch_vs_head_rotvec_x": describe_optional_numbers(
                metric("corr_signed_delta_pitch_vs_head_rotvec_x")
            ),
            "abs_delta_yaw_vs_abs_head_rotvec_y": describe_optional_numbers(
                metric("corr_abs_delta_yaw_vs_abs_head_rotvec_y")
            ),
            "abs_delta_pitch_vs_abs_head_rotvec_x": describe_optional_numbers(
                metric("corr_abs_delta_pitch_vs_abs_head_rotvec_x")
            ),
            "gaze_head_motion_alignment_2d_mean": describe_optional_numbers(
                metric("mean_gaze_head_motion_alignment_2d")
            ),
            "gaze_head_motion_opposed_fraction": describe_optional_numbers(
                metric("gaze_head_motion_opposed_fraction")
            ),
        },
        "top_sequences_by_next_step_head_rotation_correlation": [
            {
                "sequence_name": row["sequence_name"],
                "corr_next_local_velocity_vs_current_head_rotation_speed": row[
                    "corr_next_local_velocity_vs_current_head_rotation_speed"
                ],
            }
            for row in sorted(
                next_corr_rows,
                key=lambda row: abs(
                    float(row["corr_next_local_velocity_vs_current_head_rotation_speed"])
                ),
                reverse=True,
            )[:5]
        ],
    }


def write_head_gaze_analysis_rows_csv(
    path: str | Path,
    rows: Sequence[HeadGazeAnalysisRow],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [row.as_csv_row() for row in rows]
    if not materialized:
        raise ValueError("No head-gaze analysis rows to write")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def default_head_gaze_analysis_rows_csv_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[2] / "outputs" / "reports"
    )
    return base_dir / f"{sequence_name}_head_gaze_analysis_rows.csv"


def default_head_gaze_analysis_summary_json_path(csv_path: str | Path) -> Path:
    csv_file = Path(csv_path)
    stem = csv_file.stem
    if stem.endswith("_head_gaze_analysis_rows"):
        stem = stem[: -len("_head_gaze_analysis_rows")] + "_head_gaze_analysis_summary"
    else:
        stem = f"{stem}_summary"
    return csv_file.with_name(f"{stem}.json")


def write_batch_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        raise ValueError("No batch rows to write")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def summarize_rotation_strata(
    rows: Sequence[HeadGazeAnalysisRow],
    rotation_speed_values: Sequence[float],
) -> dict[str, Any]:
    if not rotation_speed_values:
        return {
            "thresholds_deg_s": {"low_to_mid": None, "mid_to_high": None},
            "groups": {},
        }

    values = np.asarray(rotation_speed_values, dtype=np.float64)
    low_to_mid = float(np.percentile(values, 33.3))
    mid_to_high = float(np.percentile(values, 66.7))

    grouped: dict[str, list[HeadGazeAnalysisRow]] = {"low": [], "mid": [], "high": []}
    for row in rows:
        speed = row.head_rotation_speed_deg_s
        if speed is None or not np.isfinite(speed):
            continue
        if speed <= low_to_mid:
            grouped["low"].append(row)
        elif speed <= mid_to_high:
            grouped["mid"].append(row)
        else:
            grouped["high"].append(row)

    return {
        "thresholds_deg_s": {"low_to_mid": low_to_mid, "mid_to_high": mid_to_high},
        "groups": {
            label: {
                "frame_count": len(group_rows),
                "gaze_head_angle_deg": describe_optional_numbers(
                    [row.gaze_head_angle_deg for row in group_rows]
                ),
                "local_velocity_deg_s": describe_optional_numbers(
                    [row.local_velocity_deg_s for row in group_rows]
                ),
                "next_local_velocity_deg_s": describe_optional_numbers(
                    [row.next_local_velocity_deg_s for row in group_rows]
                ),
                "head_translation_speed_m_s": describe_optional_numbers(
                    [row.head_translation_speed_m_s for row in group_rows]
                ),
            }
            for label, group_rows in grouped.items()
        },
    }


def summarize_directional_alignment(rows: Sequence[HeadGazeAnalysisRow]) -> dict[str, Any]:
    alignments = [
        float(row.gaze_head_motion_alignment_2d)
        for row in rows
        if row.gaze_head_motion_alignment_2d is not None
        and np.isfinite(row.gaze_head_motion_alignment_2d)
    ]
    if not alignments:
        return {
            "gaze_head_motion_alignment_2d": describe_optional_numbers([]),
            "aligned_fraction": None,
            "opposed_fraction": None,
            "weak_or_orthogonal_fraction": None,
        }
    values = np.asarray(alignments, dtype=np.float64)
    return {
        "gaze_head_motion_alignment_2d": describe_optional_numbers(alignments),
        "aligned_fraction": float(np.mean(values >= 0.5)),
        "opposed_fraction": float(np.mean(values <= -0.5)),
        "weak_or_orthogonal_fraction": float(np.mean(np.abs(values) < 0.5)),
    }


def pearson_corr(
    first_values: Sequence[float | None],
    second_values: Sequence[float | None],
) -> float | None:
    paired = [
        (float(first), float(second))
        for first, second in zip(first_values, second_values)
        if first is not None
        and second is not None
        and np.isfinite(first)
        and np.isfinite(second)
    ]
    if len(paired) < 3:
        return None
    first_array = np.asarray([pair[0] for pair in paired], dtype=np.float64)
    second_array = np.asarray([pair[1] for pair in paired], dtype=np.float64)
    if np.allclose(first_array.std(), 0.0) or np.allclose(second_array.std(), 0.0):
        return None
    return float(np.corrcoef(first_array, second_array)[0, 1])


def rotation_vector_from_matrix(rotation: np.ndarray | None) -> np.ndarray | None:
    """Return the axis-angle vector for a relative rotation matrix.

    The input is `R_prev_head_to_current_head`; therefore the returned vector is
    expressed in the previous head/CPF frame. Its magnitude is the rotation angle
    in radians and its components preserve signed rotation direction.
    """

    if rotation is None:
        return None
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None

    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew_vector = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * skew_vector
    sine = float(np.sin(angle))
    if abs(sine) < 1e-8:
        return None
    return (angle / (2.0 * sine)) * skew_vector


def signed_angular_velocity_deg_s(
    rotation_vector_rad: np.ndarray | None,
    dt_s: float | None,
) -> np.ndarray | None:
    if (
        rotation_vector_rad is None
        or dt_s is None
        or dt_s <= 0
        or not np.isfinite(dt_s)
    ):
        return None
    return np.degrees(rotation_vector_rad / dt_s)


def angular_plane_alignment(
    delta_yaw: float | None,
    delta_pitch: float | None,
    head_rotvec: np.ndarray | None,
) -> float | None:
    """Cosine similarity between 2D gaze motion and 2D head rotation.

    The gaze vector is `[delta_yaw, delta_pitch]` in CPF angular coordinates.
    The head vector is `[rotvec_y, rotvec_x]`, i.e. the relative head rotation
    components most directly corresponding to horizontal and vertical angular
    axes. Positive values mean the two 2D vectors are similarly directed under
    this coordinate convention; negative values mean they are opposed.
    """

    if delta_yaw is None or delta_pitch is None or head_rotvec is None:
        return None
    gaze_vec = np.asarray([delta_yaw, delta_pitch], dtype=np.float64)
    head_vec = np.asarray([head_rotvec[1], head_rotvec[0]], dtype=np.float64)
    if not np.isfinite(gaze_vec).all() or not np.isfinite(head_vec).all():
        return None
    gaze_norm = float(np.linalg.norm(gaze_vec))
    head_norm = float(np.linalg.norm(head_vec))
    if gaze_norm <= 0 or head_norm <= 0:
        return None
    return float(np.dot(gaze_vec, head_vec) / (gaze_norm * head_norm))


def _abs_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return abs(float(value))


def _vector_component_or_none(vector: np.ndarray | None, index: int) -> float | None:
    if vector is None or not np.isfinite(vector[index]):
        return None
    return float(vector[index])


def require_full_head_feature_schema(head_samples: Sequence[HeadSample]) -> None:
    """Fail fast on old head CSV exports.

    zh-CN:
    head-gaze relationship analysis 依赖重构后的完整 head feature layer，
    不接受旧版只包含 `origin + forward + speed` 的简化 CSV。这里直接检查关键
    绝对 pose 和相对 rotation 字段；如果缺失，要求先重跑
    `extract_head_proxy.py` / `batch_extract_head_proxy.py`。
    """

    valid_pose_samples = [sample for sample in head_samples if sample.pose_valid]
    if not valid_pose_samples:
        raise ValueError("No valid head pose samples found")

    has_absolute_axes = any(
        sample.head_right_scene_unit_x is not None
        and sample.head_up_scene_unit_x is not None
        and sample.head_rot_scene_r00 is not None
        for sample in valid_pose_samples
    )
    has_relative_rotation = any(
        sample.head_rotation_speed_deg_s is not None
        and sample.head_rotation_angle_step_deg is not None
        and sample.relative_rot_prev_to_cur_r00 is not None
        for sample in valid_pose_samples
    )
    if has_absolute_axes and has_relative_rotation:
        return

    raise ValueError(
        "Head-gaze analysis requires the refactored head feature schema. "
        "The current head_samples.csv looks like an older export. "
        "Re-run scripts/extract_head_proxy.py or scripts/batch_extract_head_proxy.py "
        "with the current code before running analyze_head_gaze_relationship.py."
    )
