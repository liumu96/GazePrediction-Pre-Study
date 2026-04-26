"""Head-feature extraction helpers based on ADT device/CPF pose.

This module is intended to play the same role for head motion that `gaze.py`
plays for gaze: a reusable feature layer. Event analysis, visualization, and
future modeling code should consume these head features instead of rebuilding
pose-derived quantities ad hoc.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HeadSample:
    """One head-feature sample aligned to a query timestamp.

    zh-CN:
    这里的 head 不是 skeleton 里的头部 joint，而是基于 ADT `device pose + CPF`
    构建的 head-mounted tracker proxy。目标不是只服务 event，而是把 head 的
    绝对姿态和相对运动都整理成可复用的数据层。
    """

    query_timestamp_ns: int
    pose_valid: bool
    pose_dt_ns: int | None
    pose_quality_score: float | None

    # Absolute Scene-frame pose.
    head_origin_scene_x_m: float | None
    head_origin_scene_y_m: float | None
    head_origin_scene_z_m: float | None
    head_right_scene_unit_x: float | None
    head_right_scene_unit_y: float | None
    head_right_scene_unit_z: float | None
    head_up_scene_unit_x: float | None
    head_up_scene_unit_y: float | None
    head_up_scene_unit_z: float | None
    head_forward_scene_unit_x: float | None
    head_forward_scene_unit_y: float | None
    head_forward_scene_unit_z: float | None
    head_rot_scene_r00: float | None
    head_rot_scene_r01: float | None
    head_rot_scene_r02: float | None
    head_rot_scene_r10: float | None
    head_rot_scene_r11: float | None
    head_rot_scene_r12: float | None
    head_rot_scene_r20: float | None
    head_rot_scene_r21: float | None
    head_rot_scene_r22: float | None

    # Relative temporal context.
    dt_from_prev_s: float | None
    translation_scene_dx_m: float | None
    translation_scene_dy_m: float | None
    translation_scene_dz_m: float | None
    translation_prev_head_dx_m: float | None
    translation_prev_head_dy_m: float | None
    translation_prev_head_dz_m: float | None
    origin_step_m: float | None
    head_translation_speed_m_s: float | None
    relative_rot_prev_to_cur_r00: float | None
    relative_rot_prev_to_cur_r01: float | None
    relative_rot_prev_to_cur_r02: float | None
    relative_rot_prev_to_cur_r10: float | None
    relative_rot_prev_to_cur_r11: float | None
    relative_rot_prev_to_cur_r12: float | None
    relative_rot_prev_to_cur_r20: float | None
    relative_rot_prev_to_cur_r21: float | None
    relative_rot_prev_to_cur_r22: float | None
    head_forward_angle_step_deg: float | None
    head_rotation_angle_step_deg: float | None
    head_rotation_speed_deg_s: float | None
    validation_notes: str

    def as_csv_row(self) -> dict[str, Any]:
        return asdict(self)


def extract_head_sample(gt_provider: Any, timestamp_ns: int) -> HeadSample:
    """Extract one aligned head sample using `T_scene_cpf`.

    zh-CN:
    这一层输出两类量：
    1. Scene frame 下的绝对 head proxy pose
    2. 后续可计算相对运动的基础姿态信息

    当前不从 skeleton 提 head，而是直接使用 ADT 的 device pose 和 CPF 校准。
    """

    pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
    if not pose_with_dt.is_valid():
        return _invalid_head_sample(timestamp_ns)

    aria_pose = pose_with_dt.data()
    device_calibration = gt_provider.raw_data_provider_ptr().get_device_calibration()
    transform_scene_cpf = (
        aria_pose.transform_scene_device @ device_calibration.get_transform_device_cpf()
    )

    origin_scene, right_scene, up_scene, forward_scene = _scene_basis(transform_scene_cpf)
    if (
        origin_scene is None
        or right_scene is None
        or up_scene is None
        or forward_scene is None
    ):
        sample = _invalid_head_sample(timestamp_ns)
        return replace(
            sample,
            pose_valid=True,
            pose_dt_ns=int(pose_with_dt.dt_ns()),
            pose_quality_score=float(aria_pose.quality_score),
            validation_notes="scene_basis_invalid",
        )

    """ The head proxy's right, up, and forward unit vectors form the rotation matrix from CPF frame to scene frame."""
    rotation_scene = np.column_stack([right_scene, up_scene, forward_scene])
    return HeadSample(
        query_timestamp_ns=int(timestamp_ns),
        pose_valid=True,
        pose_dt_ns=int(pose_with_dt.dt_ns()),
        pose_quality_score=float(aria_pose.quality_score),
        head_origin_scene_x_m=_finite_or_none(origin_scene[0]),
        head_origin_scene_y_m=_finite_or_none(origin_scene[1]),
        head_origin_scene_z_m=_finite_or_none(origin_scene[2]),
        head_right_scene_unit_x=_finite_or_none(right_scene[0]),
        head_right_scene_unit_y=_finite_or_none(right_scene[1]),
        head_right_scene_unit_z=_finite_or_none(right_scene[2]),
        head_up_scene_unit_x=_finite_or_none(up_scene[0]),
        head_up_scene_unit_y=_finite_or_none(up_scene[1]),
        head_up_scene_unit_z=_finite_or_none(up_scene[2]),
        head_forward_scene_unit_x=_finite_or_none(forward_scene[0]),
        head_forward_scene_unit_y=_finite_or_none(forward_scene[1]),
        head_forward_scene_unit_z=_finite_or_none(forward_scene[2]),
        head_rot_scene_r00=_finite_or_none(rotation_scene[0, 0]),
        head_rot_scene_r01=_finite_or_none(rotation_scene[0, 1]),
        head_rot_scene_r02=_finite_or_none(rotation_scene[0, 2]),
        head_rot_scene_r10=_finite_or_none(rotation_scene[1, 0]),
        head_rot_scene_r11=_finite_or_none(rotation_scene[1, 1]),
        head_rot_scene_r12=_finite_or_none(rotation_scene[1, 2]),
        head_rot_scene_r20=_finite_or_none(rotation_scene[2, 0]),
        head_rot_scene_r21=_finite_or_none(rotation_scene[2, 1]),
        head_rot_scene_r22=_finite_or_none(rotation_scene[2, 2]),
        dt_from_prev_s=None,
        translation_scene_dx_m=None,
        translation_scene_dy_m=None,
        translation_scene_dz_m=None,
        translation_prev_head_dx_m=None,
        translation_prev_head_dy_m=None,
        translation_prev_head_dz_m=None,
        origin_step_m=None,
        head_translation_speed_m_s=None,
        relative_rot_prev_to_cur_r00=None,
        relative_rot_prev_to_cur_r01=None,
        relative_rot_prev_to_cur_r02=None,
        relative_rot_prev_to_cur_r10=None,
        relative_rot_prev_to_cur_r11=None,
        relative_rot_prev_to_cur_r12=None,
        relative_rot_prev_to_cur_r20=None,
        relative_rot_prev_to_cur_r21=None,
        relative_rot_prev_to_cur_r22=None,
        head_forward_angle_step_deg=None,
        head_rotation_angle_step_deg=None,
        head_rotation_speed_deg_s=None,
        validation_notes="ok",
    )


def add_temporal_head_context(samples: Sequence[HeadSample]) -> list[HeadSample]:
    """Add relative head-motion features from consecutive valid samples."""

    enriched: list[HeadSample] = []
    prev_valid: HeadSample | None = None
    for sample in samples:
        if not sample.pose_valid:
            enriched.append(sample)
            continue

        dt_from_prev_s: float | None = None
        translation_scene = None
        translation_prev_head = None
        origin_step_m: float | None = None
        translation_speed_m_s: float | None = None
        relative_rotation = None
        forward_angle_step_deg: float | None = None
        rotation_angle_step_deg: float | None = None
        rotation_speed_deg_s: float | None = None
        notes = [] if sample.validation_notes == "ok" else [sample.validation_notes]

        if prev_valid is not None:
            dt_ns = sample.query_timestamp_ns - prev_valid.query_timestamp_ns
            dt_from_prev_s = dt_ns / 1e9 if dt_ns > 0 else None
            current_origin = head_origin_xyz(sample)
            prev_origin = head_origin_xyz(prev_valid)
            current_forward = head_forward_xyz(sample)
            prev_forward = head_forward_xyz(prev_valid)
            current_rotation = head_rotation_scene_matrix(sample)
            prev_rotation = head_rotation_scene_matrix(prev_valid)

            if current_origin is not None and prev_origin is not None:
                translation_scene = current_origin - prev_origin
                origin_step_m = float(np.linalg.norm(translation_scene)) # Step length in scene frame
                if dt_from_prev_s and dt_from_prev_s > 0:
                    translation_speed_m_s = origin_step_m / dt_from_prev_s # Speed in scene frame

            if current_rotation is not None and prev_rotation is not None:
                relative_rotation = prev_rotation.T @ current_rotation # Rotation from previous head frame to current head frame
                if translation_scene is not None:
                    translation_prev_head = prev_rotation.T @ translation_scene # Translation in previous head frame
                rotation_angle_step_deg = rotation_angle_deg_from_matrix(relative_rotation) # Rotation angle step in degrees
                if rotation_angle_step_deg is not None and dt_from_prev_s and dt_from_prev_s > 0:
                    rotation_speed_deg_s = rotation_angle_step_deg / dt_from_prev_s # Rotation speed in degrees per second

            if current_forward is not None and prev_forward is not None:
                forward_angle_step_deg = angle_between_unit_vectors_deg(
                    prev_forward,
                    current_forward,
                ) # Forward direction angle step in degrees

        enriched.append(
            replace(
                sample,
                dt_from_prev_s=dt_from_prev_s, # Time delta from previous valid sample
                translation_scene_dx_m=_vector_component_or_none(translation_scene, 0), # Translation in scene frame
                translation_scene_dy_m=_vector_component_or_none(translation_scene, 1), # Translation in scene frame
                translation_scene_dz_m=_vector_component_or_none(translation_scene, 2), # Translation in scene frame
                translation_prev_head_dx_m=_vector_component_or_none(translation_prev_head, 0), # Translation in previous head frame
                translation_prev_head_dy_m=_vector_component_or_none(translation_prev_head, 1), # Translation in previous head frame
                translation_prev_head_dz_m=_vector_component_or_none(translation_prev_head, 2), # Translation in previous head frame
                origin_step_m=origin_step_m, # Step length in scene frame
                head_translation_speed_m_s=translation_speed_m_s, # Speed in scene frame
                relative_rot_prev_to_cur_r00=_matrix_component_or_none(relative_rotation, 0, 0), # Relative rotation matrix from previous head frame to current head frame
                relative_rot_prev_to_cur_r01=_matrix_component_or_none(relative_rotation, 0, 1), # Relative rotation matrix from previous head frame to current head frame
                relative_rot_prev_to_cur_r02=_matrix_component_or_none(relative_rotation, 0, 2), # Relative rotation matrix from previous head frame to current head frame
                relative_rot_prev_to_cur_r10=_matrix_component_or_none(relative_rotation, 1, 0), # Relative rotation matrix from previous head frame to current head frame
                relative_rot_prev_to_cur_r11=_matrix_component_or_none(relative_rotation, 1, 1), # Relative rotation matrix from previous head frame to current head frame
                relative_rot_prev_to_cur_r12=_matrix_component_or_none(relative_rotation, 1, 2), # Relative rotation matrix from previous head frame to current head frame
                relative_rot_prev_to_cur_r20=_matrix_component_or_none(relative_rotation, 2, 0), # Relative rotation matrix from previous head frame to current head frame
                relative_rot_prev_to_cur_r21=_matrix_component_or_none(relative_rotation, 2, 1), # Relative rotation matrix from previous head frame to current head frame
                relative_rot_prev_to_cur_r22=_matrix_component_or_none(relative_rotation, 2, 2), # Relative rotation matrix from previous head frame to current head frame
                head_forward_angle_step_deg=forward_angle_step_deg, # Forward direction angle step in degrees
                head_rotation_angle_step_deg=rotation_angle_step_deg, # Rotation angle step in degrees
                head_rotation_speed_deg_s=rotation_speed_deg_s, # Rotation speed in degrees per second
                validation_notes=";".join(notes) if notes else "ok",
            )
        )
        prev_valid = sample

    return enriched


# ======================== Head sample parsing and CSV I/O ========================
def head_origin_xyz(sample: HeadSample) -> np.ndarray | None:
    """Return head origin in scene frame as a 3D vector, or None if any component is missing or invalid."""
    return _vector_from_optional_xyz(
        sample.head_origin_scene_x_m,
        sample.head_origin_scene_y_m,
        sample.head_origin_scene_z_m,
    )


def head_right_xyz(sample: HeadSample) -> np.ndarray | None:
    """Return head right direction in scene frame as a 3D unit vector, or None if any component is missing or invalid."""
    return _vector_from_optional_xyz(
        sample.head_right_scene_unit_x,
        sample.head_right_scene_unit_y,
        sample.head_right_scene_unit_z,
    )


def head_up_xyz(sample: HeadSample) -> np.ndarray | None:
    """Return head up direction in scene frame as a 3D unit vector, or None if any component is missing or invalid."""
    return _vector_from_optional_xyz(
        sample.head_up_scene_unit_x,
        sample.head_up_scene_unit_y,
        sample.head_up_scene_unit_z,
    )


def head_forward_xyz(sample: HeadSample) -> np.ndarray | None:
    """Return head forward direction in scene frame as a 3D unit vector, or None if any component is missing or invalid."""
    return _vector_from_optional_xyz(
        sample.head_forward_scene_unit_x,
        sample.head_forward_scene_unit_y,
        sample.head_forward_scene_unit_z,
    )


def head_rotation_scene_matrix(sample: HeadSample) -> np.ndarray | None:
    """Return head rotation from CPF frame to scene frame as a 3x3 matrix, or None if any component is missing or invalid."""
    matrix = np.array(
        [
            [
                sample.head_rot_scene_r00,
                sample.head_rot_scene_r01,
                sample.head_rot_scene_r02,
            ],
            [
                sample.head_rot_scene_r10,
                sample.head_rot_scene_r11,
                sample.head_rot_scene_r12,
            ],
            [
                sample.head_rot_scene_r20,
                sample.head_rot_scene_r21,
                sample.head_rot_scene_r22,
            ],
        ],
        dtype=np.float64,
    )
    return matrix if np.isfinite(matrix).all() else None


def relative_rotation_prev_to_cur_matrix(sample: HeadSample) -> np.ndarray | None:
    """Return relative rotation from previous head frame to current head frame as a 3x3 matrix, or None if any component is missing or invalid."""
    matrix = np.array(
        [
            [
                sample.relative_rot_prev_to_cur_r00,
                sample.relative_rot_prev_to_cur_r01,
                sample.relative_rot_prev_to_cur_r02,
            ],
            [
                sample.relative_rot_prev_to_cur_r10,
                sample.relative_rot_prev_to_cur_r11,
                sample.relative_rot_prev_to_cur_r12,
            ],
            [
                sample.relative_rot_prev_to_cur_r20,
                sample.relative_rot_prev_to_cur_r21,
                sample.relative_rot_prev_to_cur_r22,
            ],
        ],
        dtype=np.float64,
    )
    return matrix if np.isfinite(matrix).all() else None


def angle_between_unit_vectors_deg(first: np.ndarray, second: np.ndarray) -> float | None:
    """Return the angle between two vectors in degrees."""

    first_unit = _normalize_vector(first)
    second_unit = _normalize_vector(second)
    if first_unit is None or second_unit is None:
        return None
    cosine = float(np.clip(np.dot(first_unit, second_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rotation_angle_deg_from_matrix(rotation: np.ndarray) -> float | None:
    """Return the rotation magnitude in degrees from a 3x3 rotation matrix."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    trace = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def write_head_samples_csv(path: str | Path, samples: Sequence[HeadSample]) -> None:
    """Write head samples to a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [sample.as_csv_row() for sample in samples]
    if not rows:
        raise ValueError("No head samples to write")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_head_samples(samples: Sequence[HeadSample]) -> dict[str, Any]:
    """Compute a lightweight summary for head-feature extraction."""

    if not samples:
        raise ValueError("No head samples to summarize")

    pose_valid_count = sum(sample.pose_valid for sample in samples)
    temporal_context_count = sum(sample.origin_step_m is not None for sample in samples)
    note_counts: dict[str, int] = {}
    for sample in samples:
        if sample.validation_notes == "ok":
            continue
        for note in sample.validation_notes.split(";"):
            if note:
                note_counts[note] = note_counts.get(note, 0) + 1

    return {
        "sample_count": len(samples), # Total number of head samples
        "query_timestamp_start_ns": samples[0].query_timestamp_ns, # Query timestamp of the first head sample
        "query_timestamp_end_ns": samples[-1].query_timestamp_ns,  # Query timestamp of the last head sample
        "duration_s": (samples[-1].query_timestamp_ns - samples[0].query_timestamp_ns) / 1e9, # Duration covered by the head samples in seconds
        "pose_valid_count": pose_valid_count, # Number of head samples with valid pose
        "pose_valid_ratio": pose_valid_count / len(samples), # Ratio of head samples with valid pose
        "temporal_context_count": temporal_context_count, # Number of head samples with temporal context (i.e., origin step is not None)
        "temporal_context_ratio": temporal_context_count / len(samples), # Ratio of head samples with temporal context
        "validation_note_counts": dict(sorted(note_counts.items())), # Counts of different validation notes found in the head samples
        "pose_dt_ms": describe_optional_numbers(
            [sample.pose_dt_ns / 1e6 for sample in samples if sample.pose_dt_ns is not None]
        ), # Statistics of pose query latency in milliseconds for head samples with valid pose_dt_ns
        "pose_quality_score": describe_optional_numbers(
            [sample.pose_quality_score for sample in samples if sample.pose_quality_score is not None]
        ), # Statistics of pose quality scores for head samples with valid pose_quality_score
        "dt_from_prev_s": describe_optional_numbers(
            [sample.dt_from_prev_s for sample in samples if sample.dt_from_prev_s is not None]
        ), # Statistics of time differences from previous head samples for head samples with valid dt_from_prev_s
        "origin_step_m": describe_optional_numbers(
            [sample.origin_step_m for sample in samples if sample.origin_step_m is not None]
        ), # Statistics of step lengths in scene frame for head samples with valid origin_step_m
        "head_translation_speed_m_s": describe_optional_numbers(
            [
                sample.head_translation_speed_m_s
                for sample in samples
                if sample.head_translation_speed_m_s is not None
            ]
        ), # Statistics of head translation speeds in scene frame for head samples with valid head_translation_speed_m_s
        "head_forward_angle_step_deg": describe_optional_numbers(
            [
                sample.head_forward_angle_step_deg
                for sample in samples
                if sample.head_forward_angle_step_deg is not None
            ]
        ), # Statistics of head forward direction angle steps in degrees for head samples with valid head_forward_angle_step_deg
        "head_rotation_angle_step_deg": describe_optional_numbers(
            [
                sample.head_rotation_angle_step_deg
                for sample in samples
                if sample.head_rotation_angle_step_deg is not None
            ]
        ), # Statistics of head rotation angle steps in degrees for head samples with valid head_rotation_angle_step_deg
        "head_rotation_speed_deg_s": describe_optional_numbers(
            [
                sample.head_rotation_speed_deg_s
                for sample in samples
                if sample.head_rotation_speed_deg_s is not None
            ]
        ), # Statistics of head rotation speeds in degrees per second for head samples with valid head_rotation_speed_deg_s
    }


def write_head_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    """Write head summary to a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def read_head_samples_csv(path: str | Path) -> list[HeadSample]:
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [head_sample_from_csv_row(row) for row in reader]


def head_sample_from_csv_row(row: dict[str, str]) -> HeadSample:
    """Parse one CSV row.

    Old CSVs are still accepted; missing newer fields fall back to None so that
    downstream event code can continue reading older exports.
    """

    return HeadSample(
        query_timestamp_ns=int(_row_required(row, "query_timestamp_ns")),
        pose_valid=_csv_bool(_row_required(row, "pose_valid")),
        pose_dt_ns=_csv_optional_int(row.get("pose_dt_ns")),
        pose_quality_score=_csv_optional_float(row.get("pose_quality_score")),
        head_origin_scene_x_m=_csv_optional_float(row.get("head_origin_scene_x_m")),
        head_origin_scene_y_m=_csv_optional_float(row.get("head_origin_scene_y_m")),
        head_origin_scene_z_m=_csv_optional_float(row.get("head_origin_scene_z_m")),
        head_right_scene_unit_x=_csv_optional_float(row.get("head_right_scene_unit_x")),
        head_right_scene_unit_y=_csv_optional_float(row.get("head_right_scene_unit_y")),
        head_right_scene_unit_z=_csv_optional_float(row.get("head_right_scene_unit_z")),
        head_up_scene_unit_x=_csv_optional_float(row.get("head_up_scene_unit_x")),
        head_up_scene_unit_y=_csv_optional_float(row.get("head_up_scene_unit_y")),
        head_up_scene_unit_z=_csv_optional_float(row.get("head_up_scene_unit_z")),
        head_forward_scene_unit_x=_csv_optional_float(row.get("head_forward_scene_unit_x")),
        head_forward_scene_unit_y=_csv_optional_float(row.get("head_forward_scene_unit_y")),
        head_forward_scene_unit_z=_csv_optional_float(row.get("head_forward_scene_unit_z")),
        head_rot_scene_r00=_csv_optional_float(row.get("head_rot_scene_r00")),
        head_rot_scene_r01=_csv_optional_float(row.get("head_rot_scene_r01")),
        head_rot_scene_r02=_csv_optional_float(row.get("head_rot_scene_r02")),
        head_rot_scene_r10=_csv_optional_float(row.get("head_rot_scene_r10")),
        head_rot_scene_r11=_csv_optional_float(row.get("head_rot_scene_r11")),
        head_rot_scene_r12=_csv_optional_float(row.get("head_rot_scene_r12")),
        head_rot_scene_r20=_csv_optional_float(row.get("head_rot_scene_r20")),
        head_rot_scene_r21=_csv_optional_float(row.get("head_rot_scene_r21")),
        head_rot_scene_r22=_csv_optional_float(row.get("head_rot_scene_r22")),
        dt_from_prev_s=_csv_optional_float(row.get("dt_from_prev_s")),
        translation_scene_dx_m=_csv_optional_float(row.get("translation_scene_dx_m")),
        translation_scene_dy_m=_csv_optional_float(row.get("translation_scene_dy_m")),
        translation_scene_dz_m=_csv_optional_float(row.get("translation_scene_dz_m")),
        translation_prev_head_dx_m=_csv_optional_float(row.get("translation_prev_head_dx_m")),
        translation_prev_head_dy_m=_csv_optional_float(row.get("translation_prev_head_dy_m")),
        translation_prev_head_dz_m=_csv_optional_float(row.get("translation_prev_head_dz_m")),
        origin_step_m=_csv_optional_float(row.get("origin_step_m")),
        head_translation_speed_m_s=_csv_optional_float(row.get("head_translation_speed_m_s")),
        relative_rot_prev_to_cur_r00=_csv_optional_float(row.get("relative_rot_prev_to_cur_r00")),
        relative_rot_prev_to_cur_r01=_csv_optional_float(row.get("relative_rot_prev_to_cur_r01")),
        relative_rot_prev_to_cur_r02=_csv_optional_float(row.get("relative_rot_prev_to_cur_r02")),
        relative_rot_prev_to_cur_r10=_csv_optional_float(row.get("relative_rot_prev_to_cur_r10")),
        relative_rot_prev_to_cur_r11=_csv_optional_float(row.get("relative_rot_prev_to_cur_r11")),
        relative_rot_prev_to_cur_r12=_csv_optional_float(row.get("relative_rot_prev_to_cur_r12")),
        relative_rot_prev_to_cur_r20=_csv_optional_float(row.get("relative_rot_prev_to_cur_r20")),
        relative_rot_prev_to_cur_r21=_csv_optional_float(row.get("relative_rot_prev_to_cur_r21")),
        relative_rot_prev_to_cur_r22=_csv_optional_float(row.get("relative_rot_prev_to_cur_r22")),
        head_forward_angle_step_deg=_csv_optional_float(row.get("head_forward_angle_step_deg")),
        head_rotation_angle_step_deg=_csv_optional_float(row.get("head_rotation_angle_step_deg")),
        head_rotation_speed_deg_s=_csv_optional_float(row.get("head_rotation_speed_deg_s")),
        validation_notes=row.get("validation_notes", "ok"),
    )


def default_head_summary_json_path(csv_path: str | Path) -> Path:
    csv_file = Path(csv_path)
    stem = csv_file.stem
    if stem.endswith("_head_samples"):
        stem = stem[: -len("_head_samples")] + "_head_summary"
    else:
        stem = f"{stem}_summary"
    return csv_file.with_name(f"{stem}.json")


def default_head_csv_path(sequence_name: str, output_dir: str | Path | None = None) -> Path:
    base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[2] / "outputs" / "reports"
    )
    return base_dir / f"{sequence_name}_head_samples.csv"


def describe_optional_numbers(values: Sequence[float | None]) -> dict[str, float | int | None]:
    finite_values = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if finite_values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(finite_values.size),
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
    }


def _invalid_head_sample(timestamp_ns: int) -> HeadSample:
    """Create a head sample with all None fields except for query timestamp and pose_valid=False."""
    return HeadSample(
        query_timestamp_ns=int(timestamp_ns),
        pose_valid=False,
        pose_dt_ns=None,
        pose_quality_score=None,
        head_origin_scene_x_m=None,
        head_origin_scene_y_m=None,
        head_origin_scene_z_m=None,
        head_right_scene_unit_x=None,
        head_right_scene_unit_y=None,
        head_right_scene_unit_z=None,
        head_up_scene_unit_x=None,
        head_up_scene_unit_y=None,
        head_up_scene_unit_z=None,
        head_forward_scene_unit_x=None,
        head_forward_scene_unit_y=None,
        head_forward_scene_unit_z=None,
        head_rot_scene_r00=None,
        head_rot_scene_r01=None,
        head_rot_scene_r02=None,
        head_rot_scene_r10=None,
        head_rot_scene_r11=None,
        head_rot_scene_r12=None,
        head_rot_scene_r20=None,
        head_rot_scene_r21=None,
        head_rot_scene_r22=None,
        dt_from_prev_s=None,
        translation_scene_dx_m=None,
        translation_scene_dy_m=None,
        translation_scene_dz_m=None,
        translation_prev_head_dx_m=None,
        translation_prev_head_dy_m=None,
        translation_prev_head_dz_m=None,
        origin_step_m=None,
        head_translation_speed_m_s=None,
        relative_rot_prev_to_cur_r00=None,
        relative_rot_prev_to_cur_r01=None,
        relative_rot_prev_to_cur_r02=None,
        relative_rot_prev_to_cur_r10=None,
        relative_rot_prev_to_cur_r11=None,
        relative_rot_prev_to_cur_r12=None,
        relative_rot_prev_to_cur_r20=None,
        relative_rot_prev_to_cur_r21=None,
        relative_rot_prev_to_cur_r22=None,
        head_forward_angle_step_deg=None,
        head_rotation_angle_step_deg=None,
        head_rotation_speed_deg_s=None,
        validation_notes="pose_query_invalid",
    )


def _scene_basis(transform_scene_cpf: Any) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Compute head basis vectors in scene frame from `T_scene_cpf`."""
    origin_scene = np.asarray(transform_scene_cpf @ [0.0, 0.0, 0.0], dtype=np.float64).reshape(-1)
    right_point_scene = np.asarray(transform_scene_cpf @ [1.0, 0.0, 0.0], dtype=np.float64).reshape(-1)
    up_point_scene = np.asarray(transform_scene_cpf @ [0.0, 1.0, 0.0], dtype=np.float64).reshape(-1)
    forward_point_scene = np.asarray(transform_scene_cpf @ [0.0, 0.0, 1.0], dtype=np.float64).reshape(-1)
    right_scene = _normalize_vector(right_point_scene - origin_scene)
    up_scene = _normalize_vector(up_point_scene - origin_scene)
    forward_scene = _normalize_vector(forward_point_scene - origin_scene)
    if not np.isfinite(origin_scene).all():
        return None, None, None, None
    return origin_scene, right_scene, up_scene, forward_scene


def _vector_from_optional_xyz(
    x_value: float | None,
    y_value: float | None,
    z_value: float | None,
) -> np.ndarray | None:
    if x_value is None or y_value is None or z_value is None:
        return None
    vector = np.asarray([x_value, y_value, z_value], dtype=np.float64)
    return vector if np.isfinite(vector).all() else None


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _vector_component_or_none(vector: np.ndarray | None, index: int) -> float | None:
    if vector is None:
        return None
    return _finite_or_none(float(vector[index]))


def _matrix_component_or_none(matrix: np.ndarray | None, row_index: int, col_index: int) -> float | None:
    if matrix is None:
        return None
    return _finite_or_none(float(matrix[row_index, col_index]))


def _row_required(row: dict[str, str], key: str) -> str:
    if key not in row:
        raise KeyError(f"Missing required head CSV field: {key}")
    return row[key]


def _csv_bool(value: str | None) -> bool:
    if value is None:
        raise ValueError("Missing boolean CSV field")
    return value.strip().lower() in {"1", "true", "yes"}


def _csv_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _csv_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
