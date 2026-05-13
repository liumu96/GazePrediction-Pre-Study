"""Local gaze dynamics features derived from extracted gaze/head samples.

This module intentionally stops at feature computation. It does not produce
fixation/saccade labels because CPF-local thresholds do not define whether the
user is fixating a stable point in the scene.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .results import sequence_file_path

from .gaze import GazeSample, csv_bool, csv_int, csv_optional_float
from .head import HeadSample


@dataclass(frozen=True)
class GazeDynamicsRow:
    """One per-frame row of CPF-local gaze dynamics plus head context."""

    query_timestamp_ns: int
    frame_index: int
    gaze_valid_for_dynamics: bool
    head_valid_for_context: bool
    dynamics_input_valid: bool
    dt_s: float | None
    delta_yaw_rad: float | None
    delta_pitch_rad: float | None
    local_angle_step_deg: float | None
    local_velocity_deg_s: float | None
    window_dispersion_deg: float | None
    gaze_head_angle_deg: float | None
    origin_step_m: float | None
    head_translation_speed_m_s: float | None
    head_forward_angle_step_deg: float | None
    pose_quality_score: float | None
    gaze_validation_notes: str
    head_validation_notes: str

    def as_csv_row(self) -> dict[str, Any]:
        return asdict(self)


def compute_gaze_dynamics_features(
    gaze_samples: Sequence[GazeSample],
    head_samples: Sequence[HeadSample],
    dispersion_window_frames: int,
) -> list[GazeDynamicsRow]:
    """Compute whole-sequence CPF-local gaze dynamics features."""

    if dispersion_window_frames <= 0:
        raise ValueError("dispersion_window_frames must be positive")
    if len(gaze_samples) != len(head_samples):
        raise ValueError("gaze_samples and head_samples must have the same length")
    if not gaze_samples:
        raise ValueError("No samples provided")

    for gaze_sample, head_sample in zip(gaze_samples, head_samples):
        if gaze_sample.query_timestamp_ns != head_sample.query_timestamp_ns:
            raise ValueError(
                "Gaze/head timestamp mismatch at "
                f"{gaze_sample.query_timestamp_ns} vs {head_sample.query_timestamp_ns}"
            )

    rows: list[GazeDynamicsRow] = []
    local_dirs = [sample.gaze_dir_cpf_unit_xyz for sample in gaze_samples]

    for index, (gaze_sample, head_sample) in enumerate(zip(gaze_samples, head_samples)):
        gaze_valid = (
            gaze_sample.gaze_valid
            and gaze_sample.gaze_dir_cpf_unit_xyz is not None
            and np.isfinite(gaze_sample.gaze_dir_cpf_unit_xyz).all()
        )
        head_valid = (
            head_sample.pose_valid
            and head_sample.head_forward_scene_unit_x is not None
            and head_sample.head_forward_scene_unit_y is not None
            and head_sample.head_forward_scene_unit_z is not None
        )
        dynamics_input_valid = gaze_valid and head_valid

        dt_s = None
        delta_yaw = None
        delta_pitch = None
        local_angle_step_deg = None
        local_velocity_deg_s = None
        gaze_head_angle_deg = None

        if gaze_valid and head_valid:
            gaze_dir_scene = gaze_sample.gaze_dir_scene_unit_xyz
            head_dir_scene = np.asarray(
                [
                    head_sample.head_forward_scene_unit_x,
                    head_sample.head_forward_scene_unit_y,
                    head_sample.head_forward_scene_unit_z,
                ],
                dtype=np.float64,
            )
            if gaze_dir_scene is not None and np.isfinite(gaze_dir_scene).all():
                gaze_head_angle_deg = angular_distance_deg(gaze_dir_scene, head_dir_scene)

        if index > 0:
            prev_gaze = gaze_samples[index - 1]
            prev_dir = local_dirs[index - 1]
            curr_dir = local_dirs[index]
            dt_ns = gaze_sample.query_timestamp_ns - prev_gaze.query_timestamp_ns
            if dt_ns > 0:
                dt_s = dt_ns / 1e9
            if (
                gaze_valid
                and prev_gaze.gaze_valid
                and prev_dir is not None
                and curr_dir is not None
                and dt_s is not None
            ):
                if prev_gaze.yaw_rad is not None and gaze_sample.yaw_rad is not None:
                    delta_yaw = gaze_sample.yaw_rad - prev_gaze.yaw_rad
                if prev_gaze.pitch_rad is not None and gaze_sample.pitch_rad is not None:
                    delta_pitch = gaze_sample.pitch_rad - prev_gaze.pitch_rad
                local_angle_step_deg = angular_distance_deg(prev_dir, curr_dir)
                if local_angle_step_deg is not None and dt_s > 0:
                    local_velocity_deg_s = local_angle_step_deg / dt_s

        rows.append(
            GazeDynamicsRow(
                query_timestamp_ns=gaze_sample.query_timestamp_ns,
                frame_index=index,
                gaze_valid_for_dynamics=gaze_valid,
                head_valid_for_context=head_valid,
                dynamics_input_valid=dynamics_input_valid,
                dt_s=dt_s,
                delta_yaw_rad=delta_yaw,
                delta_pitch_rad=delta_pitch,
                local_angle_step_deg=local_angle_step_deg,
                local_velocity_deg_s=local_velocity_deg_s,
                window_dispersion_deg=centered_window_dispersion_deg(
                    local_dirs,
                    index,
                    dispersion_window_frames,
                ),
                gaze_head_angle_deg=gaze_head_angle_deg,
                origin_step_m=head_sample.origin_step_m,
                head_translation_speed_m_s=head_sample.head_translation_speed_m_s,
                head_forward_angle_step_deg=head_sample.head_forward_angle_step_deg,
                pose_quality_score=head_sample.pose_quality_score,
                gaze_validation_notes=gaze_sample.validation_notes,
                head_validation_notes=head_sample.validation_notes,
            )
        )

    return rows


def centered_window_dispersion_deg(
    directions: Sequence[np.ndarray | None],
    center_index: int,
    window_frames: int,
) -> float | None:
    """Return max pairwise angular separation in a centered CPF-direction window."""

    if window_frames <= 1:
        return 0.0
    half = window_frames // 2
    start = center_index - half
    end = start + window_frames
    if start < 0 or end > len(directions):
        return None
    window = directions[start:end]
    if any(direction is None for direction in window):
        return None

    max_angle = 0.0
    for first_index in range(window_frames):
        first = window[first_index]
        assert first is not None
        for second_index in range(first_index + 1, window_frames):
            second = window[second_index]
            assert second is not None
            angle = angular_distance_deg(first, second)
            if angle is None:
                return None
            max_angle = max(max_angle, angle)
    return max_angle


def angular_distance_deg(first: np.ndarray, second: np.ndarray) -> float | None:
    """Return the angle in degrees between two 3D vectors."""

    first_unit = normalize_vector(first)
    second_unit = normalize_vector(second)
    if first_unit is None or second_unit is None:
        return None
    cosine = float(np.clip(np.dot(first_unit, second_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def summarize_gaze_dynamics_features(
    rows: Sequence[GazeDynamicsRow],
) -> dict[str, Any]:
    """Return a lightweight summary for one per-sequence dynamics table."""

    if not rows:
        raise ValueError("No gaze dynamics rows to summarize")

    valid_count = sum(row.dynamics_input_valid for row in rows)
    return {
        "sample_count": len(rows),
        "dynamics_input_valid_count": valid_count,
        "dynamics_input_valid_ratio": valid_count / len(rows),
        "local_angle_step_deg": describe_optional_numbers(
            [row.local_angle_step_deg for row in rows]
        ),
        "local_velocity_deg_s": describe_optional_numbers(
            [row.local_velocity_deg_s for row in rows]
        ),
        "window_dispersion_deg": describe_optional_numbers(
            [row.window_dispersion_deg for row in rows]
        ),
        "gaze_head_angle_deg": describe_optional_numbers(
            [row.gaze_head_angle_deg for row in rows]
        ),
        "origin_step_m": describe_optional_numbers([row.origin_step_m for row in rows]),
        "head_translation_speed_m_s": describe_optional_numbers(
            [row.head_translation_speed_m_s for row in rows]
        ),
        "head_forward_angle_step_deg": describe_optional_numbers(
            [row.head_forward_angle_step_deg for row in rows]
        ),
        "dominant_gaze_issues": aggregate_note_counts(
            [row.gaze_validation_notes for row in rows]
        ),
        "dominant_head_issues": aggregate_note_counts(
            [row.head_validation_notes for row in rows]
        ),
    }


def write_gaze_dynamics_csv(
    path: str | Path,
    rows: Sequence[GazeDynamicsRow],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [row.as_csv_row() for row in rows]
    if not materialized:
        raise ValueError("No rows to write")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def read_gaze_dynamics_csv(path: str | Path) -> list[GazeDynamicsRow]:
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [gaze_dynamics_from_csv_row(row) for row in reader]


def gaze_dynamics_from_csv_row(row: dict[str, str]) -> GazeDynamicsRow:
    return GazeDynamicsRow(
        query_timestamp_ns=csv_int(row["query_timestamp_ns"]),
        frame_index=csv_int(row["frame_index"]),
        gaze_valid_for_dynamics=csv_bool(row["gaze_valid_for_dynamics"]),
        head_valid_for_context=csv_bool(row["head_valid_for_context"]),
        dynamics_input_valid=csv_bool(row["dynamics_input_valid"]),
        dt_s=csv_optional_float(row["dt_s"]),
        delta_yaw_rad=csv_optional_float(row["delta_yaw_rad"]),
        delta_pitch_rad=csv_optional_float(row["delta_pitch_rad"]),
        local_angle_step_deg=csv_optional_float(row["local_angle_step_deg"]),
        local_velocity_deg_s=csv_optional_float(row["local_velocity_deg_s"]),
        window_dispersion_deg=csv_optional_float(row["window_dispersion_deg"]),
        gaze_head_angle_deg=csv_optional_float(row["gaze_head_angle_deg"]),
        origin_step_m=csv_optional_float(row["origin_step_m"]),
        head_translation_speed_m_s=csv_optional_float(row["head_translation_speed_m_s"]),
        head_forward_angle_step_deg=csv_optional_float(row["head_forward_angle_step_deg"]),
        pose_quality_score=csv_optional_float(row["pose_quality_score"]),
        gaze_validation_notes=row["gaze_validation_notes"],
        head_validation_notes=row["head_validation_notes"],
    )


def default_gaze_dynamics_csv_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    return sequence_file_path(output_dir, sequence_name, "dynamics", "gaze_dynamics.csv")


def default_gaze_dynamics_summary_json_path(csv_path: str | Path) -> Path:
    csv_file = Path(csv_path)
    stem = csv_file.stem
    if stem == "gaze_dynamics":
        stem = "gaze_dynamics_summary"
    elif stem.endswith("_gaze_dynamics"):
        stem = stem[: -len("_gaze_dynamics")] + "_gaze_dynamics_summary"
    else:
        stem = f"{stem}_summary"
    return csv_file.with_name(f"{stem}.json")


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def read_summary_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        "p50": float(np.percentile(finite_values, 50)),
        "p75": float(np.percentile(finite_values, 75)),
        "p90": float(np.percentile(finite_values, 90)),
        "p95": float(np.percentile(finite_values, 95)),
        "p99": float(np.percentile(finite_values, 99)),
    }


def aggregate_note_counts(notes: Sequence[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for note_blob in notes:
        if note_blob == "ok":
            continue
        for note in note_blob.split(";"):
            if note:
                counter[note] += 1
    return dict(sorted(counter.items()))
