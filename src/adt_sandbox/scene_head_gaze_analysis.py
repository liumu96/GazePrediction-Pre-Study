"""Scene-level head-gaze relationship analysis.

This module is downstream of the extracted gaze/head feature layers and the
scene-direction event layer. It keeps CPF-local dynamics and Scene-world gaze
dynamics side by side so that we can distinguish eye-in-head motion from final
world-gaze motion.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gaze import GazeSample
from .gaze_dynamics import describe_optional_numbers
from .head import HeadSample
from .head_gaze_analysis import (
    build_head_gaze_analysis_rows,
    pearson_corr,
)
from .results import sequence_file_path
from .scene_gaze_events import SceneGazeEventFeatureRow, SceneGazeFrameLabel


@dataclass(frozen=True)
class SceneHeadGazeAnalysisRow:
    """One per-frame joined row for Scene-level head-gaze analysis."""

    query_timestamp_ns: int
    frame_index: int
    analysis_valid: bool
    scene_gaze_valid: bool
    scene_event_label: str
    scene_event_id: int | None
    scene_fixation_id: int | None
    scene_angle_step_deg: float | None
    scene_velocity_deg_s: float | None
    scene_window_dispersion_deg: float | None
    cpf_local_angle_step_deg: float | None
    cpf_local_velocity_deg_s: float | None
    cpf_delta_yaw_rad: float | None
    cpf_delta_pitch_rad: float | None
    gaze_head_angle_deg: float | None
    head_rotation_speed_deg_s: float | None
    head_translation_speed_m_s: float | None
    head_forward_angle_step_deg: float | None
    pose_quality_score: float | None
    gaze_validation_notes: str
    head_validation_notes: str

    def as_csv_row(self) -> dict[str, Any]:
        return asdict(self)


def build_scene_head_gaze_analysis_rows(
    gaze_samples: Sequence[GazeSample],
    head_samples: Sequence[HeadSample],
    scene_feature_rows: Sequence[SceneGazeEventFeatureRow],
    scene_label_rows: Sequence[SceneGazeFrameLabel],
    dispersion_window_frames: int = 5,
) -> list[SceneHeadGazeAnalysisRow]:
    """Join CPF dynamics, Scene dynamics, head dynamics, and scene event labels."""

    base_rows = build_head_gaze_analysis_rows(
        gaze_samples,
        head_samples,
        dispersion_window_frames=dispersion_window_frames,
    )
    feature_by_frame = {row.frame_index: row for row in scene_feature_rows}
    label_by_frame = {row.frame_index: row for row in scene_label_rows}

    rows: list[SceneHeadGazeAnalysisRow] = []
    for base_row in base_rows:
        scene_feature = feature_by_frame.get(base_row.frame_index)
        scene_label = label_by_frame.get(base_row.frame_index)
        if scene_feature is None or scene_label is None:
            raise ValueError(
                "Missing scene event row for frame "
                f"{base_row.frame_index}. Run detect_scene_gaze_events.py first."
            )
        if base_row.query_timestamp_ns != scene_feature.query_timestamp_ns:
            raise ValueError(
                "Timestamp mismatch between head-gaze and scene feature rows at "
                f"frame {base_row.frame_index}"
            )

        analysis_valid = (
            base_row.dynamics_input_valid
            and scene_feature.scene_gaze_valid
            and scene_feature.scene_velocity_deg_s is not None
        )
        rows.append(
            SceneHeadGazeAnalysisRow(
                query_timestamp_ns=base_row.query_timestamp_ns,
                frame_index=base_row.frame_index,
                analysis_valid=analysis_valid,
                scene_gaze_valid=scene_feature.scene_gaze_valid,
                scene_event_label=scene_label.scene_event_label,
                scene_event_id=scene_label.scene_event_id,
                scene_fixation_id=scene_label.scene_fixation_id,
                scene_angle_step_deg=scene_feature.scene_angle_step_deg,
                scene_velocity_deg_s=scene_feature.scene_velocity_deg_s,
                scene_window_dispersion_deg=scene_feature.scene_window_dispersion_deg,
                cpf_local_angle_step_deg=base_row.local_angle_step_deg,
                cpf_local_velocity_deg_s=base_row.local_velocity_deg_s,
                cpf_delta_yaw_rad=base_row.delta_yaw_rad,
                cpf_delta_pitch_rad=base_row.delta_pitch_rad,
                gaze_head_angle_deg=base_row.gaze_head_angle_deg,
                head_rotation_speed_deg_s=base_row.head_rotation_speed_deg_s,
                head_translation_speed_m_s=base_row.head_translation_speed_m_s,
                head_forward_angle_step_deg=base_row.head_forward_angle_step_deg,
                pose_quality_score=base_row.pose_quality_score,
                gaze_validation_notes=base_row.gaze_validation_notes,
                head_validation_notes=base_row.head_validation_notes,
            )
        )
    return rows


def summarize_scene_head_gaze_analysis_rows(
    rows: Sequence[SceneHeadGazeAnalysisRow],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No scene head-gaze rows to summarize")

    valid_rows = [row for row in rows if row.analysis_valid]
    head_rotation_values = [
        row.head_rotation_speed_deg_s
        for row in valid_rows
        if row.head_rotation_speed_deg_s is not None
    ]

    return {
        "sample_count": len(rows),
        "analysis_valid_count": len(valid_rows),
        "analysis_valid_ratio": len(valid_rows) / len(rows),
        "scene_event_label_counts": count_labels([row.scene_event_label for row in rows]),
        "scene_event_label_fractions": label_fractions(
            [row.scene_event_label for row in rows]
        ),
        "scene_dynamics": {
            "scene_velocity_deg_s": describe_optional_numbers(
                [row.scene_velocity_deg_s for row in valid_rows]
            ),
            "scene_window_dispersion_deg": describe_optional_numbers(
                [row.scene_window_dispersion_deg for row in valid_rows]
            ),
        },
        "cpf_dynamics": {
            "cpf_local_velocity_deg_s": describe_optional_numbers(
                [row.cpf_local_velocity_deg_s for row in valid_rows]
            ),
            "cpf_local_angle_step_deg": describe_optional_numbers(
                [row.cpf_local_angle_step_deg for row in valid_rows]
            ),
        },
        "head_dynamics": {
            "head_rotation_speed_deg_s": describe_optional_numbers(
                [row.head_rotation_speed_deg_s for row in valid_rows]
            ),
            "head_translation_speed_m_s": describe_optional_numbers(
                [row.head_translation_speed_m_s for row in valid_rows]
            ),
        },
        "correlations": {
            "scene_velocity_vs_head_rotation_speed": pearson_corr(
                [row.scene_velocity_deg_s for row in valid_rows],
                [row.head_rotation_speed_deg_s for row in valid_rows],
            ),
            "scene_velocity_vs_head_translation_speed": pearson_corr(
                [row.scene_velocity_deg_s for row in valid_rows],
                [row.head_translation_speed_m_s for row in valid_rows],
            ),
            "cpf_velocity_vs_head_rotation_speed": pearson_corr(
                [row.cpf_local_velocity_deg_s for row in valid_rows],
                [row.head_rotation_speed_deg_s for row in valid_rows],
            ),
            "cpf_velocity_vs_scene_velocity": pearson_corr(
                [row.cpf_local_velocity_deg_s for row in valid_rows],
                [row.scene_velocity_deg_s for row in valid_rows],
            ),
        },
        "event_groups": summarize_event_groups(valid_rows),
        "head_rotation_speed_groups": summarize_head_rotation_speed_groups(
            valid_rows,
            head_rotation_values,
        ),
    }


def summarize_event_groups(
    rows: Sequence[SceneHeadGazeAnalysisRow],
) -> dict[str, Any]:
    labels = ["fixation", "transition", "invalid"]
    total = len(rows)
    groups: dict[str, Any] = {}
    for label in labels:
        group_rows = [row for row in rows if row.scene_event_label == label]
        groups[label] = summarize_group(group_rows, total)
    return groups


def summarize_group(
    rows: Sequence[SceneHeadGazeAnalysisRow],
    total_count: int,
) -> dict[str, Any]:
    return {
        "frame_count": len(rows),
        "frame_fraction": len(rows) / total_count if total_count else None,
        "scene_velocity_deg_s": describe_optional_numbers(
            [row.scene_velocity_deg_s for row in rows]
        ),
        "cpf_local_velocity_deg_s": describe_optional_numbers(
            [row.cpf_local_velocity_deg_s for row in rows]
        ),
        "head_rotation_speed_deg_s": describe_optional_numbers(
            [row.head_rotation_speed_deg_s for row in rows]
        ),
        "head_translation_speed_m_s": describe_optional_numbers(
            [row.head_translation_speed_m_s for row in rows]
        ),
        "gaze_head_angle_deg": describe_optional_numbers(
            [row.gaze_head_angle_deg for row in rows]
        ),
        "corr_scene_velocity_vs_head_rotation_speed": pearson_corr(
            [row.scene_velocity_deg_s for row in rows],
            [row.head_rotation_speed_deg_s for row in rows],
        ),
        "corr_cpf_velocity_vs_head_rotation_speed": pearson_corr(
            [row.cpf_local_velocity_deg_s for row in rows],
            [row.head_rotation_speed_deg_s for row in rows],
        ),
    }


def summarize_head_rotation_speed_groups(
    rows: Sequence[SceneHeadGazeAnalysisRow],
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
    grouped: dict[str, list[SceneHeadGazeAnalysisRow]] = {
        "low": [],
        "mid": [],
        "high": [],
    }
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
                **summarize_group(group_rows, len(rows)),
                "fixation_fraction": label_fraction(group_rows, "fixation"),
                "transition_fraction": label_fraction(group_rows, "transition"),
            }
            for label, group_rows in grouped.items()
        },
    }


def count_labels(labels: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def label_fractions(labels: Sequence[str]) -> dict[str, float]:
    counts = count_labels(labels)
    total = len(labels)
    if total == 0:
        return {}
    return {label: count / total for label, count in counts.items()}


def label_fraction(rows: Sequence[SceneHeadGazeAnalysisRow], label: str) -> float | None:
    if not rows:
        return None
    return sum(row.scene_event_label == label for row in rows) / len(rows)


def write_scene_head_gaze_analysis_rows_csv(
    path: str | Path,
    rows: Sequence[SceneHeadGazeAnalysisRow],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [row.as_csv_row() for row in rows]
    if not materialized:
        raise ValueError("No scene head-gaze analysis rows to write")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


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


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def default_scene_head_gaze_analysis_rows_csv_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    return sequence_file_path(
        output_dir,
        sequence_name,
        "analysis",
        "scene_head_gaze_analysis_rows.csv",
    )


def default_scene_head_gaze_analysis_summary_json_path(csv_path: str | Path) -> Path:
    csv_file = Path(csv_path)
    stem = csv_file.stem
    if stem == "scene_head_gaze_analysis_rows":
        stem = "scene_head_gaze_analysis_summary"
    elif stem.endswith("_scene_head_gaze_analysis_rows"):
        stem = (
            stem[: -len("_scene_head_gaze_analysis_rows")]
            + "_scene_head_gaze_analysis_summary"
        )
    else:
        stem = f"{stem}_summary"
    return csv_file.with_name(f"{stem}.json")
