"""Scene-direction gaze event detection.

This module defines a first scene/world-direction event layer. It uses
Scene-frame gaze direction stability, not CPF-local eye stability, so the
result is closer to the intuitive question "is the user looking in a stable
world direction?".
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

from .gaze import GazeSample, csv_bool, csv_int, csv_optional_float, csv_optional_int
from .gaze_dynamics import (
    angular_distance_deg,
    centered_window_dispersion_deg,
    describe_optional_numbers,
)


@dataclass(frozen=True)
class SceneGazeEventFeatureRow:
    """One per-frame scene-direction gaze dynamics row."""

    query_timestamp_ns: int
    frame_index: int
    scene_gaze_valid: bool
    dt_s: float | None
    scene_angle_step_deg: float | None
    scene_velocity_deg_s: float | None
    scene_window_dispersion_deg: float | None
    gaze_validation_notes: str

    def as_csv_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneGazeFrameLabel:
    """One per-frame scene-direction event label."""

    query_timestamp_ns: int
    frame_index: int
    scene_gaze_valid: bool
    scene_event_label: str
    scene_event_id: int | None
    scene_fixation_id: int | None

    def as_csv_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneGazeEventSegment:
    """One contiguous scene-direction event segment."""

    scene_event_id: int
    scene_event_label: str
    scene_fixation_id: int | None
    start_frame_index: int
    end_frame_index: int
    frame_count: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    duration_ms: float
    mean_scene_velocity_deg_s: float | None
    max_scene_velocity_deg_s: float | None
    max_scene_window_dispersion_deg: float | None
    dominant_gaze_issue: str

    def as_csv_row(self) -> dict[str, Any]:
        return asdict(self)


def compute_scene_gaze_event_features(
    gaze_samples: Sequence[GazeSample],
    dispersion_window_frames: int,
) -> list[SceneGazeEventFeatureRow]:
    """Compute whole-sequence Scene-frame gaze direction dynamics."""

    if dispersion_window_frames <= 0:
        raise ValueError("dispersion_window_frames must be positive")
    if not gaze_samples:
        raise ValueError("No gaze samples provided")

    scene_dirs = [sample.gaze_dir_scene_unit_xyz for sample in gaze_samples]
    rows: list[SceneGazeEventFeatureRow] = []

    for index, gaze_sample in enumerate(gaze_samples):
        curr_dir = scene_dirs[index]
        scene_gaze_valid = (
            gaze_sample.gaze_valid
            and gaze_sample.pose_valid
            and curr_dir is not None
            and np.isfinite(curr_dir).all()
        )

        dt_s = None
        scene_angle_step_deg = None
        scene_velocity_deg_s = None

        if index > 0:
            prev_sample = gaze_samples[index - 1]
            prev_dir = scene_dirs[index - 1]
            dt_ns = gaze_sample.query_timestamp_ns - prev_sample.query_timestamp_ns
            if dt_ns > 0:
                dt_s = dt_ns / 1e9
            if (
                scene_gaze_valid
                and prev_sample.gaze_valid
                and prev_sample.pose_valid
                and prev_dir is not None
                and np.isfinite(prev_dir).all()
                and dt_s is not None
            ):
                scene_angle_step_deg = angular_distance_deg(prev_dir, curr_dir)
                if scene_angle_step_deg is not None and dt_s > 0:
                    scene_velocity_deg_s = scene_angle_step_deg / dt_s

        rows.append(
            SceneGazeEventFeatureRow(
                query_timestamp_ns=gaze_sample.query_timestamp_ns,
                frame_index=index,
                scene_gaze_valid=scene_gaze_valid,
                dt_s=dt_s,
                scene_angle_step_deg=scene_angle_step_deg,
                scene_velocity_deg_s=scene_velocity_deg_s,
                scene_window_dispersion_deg=centered_window_dispersion_deg(
                    scene_dirs,
                    index,
                    dispersion_window_frames,
                ),
                gaze_validation_notes=gaze_sample.validation_notes,
            )
        )

    return rows


def label_scene_gaze_events(
    feature_rows: Sequence[SceneGazeEventFeatureRow],
    velocity_threshold_deg_s: float,
    dispersion_threshold_deg: float,
    min_fixation_duration_ms: float,
) -> tuple[list[SceneGazeFrameLabel], list[SceneGazeEventSegment]]:
    """Label final scene-direction fixation/transition segments."""

    if velocity_threshold_deg_s <= 0:
        raise ValueError("velocity_threshold_deg_s must be positive")
    if dispersion_threshold_deg <= 0:
        raise ValueError("dispersion_threshold_deg must be positive")
    if min_fixation_duration_ms <= 0:
        raise ValueError("min_fixation_duration_ms must be positive")
    if not feature_rows:
        raise ValueError("No scene gaze event features provided")

    preliminary_labels = [
        classify_scene_gaze_frame(
            row,
            velocity_threshold_deg_s=velocity_threshold_deg_s,
            dispersion_threshold_deg=dispersion_threshold_deg,
        )
        for row in feature_rows
    ]
    frame_duration_ms = estimate_frame_duration_ms(feature_rows)
    final_labels = demote_short_fixation_runs(
        feature_rows,
        preliminary_labels,
        min_fixation_duration_ms=min_fixation_duration_ms,
        frame_duration_ms=frame_duration_ms,
    )
    return build_scene_gaze_outputs(feature_rows, final_labels, frame_duration_ms)


def classify_scene_gaze_frame(
    row: SceneGazeEventFeatureRow,
    velocity_threshold_deg_s: float,
    dispersion_threshold_deg: float,
) -> str:
    if not row.scene_gaze_valid:
        return "invalid"
    if row.scene_velocity_deg_s is None or row.scene_window_dispersion_deg is None:
        return "transition"
    if (
        row.scene_velocity_deg_s <= velocity_threshold_deg_s
        and row.scene_window_dispersion_deg <= dispersion_threshold_deg
    ):
        return "fixation"
    return "transition"


def demote_short_fixation_runs(
    rows: Sequence[SceneGazeEventFeatureRow],
    labels: Sequence[str],
    min_fixation_duration_ms: float,
    frame_duration_ms: float,
) -> list[str]:
    final_labels = list(labels)
    start = 0
    while start < len(labels):
        label = labels[start]
        end = start
        while end + 1 < len(labels) and labels[end + 1] == label:
            end += 1
        if label == "fixation":
            duration_ms = segment_duration_ms(rows, start, end, frame_duration_ms)
            if duration_ms < min_fixation_duration_ms:
                for index in range(start, end + 1):
                    final_labels[index] = "transition"
        start = end + 1
    return final_labels


def build_scene_gaze_outputs(
    rows: Sequence[SceneGazeEventFeatureRow],
    labels: Sequence[str],
    frame_duration_ms: float,
) -> tuple[list[SceneGazeFrameLabel], list[SceneGazeEventSegment]]:
    frame_labels: list[SceneGazeFrameLabel] = []
    segments: list[SceneGazeEventSegment] = []
    fixation_id_by_frame: dict[int, int] = {}

    event_id = 1
    fixation_id = 1
    start = 0
    while start < len(rows):
        label = labels[start]
        end = start
        while end + 1 < len(rows) and labels[end + 1] == label:
            end += 1

        segment_fixation_id = fixation_id if label == "fixation" else None
        if segment_fixation_id is not None:
            for index in range(start, end + 1):
                fixation_id_by_frame[rows[index].frame_index] = segment_fixation_id

        window = rows[start : end + 1]
        segments.append(
            SceneGazeEventSegment(
                scene_event_id=event_id,
                scene_event_label=label,
                scene_fixation_id=segment_fixation_id,
                start_frame_index=rows[start].frame_index,
                end_frame_index=rows[end].frame_index,
                frame_count=end - start + 1,
                start_timestamp_ns=rows[start].query_timestamp_ns,
                end_timestamp_ns=rows[end].query_timestamp_ns,
                duration_ms=segment_duration_ms(rows, start, end, frame_duration_ms),
                mean_scene_velocity_deg_s=mean_optional(
                    [row.scene_velocity_deg_s for row in window]
                ),
                max_scene_velocity_deg_s=max_optional(
                    [row.scene_velocity_deg_s for row in window]
                ),
                max_scene_window_dispersion_deg=max_optional(
                    [row.scene_window_dispersion_deg for row in window]
                ),
                dominant_gaze_issue=dominant_note(
                    [row.gaze_validation_notes for row in window]
                ),
            )
        )

        event_id += 1
        if segment_fixation_id is not None:
            fixation_id += 1
        start = end + 1

    event_id_by_frame: dict[int, int] = {}
    for segment in segments:
        for frame_index in range(segment.start_frame_index, segment.end_frame_index + 1):
            event_id_by_frame[frame_index] = segment.scene_event_id

    for row, label in zip(rows, labels, strict=True):
        frame_labels.append(
            SceneGazeFrameLabel(
                query_timestamp_ns=row.query_timestamp_ns,
                frame_index=row.frame_index,
                scene_gaze_valid=row.scene_gaze_valid,
                scene_event_label=label,
                scene_event_id=event_id_by_frame.get(row.frame_index),
                scene_fixation_id=fixation_id_by_frame.get(row.frame_index),
            )
        )

    return frame_labels, segments


def summarize_scene_gaze_events(
    feature_rows: Sequence[SceneGazeEventFeatureRow],
    frame_labels: Sequence[SceneGazeFrameLabel],
    segments: Sequence[SceneGazeEventSegment],
) -> dict[str, Any]:
    if not feature_rows:
        raise ValueError("No scene gaze event feature rows to summarize")

    valid_count = sum(row.scene_gaze_valid for row in feature_rows)
    fixation_segments = [
        segment for segment in segments if segment.scene_event_label == "fixation"
    ]
    transition_segments = [
        segment for segment in segments if segment.scene_event_label == "transition"
    ]
    invalid_segments = [
        segment for segment in segments if segment.scene_event_label == "invalid"
    ]
    return {
        "sample_count": len(feature_rows),
        "scene_gaze_valid_count": valid_count,
        "scene_gaze_valid_ratio": valid_count / len(feature_rows),
        "frame_label_counts": count_values(
            [row.scene_event_label for row in frame_labels]
        ),
        "segment_label_counts": count_values(
            [segment.scene_event_label for segment in segments]
        ),
        "fixation_count": len(fixation_segments),
        "transition_count": len(transition_segments),
        "invalid_count": len(invalid_segments),
        "scene_angle_step_deg": describe_optional_numbers(
            [row.scene_angle_step_deg for row in feature_rows]
        ),
        "scene_velocity_deg_s": describe_optional_numbers(
            [row.scene_velocity_deg_s for row in feature_rows]
        ),
        "scene_window_dispersion_deg": describe_optional_numbers(
            [row.scene_window_dispersion_deg for row in feature_rows]
        ),
        "fixation_duration_ms": describe_optional_numbers(
            [segment.duration_ms for segment in fixation_segments]
        ),
        "transition_duration_ms": describe_optional_numbers(
            [segment.duration_ms for segment in transition_segments]
        ),
        "dominant_gaze_issues": aggregate_note_counts(
            [row.gaze_validation_notes for row in feature_rows]
        ),
    }


def estimate_frame_duration_ms(rows: Sequence[SceneGazeEventFeatureRow]) -> float:
    dt_values = [
        float(row.dt_s) * 1000.0
        for row in rows
        if row.dt_s is not None and np.isfinite(row.dt_s) and row.dt_s > 0
    ]
    if not dt_values:
        return 0.0
    return float(np.median(np.asarray(dt_values, dtype=np.float64)))


def segment_duration_ms(
    rows: Sequence[SceneGazeEventFeatureRow],
    start_index: int,
    end_index: int,
    frame_duration_ms: float,
) -> float:
    duration = (rows[end_index].query_timestamp_ns - rows[start_index].query_timestamp_ns) / 1e6
    return float(duration + frame_duration_ms)


def mean_optional(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def max_optional(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return float(max(finite))


def aggregate_note_counts(notes: Sequence[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for note_blob in notes:
        if note_blob == "ok":
            continue
        for note in note_blob.split(";"):
            if note:
                counter[note] += 1
    return dict(sorted(counter.items()))


def dominant_note(notes: Sequence[str]) -> str:
    counts = aggregate_note_counts(notes)
    if not counts:
        return "ok"
    return max(counts.items(), key=lambda item: item[1])[0]


def count_values(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def write_scene_gaze_event_features_csv(
    path: str | Path,
    rows: Sequence[SceneGazeEventFeatureRow],
) -> None:
    write_dataclass_rows_csv(path, rows)


def write_scene_gaze_frame_labels_csv(
    path: str | Path,
    rows: Sequence[SceneGazeFrameLabel],
) -> None:
    write_dataclass_rows_csv(path, rows)


def write_scene_gaze_event_segments_csv(
    path: str | Path,
    rows: Sequence[SceneGazeEventSegment],
) -> None:
    write_dataclass_rows_csv(path, rows)


def write_dataclass_rows_csv(path: str | Path, rows: Sequence[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [row.as_csv_row() for row in rows]
    if not materialized:
        raise ValueError("No rows to write")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def read_scene_gaze_event_features_csv(
    path: str | Path,
) -> list[SceneGazeEventFeatureRow]:
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [scene_gaze_event_feature_from_csv_row(row) for row in reader]


def read_scene_gaze_frame_labels_csv(
    path: str | Path,
) -> list[SceneGazeFrameLabel]:
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [scene_gaze_frame_label_from_csv_row(row) for row in reader]


def read_scene_gaze_event_segments_csv(
    path: str | Path,
) -> list[SceneGazeEventSegment]:
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [scene_gaze_event_segment_from_csv_row(row) for row in reader]


def scene_gaze_event_feature_from_csv_row(
    row: dict[str, str],
) -> SceneGazeEventFeatureRow:
    return SceneGazeEventFeatureRow(
        query_timestamp_ns=csv_int(row["query_timestamp_ns"]),
        frame_index=csv_int(row["frame_index"]),
        scene_gaze_valid=csv_bool(row["scene_gaze_valid"]),
        dt_s=csv_optional_float(row["dt_s"]),
        scene_angle_step_deg=csv_optional_float(row["scene_angle_step_deg"]),
        scene_velocity_deg_s=csv_optional_float(row["scene_velocity_deg_s"]),
        scene_window_dispersion_deg=csv_optional_float(
            row["scene_window_dispersion_deg"]
        ),
        gaze_validation_notes=row["gaze_validation_notes"],
    )


def scene_gaze_frame_label_from_csv_row(row: dict[str, str]) -> SceneGazeFrameLabel:
    return SceneGazeFrameLabel(
        query_timestamp_ns=csv_int(row["query_timestamp_ns"]),
        frame_index=csv_int(row["frame_index"]),
        scene_gaze_valid=csv_bool(row["scene_gaze_valid"]),
        scene_event_label=row["scene_event_label"],
        scene_event_id=csv_optional_int(row["scene_event_id"]),
        scene_fixation_id=csv_optional_int(row["scene_fixation_id"]),
    )


def scene_gaze_event_segment_from_csv_row(
    row: dict[str, str],
) -> SceneGazeEventSegment:
    return SceneGazeEventSegment(
        scene_event_id=csv_int(row["scene_event_id"]),
        scene_event_label=row["scene_event_label"],
        scene_fixation_id=csv_optional_int(row["scene_fixation_id"]),
        start_frame_index=csv_int(row["start_frame_index"]),
        end_frame_index=csv_int(row["end_frame_index"]),
        frame_count=csv_int(row["frame_count"]),
        start_timestamp_ns=csv_int(row["start_timestamp_ns"]),
        end_timestamp_ns=csv_int(row["end_timestamp_ns"]),
        duration_ms=float(row["duration_ms"]),
        mean_scene_velocity_deg_s=csv_optional_float(row["mean_scene_velocity_deg_s"]),
        max_scene_velocity_deg_s=csv_optional_float(row["max_scene_velocity_deg_s"]),
        max_scene_window_dispersion_deg=csv_optional_float(
            row["max_scene_window_dispersion_deg"]
        ),
        dominant_gaze_issue=row["dominant_gaze_issue"],
    )


def default_scene_gaze_event_features_csv_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    return _base_dir(output_dir) / f"{sequence_name}_scene_gaze_event_features.csv"


def default_scene_gaze_frame_labels_csv_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    return _base_dir(output_dir) / f"{sequence_name}_scene_gaze_frame_labels.csv"


def default_scene_gaze_event_segments_csv_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    return _base_dir(output_dir) / f"{sequence_name}_scene_gaze_event_segments.csv"


def default_scene_gaze_event_summary_json_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    return _base_dir(output_dir) / f"{sequence_name}_scene_gaze_event_summary.json"


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, default=json_scalar_default),
        encoding="utf-8",
    )


def json_scalar_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _base_dir(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return Path(__file__).resolve().parents[2] / "outputs" / "reports"
