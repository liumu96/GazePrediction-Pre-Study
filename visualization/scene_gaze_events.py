"""Visualization helpers for scene-direction gaze event labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from adt_sandbox.scene_gaze_events import (
    SceneGazeEventFeatureRow,
    SceneGazeEventSegment,
    SceneGazeFrameLabel,
)

LABEL_COLORS = {
    "fixation": "#2ca25f",
    "transition": "#fdae61",
    "invalid": "#d73027",
}


@dataclass(frozen=True)
class LabelRun:
    """Contiguous frame interval with the same event label."""

    label: str
    start_frame: int
    end_frame: int


def resolve_frame_window(
    features: list[SceneGazeEventFeatureRow],
    start_frame: int,
    end_frame: int | None,
    max_frames: int,
) -> tuple[int, int]:
    """Resolve an inclusive/exclusive frame window against available rows."""

    if start_frame < 0:
        raise ValueError("start-frame must be non-negative")
    if not features:
        raise ValueError("No scene event features found")

    max_sequence_frame = max(row.frame_index for row in features)
    if end_frame is None:
        if max_frames < 0:
            raise ValueError("max-frames must be non-negative")
        end_frame = max_sequence_frame + 1 if max_frames == 0 else start_frame + max_frames
    if end_frame <= start_frame:
        raise ValueError("end-frame must be greater than start-frame")
    return start_frame, min(end_frame, max_sequence_frame + 1)


def select_feature_window(
    features: list[SceneGazeEventFeatureRow],
    start_frame: int,
    end_frame: int,
) -> list[SceneGazeEventFeatureRow]:
    """Return feature rows whose frame indices fall inside the selected window."""

    return [
        row
        for row in sorted(features, key=lambda item: item.frame_index)
        if start_frame <= row.frame_index < end_frame
    ]


def plot_scene_gaze_event_timeline(
    output_path: Path,
    sequence_name: str,
    features: list[SceneGazeEventFeatureRow],
    labels: list[SceneGazeFrameLabel],
    segments: list[SceneGazeEventSegment],
    start_frame: int,
    end_frame: int,
    velocity_threshold_deg_s: float,
    dispersion_threshold_deg: float,
    velocity_ymax: float | None,
    dispersion_ymax: float | None,
) -> None:
    """Write a three-track timeline for labels, velocity, and dispersion."""

    frames = np.asarray([row.frame_index for row in features], dtype=np.int64)
    velocities = optional_float_array([row.scene_velocity_deg_s for row in features])
    dispersions = optional_float_array(
        [row.scene_window_dispersion_deg for row in features]
    )
    label_by_frame = {row.frame_index: row.scene_event_label for row in labels}
    label_runs = event_spans_for_window(
        labels=labels,
        segments=segments,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.5, 2.0, 2.0]},
    )
    paint_label_spans(axes, label_runs)

    event_ax, velocity_ax, dispersion_ax = axes
    draw_event_track(event_ax, frames, label_by_frame)
    draw_metric_axis(
        velocity_ax,
        frames,
        velocities,
        ylabel="Scene velocity [deg/s]",
        color="#2166ac",
        threshold=velocity_threshold_deg_s,
        ymax=velocity_ymax,
    )
    draw_metric_axis(
        dispersion_ax,
        frames,
        dispersions,
        ylabel="Scene dispersion [deg]",
        color="#762a83",
        threshold=dispersion_threshold_deg,
        ymax=dispersion_ymax,
    )

    event_ax.set_title(
        f"{sequence_name}: scene-direction gaze events "
        f"(frames {start_frame}..{end_frame})"
    )
    dispersion_ax.set_xlabel("Frame index")
    fig.legend(
        handles=[
            Patch(facecolor=LABEL_COLORS["fixation"], label="fixation"),
            Patch(facecolor=LABEL_COLORS["transition"], label="transition"),
            Patch(facecolor=LABEL_COLORS["invalid"], label="invalid"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def event_spans_for_window(
    labels: list[SceneGazeFrameLabel],
    segments: list[SceneGazeEventSegment],
    start_frame: int,
    end_frame: int,
) -> list[LabelRun]:
    """Prefer segment rows, then fall back to reconstructing runs from labels."""

    overlapping_segments = [
        LabelRun(
            label=segment.scene_event_label,
            start_frame=max(segment.start_frame_index, start_frame),
            end_frame=min(segment.end_frame_index + 1, end_frame),
        )
        for segment in segments
        if segment.end_frame_index >= start_frame and segment.start_frame_index < end_frame
    ]
    if overlapping_segments:
        return overlapping_segments

    selected_labels = [
        row
        for row in sorted(labels, key=lambda item: item.frame_index)
        if start_frame <= row.frame_index < end_frame
    ]
    runs: list[LabelRun] = []
    if not selected_labels:
        return runs

    run_label = selected_labels[0].scene_event_label
    run_start = selected_labels[0].frame_index
    prev_frame = selected_labels[0].frame_index
    for row in selected_labels[1:]:
        if row.scene_event_label != run_label or row.frame_index != prev_frame + 1:
            runs.append(
                LabelRun(
                    label=run_label,
                    start_frame=run_start,
                    end_frame=prev_frame + 1,
                )
            )
            run_label = row.scene_event_label
            run_start = row.frame_index
        prev_frame = row.frame_index
    runs.append(LabelRun(label=run_label, start_frame=run_start, end_frame=prev_frame + 1))
    return runs


def paint_label_spans(axes: np.ndarray, runs: list[LabelRun]) -> None:
    """Draw event intervals behind all timeline axes."""

    for run in runs:
        color = LABEL_COLORS.get(run.label, "#bdbdbd")
        axes[0].axvspan(run.start_frame, run.end_frame, color=color, alpha=0.85)
        for axis in axes[1:]:
            axis.axvspan(run.start_frame, run.end_frame, color=color, alpha=0.12)


def draw_event_track(
    axis: plt.Axes,
    frames: np.ndarray,
    label_by_frame: dict[int, str],
) -> None:
    """Draw labels as a compact categorical step track."""

    label_values = {"invalid": -1.0, "transition": 0.0, "fixation": 1.0}
    values = np.asarray(
        [label_values.get(label_by_frame.get(int(frame), "invalid"), -1.0) for frame in frames],
        dtype=np.float64,
    )
    axis.step(frames, values, where="mid", color="#252525", linewidth=0.8)
    axis.set_ylim(-1.5, 1.5)
    axis.set_yticks([-1, 0, 1])
    axis.set_yticklabels(["invalid", "transition", "fixation"])
    axis.set_ylabel("label")
    axis.grid(axis="x", alpha=0.25)


def draw_metric_axis(
    axis: plt.Axes,
    frames: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    color: str,
    threshold: float,
    ymax: float | None,
) -> None:
    """Draw one continuous event diagnostic curve with its threshold."""

    axis.plot(frames, values, color=color, linewidth=1.1)
    axis.axhline(
        threshold,
        color="#8c510a",
        linestyle="--",
        linewidth=1.0,
        label=f"threshold={threshold:g}",
    )
    if ymax is not None:
        axis.set_ylim(0, ymax)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")


def optional_float_array(values: list[float | None]) -> np.ndarray:
    """Convert optional numeric fields into a plottable float array."""

    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )
