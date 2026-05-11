#!/usr/bin/env python
"""Visualize scene-direction gaze event labels for one sequence window.

Run `detect_scene_gaze_events.py` first. This script reads the generated
`*_scene_gaze_event_*.csv` files and writes a timeline figure with:

- final scene event labels (`fixation` / `transition` / `invalid`)
- Scene-frame angular velocity
- Scene-frame centered angular dispersion

Example:
    python scripts/visualize_scene_gaze_events.py \
      Apartment_release_decoration_skeleton_seq131_M1292 \
      --reports-dir /mnt/d/SparseGaze/ADT-Gaze \
      --start-frame 0 \
      --end-frame 600
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.scene_gaze_events import (  # noqa: E402
    SceneGazeEventFeatureRow,
    SceneGazeEventSegment,
    SceneGazeFrameLabel,
    default_scene_gaze_event_features_csv_path,
    default_scene_gaze_event_segments_csv_path,
    default_scene_gaze_frame_labels_csv_path,
    read_scene_gaze_event_features_csv,
    read_scene_gaze_event_segments_csv,
    read_scene_gaze_frame_labels_csv,
)


LABEL_COLORS = {
    "fixation": "#2ca25f",
    "transition": "#fdae61",
    "invalid": "#d73027",
}


@dataclass(frozen=True)
class LabelRun:
    label: str
    start_frame: int
    end_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", help="Sequence name used in scene event output filenames.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory containing scene event CSV files.",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=None,
        help="Optional explicit *_scene_gaze_event_features.csv path.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional explicit *_scene_gaze_frame_labels.csv path.",
    )
    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=None,
        help="Optional explicit *_scene_gaze_event_segments.csv path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "scene_gaze_events",
        help="Directory for the generated timeline figure.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Inclusive starting frame index.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive ending frame index. Default uses --max-frames from start.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Window length when --end-frame is omitted. Use 0 to plot to sequence end.",
    )
    parser.add_argument(
        "--velocity-threshold-deg-s",
        type=float,
        default=40.0,
        help="Draw the velocity threshold used for fixation labeling.",
    )
    parser.add_argument(
        "--dispersion-threshold-deg",
        type=float,
        default=2.5,
        help="Draw the dispersion threshold used for fixation labeling.",
    )
    parser.add_argument(
        "--velocity-ymax",
        type=float,
        default=None,
        help="Optional y-axis max for velocity.",
    )
    parser.add_argument(
        "--dispersion-ymax",
        type=float,
        default=None,
        help="Optional y-axis max for dispersion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = Path(args.sequence).name
    feature_csv = args.features_csv or default_scene_gaze_event_features_csv_path(
        sequence,
        output_dir=args.reports_dir,
    )
    label_csv = args.labels_csv or default_scene_gaze_frame_labels_csv_path(
        sequence,
        output_dir=args.reports_dir,
    )
    segment_csv = args.segments_csv or default_scene_gaze_event_segments_csv_path(
        sequence,
        output_dir=args.reports_dir,
    )

    require_file(feature_csv, "scene event features")
    require_file(label_csv, "scene event frame labels")
    require_file(segment_csv, "scene event segments")

    features = read_scene_gaze_event_features_csv(feature_csv)
    labels = read_scene_gaze_frame_labels_csv(label_csv)
    segments = read_scene_gaze_event_segments_csv(segment_csv)
    start_frame, end_frame = resolve_frame_window(
        features,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
    )
    selected_features = select_feature_window(features, start_frame, end_frame)
    if not selected_features:
        raise ValueError(
            f"No scene event feature rows selected for frames "
            f"{start_frame}..{end_frame}"
        )

    output_path = args.output_dir / (
        f"{sequence}_scene_gaze_events_{start_frame}_{end_frame}.png"
    )
    plot_scene_gaze_event_timeline(
        output_path,
        sequence_name=sequence,
        features=selected_features,
        labels=labels,
        segments=segments,
        start_frame=start_frame,
        end_frame=end_frame,
        velocity_threshold_deg_s=args.velocity_threshold_deg_s,
        dispersion_threshold_deg=args.dispersion_threshold_deg,
        velocity_ymax=args.velocity_ymax,
        dispersion_ymax=args.dispersion_ymax,
    )

    print(f"sequence: {sequence}")
    print(f"features_csv: {feature_csv}")
    print(f"labels_csv: {label_csv}")
    print(f"segments_csv: {segment_csv}")
    print(f"frames: {start_frame}..{end_frame}")
    print(f"figure: {output_path}")


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Run scripts/detect_scene_gaze_events.py first."
        )


def resolve_frame_window(
    features: list[SceneGazeEventFeatureRow],
    start_frame: int,
    end_frame: int | None,
    max_frames: int,
) -> tuple[int, int]:
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
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )


if __name__ == "__main__":
    main()
