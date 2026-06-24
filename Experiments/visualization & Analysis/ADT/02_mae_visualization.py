#!/usr/bin/env python
"""Visualize per-frame SparseGaze angular error for one ADT sequence.

This script is the quantitative companion to ``01_gaze_image_scanpath.py``.
It loads one SparseGaze per-sequence NPZ, attaches optional GT scene-event
labels, and writes frame-level error timelines plus CSV summaries.

python "Experiments/visualization & Analysis/ADT/02_mae_visualization.py" \
  --start-frame 0 --end-frame 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

def find_repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "src").exists() and (parent / "Experiments").exists():
            return parent
    return Path(__file__).resolve().parents[3]


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.prediction_results import load_prediction_frame, summarize_error_columns  # noqa: E402

SEQUENCE = "Apartment_release_decoration_skeleton_seq133_M1292"
DEFAULT_REPORTS_DIR = Path(
    os.environ.get("REPORTS_DIR", "/mnt/d/SparseGaze/ADT-Gaze-structured")
)
DEFAULT_PREDICTION_NPZ = Path(
    "/home/liumu/Github_Projects/SparseGaze/outputs/eval/adt/sparsegaze/test/rollout/"
    f"sequence_predictions/{SEQUENCE}/hz6_phase0.npz"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "figures" / "mae_visualization"
EVENT_COLORS = {
    "fixation": "#2ca25f",
    "transition": "#fdae61",
    "invalid": "#d73027",
    "unknown": "#bdbdbd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-npz", type=Path, default=DEFAULT_PREDICTION_NPZ)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--include-anchors",
        action="store_true",
        help="Plot anchor frames too. By default the main curves use eval_mask frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_prediction_frame(args.prediction_npz, reports_dir=args.reports_dir)
    window = select_window(frame, args.start_frame, args.end_frame)
    plotted = window.copy() if args.include_anchors else window[window["eval_mask"]].copy()
    if plotted.empty:
        raise ValueError("No frames selected for plotting; check window/eval_mask.")

    sequence = str(frame["sequence"].iloc[0])
    run_name = args.run_name or default_run_name(
        args.prediction_npz,
        args.start_frame,
        args.end_frame,
    )
    output_dir = args.output_root / sequence / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    window.to_csv(output_dir / "per_frame_errors_all.csv", index=False)
    plotted.to_csv(output_dir / "per_frame_errors_plotted.csv", index=False)
    top_errors = (
        plotted.sort_values("angular_error_deg", ascending=False)
        .head(max(args.top_k, 0))
        .reset_index(drop=True)
    )
    top_errors.to_csv(output_dir / "top_error_frames.csv", index=False)

    summary = build_summary(
        frame=frame,
        window=window,
        plotted=plotted,
        prediction_npz=args.prediction_npz,
        reports_dir=args.reports_dir,
        include_anchors=args.include_anchors,
    )
    write_json(output_dir / "summary.json", summary)

    fig = make_error_timeline_figure(window=window, plotted=plotted, summary=summary)
    fig.savefig(output_dir / "per_frame_error_timeline.png", dpi=180)
    plt.close(fig)

    print_summary(summary, output_dir)


def select_window(frame: pd.DataFrame, start_frame: int, end_frame: int | None) -> pd.DataFrame:
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame is not None and end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")
    mask = frame["frame_index"] >= start_frame
    if end_frame is not None:
        mask &= frame["frame_index"] < end_frame
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise ValueError("No frames selected; check --start-frame/--end-frame.")
    return selected


def build_summary(
    *,
    frame: pd.DataFrame,
    window: pd.DataFrame,
    plotted: pd.DataFrame,
    prediction_npz: Path,
    reports_dir: Path,
    include_anchors: bool,
) -> dict[str, Any]:
    evaluated = window[window["eval_mask"]].copy()
    missing = evaluated[~evaluated["anchor_mask"]].copy()
    anchors = window[window["anchor_mask"]].copy()
    return {
        "prediction_npz": str(prediction_npz),
        "reports_dir": str(reports_dir),
        "sequence": str(frame["sequence"].iloc[0]),
        "target_hz": int(frame["target_hz"].iloc[0]),
        "phase": int(frame["phase"].iloc[0]),
        "start_frame": int(window["frame_index"].min()),
        "end_frame_exclusive": int(window["frame_index"].max()) + 1,
        "window_frames": int(len(window)),
        "plotted_frames": int(len(plotted)),
        "include_anchors": bool(include_anchors),
        "eval_frames": int(len(evaluated)),
        "missing_eval_frames": int(len(missing)),
        "anchor_frames": int(len(anchors)),
        "window": summarize_error_columns(window),
        "eval": summarize_error_columns(evaluated),
        "missing_eval": summarize_error_columns(missing),
        "anchor": summarize_error_columns(anchors),
    }


def make_error_timeline_figure(
    *,
    window: pd.DataFrame,
    plotted: pd.DataFrame,
    summary: dict[str, Any],
) -> plt.Figure:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [0.35, 2.0, 1.4]},
    )
    event_ax, angular_ax, residual_ax = axes
    event_handles = draw_event_ribbon(event_ax, window)
    draw_angular_error_axis(angular_ax, window, plotted)
    draw_yaw_pitch_axis(residual_ax, window, plotted)
    residual_ax.set_xlabel("Frame index")
    sequence = summary["sequence"]
    hz = summary["target_hz"]
    angular_ax.set_title(
        f"{sequence} | {hz} Hz rollout | per-frame angular error "
        f"(eval MAE={summary['eval']['mean_angular_error_deg']:.3f} deg)"
    )
    if event_handles:
        fig.legend(
            handles=event_handles,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            frameon=False,
            title="GT event",
        )
        fig.tight_layout(rect=(0, 0, 0.92, 1))
    else:
        fig.tight_layout()
    return fig


def draw_event_ribbon(axis: plt.Axes, window: pd.DataFrame) -> list[Patch]:
    if "scene_event_label" not in window:
        axis.set_axis_off()
        return []
    runs = label_runs(window[["frame_index", "scene_event_label"]])
    for label, start, end in runs:
        axis.axvspan(start, end, color=EVENT_COLORS.get(label, "#bdbdbd"), alpha=0.32)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_ylabel("Event")
    axis.grid(axis="x", alpha=0.15)
    labels = [label for label in EVENT_COLORS if label in set(window["scene_event_label"])]
    return [Patch(facecolor=EVENT_COLORS[label], alpha=0.32, label=label) for label in labels]


def draw_angular_error_axis(
    axis: plt.Axes,
    window: pd.DataFrame,
    plotted: pd.DataFrame,
) -> None:
    draw_event_background(axis, window)
    draw_broken_line(
        axis,
        plotted.sort_values("frame_index"),
        value_col="angular_error_deg",
        label="Angular error",
        color="#1f77b4",
    )
    anchors = window[window["anchor_mask"]]
    if not anchors.empty:
        axis.scatter(
            anchors["frame_index"],
            np.zeros(len(anchors)),
            marker="|",
            s=80,
            color="#444444",
            label="Sparse anchors",
            alpha=0.65,
        )
    axis.set_ylabel("Angular error [deg]")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", frameon=False, fontsize=8)


def draw_yaw_pitch_axis(axis: plt.Axes, window: pd.DataFrame, plotted: pd.DataFrame) -> None:
    draw_event_background(axis, window)
    ordered = plotted.sort_values("frame_index")
    draw_broken_line(axis, ordered, value_col="yaw_error_deg", label="Yaw error", color="#ff7f0e")
    draw_broken_line(
        axis,
        ordered,
        value_col="pitch_error_deg",
        label="Pitch error",
        color="#2ca02c",
    )
    axis.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.6)
    axis.set_ylabel("Signed error [deg]")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", frameon=False, fontsize=8)


def draw_event_background(axis: plt.Axes, window: pd.DataFrame) -> None:
    if "scene_event_label" not in window:
        return
    for label, start, end in label_runs(window[["frame_index", "scene_event_label"]]):
        axis.axvspan(start, end, color=EVENT_COLORS.get(label, "#bdbdbd"), alpha=0.10)


def draw_broken_line(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    value_col: str,
    label: str,
    color: str,
) -> None:
    if frame.empty:
        return
    frames = frame["frame_index"].to_numpy(dtype=int)
    values = frame[value_col].to_numpy(dtype=float)
    breaks = np.where(np.diff(frames) > 1)[0] + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(frames)]
    for index, (start, end) in enumerate(zip(starts, ends)):
        axis.plot(
            frames[start:end],
            values[start:end],
            marker=".",
            markersize=3,
            linewidth=1.0,
            color=color,
            label=label if index == 0 else None,
        )


def label_runs(labels: pd.DataFrame) -> list[tuple[str, int, int]]:
    ordered = labels.sort_values("frame_index")
    if ordered.empty:
        return []
    records = ordered[["frame_index", "scene_event_label"]].to_records(index=False)
    runs: list[tuple[str, int, int]] = []
    run_start = int(records[0].frame_index)
    run_label = str(records[0].scene_event_label)
    previous = int(records[0].frame_index)
    for row in records[1:]:
        frame = int(row.frame_index)
        label = str(row.scene_event_label)
        if label != run_label or frame != previous + 1:
            runs.append((run_label, run_start, previous + 1))
            run_start = frame
            run_label = label
        previous = frame
    runs.append((run_label, run_start, previous + 1))
    return runs


def default_run_name(prediction_npz: Path, start_frame: int, end_frame: int | None) -> str:
    hz_phase = prediction_npz.stem
    model = prediction_npz.parents[3].name if len(prediction_npz.parents) >= 4 else "prediction"
    end_label = "end" if end_frame is None else str(end_frame)
    return f"{model}_{hz_phase}_frames_{start_frame}_{end_label}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    print(f"sequence: {summary['sequence']}")
    print(f"prediction: {summary['prediction_npz']}")
    print(f"frames: {summary['start_frame']}..{summary['end_frame_exclusive']}")
    print(f"eval_frames: {summary['eval_frames']}")
    print(f"eval_mae_deg: {summary['eval']['mean_angular_error_deg']:.4f}")
    print(f"missing_eval_mae_deg: {summary['missing_eval']['mean_angular_error_deg']:.4f}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
