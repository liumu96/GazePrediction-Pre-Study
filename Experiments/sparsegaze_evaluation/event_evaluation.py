#!/usr/bin/env python
"""SparseGaze missing-frame event-conditioned evaluation helpers.

This module is intentionally local to ``Experiments/sparsegaze_evaluation`` so
the experiment can evolve without expanding the general sandbox API surface.
It evaluates SparseGaze per-sequence NPZ outputs on ``eval_mask`` frames and
groups errors by GT scene-gaze event labels.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.results import find_sequence_file  # noqa: E402
from visualization.adt_hagi_sparsegaze_compare import (  # noqa: E402
    DEFAULT_ADT_DATA,
    DEFAULT_HAGI_DIR,
    SPARSEGAZE_MODES,
    angular_error_deg,
    load_adt_data,
    load_hagi_primary,
    load_sparsegaze_sequence,
    pitch_yaw_to_cpf,
    sequence_names,
    sparsegaze_world_to_cpf,
)

DEFAULT_REPORTS_DIR = (
    Path("/mnt/d/SparseGaze/ADT-Gaze-structured")
    if Path("/mnt/d/SparseGaze/ADT-Gaze-structured").exists()
    else REPO_ROOT / "outputs" / "reports"
)
DEFAULT_SPARSEGAZE_DIR = Path(
    os.environ.get(
        "SPARSEGAZE_ADT_EVAL_DIR",
        "/home/liumu/Github_Projects/SparseGaze/outputs/eval/adt/"
        "sparsegaze_cpf_forward_head_motion_residual_ss",
    )
)
EVENT_COLORS = {
    "fixation": "#2ca25f",
    "transition": "#fdae61",
    "invalid": "#d73027",
    "unknown": "#bdbdbd",
}
EVENT_RIBBON_ALPHA = 0.32
ERROR_EVENT_BACKGROUND_ALPHA = 0.12


def load_event_labels(reports_dir: Path, sequence: str) -> pd.DataFrame:
    """Load GT scene-gaze event labels for one sequence."""

    path = find_sequence_file(
        reports_dir,
        sequence,
        "events",
        "scene_gaze_frame_labels.csv",
    )
    labels = pd.read_csv(path)
    required = {"frame_index", "scene_event_label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    labels = labels[["frame_index", "scene_event_label"]].copy()
    labels["frame_index"] = labels["frame_index"].astype(int)
    labels["scene_event_label"] = labels["scene_event_label"].astype(str)
    return labels.sort_values("frame_index")


def resolve_window(
    labels: pd.DataFrame,
    start_frame: int,
    end_frame: int | None,
    max_frames: int,
) -> tuple[int, int]:
    """Resolve a frame window against available GT event labels."""

    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    max_label_frame = int(labels["frame_index"].max())
    if end_frame is None:
        if max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        end_frame = max_label_frame + 1 if max_frames == 0 else start_frame + max_frames
    if end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")
    return start_frame, min(end_frame, max_label_frame + 1)


def load_sparsegaze_event_errors(
    *,
    sequence: str,
    fps: int,
    modes: Iterable[str],
    adt_data: dict,
    labels: pd.DataFrame,
    sparsegaze_dir: Path,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    """Load per-frame missing-frame angular errors and attach GT event labels."""

    record = adt_data[sequence]
    gt_cpf = pitch_yaw_to_cpf(record["pitch_yaw"])
    label_window = labels[
        (labels["frame_index"] >= start_frame) & (labels["frame_index"] < end_frame)
    ]
    label_by_frame = dict(
        zip(
            label_window["frame_index"].astype(int),
            label_window["scene_event_label"].astype(str),
        )
    )

    rows: list[pd.DataFrame] = []
    for mode in modes:
        sparse = load_sparsegaze_sequence(sequence, fps, mode, sparsegaze_dir=sparsegaze_dir)
        pred_cpf = sparsegaze_world_to_cpf(
            sparse["pred_xyz"],
            record=record,
            timestamps_ns=sparse["timestamps_ns"],
        )
        eval_mask = np.asarray(sparse["eval_mask"], dtype=bool)
        max_frame = min(len(pred_cpf), len(gt_cpf), len(eval_mask), end_frame)
        frames = np.arange(start_frame, max_frame, dtype=int)
        frames = frames[eval_mask[frames]]
        if len(frames) == 0:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "frame_index": frames,
                    "method": SPARSEGAZE_MODES.get(mode, mode),
                    "mode": mode,
                    "angular_error_deg": angular_error_deg(pred_cpf[frames], gt_cpf[frames]),
                    "scene_event_label": [
                        label_by_frame.get(int(frame), "unknown") for frame in frames
                    ],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_hagi_event_errors(
    *,
    sequence: str,
    fps: int,
    labels: pd.DataFrame,
    hagi_dir: Path = DEFAULT_HAGI_DIR,
    start_frame: int,
    end_frame: int,
    nsample: int = 20,
) -> pd.DataFrame:
    """Load HAGI++ sliding errors and attach GT event labels."""

    hagi = load_hagi_primary(fps, hagi_dir=hagi_dir, nsample=nsample)
    sequence_names_arr = hagi["sequence_name"].astype(str)
    mask = sequence_names_arr == sequence
    frames = hagi["frame_index"][mask].astype(int)
    errors = hagi["angular_error_deg"][mask].astype(float)
    window_mask = (frames >= start_frame) & (frames < end_frame)
    frames = frames[window_mask]
    errors = errors[window_mask]
    if len(frames) == 0:
        return pd.DataFrame()
    label_window = labels[
        (labels["frame_index"] >= start_frame) & (labels["frame_index"] < end_frame)
    ]
    label_by_frame = dict(
        zip(
            label_window["frame_index"].astype(int),
            label_window["scene_event_label"].astype(str),
        )
    )
    return pd.DataFrame(
        {
            "frame_index": frames,
            "method": "HAGI++",
            "mode": "hagi++",
            "angular_error_deg": errors,
            "scene_event_label": [label_by_frame.get(int(frame), "unknown") for frame in frames],
        }
    )


def load_method_event_errors(
    *,
    sequence: str,
    fps: int,
    modes: Iterable[str],
    adt_data: dict,
    labels: pd.DataFrame,
    sparsegaze_dir: Path,
    start_frame: int,
    end_frame: int,
    include_hagi: bool = True,
    hagi_dir: Path = DEFAULT_HAGI_DIR,
) -> pd.DataFrame:
    """Load HAGI++ and SparseGaze errors into one method table."""

    frames = []
    if include_hagi:
        hagi_frame = load_hagi_event_errors(
            sequence=sequence,
            fps=fps,
            labels=labels,
            hagi_dir=hagi_dir,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        if not hagi_frame.empty:
            frames.append(hagi_frame)
    sparse_frame = load_sparsegaze_event_errors(
        sequence=sequence,
        fps=fps,
        modes=modes,
        adt_data=adt_data,
        labels=labels,
        sparsegaze_dir=sparsegaze_dir,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    if not sparse_frame.empty:
        frames.append(sparse_frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_by_event(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize errors by method and GT scene-gaze event label."""

    if predictions.empty:
        return pd.DataFrame()
    return (
        predictions.groupby(["method", "mode", "scene_event_label"], as_index=False)
        .agg(
            n=("angular_error_deg", "size"),
            mae_deg=("angular_error_deg", "mean"),
            median_deg=("angular_error_deg", "median"),
            p90_deg=("angular_error_deg", lambda values: float(np.percentile(values, 90))),
        )
        .sort_values(["scene_event_label", "method"])
    )


def summarize_method_event_mae(predictions: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Return one method-level table with overall, fixation, and transition errors."""

    columns = [
        "method",
        "mode",
        "overall_n",
        "overall_mae_deg",
        "overall_median_deg",
        "overall_p90_deg",
        "fixation_n",
        "fixation_mae_deg",
        "fixation_median_deg",
        "fixation_p90_deg",
        "transition_n",
        "transition_mae_deg",
        "transition_median_deg",
        "transition_p90_deg",
        "transition_minus_fixation_mae_deg",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    overall = (
        predictions.groupby(["method", "mode"], as_index=False)
        .agg(
            overall_n=("angular_error_deg", "size"),
            overall_mae_deg=("angular_error_deg", "mean"),
            overall_median_deg=("angular_error_deg", "median"),
            overall_p90_deg=("angular_error_deg", lambda values: float(np.percentile(values, 90))),
        )
        .set_index(["method", "mode"])
    )
    focus = summary[summary["scene_event_label"].isin(["fixation", "transition"])]
    rows = []
    for (method, mode), overall_row in overall.iterrows():
        row = {
            "method": method,
            "mode": mode,
            "overall_n": int(overall_row["overall_n"]),
            "overall_mae_deg": overall_row["overall_mae_deg"],
            "overall_median_deg": overall_row["overall_median_deg"],
            "overall_p90_deg": overall_row["overall_p90_deg"],
        }
        for label in ("fixation", "transition"):
            event_row = focus[
                (focus["method"] == method)
                & (focus["mode"] == mode)
                & (focus["scene_event_label"] == label)
            ]
            if event_row.empty:
                row[f"{label}_n"] = 0
                row[f"{label}_mae_deg"] = np.nan
                row[f"{label}_median_deg"] = np.nan
                row[f"{label}_p90_deg"] = np.nan
            else:
                item = event_row.iloc[0]
                row[f"{label}_n"] = int(item["n"])
                row[f"{label}_mae_deg"] = item["mae_deg"]
                row[f"{label}_median_deg"] = item["median_deg"]
                row[f"{label}_p90_deg"] = item["p90_deg"]
        row["transition_minus_fixation_mae_deg"] = (
            row["transition_mae_deg"] - row["fixation_mae_deg"]
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values("overall_mae_deg")


def make_event_comparison_figure(
    *,
    sequence: str,
    fps: int,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    error_ymax: float | None = None,
) -> plt.Figure:
    """Create the single-sequence event timeline comparison figure."""

    label_window = labels[
        (labels["frame_index"] >= start_frame) & (labels["frame_index"] < end_frame)
    ]
    runs = label_runs(label_window)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=False,
        gridspec_kw={"height_ratios": [0.38, 2.2, 1.4]},
    )
    event_ax, error_ax, bar_ax = axes
    method_colors = make_method_color_map(predictions)
    event_handles = draw_event_track(event_ax, runs)
    draw_error_timeseries(error_ax, predictions, runs, error_ymax, method_colors)
    draw_event_summary_bars(bar_ax, summary, method_colors)
    event_ax.set_title(f"{sequence} | {fps} Hz | missing-frame error by GT event")
    event_ax.set_xlim(start_frame, end_frame)
    error_ax.set_xlim(start_frame, end_frame)
    error_ax.set_xlabel("Frame index")
    fig.legend(
        handles=event_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=3,
        frameon=False,
        title="GT event",
    )
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    return fig


def label_runs(labels: pd.DataFrame) -> list[tuple[str, int, int]]:
    """Return contiguous event-label runs as ``(label, start, end)``."""

    if labels.empty:
        return []
    rows = labels.sort_values("frame_index")[["frame_index", "scene_event_label"]].to_records(index=False)
    runs: list[tuple[str, int, int]] = []
    run_start = int(rows[0].frame_index)
    run_label = str(rows[0].scene_event_label)
    prev = int(rows[0].frame_index)
    for row in rows[1:]:
        frame = int(row.frame_index)
        label = str(row.scene_event_label)
        if label != run_label or frame != prev + 1:
            runs.append((run_label, run_start, prev + 1))
            run_start = frame
            run_label = label
        prev = frame
    runs.append((run_label, run_start, prev + 1))
    return runs


def draw_event_track(axis: plt.Axes, runs: list[tuple[str, int, int]]) -> list[Patch]:
    """Draw a subtle GT event ribbon."""

    for label, start, end in runs:
        axis.axvspan(
            start,
            end,
            color=EVENT_COLORS.get(label, "#bdbdbd"),
            alpha=EVENT_RIBBON_ALPHA,
        )
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_ylabel("GT event")
    axis.grid(axis="x", alpha=0.12)
    return [
        Patch(facecolor=EVENT_COLORS[label], alpha=EVENT_RIBBON_ALPHA, label=label)
        for label in ("fixation", "transition", "invalid")
    ]


def draw_error_timeseries(
    axis: plt.Axes,
    predictions: pd.DataFrame,
    runs: list[tuple[str, int, int]],
    error_ymax: float | None,
    method_colors: dict[str, str],
) -> None:
    """Draw missing-frame error curves, broken at anchor gaps."""

    for label, start, end in runs:
        axis.axvspan(
            start,
            end,
            color=EVENT_COLORS.get(label, "#bdbdbd"),
            alpha=ERROR_EVENT_BACKGROUND_ALPHA,
        )
    for method, group in predictions.groupby("method"):
        draw_broken_error_line(
            axis,
            group.sort_values("frame_index"),
            method,
            method_colors.get(method),
        )
    axis.set_ylabel("Angular error [deg]")
    axis.set_ylim(bottom=0 if error_ymax is None else None, top=error_ymax)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=8, frameon=False)


def draw_broken_error_line(
    axis: plt.Axes,
    group: pd.DataFrame,
    label: str,
    color: str | None,
) -> None:
    """Plot missing-frame errors without connecting across anchor gaps."""

    frames = group["frame_index"].to_numpy(dtype=int)
    errors = group["angular_error_deg"].to_numpy(dtype=float)
    if len(frames) == 0:
        return
    breaks = np.where(np.diff(frames) > 1)[0] + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(frames)]
    for index, (start, end) in enumerate(zip(starts, ends)):
        axis.plot(
            frames[start:end],
            errors[start:end],
            marker=".",
            linewidth=1.0,
            markersize=3,
            label=label if index == 0 else None,
            color=color,
        )


def draw_event_summary_bars(
    axis: plt.Axes,
    summary: pd.DataFrame,
    method_colors: dict[str, str],
) -> None:
    """Draw a compact MAE bar chart by event label."""

    if summary.empty:
        axis.set_axis_off()
        return
    labels = [
        label
        for label in ("fixation", "transition", "invalid", "unknown")
        if label in set(summary["scene_event_label"])
    ]
    methods = list(dict.fromkeys(summary["method"]))
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(len(methods), 1)
    for index, method in enumerate(methods):
        values = []
        for label in labels:
            row = summary[
                (summary["method"] == method) & (summary["scene_event_label"] == label)
            ]
            values.append(float(row["mae_deg"].iloc[0]) if not row.empty else np.nan)
        offset = (index - (len(methods) - 1) / 2) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            label=method,
            color=method_colors.get(method),
        )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("MAE [deg]")
    axis.set_title("Mean angular error by event label")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right", fontsize=8, frameon=False)


def make_method_color_map(predictions: pd.DataFrame) -> dict[str, str]:
    """Assign one stable color per displayed method."""

    methods = list(dict.fromkeys(predictions["method"])) if not predictions.empty else []
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    return {method: cycle[index % len(cycle)] for index, method in enumerate(methods)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", nargs="?", default=None)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--mode", dest="modes", action="append", choices=sorted(SPARSEGAZE_MODES))
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--sparsegaze-dir", type=Path, default=DEFAULT_SPARSEGAZE_DIR)
    parser.add_argument("--hagi-dir", type=Path, default=DEFAULT_HAGI_DIR)
    parser.add_argument("--adt-data", type=Path, default=DEFAULT_ADT_DATA)
    parser.add_argument("--no-hagi", action="store_true", help="Do not include HAGI++ baseline.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--error-ymax", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adt_data = load_adt_data(args.adt_data)
    sequence = args.sequence or sequence_names(
        adt_data=adt_data,
        fps=args.fps,
        sparsegaze_dir=args.sparsegaze_dir,
    )[0]
    modes = tuple(args.modes or SPARSEGAZE_MODES.keys())
    labels = load_event_labels(args.reports_dir, sequence)
    start_frame, end_frame = resolve_window(
        labels,
        args.start_frame,
        args.end_frame,
        args.max_frames,
    )
    predictions = load_method_event_errors(
        sequence=sequence,
        fps=args.fps,
        modes=modes,
        adt_data=adt_data,
        labels=labels,
        sparsegaze_dir=args.sparsegaze_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        include_hagi=not args.no_hagi,
        hagi_dir=args.hagi_dir,
    )
    summary = summarize_by_event(predictions)
    method_summary = summarize_method_event_mae(predictions, summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sequence}_hz{args.fps}_events_{start_frame}_{end_frame}"
    method_summary.to_csv(args.output_dir / f"{stem}_method_summary.csv", index=False)
    fig = make_event_comparison_figure(
        sequence=sequence,
        fps=args.fps,
        labels=labels,
        predictions=predictions,
        summary=summary,
        start_frame=start_frame,
        end_frame=end_frame,
        error_ymax=args.error_ymax,
    )
    fig.savefig(args.output_dir / f"{stem}.png", dpi=180)
    plt.close(fig)
    print(method_summary.round(4).to_string(index=False))
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
