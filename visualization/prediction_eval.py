"""Prediction evaluation helpers for SparseGaze-style gaze outputs.

The functions here intentionally separate data loading from plotting so the
same analysis can be reused in notebooks and later report-generation scripts.
Per-frame visualizations require ``sequence_predictions/*.npz`` files.  When an
eval directory only contains aggregate CSV/JSON summaries, only aggregate
tables can be shown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adt_sandbox.results import find_sequence_file


@dataclass(frozen=True)
class PredictionRun:
    """One per-sequence prediction file discovered under an eval directory."""

    model: str
    eval_kind: str
    sequence: str
    target_hz: int
    phase: int
    path: Path


def discover_prediction_runs(eval_root: Path) -> pd.DataFrame:
    """Discover ``sequence_predictions/<sequence>/hz*_phase*.npz`` files."""

    rows: list[dict[str, Any]] = []
    for path in sorted(eval_root.rglob("sequence_predictions/*/hz*_phase*.npz")):
        try:
            run = parse_prediction_path(eval_root, path)
        except ValueError:
            continue
        rows.append(run.__dict__)
    return pd.DataFrame(rows)


def parse_prediction_path(eval_root: Path, path: Path) -> PredictionRun:
    """Parse model/eval metadata from a discovered prediction NPZ path."""

    relative = path.relative_to(eval_root)
    parts = relative.parts
    if len(parts) < 5 or parts[-3] != "sequence_predictions":
        raise ValueError(f"Not a sequence prediction path: {path}")
    model = eval_root.name
    eval_kind = parts[1]
    sequence = parts[-2]
    stem = path.stem
    hz_part, phase_part = stem.split("_phase", maxsplit=1)
    target_hz = int(hz_part.removeprefix("hz"))
    phase = int(phase_part)
    return PredictionRun(
        model=model,
        eval_kind=eval_kind,
        sequence=sequence,
        target_hz=target_hz,
        phase=phase,
        path=path,
    )


def discover_aggregate_tables(eval_root: Path) -> pd.DataFrame:
    """Load aggregate phasewise/phase-averaged CSVs from one eval directory."""

    rows: list[pd.DataFrame] = []
    for csv_path in sorted(eval_root.rglob("*_missing_phase*.csv")):
        frame = pd.read_csv(csv_path)
        relative = csv_path.relative_to(eval_root)
        parts = relative.parts
        frame.insert(0, "model", eval_root.name)
        frame.insert(1, "eval_kind", parts[1] if len(parts) > 1 else "")
        frame.insert(2, "source_file", str(csv_path))
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_prediction_frame(npz_path: Path, reports_dir: Path | None = None) -> pd.DataFrame:
    """Load one per-frame NPZ and compute angular/yaw/pitch errors."""

    with np.load(npz_path, allow_pickle=True) as data:
        pred_xyz = normalize_vectors(np.asarray(data["pred_xyz"], dtype=np.float64))
        gt_xyz = normalize_vectors(np.asarray(data["gt_xyz"], dtype=np.float64))
        timestamps_ns = np.asarray(data["timestamps_ns"], dtype=np.int64)
        eval_mask = np.asarray(data["eval_mask"], dtype=bool)
        anchor_mask = np.asarray(data["anchor_mask"], dtype=bool)
        extra = json.loads(str(data["extra_json"].item())) if "extra_json" in data.files else {}
        sequence = str(data["sequence_id"].item())
        target_hz = int(data["target_hz"].item())
        phase = int(data["phase"].item())

    pred_yaw_deg, pred_pitch_deg = direction_to_yaw_pitch_deg(pred_xyz)
    gt_yaw_deg, gt_pitch_deg = direction_to_yaw_pitch_deg(gt_xyz)
    yaw_error_deg = wrap_angle_deg(pred_yaw_deg - gt_yaw_deg)
    pitch_error_deg = pred_pitch_deg - gt_pitch_deg
    angular_error = angular_error_deg(pred_xyz, gt_xyz)

    frame = pd.DataFrame(
        {
            "frame_index": np.arange(len(pred_xyz), dtype=np.int64),
            "timestamp_ns": timestamps_ns,
            "eval_mask": eval_mask,
            "anchor_mask": anchor_mask,
            "pred_x": pred_xyz[:, 0],
            "pred_y": pred_xyz[:, 1],
            "pred_z": pred_xyz[:, 2],
            "gt_x": gt_xyz[:, 0],
            "gt_y": gt_xyz[:, 1],
            "gt_z": gt_xyz[:, 2],
            "pred_yaw_deg": pred_yaw_deg,
            "pred_pitch_deg": pred_pitch_deg,
            "gt_yaw_deg": gt_yaw_deg,
            "gt_pitch_deg": gt_pitch_deg,
            "yaw_error_deg": yaw_error_deg,
            "pitch_error_deg": pitch_error_deg,
            "angular_error_deg": angular_error,
            "sequence": sequence,
            "target_hz": target_hz,
            "phase": phase,
            "eval_kind": extra.get("eval_kind", ""),
            "correct_mode": extra.get("correct_mode", ""),
            "anchor_refresh_mode": extra.get("anchor_refresh_mode", ""),
            "feedback_writeback_mode": extra.get("feedback_writeback_mode", ""),
        }
    )

    if reports_dir is not None:
        frame = attach_scene_event_labels(frame, reports_dir, sequence)
    return frame


def attach_scene_event_labels(frame: pd.DataFrame, reports_dir: Path, sequence: str) -> pd.DataFrame:
    """Attach scene-direction event labels by frame index when available."""

    try:
        label_path = find_sequence_file(
            reports_dir,
            sequence,
            "events",
            "scene_gaze_frame_labels.csv",
        )
    except FileNotFoundError:
        frame = frame.copy()
        frame["scene_event_label"] = "unknown"
        return frame

    labels = pd.read_csv(label_path)
    if "frame_index" not in labels or "scene_event_label" not in labels:
        frame = frame.copy()
        frame["scene_event_label"] = "unknown"
        return frame
    return frame.merge(
        labels[["frame_index", "scene_event_label"]],
        on="frame_index",
        how="left",
    ).assign(scene_event_label=lambda df: df["scene_event_label"].fillna("unknown"))


def summarize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return compact mean/median metrics for all evaluated frames."""

    evaluated = frame[frame["eval_mask"]].copy()
    if evaluated.empty:
        evaluated = frame.copy()
    rows = [
        summarize_group(evaluated, "all"),
        summarize_group(evaluated[evaluated["anchor_mask"]], "anchor"),
        summarize_group(evaluated[~evaluated["anchor_mask"]], "missing"),
    ]
    if "scene_event_label" in evaluated:
        for label, group in evaluated.groupby("scene_event_label"):
            rows.append(summarize_group(group, f"event:{label}"))
    return pd.DataFrame(rows)


def summarize_group(frame: pd.DataFrame, group_name: str) -> dict[str, Any]:
    """Summarize one frame subset using only high-signal statistics."""

    if frame.empty:
        return {
            "group": group_name,
            "n": 0,
            "mean_angular_error_deg": np.nan,
            "median_angular_error_deg": np.nan,
            "mean_abs_yaw_error_deg": np.nan,
            "median_abs_yaw_error_deg": np.nan,
            "mean_abs_pitch_error_deg": np.nan,
            "median_abs_pitch_error_deg": np.nan,
        }
    return {
        "group": group_name,
        "n": len(frame),
        "mean_angular_error_deg": frame["angular_error_deg"].mean(),
        "median_angular_error_deg": frame["angular_error_deg"].median(),
        "mean_abs_yaw_error_deg": frame["yaw_error_deg"].abs().mean(),
        "median_abs_yaw_error_deg": frame["yaw_error_deg"].abs().median(),
        "mean_abs_pitch_error_deg": frame["pitch_error_deg"].abs().mean(),
        "median_abs_pitch_error_deg": frame["pitch_error_deg"].abs().median(),
    }


def plot_error_distribution(frame: pd.DataFrame, ax: Any, *, evaluated_only: bool = True) -> None:
    """Plot whole-sequence angular error distribution."""

    data = frame[frame["eval_mask"]] if evaluated_only else frame
    values = data["angular_error_deg"].dropna().to_numpy()
    ax.hist(values, bins=60, color="#4C78A8", alpha=0.82)
    ax.axvline(np.mean(values), color="#C44E52", linewidth=1.6, label="mean")
    ax.axvline(np.median(values), color="#2F4B7C", linestyle="--", linewidth=1.6, label="median")
    ax.set_xlabel("Angular error [deg]")
    ax.set_ylabel("Frame count")
    ax.set_title("Error distribution")
    ax.grid(alpha=0.25)
    ax.legend()


def plot_yaw_pitch_error(frame: pd.DataFrame, ax: Any, *, evaluated_only: bool = True) -> None:
    """Plot 2D yaw/pitch residuals for direction and bias inspection."""

    data = frame[frame["eval_mask"]] if evaluated_only else frame
    colors = np.where(data["anchor_mask"], "#4C78A8", "#F58518")
    ax.scatter(
        data["yaw_error_deg"],
        data["pitch_error_deg"],
        c=colors,
        s=10,
        alpha=0.45,
        linewidths=0,
    )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Yaw error [deg]")
    ax.set_ylabel("Pitch error [deg]")
    ax.set_title("Yaw / pitch error cloud")
    ax.grid(alpha=0.25)


def plot_error_timeseries(frame: pd.DataFrame, ax: Any, start: int, end: int) -> None:
    """Plot angular error over a selected frame window with anchor markers."""

    window = frame[(frame["frame_index"] >= start) & (frame["frame_index"] < end)]
    ax.plot(window["frame_index"], window["angular_error_deg"], color="#4C78A8", linewidth=1.1)
    anchors = window[window["anchor_mask"]]
    ax.scatter(
        anchors["frame_index"],
        anchors["angular_error_deg"],
        color="#54A24B",
        s=22,
        label="anchor",
        zorder=3,
    )
    paint_event_background(ax, window)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angular error [deg]")
    ax.set_title("Window error over time")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")


def plot_yaw_pitch_trace(frame: pd.DataFrame, axes: tuple[Any, Any], start: int, end: int) -> None:
    """Plot GT and predicted yaw/pitch traces over a selected window."""

    window = frame[(frame["frame_index"] >= start) & (frame["frame_index"] < end)]
    yaw_ax, pitch_ax = axes
    yaw_ax.plot(window["frame_index"], window["gt_yaw_deg"], color="#2F4B7C", label="GT yaw")
    yaw_ax.plot(window["frame_index"], window["pred_yaw_deg"], color="#F58518", label="Pred yaw", alpha=0.9)
    yaw_ax.set_ylabel("Yaw [deg]")
    yaw_ax.set_title("Yaw trace")
    yaw_ax.grid(alpha=0.25)
    yaw_ax.legend(loc="upper right")

    pitch_ax.plot(window["frame_index"], window["gt_pitch_deg"], color="#2F4B7C", label="GT pitch")
    pitch_ax.plot(window["frame_index"], window["pred_pitch_deg"], color="#F58518", label="Pred pitch", alpha=0.9)
    pitch_ax.set_xlabel("Frame")
    pitch_ax.set_ylabel("Pitch [deg]")
    pitch_ax.set_title("Pitch trace")
    pitch_ax.grid(alpha=0.25)
    pitch_ax.legend(loc="upper right")


def plot_cpf_scanpath(frame: pd.DataFrame, ax: Any, start: int, end: int) -> None:
    """Plot GT and predicted yaw/pitch scanpaths for a selected window."""

    window = frame[(frame["frame_index"] >= start) & (frame["frame_index"] < end)]
    ax.plot(window["gt_yaw_deg"], window["gt_pitch_deg"], color="#2F4B7C", label="GT", linewidth=1.4)
    ax.plot(window["pred_yaw_deg"], window["pred_pitch_deg"], color="#F58518", label="Pred", linewidth=1.2)
    ax.scatter(window["gt_yaw_deg"].iloc[:1], window["gt_pitch_deg"].iloc[:1], color="#54A24B", s=50, label="start")
    ax.scatter(window["gt_yaw_deg"].iloc[-1:], window["gt_pitch_deg"].iloc[-1:], color="#C44E52", marker="x", s=60, label="end")
    ax.set_xlabel("Yaw [deg]")
    ax.set_ylabel("Pitch [deg]")
    ax.set_title("CPF yaw/pitch scanpath")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")


def plot_event_error_box(frame: pd.DataFrame, ax: Any) -> None:
    """Compare evaluated-frame angular error by scene event label."""

    if "scene_event_label" not in frame:
        ax.text(0.5, 0.5, "No scene event labels attached", ha="center", va="center")
        ax.set_axis_off()
        return
    data = frame[frame["eval_mask"]]
    labels = [label for label in ["fixation", "transition", "invalid", "unknown"] if label in set(data["scene_event_label"])]
    values = [data.loc[data["scene_event_label"] == label, "angular_error_deg"].dropna() for label in labels]
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_ylabel("Angular error [deg]")
    ax.set_title("Error by scene event")
    ax.grid(axis="y", alpha=0.25)


def plot_anchor_error_box(frame: pd.DataFrame, ax: Any) -> None:
    """Compare evaluated-frame error at anchors and missing frames."""

    data = frame[frame["eval_mask"]]
    labels = ["anchor", "missing"]
    values = [
        data.loc[data["anchor_mask"], "angular_error_deg"].dropna(),
        data.loc[~data["anchor_mask"], "angular_error_deg"].dropna(),
    ]
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_ylabel("Angular error [deg]")
    ax.set_title("Error by anchor state")
    ax.grid(axis="y", alpha=0.25)


def plot_aggregate_mae_by_hz(aggregate: pd.DataFrame, ax: Any) -> None:
    """Plot aggregate MAE by target rate and eval/correction mode."""

    if aggregate.empty or "mae_missing_deg" not in aggregate:
        ax.text(0.5, 0.5, "No aggregate MAE table", ha="center", va="center")
        ax.set_axis_off()
        return
    data = aggregate.copy()
    data["mode"] = data["eval_kind"].astype(str)
    if "correct_mode" in data:
        data["mode"] = np.where(
            data["correct_mode"].notna(),
            data["mode"] + ":" + data["correct_mode"].astype(str),
            data["mode"],
        )
    for mode, group in data.groupby("mode"):
        summary = group.groupby("target_hz", as_index=False)["mae_missing_deg"].mean()
        ax.plot(summary["target_hz"], summary["mae_missing_deg"], marker="o", label=mode)
    ax.set_xlabel("Target gaze rate [Hz]")
    ax.set_ylabel("Missing-frame MAE [deg]")
    ax.set_title("Aggregate MAE by target rate")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def paint_event_background(ax: Any, window: pd.DataFrame) -> None:
    """Shade fixation/transition intervals behind a time-series plot."""

    if "scene_event_label" not in window or window.empty:
        return
    colors = {
        "fixation": "#54A24B",
        "transition": "#F58518",
        "invalid": "#C44E52",
        "unknown": "#BDBDBD",
    }
    labels = window[["frame_index", "scene_event_label"]].to_records(index=False)
    run_label = labels[0].scene_event_label
    run_start = int(labels[0].frame_index)
    prev = int(labels[0].frame_index)
    for row in labels[1:]:
        frame_index = int(row.frame_index)
        label = row.scene_event_label
        if label != run_label or frame_index != prev + 1:
            ax.axvspan(run_start, prev + 1, color=colors.get(run_label, "#BDBDBD"), alpha=0.08)
            run_label = label
            run_start = frame_index
        prev = frame_index
    ax.axvspan(run_start, prev + 1, color=colors.get(run_label, "#BDBDBD"), alpha=0.08)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize direction vectors row-wise, preserving zero rows as NaN."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def angular_error_deg(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """Compute angular distance between predicted and GT unit directions."""

    dot = np.sum(normalize_vectors(pred_xyz) * normalize_vectors(gt_xyz), axis=1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def direction_to_yaw_pitch_deg(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert CPF direction vectors to yaw/pitch degrees.

    SparseGaze ADT vectors use the negative z direction as forward in the
    observed prediction files, so yaw/pitch are computed against ``-z``.
    Angular error uses the full vector dot product and does not depend on this
    plotting convention.
    """

    x = xyz[:, 0]
    y = xyz[:, 1]
    forward = -xyz[:, 2]
    yaw = np.rad2deg(np.arctan2(x, forward))
    pitch = np.rad2deg(np.arctan2(y, forward))
    return yaw, pitch


def wrap_angle_deg(values: np.ndarray) -> np.ndarray:
    """Wrap angular residuals into [-180, 180) degrees."""

    return (values + 180.0) % 360.0 - 180.0
