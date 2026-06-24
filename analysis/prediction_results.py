"""Reusable SparseGaze/GT prediction-result analysis helpers.

This module is intentionally non-interactive. It loads per-sequence prediction
NPZ files, attaches optional ADT scene-event labels, and returns tables that can
be reused by notebooks, report scripts, or later model-comparison pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from adt_sandbox.results import find_sequence_file


@dataclass(frozen=True)
class PredictionFile:
    """One model/eval/frequency/phase NPZ file."""

    model: str
    split: str
    eval_kind: str
    sequence: str
    target_hz: int
    phase: int
    path: Path


def discover_prediction_files(
    eval_root: Path,
    *,
    target_hz: int | None = 6,
    models: Iterable[str] | None = None,
    eval_kinds: Iterable[str] | None = ("rollout",),
    split: str | None = "test",
) -> pd.DataFrame:
    """Discover per-sequence ``hz*_phase*.npz`` prediction files.

    ``eval_root`` can be either the dataset-level eval directory, e.g.
    ``.../outputs/eval/adt``, or one model directory under it. By default this
    returns only 6 Hz rollout files because that is the current SparseGaze
    analysis target; pass ``target_hz=None`` or ``eval_kinds=None`` to broaden.
    """

    model_filter = set(models) if models else None
    kind_filter = set(eval_kinds) if eval_kinds else None
    rows: list[dict[str, Any]] = []

    for path in sorted(eval_root.rglob("sequence_predictions/*/hz*_phase*.npz")):
        try:
            item = parse_prediction_file(eval_root, path)
        except ValueError:
            continue
        if target_hz is not None and item.target_hz != target_hz:
            continue
        if model_filter is not None and item.model not in model_filter:
            continue
        if kind_filter is not None and item.eval_kind not in kind_filter:
            continue
        if split is not None and item.split != split:
            continue
        rows.append(
            {
                "model": item.model,
                "split": item.split,
                "eval_kind": item.eval_kind,
                "sequence": item.sequence,
                "target_hz": item.target_hz,
                "phase": item.phase,
                "path": str(item.path),
            }
        )
    return pd.DataFrame(rows)


def parse_prediction_file(eval_root: Path, path: Path) -> PredictionFile:
    """Parse metadata from ``<model>/<split>/<eval_kind>/sequence_predictions``."""

    relative = path.relative_to(eval_root)
    parts = relative.parts
    try:
        marker_index = parts.index("sequence_predictions")
    except ValueError as exc:
        raise ValueError(f"Not a sequence prediction path: {path}") from exc

    prefix = parts[:marker_index]
    suffix = parts[marker_index + 1 :]
    if len(prefix) == 3:
        model, split, eval_kind = prefix
    elif len(prefix) == 2:
        model = eval_root.name
        split, eval_kind = prefix
    else:
        raise ValueError(f"Unexpected prediction path layout: {path}")
    if len(suffix) != 2:
        raise ValueError(f"Unexpected sequence prediction suffix: {path}")

    sequence = suffix[0]
    hz_part, phase_part = path.stem.split("_phase", maxsplit=1)
    return PredictionFile(
        model=model,
        split=split,
        eval_kind=eval_kind,
        sequence=sequence,
        target_hz=int(hz_part.removeprefix("hz")),
        phase=int(phase_part),
        path=path,
    )


def load_prediction_frame(npz_path: Path, reports_dir: Path | None = None) -> pd.DataFrame:
    """Load one NPZ and compute per-frame GT/prediction error columns."""

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
            "yaw_error_deg": wrap_angle_deg(pred_yaw_deg - gt_yaw_deg),
            "pitch_error_deg": pred_pitch_deg - gt_pitch_deg,
            "angular_error_deg": angular_error_deg(pred_xyz, gt_xyz),
            "sequence": sequence,
            "target_hz": target_hz,
            "phase": phase,
            "eval_kind": extra.get("eval_kind", ""),
            "correct_mode": extra.get("correct_mode", ""),
            "anchor_refresh_mode": extra.get("anchor_refresh_mode", ""),
            "feedback_writeback_mode": extra.get("feedback_writeback_mode", ""),
        }
    )
    frame = attach_anchor_gap_columns(frame)
    if reports_dir is not None:
        frame = attach_scene_event_labels(frame, reports_dir, sequence)
    return frame


def attach_anchor_gap_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach position-inside-anchor-interval columns.

    The valid gap rows are non-anchor frames bracketed by a previous and next
    sparse gaze anchor. Anchor frames themselves get ``anchor_gap_valid=False``
    because they are not missing-frame reconstruction targets.
    """

    if frame.empty or "anchor_mask" not in frame:
        return frame.copy()

    output = frame.copy()
    anchor_mask = output["anchor_mask"].to_numpy(dtype=bool)
    n_frames = len(output)
    frames = np.arange(n_frames, dtype=np.int64)
    anchors = np.flatnonzero(anchor_mask).astype(np.int64)

    prev_anchor = np.full(n_frames, -1, dtype=np.int64)
    next_anchor = np.full(n_frames, -1, dtype=np.int64)
    interval = np.full(n_frames, -1, dtype=np.int64)
    since_prev = np.full(n_frames, -1, dtype=np.int64)
    until_next = np.full(n_frames, -1, dtype=np.int64)
    normalized = np.full(n_frames, np.nan, dtype=np.float64)
    nearest = np.full(n_frames, -1, dtype=np.int64)

    if len(anchors):
        prev_slots = np.searchsorted(anchors, frames, side="right") - 1
        next_slots = np.searchsorted(anchors, frames, side="left")
        prev_valid = prev_slots >= 0
        next_valid = next_slots < len(anchors)

        prev_anchor[prev_valid] = anchors[prev_slots[prev_valid]]
        next_anchor[next_valid] = anchors[next_slots[next_valid]]
        bracketed = prev_valid & next_valid
        interval[bracketed] = next_anchor[bracketed] - prev_anchor[bracketed]
        valid_interval = bracketed & (interval > 0)
        since_prev[valid_interval] = frames[valid_interval] - prev_anchor[valid_interval]
        until_next[valid_interval] = next_anchor[valid_interval] - frames[valid_interval]
        normalized[valid_interval] = since_prev[valid_interval] / interval[valid_interval]
        nearest[valid_interval] = np.minimum(since_prev[valid_interval], until_next[valid_interval])

    gap_valid = (interval > 0) & ~anchor_mask
    output["prev_anchor_frame"] = prev_anchor
    output["next_anchor_frame"] = next_anchor
    output["anchor_interval_frames"] = interval
    output["missing_interval_frames"] = np.where(interval > 0, interval - 1, -1)
    output["frames_since_prev_anchor"] = since_prev
    output["frames_until_next_anchor"] = until_next
    output["nearest_anchor_distance_frames"] = nearest
    output["normalized_gap_position"] = normalized
    output["anchor_gap_valid"] = gap_valid
    return output


def attach_scene_event_labels(frame: pd.DataFrame, reports_dir: Path, sequence: str) -> pd.DataFrame:
    """Attach scene-direction fixation/transition labels by frame index."""

    try:
        label_path = find_sequence_file(
            reports_dir,
            sequence,
            "events",
            "scene_gaze_frame_labels.csv",
        )
    except FileNotFoundError:
        output = frame.copy()
        output["scene_event_label"] = "unknown"
        return output

    labels = pd.read_csv(label_path)
    required = {"frame_index", "scene_event_label"}
    if not required.issubset(labels.columns):
        output = frame.copy()
        output["scene_event_label"] = "unknown"
        return output
    return frame.merge(
        labels[["frame_index", "scene_event_label"]],
        on="frame_index",
        how="left",
    ).assign(scene_event_label=lambda df: df["scene_event_label"].fillna("unknown"))


def summarize_prediction_file(
    item: pd.Series | dict[str, Any],
    *,
    reports_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize one prediction NPZ and return sequence plus event summaries."""

    row = dict(item)
    frame = load_prediction_frame(Path(row["path"]), reports_dir=reports_dir)
    evaluated = frame[frame["eval_mask"]].copy()
    if evaluated.empty:
        evaluated = frame.copy()

    anchor_frames = frame[frame["anchor_mask"]]
    eval_anchor_frames = evaluated[evaluated["anchor_mask"]]
    eval_missing_frames = evaluated[~evaluated["anchor_mask"]]

    base = {
        "model": row["model"],
        "split": row["split"],
        "eval_kind": row["eval_kind"],
        "sequence": row["sequence"],
        "target_hz": int(row["target_hz"]),
        "phase": int(row["phase"]),
        "n_frames": len(frame),
        "n_eval": len(evaluated),
        "n_anchor_total": int(frame["anchor_mask"].sum()),
        "n_eval_anchor": int(eval_anchor_frames.shape[0]),
        "n_eval_missing": int(eval_missing_frames.shape[0]),
        "path": row["path"],
    }
    sequence_summary = base | summarize_error_columns(evaluated)
    sequence_summary |= summarize_error_columns(
        eval_missing_frames,
        prefix="missing_",
    )
    sequence_summary |= summarize_error_columns(
        anchor_frames,
        prefix="anchor_",
    )

    event_summary = pd.DataFrame()
    if "scene_event_label" in evaluated.columns:
        event_rows = []
        for label, group in evaluated.groupby("scene_event_label"):
            event_rows.append(
                base
                | {"scene_event_label": label, "n_event": len(group)}
                | summarize_error_columns(group)
            )
        event_summary = pd.DataFrame(event_rows)
    return sequence_summary, event_summary


def summarize_many_predictions(
    prediction_files: pd.DataFrame,
    *,
    reports_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize discovered predictions at sequence, model, and event levels."""

    sequence_rows: list[dict[str, Any]] = []
    event_tables: list[pd.DataFrame] = []
    for _, item in prediction_files.iterrows():
        sequence_summary, event_summary = summarize_prediction_file(item, reports_dir=reports_dir)
        sequence_rows.append(sequence_summary)
        if not event_summary.empty:
            event_tables.append(event_summary)

    sequence_summary = pd.DataFrame(sequence_rows)
    event_summary = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    model_summary = summarize_model_level(sequence_summary)
    return sequence_summary, model_summary, event_summary


def summarize_anchor_gap_position(
    frame_summary: pd.DataFrame,
    *,
    bins: int = 10,
    event_conditioned: bool = False,
) -> pd.DataFrame:
    """Summarize missing-frame error by normalized anchor-gap position.

    The summary reports both frame-weighted and sequence-macro MAE. Sequence
    macro is the paper-facing quantity because it prevents long sequences from
    dominating a gap-position bin.
    """

    columns = [
        "model",
        "split",
        "eval_kind",
        "target_hz",
        "phase",
        "gap_bin",
        "gap_bin_start",
        "gap_bin_end",
        "gap_bin_center",
        "mean_normalized_gap_position",
        "mean_frames_since_prev_anchor",
        "mean_frames_until_next_anchor",
        "sequence_n",
        "frame_n",
        "frame_weighted_mae_deg",
        "sequence_macro_mae_deg",
        "median_deg",
        "p90_deg",
    ]
    if event_conditioned:
        columns.insert(5, "scene_event_label")
    if frame_summary.empty:
        return pd.DataFrame(columns=columns)
    if "anchor_gap_valid" not in frame_summary:
        frame_summary = attach_anchor_gap_columns(frame_summary)

    mask = (
        frame_summary["eval_mask"].astype(bool)
        & ~frame_summary["anchor_mask"].astype(bool)
        & frame_summary["anchor_gap_valid"].astype(bool)
        & frame_summary["normalized_gap_position"].between(0.0, 1.0, inclusive="both")
    )
    data = frame_summary.loc[mask].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.searchsorted(edges, data["normalized_gap_position"].to_numpy(), side="right") - 1
    data["gap_bin"] = np.clip(bin_index, 0, bins - 1).astype(int)
    data["gap_bin_start"] = edges[data["gap_bin"].to_numpy()]
    data["gap_bin_end"] = edges[data["gap_bin"].to_numpy() + 1]
    data["gap_bin_center"] = (data["gap_bin_start"] + data["gap_bin_end"]) / 2.0

    group_cols = ["model", "split", "eval_kind", "target_hz", "phase"]
    if event_conditioned and "scene_event_label" in data:
        group_cols.append("scene_event_label")
    group_cols.extend(["gap_bin", "gap_bin_start", "gap_bin_end", "gap_bin_center"])

    seq_mae = (
        data.groupby([*group_cols, "sequence"], as_index=False, sort=False)
        .agg(sequence_bin_mae_deg=("angular_error_deg", "mean"))
    )
    seq_macro = (
        seq_mae.groupby(group_cols, as_index=False, sort=False)
        .agg(
            sequence_n=("sequence", "nunique"),
            sequence_macro_mae_deg=("sequence_bin_mae_deg", "mean"),
        )
    )
    frame_stats = (
        data.groupby(group_cols, as_index=False, sort=False)
        .agg(
            frame_n=("angular_error_deg", "size"),
            mean_normalized_gap_position=("normalized_gap_position", "mean"),
            mean_frames_since_prev_anchor=("frames_since_prev_anchor", "mean"),
            mean_frames_until_next_anchor=("frames_until_next_anchor", "mean"),
            frame_weighted_mae_deg=("angular_error_deg", "mean"),
            median_deg=("angular_error_deg", "median"),
            p90_deg=("angular_error_deg", lambda values: float(np.percentile(values, 90))),
        )
    )
    summary = frame_stats.merge(seq_macro, on=group_cols, how="left")
    summary = summary[columns]
    return summary.sort_values(group_cols).reset_index(drop=True)


def load_many_prediction_frames(
    prediction_files: pd.DataFrame,
    *,
    reports_dir: Path | None = None,
) -> pd.DataFrame:
    """Load per-frame predictions for all discovered files.

    This table can be large when many models/frequencies are selected, but it
    is the right input for error distributions, yaw/pitch residual clouds, and
    window-level visual diagnostics.
    """

    frames: list[pd.DataFrame] = []
    for _, item in prediction_files.iterrows():
        row = dict(item)
        frame = load_prediction_frame(Path(row["path"]), reports_dir=reports_dir)
        frame["model"] = row["model"]
        frame["split"] = row["split"]
        frame["eval_kind"] = row["eval_kind"]
        frame["target_hz"] = int(row["target_hz"])
        frame["phase"] = int(row["phase"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_model_level(sequence_summary: pd.DataFrame) -> pd.DataFrame:
    """Average sequence-level metrics per model/eval/frequency/phase."""

    if sequence_summary.empty:
        return pd.DataFrame()
    keys = ["model", "split", "eval_kind", "target_hz", "phase"]
    metric_cols = [
        col
        for col in sequence_summary.columns
        if col.endswith("_deg")
        or col in {"n_frames", "n_eval", "n_anchor_total", "n_eval_anchor", "n_eval_missing"}
    ]
    return (
        sequence_summary.groupby(keys, as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(keys)
    )


def summarize_error_columns(frame: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
    """Compute compact mean/median errors for one frame subset."""

    if frame.empty:
        return {
            f"{prefix}mean_angular_error_deg": np.nan,
            f"{prefix}median_angular_error_deg": np.nan,
            f"{prefix}mean_abs_yaw_error_deg": np.nan,
            f"{prefix}median_abs_yaw_error_deg": np.nan,
            f"{prefix}mean_abs_pitch_error_deg": np.nan,
            f"{prefix}median_abs_pitch_error_deg": np.nan,
        }
    return {
        f"{prefix}mean_angular_error_deg": frame["angular_error_deg"].mean(),
        f"{prefix}median_angular_error_deg": frame["angular_error_deg"].median(),
        f"{prefix}mean_abs_yaw_error_deg": frame["yaw_error_deg"].abs().mean(),
        f"{prefix}median_abs_yaw_error_deg": frame["yaw_error_deg"].abs().median(),
        f"{prefix}mean_abs_pitch_error_deg": frame["pitch_error_deg"].abs().mean(),
        f"{prefix}median_abs_pitch_error_deg": frame["pitch_error_deg"].abs().median(),
    }


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize direction vectors row-wise."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def angular_error_deg(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """Compute angular distance between predicted and GT unit directions."""

    dot = np.sum(normalize_vectors(pred_xyz) * normalize_vectors(gt_xyz), axis=1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def direction_to_yaw_pitch_deg(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ADT SparseGaze direction vectors to yaw/pitch degrees.

    The observed SparseGaze ADT NPZ files use negative z as forward. Angular
    error is computed from the full 3D dot product and does not depend on this
    yaw/pitch convention.
    """

    x = xyz[:, 0]
    y = xyz[:, 1]
    forward = -xyz[:, 2]
    return np.rad2deg(np.arctan2(x, forward)), np.rad2deg(np.arctan2(y, forward))


def wrap_angle_deg(values: np.ndarray) -> np.ndarray:
    """Wrap angular residuals into [-180, 180) degrees."""

    return (values + 180.0) % 360.0 - 180.0
