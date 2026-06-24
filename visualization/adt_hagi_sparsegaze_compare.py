"""Compare HAGI++ ADT imputation results with SparseGaze prediction outputs.

This module is notebook-friendly but keeps the actual loading, metric, and
plotting logic reusable from scripts. Large HAGI++ and SparseGaze outputs stay
outside this sandbox; override the default local paths with:

- ``HAGI_REPO_ROOT``
- ``HAGI_ADT_DATA``
- ``HAGI_ADT_IMPUTATION_DIR``
- ``SPARSEGAZE_ADT_EVAL_DIR``
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAGI_REPO = Path(os.environ.get("HAGI_REPO_ROOT", "/home/liumu/Github_Projects/HAGI"))
DEFAULT_ADT_DATA = Path(
    os.environ.get(
        "HAGI_ADT_DATA",
        str(DEFAULT_HAGI_REPO / "datasets" / "adt" / "gaze_head_adt.npy"),
    )
)
DEFAULT_HAGI_DIR = Path(
    os.environ.get(
        "HAGI_ADT_IMPUTATION_DIR",
        str(
            DEFAULT_HAGI_REPO
            / "save"
            / "head"
            / "hagi++_imputation"
            / "adt_low_framerate_sliding"
        ),
    )
)
DEFAULT_SPARSEGAZE_DIR = Path(
    os.environ.get(
        "SPARSEGAZE_ADT_EVAL_DIR",
        "/home/liumu/Github_Projects/SparseGaze/outputs/eval/adt/"
        "sparsegaze_cpf_forward_head_motion_residual_ss",
    )
)

SPARSEGAZE_MODES = {
    "rollout": "SparseGaze rollout",
    "rollout_linear": "SparseGaze linear",
    "rollout_pchip": "SparseGaze pchip",
    "rollout_gt": "SparseGaze gt-repair",
}

METHOD_COLORS = {
    "GT": "black",
    "Anchor GT": "#1f77b4",
    "Missing GT": "#bdbdbd",
    "HAGI++": "#2ca02c",
    "SparseGaze rollout": "#ff7f0e",
    "SparseGaze linear": "#9467bd",
    "SparseGaze pchip": "#d62728",
    "SparseGaze gt-repair": "#8c564b",
}

METHOD_STYLES = {
    "HAGI++": {"linestyle": "-", "marker": "o", "linewidth": 1.5, "markersize": 2.5},
    "SparseGaze rollout": {"linestyle": "-", "marker": ".", "linewidth": 1.2, "markersize": 3.0},
    "SparseGaze linear": {"linestyle": "--", "marker": ".", "linewidth": 1.2, "markersize": 3.0},
    "SparseGaze pchip": {"linestyle": "-.", "marker": ".", "linewidth": 1.2, "markersize": 3.0},
    "SparseGaze gt-repair": {"linestyle": ":", "marker": ".", "linewidth": 1.5, "markersize": 3.0},
}

ERROR_MARKERS = {
    "HAGI++": "o",
    "SparseGaze rollout": "x",
    "SparseGaze linear": "s",
    "SparseGaze pchip": "^",
    "SparseGaze gt-repair": "D",
}


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / np.clip(np.linalg.norm(arr, axis=-1, keepdims=True), eps, None)


def pitch_yaw_to_cpf(pitch_yaw: np.ndarray) -> np.ndarray:
    py = np.asarray(pitch_yaw, dtype=np.float64)
    gaze = np.ones((len(py), 3), dtype=np.float64)
    gaze[:, 0] = np.tan(py[:, 1])
    gaze[:, 1] = np.tan(py[:, 0])
    return _normalize_rows(gaze).astype(np.float32)


def cpf_to_pitch_yaw_deg(cpf: np.ndarray) -> np.ndarray:
    gaze = _normalize_rows(cpf)
    pitch = np.rad2deg(np.arctan2(gaze[..., 1], gaze[..., 2]))
    yaw = np.rad2deg(np.arctan2(gaze[..., 0], gaze[..., 2]))
    return np.stack([pitch, yaw], axis=-1)


def angular_error_deg(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_n = _normalize_rows(pred)
    gt_n = _normalize_rows(gt)
    dot = np.sum(pred_n * gt_n, axis=-1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def _plot_broken_missing_line(
    ax,
    frames: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    marker: str = "o",
    linewidth: float = 1.2,
    markersize: float = 2.5,
    alpha: float = 1.0,
    zorder: int = 3,
) -> None:
    frames = np.asarray(frames, dtype=int)
    values = np.asarray(values, dtype=np.float64)
    if len(frames) == 0:
        return
    breaks = np.where(np.diff(frames) > 1)[0] + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(frames)]
    for i, (start, end) in enumerate(zip(starts, ends)):
        ax.plot(
            frames[start:end],
            values[start:end],
            color=color,
            label=label if i == 0 else None,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            alpha=alpha,
            zorder=zorder,
        )


def _legend_if_handles(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **kwargs)


def load_adt_data(data_path: Path = DEFAULT_ADT_DATA) -> dict:
    return np.load(data_path, allow_pickle=True).item()


def load_hagi_primary(
    fps: int,
    *,
    hagi_dir: Path = DEFAULT_HAGI_DIR,
    nsample: int = 20,
) -> dict:
    path = hagi_dir / f"sliding_primary_nsample{nsample}_framerate_{int(fps)}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def load_sparsegaze_sequence(
    sequence_name: str,
    fps: int,
    mode: str,
    *,
    sparsegaze_dir: Path = DEFAULT_SPARSEGAZE_DIR,
) -> dict:
    path = (
        sparsegaze_dir
        / "test"
        / mode
        / "sequence_predictions"
        / sequence_name
        / f"hz{int(fps)}_phase0.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def sparsegaze_world_to_cpf(
    world_xyz: np.ndarray,
    *,
    record: dict,
    timestamps_ns: np.ndarray | None = None,
) -> np.ndarray:
    rot_world_from_cpf = np.asarray(record["T_world_CPF"], dtype=np.float64)[:, :3, :3]
    world = np.asarray(world_xyz, dtype=np.float64)
    if len(world) > len(rot_world_from_cpf):
        # HAGI++ ADT cache intentionally drops the final gaze row because its
        # head-motion condition is delta[t] = inv(T[t]) @ T[t + 1]. Structured
        # ADT/SparseGaze rollout outputs may keep that last frame.
        world = world[: len(rot_world_from_cpf)]
        if timestamps_ns is not None:
            timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)[: len(rot_world_from_cpf)]
    if timestamps_ns is not None:
        expected = np.asarray(record["query_timestamp_ns"][: len(world)], dtype=np.int64)
        actual = np.asarray(timestamps_ns, dtype=np.int64)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError("SparseGaze timestamps do not align with the HAGI++ ADT record.")
    cpf = np.einsum("tji,tj->ti", rot_world_from_cpf[: len(world)], world)
    return _normalize_rows(cpf).astype(np.float32)


def available_frame_rates(
    *,
    hagi_dir: Path = DEFAULT_HAGI_DIR,
    sparsegaze_dir: Path = DEFAULT_SPARSEGAZE_DIR,
    nsample: int = 20,
) -> list[int]:
    hagi_rates = {
        int(p.stem.rsplit("_", 1)[-1])
        for p in hagi_dir.glob(f"sliding_primary_nsample{nsample}_framerate_*.npz")
    }
    sparse_rates: set[int] = set()
    seq_root = sparsegaze_dir / "test" / "rollout" / "sequence_predictions"
    for path in seq_root.glob("*/*.npz"):
        stem = path.stem
        if stem.startswith("hz") and "_phase" in stem:
            sparse_rates.add(int(stem.split("_phase", 1)[0][2:]))
    return sorted(hagi_rates & sparse_rates, reverse=True)


def sequence_names(
    *,
    adt_data: dict | None = None,
    hagi_dir: Path = DEFAULT_HAGI_DIR,
    sparsegaze_dir: Path = DEFAULT_SPARSEGAZE_DIR,
    fps: int | None = None,
) -> list[str]:
    if adt_data is None:
        adt_data = load_adt_data()
    names = set(adt_data.keys())
    if fps is not None:
        hagi = load_hagi_primary(fps, hagi_dir=hagi_dir)
        names &= set(map(str, hagi["sequence_name"]))
        sparse_root = sparsegaze_dir / "test" / "rollout" / "sequence_predictions"
        names &= {p.name for p in sparse_root.iterdir() if (p / f"hz{int(fps)}_phase0.npz").exists()}
    return sorted(names)


def _hagi_frame_table(hagi: dict, sequence_name: str) -> pd.DataFrame:
    seq = hagi["sequence_name"].astype(str)
    idx = np.where(seq == sequence_name)[0]
    return pd.DataFrame(
        {
            "frame": hagi["frame_index"][idx].astype(int),
            "pred_x": hagi["pred"][idx, 0],
            "pred_y": hagi["pred"][idx, 1],
            "pred_z": hagi["pred"][idx, 2],
            "gt_x": hagi["gt"][idx, 0],
            "gt_y": hagi["gt"][idx, 1],
            "gt_z": hagi["gt"][idx, 2],
            "error_deg": hagi["angular_error_deg"][idx],
        }
    ).sort_values("frame")


def comparison_metrics(
    fps: int,
    *,
    sparse_modes: Iterable[str] = SPARSEGAZE_MODES.keys(),
    adt_data: dict | None = None,
    hagi_dir: Path = DEFAULT_HAGI_DIR,
    sparsegaze_dir: Path = DEFAULT_SPARSEGAZE_DIR,
    nsample: int = 20,
) -> pd.DataFrame:
    if adt_data is None:
        adt_data = load_adt_data()
    hagi = load_hagi_primary(fps, hagi_dir=hagi_dir, nsample=nsample)
    rows: list[dict] = []

    hagi_errors = []
    by_seq: dict[str, np.ndarray] = {}
    for seq in sorted(set(map(str, hagi["sequence_name"]))):
        table = _hagi_frame_table(hagi, seq)
        by_seq[seq] = table["frame"].to_numpy(dtype=int)
        hagi_errors.append(table["error_deg"].to_numpy(dtype=np.float64))
    hagi_errors_arr = np.concatenate(hagi_errors) if hagi_errors else np.array([], dtype=np.float64)
    rows.append(_metric_row("HAGI++", hagi_errors_arr))

    for mode in sparse_modes:
        label = SPARSEGAZE_MODES.get(mode, mode)
        errors = []
        for seq, hagi_frames in by_seq.items():
            record = adt_data[seq]
            sparse = load_sparsegaze_sequence(seq, fps, mode, sparsegaze_dir=sparsegaze_dir)
            pred_cpf = sparsegaze_world_to_cpf(
                sparse["pred_xyz"],
                record=record,
                timestamps_ns=sparse["timestamps_ns"],
            )
            gt_cpf = pitch_yaw_to_cpf(record["pitch_yaw"][: len(pred_cpf)])
            eval_frames = set(map(int, np.flatnonzero(sparse["eval_mask"])))
            frames = np.array(
                [
                    f
                    for f in hagi_frames
                    if int(f) in eval_frames and int(f) < len(pred_cpf) and int(f) < len(gt_cpf)
                ],
                dtype=int,
            )
            if len(frames):
                errors.append(angular_error_deg(pred_cpf[frames], gt_cpf[frames]))
        err = np.concatenate(errors) if errors else np.array([], dtype=np.float64)
        rows.append(_metric_row(label, err))

    return pd.DataFrame(rows)


def _metric_row(method: str, errors: np.ndarray) -> dict:
    if len(errors) == 0:
        return {"method": method, "n_common": 0, "mae_deg": np.nan, "median_deg": np.nan, "p90_deg": np.nan}
    return {
        "method": method,
        "n_common": int(len(errors)),
        "mae_deg": float(np.mean(errors)),
        "median_deg": float(np.median(errors)),
        "p90_deg": float(np.percentile(errors, 90)),
    }


def _reported_metric_lookup(
    fps: int,
    *,
    hagi_dir: Path = DEFAULT_HAGI_DIR,
    sparsegaze_dir: Path = DEFAULT_SPARSEGAZE_DIR,
    nsample: int = 20,
) -> dict[str, dict]:
    summary = reported_summary(hagi_dir=hagi_dir, sparsegaze_dir=sparsegaze_dir, nsample=nsample)
    if summary.empty:
        return {}
    rows = summary[summary["fps"] == int(fps)]
    return {str(row["method"]): dict(row) for _, row in rows.iterrows()}


def sequence_summary(
    fps: int,
    sequence_name: str | None = None,
    *,
    include_hagi: bool = True,
    sparse_modes: Iterable[str] = SPARSEGAZE_MODES.keys(),
    adt_data: dict | None = None,
    hagi_dir: Path = DEFAULT_HAGI_DIR,
    sparsegaze_dir: Path = DEFAULT_SPARSEGAZE_DIR,
    nsample: int = 20,
) -> pd.DataFrame:
    """Summarize whole-sequence MAE and available frequency-level JS metrics."""

    if adt_data is None:
        adt_data = load_adt_data()
    if sequence_name is None:
        sequence_name = sequence_names(
            adt_data=adt_data,
            fps=fps,
            hagi_dir=hagi_dir,
            sparsegaze_dir=sparsegaze_dir,
        )[0]
    record = adt_data[sequence_name]
    hagi = load_hagi_primary(fps, hagi_dir=hagi_dir, nsample=nsample)
    hagi_table = _hagi_frame_table(hagi, sequence_name)
    hagi_frames = hagi_table["frame"].to_numpy(dtype=int)
    reported = _reported_metric_lookup(
        fps,
        hagi_dir=hagi_dir,
        sparsegaze_dir=sparsegaze_dir,
        nsample=nsample,
    )

    rows = []
    if include_hagi:
        hagi_row = _metric_row("HAGI++", hagi_table["error_deg"].to_numpy(dtype=np.float64))
        hagi_row.update(
            {
                "sequence": sequence_name,
                "fps": int(fps),
                "velocity_js": reported.get("HAGI++", {}).get("velocity_js", np.nan),
                "js_scope": "fps aggregate",
                "mae_scope": "whole HAGI++ sequence output",
            }
        )
        rows.append(hagi_row)

    gt_cpf = pitch_yaw_to_cpf(record["pitch_yaw"])
    for mode in sparse_modes:
        label = SPARSEGAZE_MODES.get(mode, mode)
        sparse = load_sparsegaze_sequence(sequence_name, fps, mode, sparsegaze_dir=sparsegaze_dir)
        pred_cpf = sparsegaze_world_to_cpf(
            sparse["pred_xyz"],
            record=record,
            timestamps_ns=sparse["timestamps_ns"],
        )
        eval_frames = set(map(int, np.flatnonzero(sparse["eval_mask"])))
        frames = np.array(
            [
                f
                for f in hagi_frames
                if int(f) in eval_frames and int(f) < len(pred_cpf) and int(f) < len(gt_cpf)
            ],
            dtype=int,
        )
        errors = (
            angular_error_deg(pred_cpf[frames], gt_cpf[frames])
            if len(frames)
            else np.array([], dtype=np.float64)
        )
        row = _metric_row(label, errors)
        row.update(
            {
                "sequence": sequence_name,
                "fps": int(fps),
                "velocity_js": reported.get(label, {}).get("velocity_js", np.nan),
                "js_scope": "fps aggregate",
                "mae_scope": "whole-sequence common missing frames",
            }
        )
        rows.append(row)

    columns = [
        "sequence",
        "fps",
        "method",
        "n_common",
        "mae_deg",
        "median_deg",
        "p90_deg",
        "velocity_js",
        "js_scope",
        "mae_scope",
    ]
    return pd.DataFrame(rows, columns=columns)


def reported_summary(
    *,
    hagi_dir: Path = DEFAULT_HAGI_DIR,
    sparsegaze_dir: Path = DEFAULT_SPARSEGAZE_DIR,
    nsample: int = 20,
) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(hagi_dir.glob(f"sliding_metrics_nsample{nsample}_framerate_*.json")):
        fps = int(path.stem.rsplit("_", 1)[-1])
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "method": "HAGI++",
                "fps": fps,
                "n": int(data.get("count", 0)),
                "mae_deg": data.get("mean_error"),
                "p90_deg": data.get("p90_error"),
                "velocity_js": data.get("js_dense_sequence"),
            }
        )

    for mode, label in SPARSEGAZE_MODES.items():
        path = sparsegaze_dir / "test" / mode / "rollout_missing_summary.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data.get("phase_averaged", []):
            rows.append(
                {
                    "method": label,
                    "fps": int(row["target_hz"]),
                    "n": int(row["n_missing_total"]),
                    "mae_deg": row.get("mae_missing_deg"),
                    "p90_deg": row.get("p90_missing_deg"),
                    "velocity_js": row.get("velocity_js_divergence"),
                }
            )
    return pd.DataFrame(rows).sort_values(["fps", "method"], ascending=[False, True])


def plot_sequence(
    fps: int = 6,
    sequence_name: str | None = None,
    start_frame: int = 149,
    length: int = 240,
    *,
    show_hagi: bool = True,
    sparse_modes: Iterable[str] = ("rollout",),
    show_anchors: bool = True,
    show_missing_gt: bool = True,
    hagi_dir: Path = DEFAULT_HAGI_DIR,
    sparsegaze_dir: Path = DEFAULT_SPARSEGAZE_DIR,
    adt_data: dict | None = None,
    nsample: int = 20,
) -> plt.Figure:
    if adt_data is None:
        adt_data = load_adt_data()
    if sequence_name is None:
        sequence_name = sequence_names(adt_data=adt_data, fps=fps, hagi_dir=hagi_dir, sparsegaze_dir=sparsegaze_dir)[0]
    record = adt_data[sequence_name]
    hagi = load_hagi_primary(fps, hagi_dir=hagi_dir, nsample=nsample)
    hagi_table = _hagi_frame_table(hagi, sequence_name)

    gt_cpf = pitch_yaw_to_cpf(record["pitch_yaw"])
    gt_py = cpf_to_pitch_yaw_deg(gt_cpf)
    frame_end = min(int(start_frame) + int(length), len(gt_py))
    window = np.arange(int(start_frame), frame_end, dtype=int)
    hagi_frame_set = set(map(int, hagi_table["frame"]))
    sparse_modes = tuple(sparse_modes)

    rows = 3
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(14, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.7]},
    )
    pitch_ax, yaw_ax = axes[0], axes[1]
    error_ax = axes[2]

    anchor_mask = None
    eval_mask_for_missing_gt = None
    for mode in sparse_modes:
        sparse = load_sparsegaze_sequence(sequence_name, fps, mode, sparsegaze_dir=sparsegaze_dir)
        anchor_mask = np.asarray(sparse["anchor_mask"], dtype=bool)
        eval_mask_for_missing_gt = np.asarray(sparse["eval_mask"], dtype=bool)
        eval_frames = set(map(int, np.flatnonzero(sparse["eval_mask"])))
        frames = np.array([f for f in window if f in hagi_frame_set and f in eval_frames], dtype=int)
        if len(frames) == 0:
            continue
        pred_cpf = sparsegaze_world_to_cpf(
            sparse["pred_xyz"],
            record=record,
            timestamps_ns=sparse["timestamps_ns"],
        )
        max_pred_frame = min(len(pred_cpf), len(gt_cpf))
        frames = frames[frames < max_pred_frame]
        if len(frames) == 0:
            continue
        pred_py = cpf_to_pitch_yaw_deg(pred_cpf)
        label = SPARSEGAZE_MODES.get(mode, mode)
        color = METHOD_COLORS.get(label, None)
        style = METHOD_STYLES.get(label, {})
        _plot_broken_missing_line(pitch_ax, frames, pred_py[frames, 0], color=color, label=label, **style)
        _plot_broken_missing_line(yaw_ax, frames, pred_py[frames, 1], color=color, label=label, **style)
        err = angular_error_deg(pred_cpf[frames], gt_cpf[frames])
        error_ax.scatter(
            frames,
            err,
            s=18,
            color=color,
            alpha=0.65,
            marker=ERROR_MARKERS.get(label, "o"),
            label=label,
        )

    hagi_window = hagi_table[(hagi_table["frame"] >= int(start_frame)) & (hagi_table["frame"] < frame_end)]
    if show_hagi and not hagi_window.empty:
        frames = hagi_window["frame"].to_numpy(dtype=int)
        pred = hagi_window[["pred_x", "pred_y", "pred_z"]].to_numpy(dtype=np.float64)
        pred_py = cpf_to_pitch_yaw_deg(pred)
        style = METHOD_STYLES["HAGI++"]
        _plot_broken_missing_line(pitch_ax, frames, pred_py[:, 0], color=METHOD_COLORS["HAGI++"], label="HAGI++", **style)
        _plot_broken_missing_line(yaw_ax, frames, pred_py[:, 1], color=METHOD_COLORS["HAGI++"], label="HAGI++", **style)
        error_ax.scatter(
            frames,
            hagi_window["error_deg"],
            s=20,
            color=METHOD_COLORS["HAGI++"],
            alpha=0.7,
            marker=ERROR_MARKERS["HAGI++"],
            label="HAGI++",
        )

    if show_anchors and anchor_mask is not None:
        anchor_window = window[window < len(anchor_mask)]
        anchor_frames = anchor_window[anchor_mask[anchor_window]]
        pitch_ax.scatter(anchor_frames, gt_py[anchor_frames, 0], s=22, color=METHOD_COLORS["Anchor GT"], label="Anchor GT", zorder=4)
        yaw_ax.scatter(anchor_frames, gt_py[anchor_frames, 1], s=22, color=METHOD_COLORS["Anchor GT"], label="Anchor GT", zorder=4)

    if show_missing_gt and eval_mask_for_missing_gt is not None:
        missing_gt_frames = np.array(
            [
                f
                for f in window
                if f < len(eval_mask_for_missing_gt)
                and f in hagi_frame_set
                and eval_mask_for_missing_gt[f]
            ],
            dtype=int,
        )
        _plot_broken_missing_line(
            pitch_ax,
            missing_gt_frames,
            gt_py[missing_gt_frames, 0],
            color=METHOD_COLORS["Missing GT"],
            label="Missing GT",
            linestyle="--",
            marker=".",
            linewidth=1.0,
            markersize=3.0,
            alpha=0.45,
            zorder=2,
        )
        _plot_broken_missing_line(
            yaw_ax,
            missing_gt_frames,
            gt_py[missing_gt_frames, 1],
            color=METHOD_COLORS["Missing GT"],
            label="Missing GT",
            linestyle="--",
            marker=".",
            linewidth=1.0,
            markersize=3.0,
            alpha=0.45,
            zorder=2,
        )

    pitch_ax.set_ylabel("Pitch (deg)")
    yaw_ax.set_ylabel("Yaw (deg)")
    error_ax.set_ylabel("Angular error (deg)")
    error_ax.set_xlabel("Frame index")
    error_ax.set_ylim(bottom=0)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    for ax in (pitch_ax, yaw_ax):
        _legend_if_handles(ax, loc="upper right", ncol=3, fontsize=8, frameon=False)
    _legend_if_handles(error_ax, loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{sequence_name} | {fps} Hz | common HAGI++/SparseGaze missing frames")
    fig.tight_layout()
    return fig


def make_widget():
    import ipywidgets as widgets
    from IPython.display import display

    adt_data = load_adt_data()
    fps_options = available_frame_rates()
    fps_dropdown = widgets.Dropdown(options=fps_options, value=6 if 6 in fps_options else fps_options[0], description="FPS")
    seq_options = sequence_names(adt_data=adt_data, fps=fps_dropdown.value)
    seq_dropdown = widgets.Dropdown(options=seq_options, value=seq_options[0], description="Sequence")
    def _max_frame_for_sequence(sequence_name: str, fps: int) -> int:
        max_frame = len(adt_data[sequence_name]["pitch_yaw"]) - 1
        try:
            sparse = load_sparsegaze_sequence(sequence_name, fps, "rollout")
        except FileNotFoundError:
            return max(max_frame, 0)
        return max(min(max_frame, len(sparse["eval_mask"]) - 1), 0)

    first_seq_max = _max_frame_for_sequence(seq_options[0], fps_dropdown.value)
    start_input = widgets.BoundedIntText(
        value=149,
        min=0,
        max=first_seq_max,
        step=1,
        description="Start",
        layout=widgets.Layout(width="180px"),
    )
    length_input = widgets.BoundedIntText(
        value=240,
        min=1,
        max=max(first_seq_max + 1, 1),
        step=1,
        description="Length",
        layout=widgets.Layout(width="180px"),
    )
    anchors_checkbox = widgets.Checkbox(value=True, description="Anchors")
    missing_gt_checkbox = widgets.Checkbox(value=True, description="Missing GT")
    hagi_checkbox = widgets.Checkbox(value=True, description="HAGI++", indent=False)
    sparse_checkboxes = {
        mode: widgets.Checkbox(
            value=(mode == "rollout"),
            description=label,
            indent=False,
            layout=widgets.Layout(width="220px"),
        )
        for mode, label in SPARSEGAZE_MODES.items()
    }
    method_box = widgets.VBox(
        [hagi_checkbox, *sparse_checkboxes.values()],
        layout=widgets.Layout(border="1px solid #dddddd", padding="8px", width="260px"),
    )

    def _on_fps_change(change):
        names = sequence_names(adt_data=adt_data, fps=change["new"])
        seq_dropdown.options = names
        if names:
            seq_dropdown.value = names[0]
            _update_frame_inputs(names[0], change["new"])

    def _on_seq_change(change):
        if change["new"] in adt_data:
            _update_frame_inputs(change["new"], fps_dropdown.value)

    def _update_frame_inputs(sequence_name: str, fps: int) -> None:
        max_frame = _max_frame_for_sequence(sequence_name, fps)
        start_input.max = max_frame
        length_input.max = max(max_frame + 1, 1)
        start_input.value = min(start_input.value, start_input.max)
        length_input.value = min(length_input.value, length_input.max)

    fps_dropdown.observe(_on_fps_change, names="value")
    seq_dropdown.observe(_on_seq_change, names="value")

    def _render(
        fps,
        sequence_name,
        start_frame,
        length,
        show_anchors,
        show_missing_gt,
        show_hagi,
        **mode_flags,
    ):
        sparse_modes = tuple(mode for mode in SPARSEGAZE_MODES if mode_flags.get(mode, False))
        summary = sequence_summary(
            fps=fps,
            sequence_name=sequence_name,
            include_hagi=show_hagi,
            sparse_modes=sparse_modes,
            adt_data=adt_data,
        )
        if summary.empty:
            display("Select at least one method.")
        else:
            display(summary.round({"mae_deg": 4, "median_deg": 4, "p90_deg": 4, "velocity_js": 5}))
        fig = plot_sequence(
            fps=fps,
            sequence_name=sequence_name,
            start_frame=start_frame,
            length=length,
            show_hagi=show_hagi,
            show_anchors=show_anchors,
            show_missing_gt=show_missing_gt,
            sparse_modes=sparse_modes,
            adt_data=adt_data,
        )
        display(fig)
        plt.close(fig)

    out = widgets.interactive_output(
        _render,
        {
            "fps": fps_dropdown,
            "sequence_name": seq_dropdown,
            "start_frame": start_input,
            "length": length_input,
            "show_anchors": anchors_checkbox,
            "show_missing_gt": missing_gt_checkbox,
            "show_hagi": hagi_checkbox,
            **sparse_checkboxes,
        },
    )
    ui = widgets.VBox(
        [
            widgets.HBox([fps_dropdown, seq_dropdown]),
            widgets.HBox([start_input, length_input, anchors_checkbox, missing_gt_checkbox]),
            method_box,
        ]
    )
    display(ui, out)
    return ui, out
