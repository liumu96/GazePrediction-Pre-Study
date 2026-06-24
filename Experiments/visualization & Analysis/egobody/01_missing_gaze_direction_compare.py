#!/usr/bin/env python
"""Compare EgoBody GT and predicted gaze directions on missing frames.

This is the EgoBody counterpart to image-space scanpath inspection when RGB
frames are unavailable. It reads SparseGaze per-sequence prediction NPZ files,
HAGI++ low-framerate outputs, and EgoBody feature caches, then plots
missing-frame direction errors and 3D gaze endpoint scanpaths.

Example:

    python "Experiments/visualization & Analysis/egobody/01_missing_gaze_direction_compare.py" \
      recording_20210907_S03_S04_01 \
      --target-hz 6 \
      --start-frame 149 \
      --end-frame 300
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

def find_repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "src").exists() and (parent / "Experiments").exists():
            return parent
    return Path(__file__).resolve().parents[3]


REPO_ROOT = find_repo_root()
DEFAULT_EVAL_ROOT = Path("/home/liumu/Github_Projects/SparseGaze/outputs/eval/egobody")
DEFAULT_CACHE_ROOT = Path("/mnt/d/SparseGaze/feature_cache/egobody/sparsegaze")
DEFAULT_HAGI_DATA_PATH = Path("/home/liumu/Github_Projects/HAGI/datasets/egobody/gaze_head_egobody.npy")
DEFAULT_HAGI_OUTPUT_ROOT = Path(
    "/home/liumu/Github_Projects/HAGI/save/head/hagi++_imputation/egobody_low_framerate_sliding"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "figures" / "egobody_missing_gaze_direction"

DEFAULT_METHODS = [
    (
        "SparseGaze residual",
        "sparsegaze_cpf_forward_head_motion_residual_ss/test/rollout",
    ),
]


@dataclass
class Track:
    label: str
    pred_xyz: np.ndarray
    source: str
    angular_error_deg: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", help="EgoBody recording id.")
    parser.add_argument("--target-hz", type=int, default=6)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=240)
    parser.add_argument("--base-fps", type=int, default=30)
    parser.add_argument("--ray-length", type=float, default=2.0)
    parser.add_argument("--max-3d-points", type=int, default=180)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--hagi-data-path", type=Path, default=DEFAULT_HAGI_DATA_PATH)
    parser.add_argument(
        "--hagi-npz",
        type=Path,
        default=None,
        help=(
            "HAGI++ sliding low-framerate result NPZ. Defaults to "
            "sliding_primary_nsample20_framerate_<target_hz>.npz under the HAGI output root."
        ),
    )
    parser.add_argument("--hagi-output-root", type=Path, default=DEFAULT_HAGI_OUTPUT_ROOT)
    parser.add_argument("--no-hagi", action="store_true", help="Do not add the default HAGI++ comparison track.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-3d", action="store_true", help="Skip the diagnostic 3D endpoint plot.")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help=(
            "Prediction track as 'Label=relative_eval_dir'. Relative dir is under "
            "--eval-root and should contain sequence_predictions/<sequence>/hzX_phaseY.npz. "
            "If omitted, the default SparseGaze residual rollout is used when present."
        ),
    )
    parser.add_argument(
        "--include-synthetic-baselines",
        action="store_true",
        help="Also include reconstructed repeat/head-direction sanity baselines.",
    )
    parser.add_argument("--no-synthetic-baselines", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-12)
    out[valid] = arr[valid] / norms[valid]
    return out


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = normalize_rows(a)
    bb = normalize_rows(b)
    dot = np.sum(aa * bb, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    err = np.degrees(np.arccos(dot))
    err[~np.isfinite(dot)] = np.nan
    return err


def method_specs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.method:
        return DEFAULT_METHODS
    specs = []
    for item in args.method:
        if "=" not in item:
            raise ValueError(f"--method must be 'Label=relative_eval_dir', got: {item}")
        label, rel = item.split("=", maxsplit=1)
        specs.append((label.strip(), rel.strip()))
    return specs


def prediction_path(
    eval_root: Path,
    relative_eval_dir: str,
    sequence: str,
    target_hz: int,
    phase: int,
) -> Path:
    return (
        eval_root
        / relative_eval_dir
        / "sequence_predictions"
        / sequence
        / f"hz{target_hz}_phase{phase}.npz"
    )


def load_prediction_track(path: Path, label: str) -> tuple[Track, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        track = Track(
            label=label,
            pred_xyz=normalize_rows(np.asarray(data["pred_xyz"], dtype=np.float64)),
            source=str(path),
        )
        meta = {
            "gt_xyz": normalize_rows(np.asarray(data["gt_xyz"], dtype=np.float64)),
            "timestamps_ns": np.asarray(data["timestamps_ns"], dtype=np.int64),
            "anchor_mask": np.asarray(data["anchor_mask"], dtype=bool),
            "eval_mask": np.asarray(data["eval_mask"], dtype=bool),
        }
    return track, meta


def default_hagi_npz(output_root: Path, target_hz: int) -> Path:
    return output_root / f"sliding_primary_nsample20_framerate_{int(target_hz)}.npz"


def load_hagi_record(data_path: Path, sequence: str) -> dict[str, np.ndarray]:
    records = np.load(data_path, allow_pickle=True).item()
    if sequence not in records:
        raise KeyError(f"HAGI++ data {data_path} does not contain sequence {sequence}")
    return {key: np.asarray(value) for key, value in records[sequence].items()}


def load_hagi_track(
    result_path: Path,
    data_path: Path,
    sequence: str,
    n_frames: int,
) -> Track:
    with np.load(result_path, allow_pickle=True) as data:
        seq_names = np.asarray(data["sequence_name"])
        mask = seq_names == sequence
        frame_index = np.asarray(data["frame_index"][mask], dtype=np.int64)
        pred_cpf = normalize_rows(np.asarray(data["pred"][mask], dtype=np.float64))
        saved_error = np.asarray(data["angular_error_deg"][mask], dtype=np.float64)

    if not np.any(mask):
        raise KeyError(f"HAGI++ result {result_path} does not contain sequence {sequence}")

    record = load_hagi_record(data_path, sequence)
    rotations = np.asarray(record["T_world_CPF"], dtype=np.float64)[:, :3, :3]

    pred_world = np.full((n_frames, 3), np.nan, dtype=np.float64)
    error_full = np.full(n_frames, np.nan, dtype=np.float64)
    valid = (frame_index >= 0) & (frame_index < n_frames) & (frame_index < len(rotations))
    if np.any(valid):
        frames = frame_index[valid]
        # HAGI++ stores gaze in its CPF convention. In this EgoBody cache, -R_world_CPF
        # aligns HAGI++ CPF gaze with SparseGaze world gaze.
        pred_world[frames] = -np.einsum("nij,nj->ni", rotations[frames], pred_cpf[valid])
        error_full[frames] = saved_error[valid]

    return Track(
        "HAGI++",
        normalize_rows(pred_world),
        f"{result_path} + {data_path}:T_world_CPF",
        angular_error_deg=error_full,
    )


def world_to_hagi_cpf(world_xyz: np.ndarray, hagi_record: dict[str, np.ndarray], n_frames: int) -> np.ndarray:
    rotations = np.asarray(hagi_record["T_world_CPF"], dtype=np.float64)[:, :3, :3]
    world = normalize_rows(np.asarray(world_xyz, dtype=np.float64)[:n_frames])
    count = min(len(world), len(rotations))
    cpf = np.full((n_frames, 3), np.nan, dtype=np.float64)
    if count:
        # EgoBody/HAGI++ stores gaze in the opposite CPF ray direction.
        cpf[:count] = -np.einsum("tji,tj->ti", rotations[:count], world[:count])
    return normalize_rows(cpf)


def cpf_to_pitch_yaw_deg(cpf: np.ndarray) -> np.ndarray:
    gaze = normalize_rows(cpf)
    pitch = np.degrees(np.arctan2(gaze[:, 1], gaze[:, 2]))
    yaw = np.degrees(np.arctan2(gaze[:, 0], gaze[:, 2]))
    return np.stack([pitch, yaw], axis=1)


def find_cache_npz(cache_root: Path, sequence: str) -> Path:
    matches = sorted(cache_root.glob(f"*/{sequence}.npz"))
    if not matches:
        raise FileNotFoundError(f"Could not find EgoBody cache NPZ for {sequence} under {cache_root}")
    return matches[0]


def load_cache(cache_root: Path, sequence: str) -> dict[str, np.ndarray]:
    path = find_cache_npz(cache_root, sequence)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def generated_masks(n: int, target_hz: int, phase: int, base_fps: int) -> tuple[np.ndarray, np.ndarray]:
    stride = max(int(round(base_fps / target_hz)), 1)
    idx = np.arange(n, dtype=np.int64)
    anchor = (idx % stride) == int(phase)
    eval_mask = ~anchor
    eval_mask[: stride * 3] = False
    return anchor, eval_mask


def last_anchor_indices(anchor_mask: np.ndarray) -> np.ndarray:
    out = np.full(len(anchor_mask), -1, dtype=np.int64)
    last = -1
    for idx, is_anchor in enumerate(anchor_mask):
        if bool(is_anchor):
            last = idx
        out[idx] = last
    return out


def synthetic_tracks(cache: dict[str, np.ndarray], gt_xyz: np.ndarray, anchor_mask: np.ndarray) -> list[Track]:
    tracks: list[Track] = []
    last_anchor = last_anchor_indices(anchor_mask)
    valid_anchor = last_anchor >= 0

    repeat = np.full_like(gt_xyz, np.nan, dtype=np.float64)
    repeat[valid_anchor] = gt_xyz[last_anchor[valid_anchor]]
    tracks.append(Track("repeat", normalize_rows(repeat), "synthetic:last_anchor_world_gaze"))

    if "head_dir_xyz" in cache:
        tracks.append(
            Track(
                "head_direction",
                normalize_rows(np.asarray(cache["head_dir_xyz"], dtype=np.float64)[: len(gt_xyz)]),
                "cache:head_dir_xyz",
            )
        )

    return tracks


def frame_window_mask(n: int, start_frame: int, end_frame: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    return (idx >= int(start_frame)) & (idx < int(end_frame))


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_").lower()


def write_error_csv(
    path: Path,
    *,
    gt_xyz: np.ndarray,
    timestamps_ns: np.ndarray,
    anchor_mask: np.ndarray,
    eval_mask: np.ndarray,
    tracks: list[Track],
    errors: dict[str, np.ndarray],
    rows_mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "timestamp_ns",
        "is_anchor",
        "is_eval_missing",
        "gt_x",
        "gt_y",
        "gt_z",
    ]
    for track in tracks:
        name = safe_name(track.label)
        fieldnames.extend([f"{name}_pred_x", f"{name}_pred_y", f"{name}_pred_z", f"{name}_angular_error_deg"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in np.flatnonzero(rows_mask):
            row: dict[str, Any] = {
                "frame_index": int(idx),
                "timestamp_ns": int(timestamps_ns[idx]),
                "is_anchor": bool(anchor_mask[idx]),
                "is_eval_missing": bool(eval_mask[idx]),
                "gt_x": float(gt_xyz[idx, 0]),
                "gt_y": float(gt_xyz[idx, 1]),
                "gt_z": float(gt_xyz[idx, 2]),
            }
            for track in tracks:
                name = safe_name(track.label)
                pred = track.pred_xyz[idx]
                row[f"{name}_pred_x"] = float(pred[0])
                row[f"{name}_pred_y"] = float(pred[1])
                row[f"{name}_pred_z"] = float(pred[2])
                row[f"{name}_angular_error_deg"] = float(errors[track.label][idx])
            writer.writerow(row)


def plot_error_timeline(
    path: Path,
    *,
    sequence: str,
    target_hz: int,
    phase: int,
    tracks: list[Track],
    errors: dict[str, np.ndarray],
    anchor_mask: np.ndarray,
    eval_mask: np.ndarray,
    rows_mask: np.ndarray,
) -> None:
    x = np.arange(len(anchor_mask), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(13, 5))
    for track in tracks:
        y = np.full(len(x), np.nan, dtype=np.float64)
        mask = rows_mask & eval_mask
        y[mask] = errors[track.label][mask]
        ax.plot(x, y, marker=".", linewidth=1.1, markersize=3.0, label=track.label)

    anchor_x = x[rows_mask & anchor_mask]
    if len(anchor_x):
        ax.scatter(anchor_x, np.zeros_like(anchor_x), s=8, color="black", alpha=0.35, label="anchors")
    ax.set_title(f"{sequence} | missing-frame angular error | {target_hz}Hz phase={phase}")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Angular error [deg]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def set_axes_equal(ax: Any) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=np.float64)
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_3d_scanpath(
    path: Path,
    *,
    sequence: str,
    target_hz: int,
    phase: int,
    gt_xyz: np.ndarray,
    head_pos: np.ndarray,
    tracks: list[Track],
    eval_mask: np.ndarray,
    rows_mask: np.ndarray,
    ray_length: float,
    max_points: int,
) -> None:
    valid = np.flatnonzero(rows_mask & eval_mask)
    if len(valid) > max_points:
        take = np.linspace(0, len(valid) - 1, max_points).round().astype(np.int64)
        valid = valid[take]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(valid):
        head = head_pos[valid]
        gt_end = head + gt_xyz[valid] * float(ray_length)
        ax.plot(head[:, 0], head[:, 1], head[:, 2], color="#666666", linewidth=1.2, label="head trajectory")
        ax.plot(gt_end[:, 0], gt_end[:, 1], gt_end[:, 2], color="black", linewidth=2.2, label="GT gaze endpoint")
        ax.scatter(gt_end[:, 0], gt_end[:, 1], gt_end[:, 2], color="black", s=12)

        for track in tracks:
            pred = head + track.pred_xyz[valid] * float(ray_length)
            ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], linewidth=1.5, label=track.label)
            ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=10)

        stride = max(len(valid) // 30, 1)
        for idx in valid[::stride]:
            origin = head_pos[idx]
            gt_tip = origin + gt_xyz[idx] * float(ray_length)
            ax.plot([origin[0], gt_tip[0]], [origin[1], gt_tip[1]], [origin[2], gt_tip[2]], color="black", alpha=0.18, linewidth=0.7)

    ax.set_title(f"{sequence} | missing-frame 3D gaze endpoints | {target_hz}Hz phase={phase}")
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Y [m]")
    ax.set_zlabel("World Z [m]")
    ax.legend(loc="upper left", fontsize=8)
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cpf_pitch_yaw_timeline(
    path: Path,
    *,
    sequence: str,
    target_hz: int,
    phase: int,
    gt_xyz: np.ndarray,
    tracks: list[Track],
    hagi_record: dict[str, np.ndarray],
    anchor_mask: np.ndarray,
    eval_mask: np.ndarray,
    rows_mask: np.ndarray,
) -> None:
    n = min(len(gt_xyz), len(hagi_record["T_world_CPF"]), len(anchor_mask))
    frames = np.arange(n, dtype=np.int64)
    rows_mask = rows_mask[:n]
    anchor_mask = anchor_mask[:n]
    eval_mask = eval_mask[:n]

    gt_py = cpf_to_pitch_yaw_deg(world_to_hagi_cpf(gt_xyz, hagi_record, n))
    track_py = {
        track.label: cpf_to_pitch_yaw_deg(world_to_hagi_cpf(track.pred_xyz, hagi_record, n))
        for track in tracks
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    labels = [("Pitch", 0), ("Yaw", 1)]
    for ax, (axis_name, dim) in zip(axes, labels):
        plot_mask = rows_mask
        ax.plot(frames[plot_mask], gt_py[plot_mask, dim], color="black", linewidth=2.0, label=f"GT {axis_name.lower()}")
        for track in tracks:
            values = track_py[track.label][:n, dim]
            mask = rows_mask & eval_mask & np.isfinite(values)
            ax.plot(frames[mask], values[mask], marker=".", linewidth=1.2, markersize=3.0, label=track.label)

        anchor_x = frames[rows_mask & anchor_mask]
        if len(anchor_x):
            anchor_y = gt_py[anchor_x, dim]
            ax.scatter(anchor_x, anchor_y, s=10, color="#777777", alpha=0.5, label="anchor GT")
        ax.set_ylabel(f"{axis_name} [deg]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncols=2, fontsize=8)

    axes[0].set_title(f"{sequence} | CPF pitch/yaw direction | {target_hz}Hz phase={phase}")
    axes[-1].set_xlabel("Frame index")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tracks: list[Track] = []
    common_meta: dict[str, np.ndarray] | None = None
    missing_predictions: list[str] = []

    for label, rel_dir in method_specs(args):
        path = prediction_path(args.eval_root, rel_dir, args.sequence, args.target_hz, args.phase)
        if not path.exists():
            missing_predictions.append(f"{label}: {path}")
            continue
        track, meta = load_prediction_track(path, label)
        tracks.append(track)
        if common_meta is None:
            common_meta = meta

    cache = load_cache(args.cache_root, args.sequence)
    if common_meta is None:
        n = len(cache["gaze_world"]) if "gaze_world" in cache else len(cache["gaze_xyz"])
        anchor_mask, eval_mask = generated_masks(n, args.target_hz, args.phase, args.base_fps)
        gt_xyz = normalize_rows(np.asarray(cache.get("gaze_world", cache["gaze_xyz"]), dtype=np.float64))
        timestamps_ns = np.asarray(cache.get("timestamps_ns", cache.get("frame_timestamps_ns", np.arange(n))), dtype=np.int64)
    else:
        gt_xyz = common_meta["gt_xyz"]
        timestamps_ns = common_meta["timestamps_ns"]
        anchor_mask = common_meta["anchor_mask"]
        eval_mask = common_meta["eval_mask"]

    n = len(gt_xyz)
    tracks = [Track(t.label, t.pred_xyz[:n], t.source, t.angular_error_deg) for t in tracks]
    hagi_record: dict[str, np.ndarray] | None = None
    if args.hagi_data_path.exists():
        try:
            hagi_record = load_hagi_record(args.hagi_data_path, args.sequence)
        except Exception as exc:
            missing_predictions.append(f"HAGI++ data: {exc}")

    if not args.no_hagi:
        if args.phase != 0:
            missing_predictions.append("HAGI++: available low-framerate result assumes phase=0")
        else:
            hagi_npz = args.hagi_npz or default_hagi_npz(args.hagi_output_root, args.target_hz)
            if hagi_npz.exists() and args.hagi_data_path.exists():
                try:
                    tracks.append(load_hagi_track(hagi_npz, args.hagi_data_path, args.sequence, n))
                except Exception as exc:
                    missing_predictions.append(f"HAGI++: {exc}")
            else:
                missing_predictions.append(f"HAGI++: missing {hagi_npz} or {args.hagi_data_path}")

    if args.include_synthetic_baselines and not args.no_synthetic_baselines:
        tracks.extend(synthetic_tracks(cache, gt_xyz, anchor_mask))

    if not tracks:
        raise RuntimeError("No prediction tracks available.")

    rows_mask = frame_window_mask(n, args.start_frame, args.end_frame)
    errors: dict[str, np.ndarray] = {}
    for track in tracks:
        if track.angular_error_deg is None:
            errors[track.label] = angular_error_deg(track.pred_xyz[:n], gt_xyz)
        else:
            track_error = np.full(n, np.nan, dtype=np.float64)
            count = min(n, len(track.angular_error_deg))
            track_error[:count] = track.angular_error_deg[:count]
            errors[track.label] = track_error
    out_dir = args.output_root / args.sequence / f"hz{args.target_hz}_phase{args.phase}_frames_{args.start_frame}_{args.end_frame}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "missing_gaze_direction_errors.csv"
    timeline_path = out_dir / "missing_gaze_direction_error_timeline.png"
    scanpath_path = out_dir / "missing_gaze_direction_3d_scanpath.png"
    pitch_yaw_path = out_dir / "cpf_pitch_yaw_timeline.png"
    write_error_csv(
        csv_path,
        gt_xyz=gt_xyz,
        timestamps_ns=timestamps_ns,
        anchor_mask=anchor_mask,
        eval_mask=eval_mask,
        tracks=tracks,
        errors=errors,
        rows_mask=rows_mask,
    )
    plot_error_timeline(
        timeline_path,
        sequence=args.sequence,
        target_hz=args.target_hz,
        phase=args.phase,
        tracks=tracks,
        errors=errors,
        anchor_mask=anchor_mask,
        eval_mask=eval_mask,
        rows_mask=rows_mask,
    )

    head_pos = np.asarray(cache["head_pos_xyz"], dtype=np.float64)[:n]
    if hagi_record is not None:
        plot_cpf_pitch_yaw_timeline(
            pitch_yaw_path,
            sequence=args.sequence,
            target_hz=args.target_hz,
            phase=args.phase,
            gt_xyz=gt_xyz,
            tracks=tracks,
            hagi_record=hagi_record,
            anchor_mask=anchor_mask,
            eval_mask=eval_mask,
            rows_mask=rows_mask,
        )
    if not args.skip_3d:
        plot_3d_scanpath(
            scanpath_path,
            sequence=args.sequence,
            target_hz=args.target_hz,
            phase=args.phase,
            gt_xyz=gt_xyz,
            head_pos=head_pos,
            tracks=tracks,
            eval_mask=eval_mask,
            rows_mask=rows_mask,
            ray_length=args.ray_length,
            max_points=args.max_3d_points,
        )

    print(f"tracks: {[track.label for track in tracks]}")
    print(f"missing eval frames in window: {int(np.sum(rows_mask & eval_mask))}")
    for track in tracks:
        valid_error = rows_mask & eval_mask & np.isfinite(errors[track.label])
        if np.any(valid_error):
            print(
                f"{track.label}: valid={int(np.sum(valid_error))}, "
                f"mean_error_deg={float(np.nanmean(errors[track.label][valid_error])):.3f}"
            )
        else:
            print(f"{track.label}: valid=0 in selected window")
    if missing_predictions:
        print("missing prediction tracks:")
        for item in missing_predictions:
            print(f"  {item}")
    print(f"csv: {csv_path}")
    print(f"timeline: {timeline_path}")
    if hagi_record is not None:
        print(f"cpf_pitch_yaw: {pitch_yaw_path}")
    if not args.skip_3d:
        print(f"scanpath_3d: {scanpath_path}")


if __name__ == "__main__":
    main()
