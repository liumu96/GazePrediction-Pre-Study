"""Paper-facing SparseGaze missing-frame analyses.

This script collects the analyses that are stable enough to support paper
writing: model ablations, frequency sensitivity, long-gap/event subsets,
GT-depth scene projected point error, and simple scanpath dynamics metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.prediction_plots import dataframe_to_markdown, existing  # noqa: E402
from analysis.prediction_results import (  # noqa: E402
    angular_error_deg,
    attach_anchor_gap_columns,
    discover_prediction_files,
    load_many_prediction_frames,
    load_prediction_frame,
    normalize_vectors,
)
from visualization.adt_hagi_sparsegaze_compare import (  # noqa: E402
    DEFAULT_ADT_DATA,
    DEFAULT_HAGI_DIR,
    load_adt_data,
    load_hagi_primary,
)


DEFAULT_EVAL_ROOT = Path("/home/liumu/Github_Projects/SparseGaze/outputs/eval/adt")
DEFAULT_REPORTS_DIR = Path("/mnt/d/SparseGaze/ADT-Gaze-structured")
DEFAULT_MODELS = [
    "sparsegaze_cpf_gaze_only_ss",
    "sparsegaze_cpf_local_head_motion_ss",
    "sparsegaze_cpf_rotation_only_ss",
    "sparsegaze_cpf_rotation_translation_ss",
    "sparsegaze_cpf_forward_head_motion_ss",
    "sparsegaze_cpf_forward_head_motion_residual_ss",
]
MODEL_LABELS = {
    "HAGI++": "HAGI++",
    "sparsegaze_cpf_gaze_only_ss": "Gaze-only",
    "sparsegaze_cpf_local_head_motion_ss": "Local head",
    "sparsegaze_cpf_rotation_only_ss": "Rotation-only",
    "sparsegaze_cpf_rotation_translation_ss": "Rotation+translation",
    "sparsegaze_cpf_forward_head_motion_ss": "Forward head",
    "sparsegaze_cpf_forward_head_motion_residual_ss": "Residual",
}
EVENT_ORDER = ["fixation", "transition", "invalid", "unknown"]
GRID_ALPHA = 0.22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--adt-data", type=Path, default=DEFAULT_ADT_DATA)
    parser.add_argument("--hagi-dir", type=Path, default=DEFAULT_HAGI_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/paper_missing_results"))
    parser.add_argument("--model", action="append", dest="models", help="Model directory. Repeatable.")
    parser.add_argument("--target-hz", type=int, default=None, help="Restrict to one Hz. Default: all Hz.")
    parser.add_argument("--eval-kind", action="append", dest="eval_kinds", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--long-gap-threshold", type=float, default=0.6)
    parser.add_argument("--no-hagi", action="store_true")
    parser.add_argument("--all-available", action="store_true", help="Disable common-frame filtering.")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = args.models or DEFAULT_MODELS
    eval_kinds = args.eval_kinds or ["rollout"]

    prediction_files = discover_prediction_files(
        args.eval_root,
        target_hz=args.target_hz,
        models=models,
        eval_kinds=eval_kinds,
        split=args.split,
    )
    if prediction_files.empty:
        raise FileNotFoundError("No prediction NPZ files found for the selected models.")

    frames = load_many_prediction_frames(prediction_files, reports_dir=args.reports_dir)
    if not args.no_hagi:
        hagi_frames = load_hagi_frames_with_reference_schedule(
            prediction_files,
            reports_dir=args.reports_dir,
            hagi_dir=args.hagi_dir,
            adt_data_path=args.adt_data,
            split=args.split,
        )
        if not hagi_frames.empty:
            frames = pd.concat([frames, hagi_frames], ignore_index=True)

    frames = attach_scene_gt_depth_points(frames, reports_dir=args.reports_dir, adt_data_path=args.adt_data)
    if not args.all_available:
        frames = keep_common_missing_frames(frames)

    evaluated = evaluated_missing_gap_frames(frames)
    overall = summarize_subset(evaluated, subset_name="overall")
    event = summarize_event(evaluated)
    long_gap = summarize_subset(
        evaluated[evaluated["normalized_gap_position"] >= args.long_gap_threshold],
        subset_name=f"long_gap_ge_{args.long_gap_threshold:.2f}",
    )
    frequency = summarize_frequency(evaluated)
    scene_point = summarize_scene_point(evaluated)
    scanpath_sequence, scanpath = summarize_scanpath(evaluated)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_files.to_csv(args.output_dir / "prediction_files.csv", index=False)
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False)
    event.to_csv(args.output_dir / "event_summary.csv", index=False)
    long_gap.to_csv(args.output_dir / "long_gap_summary.csv", index=False)
    frequency.to_csv(args.output_dir / "frequency_summary.csv", index=False)
    scene_point.to_csv(args.output_dir / "scene_point_summary.csv", index=False)
    scanpath_sequence.to_csv(args.output_dir / "scanpath_sequence_summary.csv", index=False)
    scanpath.to_csv(args.output_dir / "scanpath_summary.csv", index=False)

    figure_paths: dict[str, Path] = {}
    if not args.no_figures:
        figure_paths = write_figures(
            overall=overall,
            event=event,
            long_gap=long_gap,
            frequency=frequency,
            scene_point=scene_point,
            scanpath=scanpath,
            output_dir=args.output_dir,
        )
    write_report(
        overall=overall,
        event=event,
        long_gap=long_gap,
        frequency=frequency,
        scene_point=scene_point,
        scanpath=scanpath,
        figure_paths=figure_paths,
        output_path=args.output_dir / "REPORT.md",
    )
    config = {
        "eval_root": str(args.eval_root),
        "reports_dir": str(args.reports_dir),
        "adt_data": str(args.adt_data),
        "hagi_dir": str(args.hagi_dir),
        "output_dir": str(args.output_dir),
        "models": models,
        "eval_kinds": eval_kinds,
        "target_hz": args.target_hz,
        "common_frames": not args.all_available,
        "hagi": not args.no_hagi,
        "long_gap_threshold": args.long_gap_threshold,
        "n_prediction_files": len(prediction_files),
        "n_sequences": int(prediction_files["sequence"].nunique()),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(f"prediction_files: {len(prediction_files)}")
    print(f"sequences: {config['n_sequences']}")
    print(f"evaluated_rows: {len(evaluated)}")
    print(f"report: {args.output_dir / 'REPORT.md'}")


def load_hagi_frames_with_reference_schedule(
    prediction_files: pd.DataFrame,
    *,
    reports_dir: Path,
    hagi_dir: Path,
    adt_data_path: Path,
    split: str,
) -> pd.DataFrame:
    """Load HAGI++ predictions and attach the SparseGaze anchor schedule."""

    adt_data = load_adt_data(adt_data_path)
    references = prediction_files.drop_duplicates(["sequence", "target_hz", "phase"])
    hagi_cache: dict[int, dict[str, Any]] = {}
    rows = []
    for _, item in references.iterrows():
        fps = int(item["target_hz"])
        sequence = str(item["sequence"])
        phase = int(item["phase"])
        if fps not in hagi_cache:
            hagi_cache[fps] = load_hagi_primary(fps, hagi_dir=hagi_dir)
        hagi = hagi_cache[fps]
        mask = hagi["sequence_name"].astype(str) == sequence
        if not np.any(mask):
            continue
        frame_indices = hagi["frame_index"][mask].astype(int)
        pred = hagi["pred"][mask].astype(float)
        gt = hagi["gt"][mask].astype(float)
        record = adt_data.get(sequence)
        if record is not None:
            rotations = np.asarray(record["T_world_CPF"], dtype=float)[:, :3, :3]
            valid = (frame_indices >= 0) & (frame_indices < len(rotations))
            pred_scene = pred.copy()
            gt_scene = gt.copy()
            pred_scene[valid] = np.einsum("tij,tj->ti", rotations[frame_indices[valid]], pred[valid])
            gt_scene[valid] = np.einsum("tij,tj->ti", rotations[frame_indices[valid]], gt[valid])
            pred = normalize_vectors(pred_scene)
            gt = normalize_vectors(gt_scene)
        reference = load_prediction_frame(Path(item["path"]), reports_dir=reports_dir)
        keep_cols = [
            "frame_index",
            "anchor_mask",
            "prev_anchor_frame",
            "next_anchor_frame",
            "anchor_interval_frames",
            "missing_interval_frames",
            "frames_since_prev_anchor",
            "frames_until_next_anchor",
            "nearest_anchor_distance_frames",
            "normalized_gap_position",
            "anchor_gap_valid",
        ]
        if "scene_event_label" in reference:
            keep_cols.append("scene_event_label")
        hagi_frame = pd.DataFrame(
            {
                "frame_index": frame_indices,
                "pred_x": pred[:, 0],
                "pred_y": pred[:, 1],
                "pred_z": pred[:, 2],
                "gt_x": gt[:, 0],
                "gt_y": gt[:, 1],
                "gt_z": gt[:, 2],
                "angular_error_deg": hagi["angular_error_deg"][mask].astype(float),
            }
        )
        hagi_frame = hagi_frame.merge(reference[keep_cols], on="frame_index", how="inner")
        if hagi_frame.empty:
            continue
        hagi_frame["eval_mask"] = True
        hagi_frame["model"] = "HAGI++"
        hagi_frame["split"] = split
        hagi_frame["eval_kind"] = "hagi++"
        hagi_frame["target_hz"] = fps
        hagi_frame["phase"] = phase
        hagi_frame["sequence"] = sequence
        hagi_frame["direction_frame"] = "scene"
        rows.append(hagi_frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def attach_scene_gt_depth_points(
    frames: pd.DataFrame,
    *,
    reports_dir: Path,
    adt_data_path: Path,
) -> pd.DataFrame:
    """Attach GT depth/origin/point and compute GT-depth scene point error."""

    if frames.empty:
        return frames
    adt_data = load_adt_data(adt_data_path)
    out = frames.copy()
    out["direction_frame"] = out.get("direction_frame", "scene")
    merged = []
    for sequence, group in out.groupby("sequence", sort=False):
        gaze_path = reports_dir / "sequences" / sequence / "gaze" / "gaze_samples.csv"
        if not gaze_path.exists():
            group = group.copy()
            group["scene_point_error_m"] = np.nan
            merged.append(group)
            continue
        gaze = pd.read_csv(gaze_path)
        gaze = gaze.reset_index(names="frame_index")
        cols = [
            "frame_index",
            "depth_m",
            "gaze_origin_scene_x_m",
            "gaze_origin_scene_y_m",
            "gaze_origin_scene_z_m",
            "gaze_point_scene_x_m",
            "gaze_point_scene_y_m",
            "gaze_point_scene_z_m",
        ]
        group = group.merge(gaze[cols], on="frame_index", how="left")
        group = compute_scene_point_error(group, adt_data.get(sequence))
        merged.append(group)
    return pd.concat(merged, ignore_index=True)


def compute_scene_point_error(group: pd.DataFrame, record: dict[str, Any] | None) -> pd.DataFrame:
    """Compute predicted scene point at GT depth and compare with GT gaze point."""

    output = group.copy()
    pred = normalize_vectors(output[["pred_x", "pred_y", "pred_z"]].to_numpy(dtype=float))
    frame_kind = output["direction_frame"].fillna("scene").astype(str).to_numpy()
    if record is not None and np.any(frame_kind == "cpf"):
        rotations = np.asarray(record["T_world_CPF"], dtype=float)[:, :3, :3]
        frames = output["frame_index"].to_numpy(dtype=int)
        valid = (frame_kind == "cpf") & (frames >= 0) & (frames < len(rotations))
        pred_scene = pred.copy()
        pred_scene[valid] = np.einsum("tij,tj->ti", rotations[frames[valid]], pred[valid])
        pred = normalize_vectors(pred_scene)

    origin = output[
        ["gaze_origin_scene_x_m", "gaze_origin_scene_y_m", "gaze_origin_scene_z_m"]
    ].to_numpy(dtype=float)
    gt_point = output[
        ["gaze_point_scene_x_m", "gaze_point_scene_y_m", "gaze_point_scene_z_m"]
    ].to_numpy(dtype=float)
    depth = output["depth_m"].to_numpy(dtype=float)
    pred_point = origin + pred * depth[:, None]
    valid = np.isfinite(pred_point).all(axis=1) & np.isfinite(gt_point).all(axis=1)
    error = np.full(len(output), np.nan, dtype=float)
    error[valid] = np.linalg.norm(pred_point[valid] - gt_point[valid], axis=1)
    output["scene_point_error_m"] = error
    return output


def keep_common_missing_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Keep evaluated missing gap frames shared by every method per sequence/frequency."""

    eligible = evaluated_missing_gap_frames(frames)
    if eligible.empty:
        return eligible
    kept = []
    method_cols = ["model", "eval_kind"]
    for _, group in eligible.groupby(["sequence", "target_hz", "phase"], sort=False):
        methods = list(group[method_cols].drop_duplicates().itertuples(index=False, name=None))
        frame_sets = [
            set(
                group.loc[
                    (group["model"] == model) & (group["eval_kind"] == eval_kind),
                    "frame_index",
                ].astype(int)
            )
            for model, eval_kind in methods
        ]
        common = set.intersection(*frame_sets) if frame_sets else set()
        if common:
            kept.append(group[group["frame_index"].isin(common)])
    return pd.concat(kept, ignore_index=True) if kept else eligible.iloc[0:0].copy()


def evaluated_missing_gap_frames(frames: pd.DataFrame) -> pd.DataFrame:
    if frames.empty:
        return frames
    if "anchor_gap_valid" not in frames:
        frames = attach_anchor_gap_columns(frames)
    return frames[
        frames["eval_mask"].astype(bool)
        & ~frames["anchor_mask"].astype(bool)
        & frames["anchor_gap_valid"].astype(bool)
    ].copy()


def method_keys() -> list[str]:
    return ["model", "eval_kind", "target_hz", "phase"]


def summarize_subset(data: pd.DataFrame, *, subset_name: str) -> pd.DataFrame:
    cols = [
        "subset",
        *method_keys(),
        "method_label",
        "sequence_n",
        "frame_n",
        "frame_weighted_mae_deg",
        "sequence_macro_mae_deg",
        "median_deg",
        "p90_deg",
    ]
    if data.empty:
        return pd.DataFrame(columns=cols)
    seq = (
        data.groupby([*method_keys(), "sequence"], as_index=False, sort=False)
        .agg(mae_deg=("angular_error_deg", "mean"))
    )
    rows = []
    for key, group in data.groupby(method_keys(), sort=False):
        seq_group = seq
        for col, value in zip(method_keys(), key):
            seq_group = seq_group[seq_group[col] == value]
        rows.append(
            {
                "subset": subset_name,
                **dict(zip(method_keys(), key)),
                "method_label": model_label(key[0]),
                "sequence_n": int(seq_group["sequence"].nunique()),
                "frame_n": int(len(group)),
                "frame_weighted_mae_deg": float(group["angular_error_deg"].mean()),
                "sequence_macro_mae_deg": float(seq_group["mae_deg"].mean()),
                "median_deg": float(group["angular_error_deg"].median()),
                "p90_deg": float(np.percentile(group["angular_error_deg"], 90)),
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values(["target_hz", "sequence_macro_mae_deg"])


def summarize_event(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "scene_event_label" not in data:
        return pd.DataFrame()
    rows = []
    for event, group in data.groupby("scene_event_label", sort=False):
        summary = summarize_subset(group, subset_name=f"event_{event}")
        summary.insert(1, "scene_event_label", event)
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    out["event_order"] = out["scene_event_label"].map(
        {event: i for i, event in enumerate(EVENT_ORDER)}
    ).fillna(len(EVENT_ORDER))
    return out.sort_values(["target_hz", "event_order", "sequence_macro_mae_deg"]).drop(columns="event_order")


def summarize_frequency(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    summary = summarize_subset(data, subset_name="frequency")
    return summary.sort_values(["model", "eval_kind", "target_hz"])


def summarize_scene_point(data: pd.DataFrame) -> pd.DataFrame:
    cols = [
        *method_keys(),
        "method_label",
        "sequence_n",
        "frame_n",
        "mean_scene_point_error_m",
        "median_scene_point_error_m",
        "p90_scene_point_error_m",
    ]
    data = data.dropna(subset=["scene_point_error_m"])
    if data.empty:
        return pd.DataFrame(columns=cols)
    seq = (
        data.groupby([*method_keys(), "sequence"], as_index=False, sort=False)
        .agg(mean_scene_point_error_m=("scene_point_error_m", "mean"))
    )
    rows = []
    for key, group in data.groupby(method_keys(), sort=False):
        seq_group = seq
        for col, value in zip(method_keys(), key):
            seq_group = seq_group[seq_group[col] == value]
        rows.append(
            {
                **dict(zip(method_keys(), key)),
                "method_label": model_label(key[0]),
                "sequence_n": int(seq_group["sequence"].nunique()),
                "frame_n": int(len(group)),
                "mean_scene_point_error_m": float(seq_group["mean_scene_point_error_m"].mean()),
                "median_scene_point_error_m": float(group["scene_point_error_m"].median()),
                "p90_scene_point_error_m": float(np.percentile(group["scene_point_error_m"], 90)),
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values(["target_hz", "mean_scene_point_error_m"])


def summarize_scanpath(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize simple scanpath dynamics on contiguous evaluated frame runs."""

    rows = []
    for key, group in data.groupby([*method_keys(), "sequence"], sort=False):
        group = group.sort_values("frame_index")
        pred = group[["pred_x", "pred_y", "pred_z"]].to_numpy(dtype=float)
        gt = group[["gt_x", "gt_y", "gt_z"]].to_numpy(dtype=float)
        frames = group["frame_index"].to_numpy(dtype=int)
        breaks = np.where(np.diff(frames) > 1)[0] + 1
        starts = np.r_[0, breaks]
        ends = np.r_[breaks, len(frames)]
        pred_steps = []
        gt_steps = []
        for start, end in zip(starts, ends):
            if end - start < 2:
                continue
            pred_steps.append(angular_error_deg(pred[start + 1 : end], pred[start : end - 1]))
            gt_steps.append(angular_error_deg(gt[start + 1 : end], gt[start : end - 1]))
        if not pred_steps:
            continue
        pred_step = np.concatenate(pred_steps)
        gt_step = np.concatenate(gt_steps)
        pred_path = float(pred_step.sum())
        gt_path = float(gt_step.sum())
        rows.append(
            {
                **dict(zip([*method_keys(), "sequence"], key)),
                "method_label": model_label(key[0]),
                "step_n": int(len(pred_step)),
                "pred_path_length_deg": pred_path,
                "gt_path_length_deg": gt_path,
                "path_length_ratio": pred_path / gt_path if gt_path > 1e-9 else np.nan,
                "step_magnitude_mae_deg": float(np.mean(np.abs(pred_step - gt_step))),
                "pred_mean_step_deg": float(pred_step.mean()),
                "gt_mean_step_deg": float(gt_step.mean()),
            }
        )
    seq = pd.DataFrame(rows)
    if seq.empty:
        return seq, pd.DataFrame()
    summary = (
        seq.groupby(method_keys(), as_index=False, sort=False)
        .agg(
            method_label=("method_label", "first"),
            sequence_n=("sequence", "nunique"),
            step_n=("step_n", "sum"),
            mean_path_length_ratio=("path_length_ratio", "mean"),
            median_path_length_ratio=("path_length_ratio", "median"),
            mean_step_magnitude_mae_deg=("step_magnitude_mae_deg", "mean"),
            mean_pred_step_deg=("pred_mean_step_deg", "mean"),
            mean_gt_step_deg=("gt_mean_step_deg", "mean"),
        )
        .sort_values(["target_hz", "mean_step_magnitude_mae_deg"])
    )
    return seq, summary


def write_figures(
    *,
    overall: pd.DataFrame,
    event: pd.DataFrame,
    long_gap: pd.DataFrame,
    frequency: pd.DataFrame,
    scene_point: pd.DataFrame,
    scanpath: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "overall_ablation": figure_dir / "overall_ablation.png",
        "event_ablation": figure_dir / "event_ablation.png",
        "long_gap_ablation": figure_dir / "long_gap_ablation.png",
        "frequency_sensitivity": figure_dir / "frequency_sensitivity.png",
        "scene_point_error": figure_dir / "scene_point_error.png",
        "scanpath_metrics": figure_dir / "scanpath_metrics.png",
    }
    plot_bar_metric(overall, "sequence_macro_mae_deg", paths["overall_ablation"], "Overall missing-frame MAE")
    plot_event_metric(event, paths["event_ablation"])
    plot_bar_metric(long_gap, "sequence_macro_mae_deg", paths["long_gap_ablation"], "Long-gap missing-frame MAE")
    plot_frequency(frequency, paths["frequency_sensitivity"])
    plot_bar_metric(scene_point, "mean_scene_point_error_m", paths["scene_point_error"], "GT-depth scene point error")
    plot_scanpath(scanpath, paths["scanpath_metrics"])
    return paths


def plot_bar_metric(data: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.58 * len(data)), 4.8))
    if data.empty or metric not in data:
        draw_empty(ax, "No data")
    else:
        plot_data = data[data["target_hz"] == data["target_hz"].min()].copy()
        plot_data = plot_data.sort_values(metric)
        labels = [model_label(m) for m in plot_data["model"]]
        ax.bar(np.arange(len(plot_data)), plot_data[metric], color="#4C78A8")
        ax.set_xticks(np.arange(len(plot_data)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{title} ({int(plot_data['target_hz'].iloc[0])}Hz)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(axis="y", alpha=GRID_ALPHA)
    save(fig, path)


def plot_event_metric(event: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if event.empty:
        draw_empty(ax, "No event data")
    else:
        data = event[
            (event["target_hz"] == event["target_hz"].min())
            & event["scene_event_label"].isin(["fixation", "transition"])
        ].copy()
        methods = list(dict.fromkeys(data["model"]))
        events = ["fixation", "transition"]
        x = np.arange(len(methods), dtype=float)
        width = 0.36
        for idx, event_name in enumerate(events):
            vals = []
            for model in methods:
                row = data[(data["model"] == model) & (data["scene_event_label"] == event_name)]
                vals.append(np.nan if row.empty else float(row["sequence_macro_mae_deg"].iloc[0]))
            ax.bar(x + (idx - 0.5) * width, vals, width=width, label=event_name)
        ax.set_xticks(x)
        ax.set_xticklabels([model_label(m) for m in methods], rotation=30, ha="right")
        ax.set_ylabel("Sequence-macro MAE [deg]")
        ax.set_title("Event-conditioned missing-frame MAE")
        ax.grid(axis="y", alpha=GRID_ALPHA)
        ax.legend()
    save(fig, path)


def plot_frequency(frequency: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if frequency.empty:
        draw_empty(ax, "No frequency data")
    else:
        for model, group in frequency.groupby("model", sort=False):
            group = group.sort_values("target_hz")
            ax.plot(group["target_hz"], group["sequence_macro_mae_deg"], marker="o", label=model_label(model))
        ax.set_xlabel("Target gaze rate [Hz]")
        ax.set_ylabel("Sequence-macro MAE [deg]")
        ax.set_title("Frequency sensitivity")
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(fontsize=8)
    save(fig, path)


def plot_scanpath(scanpath: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if scanpath.empty:
        for ax in axes:
            draw_empty(ax, "No scanpath data")
    else:
        data = scanpath[scanpath["target_hz"] == scanpath["target_hz"].min()].copy()
        data = data.sort_values("mean_step_magnitude_mae_deg")
        labels = [model_label(m) for m in data["model"]]
        x = np.arange(len(data))
        axes[0].bar(x, data["mean_step_magnitude_mae_deg"], color="#4C78A8")
        axes[0].set_title("Step magnitude MAE")
        axes[0].set_ylabel("deg/frame")
        axes[1].bar(x, data["mean_path_length_ratio"], color="#F58518")
        axes[1].axhline(1.0, color="black", linewidth=0.9)
        axes[1].set_title("Pred/GT path length ratio")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.grid(axis="y", alpha=GRID_ALPHA)
    save(fig, path)


def write_report(
    *,
    overall: pd.DataFrame,
    event: pd.DataFrame,
    long_gap: pd.DataFrame,
    frequency: pd.DataFrame,
    scene_point: pd.DataFrame,
    scanpath: pd.DataFrame,
    figure_paths: dict[str, Path],
    output_path: Path,
) -> None:
    lines = [
        "# SparseGaze Paper Missing-Frame Results",
        "",
        "This report aggregates paper-facing analyses computed from per-sequence prediction NPZ files.",
        "All primary comparisons use evaluated missing frames inside valid anchor gaps. By default, rows are restricted to frames shared by every selected method.",
        "",
        "## Figures",
        "",
    ]
    for key, path in figure_paths.items():
        if path.exists():
            lines.extend([f"### {key}", "", f"![{key}]({path.relative_to(output_path.parent).as_posix()})", ""])

    tables = [
        ("Overall Missing-Frame Summary", overall),
        ("Event Summary", event[event["scene_event_label"].isin(["fixation", "transition"])] if not event.empty else event),
        ("Long-Gap Summary", long_gap),
        ("Frequency Summary", frequency),
        ("GT-Depth Scene Point Summary", scene_point),
        ("Scanpath Summary", scanpath),
    ]
    for title, table in tables:
        lines.extend(["", f"## {title}", ""])
        if table.empty:
            lines.append("No rows.")
            continue
        cols = existing(
            [
                "subset",
                "scene_event_label",
                "model",
                "method_label",
                "eval_kind",
                "target_hz",
                "phase",
                "sequence_n",
                "frame_n",
                "sequence_macro_mae_deg",
                "median_deg",
                "p90_deg",
                "mean_scene_point_error_m",
                "p90_scene_point_error_m",
                "mean_step_magnitude_mae_deg",
                "mean_path_length_ratio",
            ],
            table,
        )
        lines.extend(dataframe_to_markdown(table[cols].head(120)))

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `scene_point_error_m` is a GT-depth projected endpoint diagnostic; it does not mean the model predicts true fixation depth.",
            "- Scanpath metrics are simple contiguous-frame dynamics summaries, not DTW/Frechet distances.",
            "- `rollout_gt` or other oracle-like repair modes should not be interpreted as deployable model results.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model).replace("sparsegaze_cpf_", "").replace("_ss", ""))


def draw_empty(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()


def save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
