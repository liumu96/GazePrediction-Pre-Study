"""Visualization adapters for per-sequence prediction NPZ gaze outputs.

SparseGaze evaluation NPZ files store `pred_xyz` and `gt_xyz` as Scene-frame
unit gaze directions.  They do not store ADT depth-defined gaze points.  To
reuse the CSV-style visualization path, this module combines NPZ directions
with extracted `gaze_samples.csv` origins and depth values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adt_sandbox.gaze import GazeSample, project_scene_points_to_rgb, read_samples_csv
from adt_sandbox.results import find_sequence_file
from visualization.gaze_outputs import (
    reference_scanpath_from_samples,
    write_overlay_frames,
    write_reference_frame_scanpath_clean,
    write_reference_frame_scanpath_overlay,
    write_scene_rays_plot,
    write_video_from_overlay_frames,
)
from visualization.prediction_eval import normalize_vectors


def load_prediction_npz(npz_path: Path) -> dict[str, Any]:
    """Load one prediction NPZ into plain arrays and metadata."""

    with np.load(npz_path, allow_pickle=True) as data:
        return {
            "sequence": str(data["sequence_id"].item()),
            "target_hz": int(data["target_hz"].item()),
            "phase": int(data["phase"].item()),
            "timestamps_ns": np.asarray(data["timestamps_ns"], dtype=np.int64),
            "pred_xyz": normalize_vectors(np.asarray(data["pred_xyz"], dtype=np.float64)),
            "gt_xyz": normalize_vectors(np.asarray(data["gt_xyz"], dtype=np.float64)),
            "eval_mask": np.asarray(data["eval_mask"], dtype=bool),
            "anchor_mask": np.asarray(data["anchor_mask"], dtype=bool),
        }


def load_context_samples(reports_dir: Path, sequence: str) -> list[GazeSample]:
    """Load extracted gaze samples that provide origin/depth/image metadata."""

    csv_path = find_sequence_file(reports_dir, sequence, "gaze", "gaze_samples.csv")
    return read_samples_csv(csv_path)


def prediction_samples_from_npz(
    *,
    gt_provider: Any,
    npz_path: Path,
    reports_dir: Path,
    track: str,
    depth_mode: str,
    fixed_depth_m: float,
    stream_id_value: str = "214-1",
    make_upright: bool = True,
) -> list[GazeSample]:
    """Build `GazeSample` rows from an NPZ prediction or GT direction track.

    `depth_mode` controls where the ray endpoint is placed:
    - `gt_depth`: use the extracted ADT gaze depth for each timestamp;
    - `fixed`: use the same fixed metric length for every ray.
    """

    prediction = load_prediction_npz(npz_path)
    sequence = prediction["sequence"]
    directions = prediction[f"{track}_xyz"]
    timestamps = prediction["timestamps_ns"]
    context_by_timestamp = {
        int(sample.query_timestamp_ns): sample
        for sample in load_context_samples(reports_dir, sequence)
    }

    raw_samples: list[GazeSample] = []
    scene_points: list[np.ndarray] = []
    for index, timestamp_ns in enumerate(timestamps):
        context = context_by_timestamp.get(int(timestamp_ns))
        if context is None or context.gaze_origin_scene_x_m is None:
            continue
        origin = np.array(
            [
                context.gaze_origin_scene_x_m,
                context.gaze_origin_scene_y_m,
                context.gaze_origin_scene_z_m,
            ],
            dtype=np.float64,
        )
        direction = directions[index]
        depth = resolve_ray_depth(context, depth_mode=depth_mode, fixed_depth_m=fixed_depth_m)
        point = origin + direction * depth
        scene_points.append(point)
        raw_samples.append(
            GazeSample(
                query_timestamp_ns=int(timestamp_ns),
                gaze_valid=True,
                gaze_dt_ns=context.gaze_dt_ns,
                yaw_rad=None,
                pitch_rad=None,
                depth_m=float(depth),
                gaze_dir_cpf_unit_x=None,
                gaze_dir_cpf_unit_y=None,
                gaze_dir_cpf_unit_z=None,
                yaw_confidence_width_rad=None,
                pitch_confidence_width_rad=None,
                projection_valid=False,
                gaze_u_px=None,
                gaze_v_px=None,
                projection_in_image=False,
                image_width_px=context.image_width_px,
                image_height_px=context.image_height_px,
                pose_valid=context.pose_valid,
                pose_dt_ns=context.pose_dt_ns,
                pose_quality_score=context.pose_quality_score,
                gaze_origin_scene_x_m=float(origin[0]),
                gaze_origin_scene_y_m=float(origin[1]),
                gaze_origin_scene_z_m=float(origin[2]),
                gaze_point_scene_x_m=float(point[0]),
                gaze_point_scene_y_m=float(point[1]),
                gaze_point_scene_z_m=float(point[2]),
                gaze_dir_scene_unit_x=float(direction[0]),
                gaze_dir_scene_unit_y=float(direction[1]),
                gaze_dir_scene_unit_z=float(direction[2]),
                validation_notes=f"npz_{track};depth_mode={depth_mode}",
            )
        )

    projected_samples: list[GazeSample] = []
    for sample, point in zip(raw_samples, scene_points, strict=True):
        projection, image_size = project_scene_points_to_rgb(
            gt_provider,
            [point],
            sample.query_timestamp_ns,
            stream_id_value=stream_id_value,
            make_upright=make_upright,
        )
        u_v = projection[0]
        width, height = image_size
        in_image = (
            u_v is not None
            and 0 <= float(u_v[0]) < width
            and 0 <= float(u_v[1]) < height
        )
        projected_samples.append(
            GazeSample(
                **{
                    **sample.as_csv_row(),
                    "projection_valid": u_v is not None,
                    "gaze_u_px": float(u_v[0]) if u_v is not None else None,
                    "gaze_v_px": float(u_v[1]) if u_v is not None else None,
                    "projection_in_image": bool(in_image),
                    "image_width_px": width,
                    "image_height_px": height,
                }
            )
        )
    return projected_samples


def resolve_ray_depth(context: GazeSample, depth_mode: str, fixed_depth_m: float) -> float:
    """Choose a metric endpoint distance for a prediction ray."""

    if depth_mode == "fixed":
        return float(fixed_depth_m)
    if depth_mode == "gt_depth":
        if context.depth_m is not None and np.isfinite(context.depth_m) and context.depth_m > 0:
            return float(context.depth_m)
        return float(fixed_depth_m)
    raise ValueError(f"Unsupported depth_mode: {depth_mode}")


def generate_npz_gaze_visualizations(
    *,
    gt_provider: Any,
    npz_path: Path,
    reports_dir: Path,
    output_root: Path,
    track: str,
    start_row: int,
    end_row: int | None,
    stride: int,
    depth_mode: str = "gt_depth",
    fixed_depth_m: float = 2.0,
    stream_id_value: str = "214-1",
    make_upright: bool = True,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Generate CSV-style visualization outputs from an NPZ gaze direction track."""

    prediction = load_prediction_npz(npz_path)
    sequence = prediction["sequence"]
    samples = prediction_samples_from_npz(
        gt_provider=gt_provider,
        npz_path=npz_path,
        reports_dir=reports_dir,
        track=track,
        depth_mode=depth_mode,
        fixed_depth_m=fixed_depth_m,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    indexed = list(enumerate(samples))
    selected = indexed[start_row:end_row]
    if stride <= 0:
        raise ValueError("stride must be positive")
    viz_pairs = selected[::stride]
    if selected and viz_pairs[-1] != selected[-1]:
        viz_pairs.append(selected[-1])
    viz_orders = [idx for idx, _ in viz_pairs]
    viz_samples = [sample for _, sample in viz_pairs]
    if not viz_samples:
        raise ValueError("No NPZ gaze samples selected")

    end_label = end_row if end_row is not None else len(samples)
    resolved_run_name = run_name or (
        f"{npz_path.parent.parent.parent.name}_{track}_{depth_mode}_"
        f"rows_{start_row}_{end_label}_stride_{stride}"
    )
    output_dir = output_root / sequence / "npz_visualizations" / resolved_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    write_scene_rays_plot(output_dir / f"{track}_scene_rays.png", viz_samples)
    scanpath = reference_scanpath_from_samples(
        gt_provider,
        viz_samples,
        viz_orders,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    write_reference_frame_scanpath_overlay(
        output_dir / f"{track}_reference_frame_scanpath_overlay.png",
        scanpath,
    )
    write_reference_frame_scanpath_clean(
        output_dir / f"{track}_reference_frame_scanpath_clean.png",
        scanpath,
    )
    overlay_paths = write_overlay_frames(
        gt_provider,
        viz_pairs,
        output_dir / f"{track}_overlays",
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    write_video_from_overlay_frames(output_dir / f"{track}_overlay_video.mp4", overlay_paths)
    return {
        "output_dir": output_dir,
        "sequence": sequence,
        "track": track,
        "depth_mode": depth_mode,
        "window_samples": len(selected),
        "viz_samples": len(viz_samples),
    }


def prediction_metadata_table(npz_path: Path) -> pd.DataFrame:
    """Return a one-row table with useful NPZ metadata."""

    prediction = load_prediction_npz(npz_path)
    return pd.DataFrame(
        [
            {
                "sequence": prediction["sequence"],
                "target_hz": prediction["target_hz"],
                "phase": prediction["phase"],
                "n_frames": len(prediction["timestamps_ns"]),
                "n_eval": int(prediction["eval_mask"].sum()),
                "n_anchor": int(prediction["anchor_mask"].sum()),
            }
        ]
    )
