#!/usr/bin/env python
"""Compare GT, SparseGaze, and HAGI++ scanpaths in ADT Scene 3D space.

This script focuses on Scene-frame scanpath inspection.  GT gaze points come
from the structured ADT gaze CSV.  SparseGaze and HAGI++ provide gaze
directions, not depth; their 3D endpoints are therefore placed at the aligned
GT gaze depth:

    predicted_point = GT_origin + predicted_direction * GT_depth

The output includes static 3D figures, a dynamic scanpath-tail video, optional
Plotly HTML, and per-frame point/angular error CSVs.

conda run -n adt python "Experiments/visualization & Analysis/ADT/04_scene_scanpath_3d.py" \
  --start-frame 149 \
  --end-frame 300 \
  --stride 5 \
  --tail-frames 30 \
  --run-name gt_sparsegaze_hagi_scene3d_frames_149_300_stride5_tail30 \
  --no-plotly
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "src").exists() and (parent / "Experiments").exists():
            return parent
    return Path(__file__).resolve().parents[3]


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.results import find_sequence_file  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

SEQUENCE = "Apartment_release_decoration_skeleton_seq133_M1292"
DEFAULT_REPORTS_DIR = Path(
    os.environ.get("REPORTS_DIR", "/mnt/d/SparseGaze/ADT-Gaze-structured")
)
DEFAULT_SPARSEGAZE_NPZ = Path(
    "/home/liumu/Github_Projects/SparseGaze/outputs/eval/adt/sparsegaze/test/rollout/"
    f"sequence_predictions/{SEQUENCE}/hz6_phase0.npz"
)
DEFAULT_HAGI_NPZ = Path(
    "/home/liumu/Github_Projects/HAGI/save/head/hagi++_imputation/"
    "adt_low_framerate_sliding/sliding_primary_nsample20_framerate_6.npz"
)
DEFAULT_HAGI_ADT_DATA = Path(
    "/home/liumu/Github_Projects/HAGI/datasets/adt/gaze_head_adt.npy"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "figures" / "gaze_scene_scanpath"

BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)
TRACK_COLORS = {
    "GT": "#111111",
    "SparseGaze": "#ff7f0e",
    "HAGI++": "#2ca02c",
}


@dataclass(frozen=True)
class SceneContext:
    sequence: str
    gaze: pd.DataFrame
    head: pd.DataFrame | None
    objects: pd.DataFrame | None
    hits: pd.DataFrame | None
    skeleton: pd.DataFrame | None
    skeleton_summary: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        nargs="?",
        default=SEQUENCE,
        help="ADT sequence id in the structured reports directory.",
    )
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--prediction-npz", type=Path, default=DEFAULT_SPARSEGAZE_NPZ)
    parser.add_argument("--pred-key", default="pred_xyz")
    parser.add_argument("--hagi-npz", type=Path, default=DEFAULT_HAGI_NPZ)
    parser.add_argument("--hagi-adt-data", type=Path, default=DEFAULT_HAGI_ADT_DATA)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=149,
        help="Inclusive structured GT row/frame start. Default starts where HAGI++ exists.",
    )
    parser.add_argument("--end-frame", type=int, default=300)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--tail-frames",
        type=int,
        default=30,
        help="Dynamic video tail length in original GT row/frame units.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-plotly", action="store_true")
    parser.add_argument("--show-rays", action="store_true", help="Draw current-frame gaze rays.")
    parser.add_argument("--ray-length-m", type=float, default=1.25)
    parser.add_argument(
        "--mirror-horizontal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mirror display Scene X for all rendered objects/tracks. This fixes "
            "the left-right handedness mismatch against the RGB/image-space view."
        ),
    )
    parser.add_argument(
        "--vertical-axis",
        choices=["scene_x", "scene_y", "scene_z"],
        default="scene_y",
        help="Physical Scene axis to use as visual up. ADT Scene viewers use Scene Y-up.",
    )
    parser.add_argument("--view-elev", type=float, default=22.0)
    parser.add_argument("--view-azim", type=float, default=-58.0)
    parser.add_argument("--max-static-objects", type=int, default=80)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--exclude-category-filter", default="shelter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.tail_frames < 0:
        raise ValueError("--tail-frames must be non-negative")
    if args.end_frame <= args.start_frame:
        raise ValueError("--end-frame must be greater than --start-frame")
    args.effective_mirror_horizontal = bool(args.mirror_horizontal)

    context = load_scene_context(args.reports_dir, args.sequence)
    tracks, table = build_track_table(
        context=context,
        prediction_npz=args.prediction_npz,
        pred_key=args.pred_key,
        hagi_npz=args.hagi_npz,
        hagi_adt_data=args.hagi_adt_data,
    )
    window = select_window(table, args.start_frame, args.end_frame)
    viz_frames = downsample_frames(window["frame_index"].to_numpy(dtype=int), args.stride)

    run_name = args.run_name or default_run_name(
        args.start_frame,
        args.end_frame,
        args.stride,
        args.tail_frames,
    )
    output_dir = args.output_root / args.sequence / run_name
    frame_dir = output_dir / "scene_scanpath_3d_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    window.to_csv(output_dir / "scene_scanpath_points.csv", index=False)
    summary = build_summary(
        context=context,
        table=table,
        window=window,
        tracks=tracks,
        args=args,
        output_dir=output_dir,
    )
    write_json(output_dir / "scene_scanpath_summary.json", summary)

    object_rows = select_object_rows(
        context.objects,
        focus_timestamp_ns=int(window["query_timestamp_ns"].iloc[-1]),
        category_filter=args.category_filter,
        exclude_category_filter=args.exclude_category_filter,
        max_static_objects=args.max_static_objects,
    )

    static_fig = make_scene_scanpath_figure(
        context=context,
        rows=window,
        tracks=tracks,
        object_rows=object_rows,
        current_frame=int(window["frame_index"].iloc[-1]),
        tail_frames=args.end_frame - args.start_frame,
        title="Scene 3D scanpath comparison",
        show_rays=args.show_rays,
        ray_length_m=args.ray_length_m,
        mirror_horizontal=args.effective_mirror_horizontal,
        vertical_axis=args.vertical_axis,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )
    static_fig.savefig(output_dir / "scene_scanpath_3d.png", dpi=180)
    plt.close(static_fig)

    error_fig = make_error_figure(window)
    error_fig.savefig(output_dir / "scene_point_error_timeline.png", dpi=180)
    plt.close(error_fig)

    frame_paths: list[Path] = []
    if not args.no_video:
        frame_paths = write_dynamic_tail_frames(
            context=context,
            rows=window,
            tracks=tracks,
            frame_indices=viz_frames,
            output_dir=frame_dir,
            tail_frames=args.tail_frames,
            args=args,
        )
        write_video(output_dir / "scene_scanpath_3d_video.mp4", frame_paths, fps=args.video_fps)

    plotly_path: Path | None = None
    if not args.no_plotly:
        plotly_path = write_plotly_scene(
            output_dir / "scene_scanpath_interactive.html",
            context=context,
            rows=window,
            tracks=tracks,
            object_rows=object_rows,
            mirror_horizontal=args.effective_mirror_horizontal,
            vertical_axis=args.vertical_axis,
            view_elev=args.view_elev,
            view_azim=args.view_azim,
        )

    summary.update(
        {
            "static_figure": str(output_dir / "scene_scanpath_3d.png"),
            "error_figure": str(output_dir / "scene_point_error_timeline.png"),
            "video": str(output_dir / "scene_scanpath_3d_video.mp4")
            if frame_paths and not args.no_video
            else None,
            "video_frames": len(frame_paths),
            "plotly_html": str(plotly_path) if plotly_path is not None else None,
        }
    )
    write_json(output_dir / "scene_scanpath_summary.json", summary)

    print(f"sequence: {args.sequence}")
    print(f"reports_dir: {args.reports_dir}")
    print(f"frames: {args.start_frame}..{args.end_frame} stride={args.stride}")
    print(f"tracks: {', '.join(tracks)}")
    print(f"selected_rows: {len(window)}")
    print(f"figures: {output_dir}")
    if frame_paths and not args.no_video:
        print(f"video: {output_dir / 'scene_scanpath_3d_video.mp4'}")
    if plotly_path is not None:
        print(f"plotly: {plotly_path}")


def load_scene_context(reports_dir: Path, sequence: str) -> SceneContext:
    gaze = pd.read_csv(find_sequence_file(reports_dir, sequence, "gaze", "gaze_samples.csv"))
    gaze = gaze.copy()
    gaze["frame_index"] = np.arange(len(gaze), dtype=int)
    return SceneContext(
        sequence=sequence,
        gaze=gaze,
        head=read_optional_csv(reports_dir, sequence, "head", "head_samples.csv"),
        objects=read_optional_csv(reports_dir, sequence, "scene", "scene_object_boxes.csv"),
        hits=read_optional_csv(reports_dir, sequence, "scene", "gaze_object_hits.csv"),
        skeleton=read_optional_csv(reports_dir, sequence, "skeleton", "skeleton_samples.csv"),
        skeleton_summary=read_optional_json(
            reports_dir,
            sequence,
            "skeleton",
            "skeleton_summary.json",
        ),
    )


def read_optional_csv(
    reports_dir: Path,
    sequence: str,
    layer: str,
    filename: str,
) -> pd.DataFrame | None:
    try:
        return pd.read_csv(find_sequence_file(reports_dir, sequence, layer, filename))
    except FileNotFoundError:
        return None


def read_optional_json(
    reports_dir: Path,
    sequence: str,
    layer: str,
    filename: str,
) -> dict[str, Any] | None:
    try:
        path = find_sequence_file(reports_dir, sequence, layer, filename)
    except FileNotFoundError:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_track_table(
    *,
    context: SceneContext,
    prediction_npz: Path | None,
    pred_key: str,
    hagi_npz: Path | None,
    hagi_adt_data: Path,
) -> tuple[list[str], pd.DataFrame]:
    gaze = context.gaze
    table = pd.DataFrame(
        {
            "frame_index": gaze["frame_index"].astype(int),
            "query_timestamp_ns": gaze["query_timestamp_ns"].astype(np.int64),
            "depth_m": numeric(gaze["depth_m"]),
            "origin_x": numeric(gaze["gaze_origin_scene_x_m"]),
            "origin_y": numeric(gaze["gaze_origin_scene_y_m"]),
            "origin_z": numeric(gaze["gaze_origin_scene_z_m"]),
            "gt_x": numeric(gaze["gaze_point_scene_x_m"]),
            "gt_y": numeric(gaze["gaze_point_scene_y_m"]),
            "gt_z": numeric(gaze["gaze_point_scene_z_m"]),
            "gt_dir_x": numeric(gaze["gaze_dir_scene_unit_x"]),
            "gt_dir_y": numeric(gaze["gaze_dir_scene_unit_y"]),
            "gt_dir_z": numeric(gaze["gaze_dir_scene_unit_z"]),
        }
    )
    tracks = ["GT"]

    if prediction_npz is not None and prediction_npz.exists():
        pred_dirs = load_sparsegaze_directions(prediction_npz, pred_key, table)
        table = attach_direction_track(table, pred_dirs, "sparsegaze")
        tracks.append("SparseGaze")
    elif prediction_npz is not None:
        print(f"warning: SparseGaze NPZ not found, skipping: {prediction_npz}")

    if hagi_npz is not None and hagi_npz.exists():
        hagi_dirs = load_hagi_world_directions(
            hagi_npz=hagi_npz,
            hagi_adt_data=hagi_adt_data,
            sequence_name=context.sequence,
        )
        table = attach_direction_track(table, hagi_dirs, "hagi")
        tracks.append("HAGI++")
    elif hagi_npz is not None:
        print(f"warning: HAGI++ NPZ not found, skipping: {hagi_npz}")

    if context.hits is not None and not context.hits.empty:
        hit_cols = [
            "query_timestamp_ns",
            "object_hit",
            "hit_instance_name",
            "hit_category",
            "hit_distance_m",
            "hit_x_m",
            "hit_y_m",
            "hit_z_m",
        ]
        available = [column for column in hit_cols if column in context.hits.columns]
        table = table.merge(
            context.hits[available],
            on="query_timestamp_ns",
            how="left",
        )
    return tracks, table


def load_sparsegaze_directions(
    npz_path: Path,
    pred_key: str,
    table: pd.DataFrame,
) -> dict[int, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        if pred_key not in data.files:
            raise KeyError(f"{npz_path} does not contain {pred_key!r}")
        directions = normalize_vectors(np.asarray(data[pred_key], dtype=np.float64))
        if "timestamps_ns" in data.files:
            timestamps = np.asarray(data["timestamps_ns"], dtype=np.int64)
            by_timestamp = {
                int(timestamp): directions[index] for index, timestamp in enumerate(timestamps)
            }
            return {
                int(row.frame_index): by_timestamp[int(row.query_timestamp_ns)]
                for row in table.itertuples()
                if int(row.query_timestamp_ns) in by_timestamp
            }
    return {
        int(index): directions[int(index)]
        for index in table["frame_index"].to_numpy(dtype=int)
        if 0 <= int(index) < len(directions)
    }


def load_hagi_world_directions(
    *,
    hagi_npz: Path,
    hagi_adt_data: Path,
    sequence_name: str,
) -> dict[int, np.ndarray]:
    adt_data = np.load(hagi_adt_data, allow_pickle=True).item()
    if sequence_name not in adt_data:
        raise KeyError(f"{sequence_name} not found in HAGI ADT cache: {hagi_adt_data}")
    transforms = np.asarray(adt_data[sequence_name]["T_world_CPF"], dtype=np.float64)
    with np.load(hagi_npz, allow_pickle=True) as data:
        sequence_mask = data["sequence_name"].astype(str) == sequence_name
        frames = np.asarray(data["frame_index"][sequence_mask], dtype=np.int64)
        pred_cpf = normalize_vectors(np.asarray(data["pred"][sequence_mask], dtype=np.float64))

    valid = (frames >= 0) & (frames < len(transforms))
    frames = frames[valid]
    pred_cpf = pred_cpf[valid]
    rotations = transforms[frames, :3, :3]
    pred_world = normalize_vectors(np.einsum("fij,fj->fi", rotations, pred_cpf))
    return {int(frame): pred_world[index] for index, frame in enumerate(frames)}


def attach_direction_track(
    table: pd.DataFrame,
    directions_by_frame: dict[int, np.ndarray],
    prefix: str,
) -> pd.DataFrame:
    output = table.copy()
    direction = np.full((len(output), 3), np.nan, dtype=np.float64)
    for row_position, frame_index in enumerate(output["frame_index"].to_numpy(dtype=int)):
        value = directions_by_frame.get(int(frame_index))
        if value is not None:
            direction[row_position] = value

    origin = output[["origin_x", "origin_y", "origin_z"]].to_numpy(dtype=np.float64)
    depth = output["depth_m"].to_numpy(dtype=np.float64)
    points = origin + direction * depth[:, None]
    valid = (
        np.isfinite(origin).all(axis=1)
        & np.isfinite(direction).all(axis=1)
        & np.isfinite(depth)
        & (depth > 0)
    )
    points[~valid] = np.nan

    gt_points = output[["gt_x", "gt_y", "gt_z"]].to_numpy(dtype=np.float64)
    gt_dirs = output[["gt_dir_x", "gt_dir_y", "gt_dir_z"]].to_numpy(dtype=np.float64)
    point_error = np.linalg.norm(points - gt_points, axis=1)
    point_error[~valid | ~np.isfinite(gt_points).all(axis=1)] = np.nan
    angular = angular_error_deg(direction, gt_dirs)
    angular[~valid | ~np.isfinite(gt_dirs).all(axis=1)] = np.nan

    output[f"{prefix}_dir_x"] = direction[:, 0]
    output[f"{prefix}_dir_y"] = direction[:, 1]
    output[f"{prefix}_dir_z"] = direction[:, 2]
    output[f"{prefix}_x"] = points[:, 0]
    output[f"{prefix}_y"] = points[:, 1]
    output[f"{prefix}_z"] = points[:, 2]
    output[f"{prefix}_available"] = valid
    output[f"{prefix}_point_error_m"] = point_error
    output[f"{prefix}_angular_error_deg"] = angular
    return output


def select_window(table: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    selected = table[
        (table["frame_index"] >= int(start_frame))
        & (table["frame_index"] < int(end_frame))
    ].copy()
    if selected.empty:
        raise ValueError("Selected frame window is empty.")
    return selected.reset_index(drop=True)


def downsample_frames(frames: np.ndarray, stride: int) -> list[int]:
    selected = list(np.asarray(frames, dtype=int)[::stride])
    if len(frames) and selected[-1] != int(frames[-1]):
        selected.append(int(frames[-1]))
    return selected


def make_scene_scanpath_figure(
    *,
    context: SceneContext,
    rows: pd.DataFrame,
    tracks: list[str],
    object_rows: pd.DataFrame | None,
    current_frame: int,
    tail_frames: int,
    title: str,
    show_rays: bool,
    ray_length_m: float,
    mirror_horizontal: bool,
    vertical_axis: str,
    view_elev: float,
    view_azim: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 9))
    axis = fig.add_subplot(111, projection="3d")
    plotted_points: list[np.ndarray] = []

    draw_object_boxes(
        axis,
        object_rows,
        plotted_points,
        vertical_axis,
        mirror_horizontal,
    )
    draw_head_trajectory(
        axis,
        context.head,
        rows,
        current_frame,
        plotted_points,
        vertical_axis,
        mirror_horizontal,
    )
    draw_skeleton_root_trajectory(
        axis,
        context.skeleton,
        rows,
        current_frame,
        plotted_points,
        vertical_axis,
        mirror_horizontal,
    )
    draw_skeleton(
        axis,
        context.skeleton,
        context.skeleton_summary,
        current_frame,
        plotted_points,
        vertical_axis,
        mirror_horizontal,
    )

    tail = rows[
        (rows["frame_index"] >= current_frame - tail_frames)
        & (rows["frame_index"] <= current_frame)
    ].copy()
    draw_tracks(
        axis,
        tail,
        tracks,
        current_frame,
        show_rays,
        ray_length_m,
        plotted_points,
        vertical_axis,
        mirror_horizontal,
    )
    draw_current_hit(
        axis,
        context.hits,
        rows,
        current_frame,
        plotted_points,
        vertical_axis,
        mirror_horizontal,
    )

    set_equal_axes(axis, plotted_points)
    x_label, y_label, z_label = display_axis_labels(vertical_axis, mirror_horizontal)
    axis.set_xlabel(f"{x_label} [m]")
    axis.set_ylabel(f"{y_label} [m]")
    axis.set_zlabel(f"{z_label} [m]")
    apply_matplotlib_scene_view(axis, view_elev, view_azim)
    axis.set_title(f"{title} | frame={current_frame} | tail={tail_frames}")
    axis.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def draw_object_boxes(
    axis: Any,
    object_rows: pd.DataFrame | None,
    plotted_points: list[np.ndarray],
    vertical_axis: str,
    mirror_horizontal: bool,
) -> None:
    if object_rows is None or object_rows.empty:
        return
    static_rows = object_rows[object_rows["timestamp_ns"] == -1]
    dynamic_rows = object_rows[object_rows["timestamp_ns"] != -1]
    for rows, color, alpha, label in [
        (static_rows, "#6b7280", 0.22, "static object boxes"),
        (dynamic_rows, "#2563eb", 0.55, "dynamic object boxes"),
    ]:
        if rows.empty:
            continue
        first = True
        for _, row in rows.iterrows():
            corners = display_points(row_corners(row), vertical_axis, mirror_horizontal)
            for edge in BOX_EDGES:
                pts = corners[list(edge)]
                axis.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=color,
                    alpha=alpha,
                    linewidth=0.8,
                    label=label if first else None,
                )
                first = False
            plotted_points.append(corners)


def draw_head_trajectory(
    axis: Any,
    head: pd.DataFrame | None,
    rows: pd.DataFrame,
    current_frame: int,
    plotted_points: list[np.ndarray],
    vertical_axis: str,
    mirror_horizontal: bool,
) -> None:
    if head is None or head.empty:
        return
    timestamps = rows[rows["frame_index"] <= current_frame]["query_timestamp_ns"]
    head_rows = head[head["query_timestamp_ns"].isin(set(timestamps.astype(np.int64)))]
    points = finite_xyz(
        head_rows,
        ["head_origin_scene_x_m", "head_origin_scene_y_m", "head_origin_scene_z_m"],
    )
    if len(points) == 0:
        return
    points = display_points(points, vertical_axis, mirror_horizontal)
    axis.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="#0f766e",
        linewidth=1.8,
        label="head/device trajectory",
    )
    axis.scatter(points[-1:, 0], points[-1:, 1], points[-1:, 2], color="#0f766e", s=24)
    plotted_points.append(points)


def draw_skeleton(
    axis: Any,
    skeleton: pd.DataFrame | None,
    skeleton_summary: dict[str, Any] | None,
    current_frame: int,
    plotted_points: list[np.ndarray],
    vertical_axis: str,
    mirror_horizontal: bool,
) -> None:
    if skeleton is None or skeleton.empty or not skeleton_summary:
        return
    if "frame_index" not in skeleton.columns:
        return
    nearest = skeleton.iloc[
        int(np.argmin(np.abs(skeleton["frame_index"].to_numpy(dtype=int) - current_frame)))
    ]
    labels = skeleton_summary.get("joint_labels", [])
    connections = skeleton_summary.get("joint_connections", [])
    points = []
    for joint_index, label in enumerate(labels):
        safe = safe_joint_label(str(label))
        columns = [
            f"joint_{joint_index:02d}_{safe}_scene_x_m",
            f"joint_{joint_index:02d}_{safe}_scene_y_m",
            f"joint_{joint_index:02d}_{safe}_scene_z_m",
        ]
        if not all(column in nearest.index for column in columns):
            points.append([np.nan, np.nan, np.nan])
            continue
        points.append([float(nearest[column]) for column in columns])
    point_array = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(point_array).all(axis=1)
    if not finite.any():
        return
    point_array = display_points(point_array, vertical_axis, mirror_horizontal)
    first = True
    for first_joint, second_joint in connections:
        if first_joint >= len(point_array) or second_joint >= len(point_array):
            continue
        if not (finite[first_joint] and finite[second_joint]):
            continue
        pts = point_array[[first_joint, second_joint]]
        axis.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color="#7c3aed",
            alpha=0.9,
            linewidth=1.2,
            label="current skeleton" if first else None,
        )
        first = False
    axis.scatter(
        point_array[finite, 0],
        point_array[finite, 1],
        point_array[finite, 2],
        color="#a78bfa",
        s=8,
        alpha=0.9,
    )
    plotted_points.append(point_array[finite])


def draw_skeleton_root_trajectory(
    axis: Any,
    skeleton: pd.DataFrame | None,
    rows: pd.DataFrame,
    current_frame: int,
    plotted_points: list[np.ndarray],
    vertical_axis: str,
    mirror_horizontal: bool,
) -> None:
    if skeleton is None or skeleton.empty or "frame_index" not in skeleton.columns:
        return
    frame_set = set(rows[rows["frame_index"] <= current_frame]["frame_index"].astype(int))
    skel_rows = skeleton[skeleton["frame_index"].astype(int).isin(frame_set)]
    points = finite_xyz(
        skel_rows,
        ["root_joint_scene_x_m", "root_joint_scene_y_m", "root_joint_scene_z_m"],
    )
    if len(points) == 0:
        return
    points = display_points(points, vertical_axis, mirror_horizontal)
    axis.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="#9333ea",
        alpha=0.85,
        linewidth=1.6,
        label="body/root trajectory",
    )
    plotted_points.append(points)


def draw_tracks(
    axis: Any,
    rows: pd.DataFrame,
    tracks: list[str],
    current_frame: int,
    show_rays: bool,
    ray_length_m: float,
    plotted_points: list[np.ndarray],
    vertical_axis: str,
    mirror_horizontal: bool,
) -> None:
    for track in tracks:
        prefix = track_prefix(track)
        columns = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
        points = finite_xyz(rows, columns)
        if len(points) == 0:
            continue
        points = display_points(points, vertical_axis, mirror_horizontal)
        color = TRACK_COLORS[track]
        axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            linewidth=2.4,
            alpha=0.9,
            label=f"{track} scanpath",
        )
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            s=18,
            alpha=0.9,
        )
        current = rows[rows["frame_index"] == current_frame]
        current_points = finite_xyz(current, columns)
        if len(current_points):
            current_points = display_points(current_points, vertical_axis, mirror_horizontal)
            axis.scatter(
                current_points[:, 0],
                current_points[:, 1],
                current_points[:, 2],
                color=color,
                marker="X",
                s=92,
                edgecolors="black",
                linewidths=0.8,
            )
        plotted_points.append(points)
        if show_rays:
            draw_current_ray(
                axis,
                rows,
                prefix,
                current_frame,
                color,
                ray_length_m,
                plotted_points,
                vertical_axis,
                mirror_horizontal,
            )


def draw_current_ray(
    axis: Any,
    rows: pd.DataFrame,
    prefix: str,
    current_frame: int,
    color: str,
    ray_length_m: float,
    plotted_points: list[np.ndarray],
    vertical_axis: str,
    mirror_horizontal: bool,
) -> None:
    current = rows[rows["frame_index"] == current_frame]
    if current.empty:
        return
    row = current.iloc[-1]
    origin = row[["origin_x", "origin_y", "origin_z"]].to_numpy(dtype=np.float64)
    direction = row[[f"{prefix}_dir_x", f"{prefix}_dir_y", f"{prefix}_dir_z"]].to_numpy(
        dtype=np.float64
    )
    if not (np.isfinite(origin).all() and np.isfinite(direction).all()):
        return
    end = origin + normalize_vectors(direction.reshape(1, 3))[0] * ray_length_m
    ray = display_points(np.vstack([origin, end]), vertical_axis, mirror_horizontal)
    axis.plot(ray[:, 0], ray[:, 1], ray[:, 2], color=color, alpha=0.45, linewidth=1.5)
    plotted_points.append(ray)


def draw_current_hit(
    axis: Any,
    hits: pd.DataFrame | None,
    rows: pd.DataFrame,
    current_frame: int,
    plotted_points: list[np.ndarray],
    vertical_axis: str,
    mirror_horizontal: bool,
) -> None:
    if hits is None or hits.empty:
        return
    current = rows[rows["frame_index"] == current_frame]
    if current.empty:
        return
    timestamp = int(current["query_timestamp_ns"].iloc[-1])
    hit = hits[hits["query_timestamp_ns"] == timestamp]
    if hit.empty:
        return
    row = hit.iloc[-1]
    if not row_bool(row.get("object_hit")):
        return
    point = np.asarray([row.get("hit_x_m"), row.get("hit_y_m"), row.get("hit_z_m")], dtype=float)
    if not np.isfinite(point).all():
        return
    point = display_points(point.reshape(1, 3), vertical_axis, mirror_horizontal)[0]
    axis.scatter(
        [point[0]],
        [point[1]],
        [point[2]],
        color="#dc2626",
        marker="D",
        s=74,
        label="current GT ray-box hit",
    )
    plotted_points.append(point.reshape(1, 3))


def make_error_figure(rows: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for prefix, label, color in [
        ("sparsegaze", "SparseGaze", TRACK_COLORS["SparseGaze"]),
        ("hagi", "HAGI++", TRACK_COLORS["HAGI++"]),
    ]:
        point_col = f"{prefix}_point_error_m"
        angular_col = f"{prefix}_angular_error_deg"
        if point_col in rows.columns:
            axes[0].plot(rows["frame_index"], rows[point_col], color=color, label=label)
        if angular_col in rows.columns:
            axes[1].plot(rows["frame_index"], rows[angular_col], color=color, label=label)
    axes[0].set_ylabel("Point error [m]")
    axes[1].set_ylabel("Angular error [deg]")
    axes[1].set_xlabel("Frame index")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[0].set_title("Scene point and direction error")
    fig.tight_layout()
    return fig


def write_dynamic_tail_frames(
    *,
    context: SceneContext,
    rows: pd.DataFrame,
    tracks: list[str],
    frame_indices: list[int],
    output_dir: Path,
    tail_frames: int,
    args: argparse.Namespace,
) -> list[Path]:
    frame_paths: list[Path] = []
    for frame_index in frame_indices:
        current_rows = rows[rows["frame_index"] <= frame_index]
        if current_rows.empty:
            continue
        focus_timestamp = int(current_rows["query_timestamp_ns"].iloc[-1])
        object_rows = select_object_rows(
            context.objects,
            focus_timestamp_ns=focus_timestamp,
            category_filter=args.category_filter,
            exclude_category_filter=args.exclude_category_filter,
            max_static_objects=args.max_static_objects,
        )
        fig = make_scene_scanpath_figure(
            context=context,
            rows=rows,
            tracks=tracks,
            object_rows=object_rows,
            current_frame=int(frame_index),
            tail_frames=tail_frames,
            title="Scene 3D dynamic scanpath tail",
            show_rays=args.show_rays,
            ray_length_m=args.ray_length_m,
            mirror_horizontal=args.effective_mirror_horizontal,
            vertical_axis=args.vertical_axis,
            view_elev=args.view_elev,
            view_azim=args.view_azim,
        )
        path = output_dir / f"scene3d_row{frame_index:04d}.png"
        fig.savefig(path, dpi=128)
        plt.close(fig)
        frame_paths.append(path)
    return frame_paths


def write_video(path: Path, frame_paths: Sequence[Path], fps: float) -> None:
    if not frame_paths:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))


def write_plotly_scene(
    path: Path,
    *,
    context: SceneContext,
    rows: pd.DataFrame,
    tracks: list[str],
    object_rows: pd.DataFrame | None,
    mirror_horizontal: bool,
    vertical_axis: str,
    view_elev: float,
    view_azim: float,
) -> Path | None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("warning: plotly is not installed; skipping interactive HTML.")
        return None

    fig = go.Figure()
    plotted: list[np.ndarray] = []
    if object_rows is not None and not object_rows.empty:
        add_plotly_boxes(fig, object_rows, plotted, vertical_axis, mirror_horizontal)
    for track in tracks:
        prefix = track_prefix(track)
        points = finite_xyz(rows, [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"])
        if len(points) == 0:
            continue
        points = display_points(points, vertical_axis, mirror_horizontal)
        color = TRACK_COLORS[track]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines+markers",
                name=f"{track} scanpath",
                line=dict(color=color, width=5),
                marker=dict(size=3, color=color),
            )
        )
        plotted.append(points)
    head_points = finite_head_points(context.head, rows)
    if len(head_points):
        head_points = display_points(head_points, vertical_axis, mirror_horizontal)
        fig.add_trace(
            go.Scatter3d(
                x=head_points[:, 0],
                y=head_points[:, 1],
                z=head_points[:, 2],
                mode="lines",
                name="head/device trajectory",
                line=dict(color="#0f766e", width=4),
            )
        )
        plotted.append(head_points)
    ranges = axis_ranges(plotted)
    x_label, y_label, z_label = display_axis_labels(vertical_axis, mirror_horizontal)
    fig.update_layout(
        title=f"{context.sequence}: Scene 3D scanpath comparison",
        scene=dict(
            xaxis=dict(title=f"{x_label} [m]", range=ranges[0]),
            yaxis=dict(title=f"{y_label} [m]", range=ranges[1]),
            zaxis=dict(title=f"{z_label} [m]", range=ranges[2]),
            aspectmode="data",
            camera=plotly_camera(vertical_axis, view_elev, view_azim),
        ),
        height=820,
        margin=dict(l=0, r=0, t=48, b=0),
    )
    fig.write_html(path)
    return path


def add_plotly_boxes(
    fig: Any,
    object_rows: pd.DataFrame,
    plotted: list[np.ndarray],
    vertical_axis: str,
    mirror_horizontal: bool,
) -> None:
    import plotly.graph_objects as go

    for rows, color, name in [
        (object_rows[object_rows["timestamp_ns"] == -1], "#6b7280", "static object boxes"),
        (object_rows[object_rows["timestamp_ns"] != -1], "#2563eb", "dynamic object boxes"),
    ]:
        if rows.empty:
            continue
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        for _, row in rows.iterrows():
            corners = display_points(row_corners(row), vertical_axis, mirror_horizontal)
            for first, second in BOX_EDGES:
                xs.extend([corners[first, 0], corners[second, 0], None])
                ys.extend([corners[first, 1], corners[second, 1], None])
                zs.extend([corners[first, 2], corners[second, 2], None])
            plotted.append(corners)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                opacity=0.35,
                hoverinfo="skip",
            )
        )


def select_object_rows(
    objects: pd.DataFrame | None,
    *,
    focus_timestamp_ns: int,
    category_filter: str,
    exclude_category_filter: str,
    max_static_objects: int,
) -> pd.DataFrame | None:
    if objects is None or objects.empty:
        return None
    categories = parse_filter(category_filter)
    excluded = parse_filter(exclude_category_filter)
    static_rows = objects[objects["timestamp_ns"] == -1].copy()
    dynamic_rows = objects[objects["timestamp_ns"] != -1].copy()
    static_rows = filter_object_categories(static_rows, categories, excluded)
    dynamic_rows = filter_object_categories(dynamic_rows, categories, excluded)
    if max_static_objects > 0:
        static_rows = static_rows.head(max_static_objects)
    if not dynamic_rows.empty:
        timestamps = dynamic_rows["timestamp_ns"].to_numpy(dtype=np.int64)
        nearest = int(timestamps[np.argmin(np.abs(timestamps - int(focus_timestamp_ns)))])
        dynamic_rows = dynamic_rows[dynamic_rows["timestamp_ns"] == nearest]
    return pd.concat([static_rows, dynamic_rows], ignore_index=True)


def filter_object_categories(
    rows: pd.DataFrame,
    categories: set[str] | None,
    excluded: set[str] | None,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    output = rows
    if categories:
        output = output[output["category"].isin(categories)]
    if excluded:
        output = output[~output["category"].isin(excluded)]
    return output


def build_summary(
    *,
    context: SceneContext,
    table: pd.DataFrame,
    window: pd.DataFrame,
    tracks: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sequence": context.sequence,
        "reports_dir": str(args.reports_dir),
        "prediction_npz": str(args.prediction_npz) if args.prediction_npz else None,
        "hagi_npz": str(args.hagi_npz) if args.hagi_npz else None,
        "hagi_adt_data": str(args.hagi_adt_data),
        "output_dir": str(output_dir),
        "start_frame": int(args.start_frame),
        "end_frame_exclusive": int(args.end_frame),
        "stride": int(args.stride),
        "tail_frames": int(args.tail_frames),
        "mirror_horizontal": bool(args.effective_mirror_horizontal),
        "vertical_axis": str(args.vertical_axis),
        "view_elev": float(args.view_elev),
        "view_azim": float(args.view_azim),
        "tracks": tracks,
        "total_frames": int(len(table)),
        "selected_frames": int(len(window)),
        "object_context_available": context.objects is not None,
        "skeleton_context_available": context.skeleton is not None,
        "head_context_available": context.head is not None,
        "note": (
            "SparseGaze/HAGI++ 3D points use aligned GT depth, not predicted depth. "
            "Rendered display coordinates use Scene Y-up and optional mirrored Scene X; "
            "raw CSV coordinates are unchanged."
        ),
    }
    for prefix in ["sparsegaze", "hagi"]:
        error_col = f"{prefix}_point_error_m"
        angular_col = f"{prefix}_angular_error_deg"
        available_col = f"{prefix}_available"
        if error_col not in window.columns:
            continue
        valid_errors = window[error_col].dropna()
        valid_angular = window[angular_col].dropna()
        summary[prefix] = {
            "available_frames": int(window[available_col].sum()),
            "mean_point_error_m": float(valid_errors.mean()) if not valid_errors.empty else None,
            "median_point_error_m": float(valid_errors.median()) if not valid_errors.empty else None,
            "p90_point_error_m": float(valid_errors.quantile(0.9))
            if not valid_errors.empty
            else None,
            "mean_angular_error_deg": float(valid_angular.mean())
            if not valid_angular.empty
            else None,
        }
    return summary


def set_equal_axes(axis: Any, points_list: Sequence[np.ndarray]) -> None:
    ranges = axis_ranges(points_list)
    axis.set_xlim(*ranges[0])
    axis.set_ylim(*ranges[1])
    axis.set_zlim(*ranges[2])
    try:
        axis.set_box_aspect(
            (
                ranges[0][1] - ranges[0][0],
                ranges[1][1] - ranges[1][0],
                ranges[2][1] - ranges[2][0],
            )
        )
    except AttributeError:
        pass


def apply_matplotlib_scene_view(
    axis: Any,
    elevation_deg: float,
    azimuth_deg: float,
) -> None:
    axis.view_init(elev=elevation_deg, azim=azimuth_deg)


def display_points(
    points: np.ndarray,
    vertical_axis: str,
    mirror_horizontal: bool = False,
) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {arr.shape}")
    if vertical_axis == "scene_z":
        display = arr.copy()
    elif vertical_axis == "scene_y":
        display = arr[:, [0, 2, 1]].copy()
    elif vertical_axis == "scene_x":
        display = arr[:, [1, 2, 0]].copy()
    else:
        raise ValueError(f"Unsupported vertical_axis: {vertical_axis}")
    if mirror_horizontal:
        display[:, 0] *= -1.0
    return display


def display_axis_labels(
    vertical_axis: str,
    mirror_horizontal: bool = False,
) -> tuple[str, str, str]:
    if vertical_axis == "scene_z":
        labels = ["Scene X", "Scene Y", "Scene Z"]
    elif vertical_axis == "scene_y":
        labels = ["Scene X", "Scene Z", "Scene Y"]
    elif vertical_axis == "scene_x":
        labels = ["Scene Y", "Scene Z", "Scene X"]
    else:
        raise ValueError(f"Unsupported vertical_axis: {vertical_axis}")
    if mirror_horizontal:
        labels[0] = f"-{labels[0]}"
    return tuple(labels)


def plotly_camera(vertical_axis: str, elevation_deg: float, azimuth_deg: float) -> dict[str, Any]:
    if vertical_axis not in {"scene_x", "scene_y", "scene_z"}:
        raise ValueError(f"Unsupported vertical_axis: {vertical_axis}")
    up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    plane_axis_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    plane_axis_b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    elevation_rad = np.deg2rad(float(elevation_deg))
    distance = 2.1
    horizontal_radius = distance * np.cos(elevation_rad)
    vertical_offset = distance * np.sin(elevation_rad)
    eye = (
        horizontal_radius * np.cos(azimuth_rad) * plane_axis_a
        + horizontal_radius * np.sin(azimuth_rad) * plane_axis_b
        + vertical_offset * up_vec
    )
    return {
        "up": {"x": float(up_vec[0]), "y": float(up_vec[1]), "z": float(up_vec[2])},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
    }


def axis_ranges(points_list: Sequence[np.ndarray]) -> list[list[float]]:
    finite = [
        points[np.isfinite(points).all(axis=1)]
        for points in points_list
        if points is not None and len(points)
    ]
    if not finite:
        return [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    points = np.vstack(finite)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    ranges: list[list[float]] = []
    for axis in range(3):
        span = float(maxs[axis] - mins[axis])
        if span <= 1e-6:
            center = float((mins[axis] + maxs[axis]) / 2.0)
            ranges.append([center - 0.5, center + 0.5])
            continue
        pad = max(span * 0.1, 0.25)
        ranges.append([float(mins[axis] - pad), float(maxs[axis] + pad)])
    return ranges


def finite_head_points(head: pd.DataFrame | None, rows: pd.DataFrame) -> np.ndarray:
    if head is None or head.empty:
        return np.zeros((0, 3), dtype=np.float64)
    timestamps = set(rows["query_timestamp_ns"].astype(np.int64))
    return finite_xyz(
        head[head["query_timestamp_ns"].isin(timestamps)],
        ["head_origin_scene_x_m", "head_origin_scene_y_m", "head_origin_scene_z_m"],
    )


def finite_xyz(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if frame.empty or not all(column in frame.columns for column in columns):
        return np.zeros((0, 3), dtype=np.float64)
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    return values[np.isfinite(values).all(axis=1)]


def row_corners(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            [
                float(row[f"scene_corner_{index}_x_m"]),
                float(row[f"scene_corner_{index}_y_m"]),
                float(row[f"scene_corner_{index}_z_m"]),
            ]
            for index in range(8)
        ],
        dtype=np.float64,
    )


def track_prefix(track: str) -> str:
    if track == "GT":
        return "gt"
    if track == "SparseGaze":
        return "sparsegaze"
    if track == "HAGI++":
        return "hagi"
    raise ValueError(f"Unknown track: {track}")


def normalize_vectors(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr / np.clip(np.linalg.norm(arr, axis=-1, keepdims=True), eps, None)


def angular_error_deg(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_n = normalize_vectors(pred)
    gt_n = normalize_vectors(gt)
    dot = np.sum(pred_n * gt_n, axis=-1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_filter(value: str) -> set[str] | None:
    parsed = {item.strip() for item in str(value).split(",") if item.strip()}
    return parsed or None


def safe_joint_label(label: str) -> str:
    import re

    safe = re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_")
    return safe or "joint"


def row_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    if value is None or pd.isna(value):
        return False
    return bool(value)


def default_run_name(start_frame: int, end_frame: int, stride: int, tail_frames: int) -> str:
    return f"frames_{start_frame}_{end_frame}_stride{stride}_tail{tail_frames}_scene3d"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
