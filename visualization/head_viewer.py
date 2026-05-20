"""Helpers for notebook-based 3D visualization of gaze and head proxy data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adt_sandbox.results import discover_sequence_files, find_sequence_file


def discover_sequence_ids(reports_dir: str | Path) -> list[str]:
    """Return sequence ids that have both gaze and head CSVs."""

    root = Path(reports_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected reports directory: {root}")

    gaze_ids = {
        item.sequence_name
        for item in discover_sequence_files(root, "gaze", "gaze_samples.csv")
    }
    head_ids = {
        item.sequence_name
        for item in discover_sequence_files(root, "head", "head_samples.csv")
    }
    sequence_ids = sorted(gaze_ids & head_ids)
    if not sequence_ids:
        raise ValueError(f"No sequence ids with both gaze/head CSVs found in: {root}")
    return sequence_ids


def load_gaze_head_frame(reports_dir: str | Path, sequence_id: str) -> pd.DataFrame:
    """Load and align one sequence's gaze/head CSVs on `query_timestamp_ns`."""

    root = Path(reports_dir).expanduser()
    gaze_csv = find_sequence_file(root, sequence_id, "gaze", "gaze_samples.csv")
    head_csv = find_sequence_file(root, sequence_id, "head", "head_samples.csv")

    gaze_df = pd.read_csv(gaze_csv)
    head_df = pd.read_csv(head_csv)
    merged = gaze_df.merge(
        head_df,
        on="query_timestamp_ns",
        how="inner",
        suffixes=("_gaze", "_head"),
    )
    if merged.empty:
        raise ValueError(f"No aligned gaze/head rows found for: {sequence_id}")
    return merged.sort_values("query_timestamp_ns").reset_index(drop=True)


def slice_frame_window(
    frame: pd.DataFrame,
    start_row: int,
    end_row: int | None,
    stride: int,
) -> pd.DataFrame:
    """Return one row window with optional stride."""

    if start_row < 0:
        raise ValueError("start_row must be non-negative")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if end_row is not None and end_row <= start_row:
        raise ValueError("end_row must be greater than start_row")

    window = frame.iloc[start_row:end_row].copy()
    if window.empty:
        raise ValueError("Selected row window is empty")
    return window.iloc[::stride].reset_index(drop=True)


def plot_gaze_head_scene_window(
    window: pd.DataFrame,
    head_scale_m: float = 0.35,
    gaze_scale_mode: str = "fixed",
    fixed_gaze_scale_m: float = 0.35,
    show_trajectory: bool = True,
    vertical_axis: str = "scene_y",
    azimuth_deg: float = 35.0,
    elevation_deg: float = 18.0,
) -> Any:
    """Plot one 3D Scene-frame window with gaze and head directions.

    zh-CN:
    - 红色：gaze direction
    - 蓝色：head forward direction
    - 黑色折线：CPF/head origin trajectory
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    origins = window[
        ["head_origin_scene_x_m", "head_origin_scene_y_m", "head_origin_scene_z_m"]
    ].to_numpy(dtype=np.float64)
    head_dirs = window[
        [
            "head_forward_scene_unit_x",
            "head_forward_scene_unit_y",
            "head_forward_scene_unit_z",
        ]
    ].to_numpy(dtype=np.float64)
    gaze_dirs = window[
        [
            "gaze_dir_scene_unit_x",
            "gaze_dir_scene_unit_y",
            "gaze_dir_scene_unit_z",
        ]
    ].to_numpy(dtype=np.float64)

    if show_trajectory:
        ax.plot(
            origins[:, 0],
            origins[:, 1],
            origins[:, 2],
            color="black",
            linewidth=1.2,
            alpha=0.8,
            label="head/gaze origin",
        )

    if gaze_scale_mode == "depth":
        depth_values = (
            window["depth_m"].to_numpy(dtype=np.float64)
            if "depth_m" in window.columns
            else np.full(len(window), fixed_gaze_scale_m, dtype=np.float64)
        )
        gaze_lengths = np.where(
            np.isfinite(depth_values) & (depth_values > 0),
            depth_values,
            fixed_gaze_scale_m,
        )
    else:
        gaze_lengths = np.full(len(window), fixed_gaze_scale_m, dtype=np.float64)

    for idx, origin in enumerate(origins):
        if np.isfinite(head_dirs[idx]).all():
            head_end = origin + head_dirs[idx] * head_scale_m
            ax.plot(
                [origin[0], head_end[0]],
                [origin[1], head_end[1]],
                [origin[2], head_end[2]],
                color="royalblue",
                alpha=0.8,
                linewidth=1.6,
            )
        if np.isfinite(gaze_dirs[idx]).all():
            gaze_end = origin + gaze_dirs[idx] * gaze_lengths[idx]
            ax.plot(
                [origin[0], gaze_end[0]],
                [origin[1], gaze_end[1]],
                [origin[2], gaze_end[2]],
                color="crimson",
                alpha=0.65,
                linewidth=1.4,
            )

    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c="black", s=12)
    ax.scatter(origins[0, 0], origins[0, 1], origins[0, 2], c="green", s=70, marker="o", label="start")
    ax.scatter(origins[-1, 0], origins[-1, 1], origins[-1, 2], c="gold", s=90, marker="x", label="end")

    plotted_points = _collect_plotted_points(
        origins=origins,
        head_dirs=head_dirs,
        head_scale_m=head_scale_m,
        gaze_dirs=gaze_dirs,
        gaze_lengths=gaze_lengths,
    )

    ax.set_xlabel("Scene X [m]")
    ax.set_ylabel("Scene Y [m]")
    ax.set_zlabel("Scene Z [m]")
    ax.set_title("Gaze (red) and head forward (blue) in Scene frame")
    ax.legend(loc="upper right")
    _set_equal_axes(ax, plotted_points)
    _apply_matplotlib_scene_view(
        ax,
        vertical_axis=vertical_axis,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )
    return fig


def plot_gaze_head_scene_window_plotly(
    window: pd.DataFrame,
    head_scale_m: float = 0.35,
    gaze_scale_mode: str = "fixed",
    fixed_gaze_scale_m: float = 0.35,
    show_trajectory: bool = True,
    vertical_axis: str = "scene_y",
    azimuth_deg: float = 35.0,
    elevation_deg: float = 18.0,
    camera_distance_scale: float = 2.4,
    mouse_drag_mode: str = "zoom",
    equalize_axes: bool = True,
) -> Any:
    """Return an interactive Plotly 3D figure for one selected window.

    zh-CN:
    这是给 notebook 用的可旋转版本。语义和 matplotlib 版本保持一致：
    - 黑色：head/gaze origin trajectory
    - 蓝色：head forward direction
    - 红色：gaze direction
    """

    import plotly.graph_objects as go

    origins = window[
        ["head_origin_scene_x_m", "head_origin_scene_y_m", "head_origin_scene_z_m"]
    ].to_numpy(dtype=np.float64)
    head_dirs = window[
        [
            "head_forward_scene_unit_x",
            "head_forward_scene_unit_y",
            "head_forward_scene_unit_z",
        ]
    ].to_numpy(dtype=np.float64)
    gaze_dirs = window[
        [
            "gaze_dir_scene_unit_x",
            "gaze_dir_scene_unit_y",
            "gaze_dir_scene_unit_z",
        ]
    ].to_numpy(dtype=np.float64)

    if gaze_scale_mode == "depth":
        depth_values = (
            window["depth_m"].to_numpy(dtype=np.float64)
            if "depth_m" in window.columns
            else np.full(len(window), fixed_gaze_scale_m, dtype=np.float64)
        )
        gaze_lengths = np.where(
            np.isfinite(depth_values) & (depth_values > 0),
            depth_values,
            fixed_gaze_scale_m,
        )
    else:
        gaze_lengths = np.full(len(window), fixed_gaze_scale_m, dtype=np.float64)

    fig = go.Figure()

    if show_trajectory:
        fig.add_trace(
            go.Scatter3d(
                x=origins[:, 0],
                y=origins[:, 1],
                z=origins[:, 2],
                mode="lines+markers",
                name="head/gaze origin",
                line={"color": "black", "width": 4},
                marker={"color": "black", "size": 3},
                hovertemplate=(
                    "frame=%{customdata[0]}<br>"
                    "t=%{customdata[1]}<br>"
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
                customdata=np.column_stack(
                    [
                        window["query_timestamp_ns"].to_numpy(dtype=np.int64),
                        window.index.to_numpy(dtype=np.int64),
                    ]
                ),
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[origins[0, 0]],
            y=[origins[0, 1]],
            z=[origins[0, 2]],
            mode="markers",
            name="start",
            marker={"color": "green", "size": 6, "symbol": "circle"},
            hovertemplate="start<x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[origins[-1, 0]],
            y=[origins[-1, 1]],
            z=[origins[-1, 2]],
            mode="markers",
            name="end",
            marker={"color": "gold", "size": 7, "symbol": "x"},
            hovertemplate="end<x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )

    head_x, head_y, head_z = _plotly_segment_coords(origins, head_dirs, head_scale_m)
    fig.add_trace(
        go.Scatter3d(
            x=head_x,
            y=head_y,
            z=head_z,
            mode="lines",
            name="head forward",
            line={"color": "royalblue", "width": 5},
            hoverinfo="skip",
        )
    )

    gaze_x, gaze_y, gaze_z = _plotly_segment_coords(origins, gaze_dirs, gaze_lengths)
    fig.add_trace(
        go.Scatter3d(
            x=gaze_x,
            y=gaze_y,
            z=gaze_z,
            mode="lines",
            name="gaze direction",
            line={"color": "crimson", "width": 4},
            hoverinfo="skip",
        )
    )

    plotted_points = _collect_plotted_points(
        origins=origins,
        head_dirs=head_dirs,
        head_scale_m=head_scale_m,
        gaze_dirs=gaze_dirs,
        gaze_lengths=gaze_lengths,
    )
    scene_ranges, aspect_ratio = _scene_ranges_from_points(
        plotted_points,
        equalize_axes=equalize_axes,
    )

    scene_camera = _scene_camera_from_vertical_axis(
        points=plotted_points,
        vertical_axis=vertical_axis,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        camera_distance_scale=camera_distance_scale,
    )

    fig.update_layout(
        title="Interactive gaze (red) and head forward (blue) in Scene frame",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        scene_dragmode=mouse_drag_mode,
        scene={
            "xaxis_title": "Scene X [m]",
            "yaxis_title": "Scene Y [m]",
            "zaxis_title": "Scene Z [m]",
            "xaxis": {"range": scene_ranges["x"]},
            "yaxis": {"range": scene_ranges["y"]},
            "zaxis": {"range": scene_ranges["z"]},
            "aspectmode": "cube" if equalize_axes else "manual",
            "aspectratio": aspect_ratio,
            "camera": scene_camera,
        },
        legend={"x": 0.01, "y": 0.99},
    )
    return fig


def _set_equal_axes(ax: Any, points: np.ndarray) -> None:
    """Set equal-ish axis limits around the observed trajectory."""

    xyz_min = np.nanmin(points, axis=0)
    xyz_max = np.nanmax(points, axis=0)
    center = (xyz_min + xyz_max) / 2.0
    radius = float(np.max(xyz_max - xyz_min) / 2.0)
    radius = max(radius, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _apply_matplotlib_scene_view(
    ax: Any,
    vertical_axis: str,
    azimuth_deg: float,
    elevation_deg: float,
) -> None:
    """Apply a default static view that matches the chosen physical up axis."""

    vertical_axis_map = {
        "scene_x": "x",
        "scene_y": "y",
        "scene_z": "z",
    }
    axis_name = vertical_axis_map.get(vertical_axis)
    if axis_name is None:
        raise ValueError(
            f"Unsupported vertical_axis: {vertical_axis}. "
            "Expected one of: scene_x, scene_y, scene_z."
        )
    try:
        ax.view_init(elev=elevation_deg, azim=azimuth_deg, vertical_axis=axis_name)
    except TypeError:
        ax.view_init(elev=elevation_deg, azim=azimuth_deg)


def _plotly_segment_coords(
    origins: np.ndarray,
    directions: np.ndarray,
    lengths: float | np.ndarray,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return line-segment coordinates with `None` separators for Plotly."""

    if np.isscalar(lengths):
        length_array = np.full(len(origins), float(lengths), dtype=np.float64)
    else:
        length_array = np.asarray(lengths, dtype=np.float64)

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for idx, origin in enumerate(origins):
        if not np.isfinite(origin).all() or not np.isfinite(directions[idx]).all():
            continue
        end = origin + directions[idx] * length_array[idx]
        xs.extend([float(origin[0]), float(end[0]), None])
        ys.extend([float(origin[1]), float(end[1]), None])
        zs.extend([float(origin[2]), float(end[2]), None])
    return xs, ys, zs


def _collect_plotted_points(
    origins: np.ndarray,
    head_dirs: np.ndarray,
    head_scale_m: float,
    gaze_dirs: np.ndarray,
    gaze_lengths: np.ndarray,
) -> np.ndarray:
    """Collect all finite points that should influence the 3D bounds."""

    points: list[np.ndarray] = []
    for idx, origin in enumerate(origins):
        if not np.isfinite(origin).all():
            continue
        points.append(origin)
        if np.isfinite(head_dirs[idx]).all():
            points.append(origin + head_dirs[idx] * head_scale_m)
        if np.isfinite(gaze_dirs[idx]).all():
            points.append(origin + gaze_dirs[idx] * gaze_lengths[idx])
    if not points:
        raise ValueError("No finite points available to define 3D bounds.")
    return np.vstack(points)


def _scene_ranges_from_points(
    points: np.ndarray,
    padding_ratio: float = 0.08,
    min_axis_span_m: float = 0.6,
    equalize_axes: bool = True,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Return explicit axis ranges and Plotly aspect ratio from plotted points."""

    xyz_min = np.nanmin(points, axis=0)
    xyz_max = np.nanmax(points, axis=0)
    spans = xyz_max - xyz_min
    spans = np.where(spans < min_axis_span_m, min_axis_span_m, spans)
    centers = (xyz_min + xyz_max) / 2.0
    padded_spans = spans * (1.0 + padding_ratio)
    if equalize_axes:
        max_span = float(np.max(padded_spans))
        padded_spans = np.full(3, max_span, dtype=np.float64)
    half_spans = padded_spans / 2.0
    ranges = {
        "x": [float(centers[0] - half_spans[0]), float(centers[0] + half_spans[0])],
        "y": [float(centers[1] - half_spans[1]), float(centers[1] + half_spans[1])],
        "z": [float(centers[2] - half_spans[2]), float(centers[2] + half_spans[2])],
    }
    max_span = float(np.max(padded_spans))
    aspect_ratio = {
        "x": float(padded_spans[0] / max_span),
        "y": float(padded_spans[1] / max_span),
        "z": float(padded_spans[2] / max_span),
    }
    return ranges, aspect_ratio


def _scene_camera_from_vertical_axis(
    points: np.ndarray,
    vertical_axis: str,
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance_scale: float,
) -> dict[str, dict[str, float]]:
    """Build a camera that rotates only around a chosen scene vertical axis."""

    xyz_min = np.nanmin(points, axis=0)
    xyz_max = np.nanmax(points, axis=0)
    center = (xyz_min + xyz_max) / 2.0
    radius = float(np.max(xyz_max - xyz_min) / 2.0)
    radius = max(radius, 0.5)
    distance = radius * camera_distance_scale

    axis_defs = {
        "scene_z": (
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        ),
        "scene_y": (
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ),
        "scene_x": (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ),
    }
    if vertical_axis not in axis_defs:
        raise ValueError(
            f"Unsupported vertical_axis: {vertical_axis}. "
            "Expected one of: scene_x, scene_y, scene_z."
        )

    up_vec, plane_axis_a, plane_axis_b = axis_defs[vertical_axis]
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    elevation_rad = np.deg2rad(float(elevation_deg))
    horizontal_radius = distance * np.cos(elevation_rad)
    vertical_offset = distance * np.sin(elevation_rad)
    eye = (
        center
        + horizontal_radius * np.cos(azimuth_rad) * plane_axis_a
        + horizontal_radius * np.sin(azimuth_rad) * plane_axis_b
        + vertical_offset * up_vec
    )
    return {
        "up": {"x": float(up_vec[0]), "y": float(up_vec[1]), "z": float(up_vec[2])},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
    }
