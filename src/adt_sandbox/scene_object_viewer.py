"""Interactive Scene-frame visualization for objects, skeleton, head, and gaze."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .results import discover_sequence_files, find_sequence_file
from .viz_palette import SEMANTIC_COLORS


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

STATIC_BOX_COLOR = SEMANTIC_COLORS["static_object"]
DYNAMIC_BOX_COLOR = SEMANTIC_COLORS["dynamic_object"]
HIT_BOX_COLOR = SEMANTIC_COLORS["hit_object"]
HIT_POINT_COLOR = SEMANTIC_COLORS["hit"]
HEAD_TRAJECTORY_COLOR = SEMANTIC_COLORS["head"]
SKELETON_BONE_COLOR = SEMANTIC_COLORS["skeleton"]
SKELETON_JOINT_COLOR = SEMANTIC_COLORS["skeleton_joint"]
GAZE_RAY_COLOR = SEMANTIC_COLORS["gaze_ray"]
GAZE_POINT_COLOR = SEMANTIC_COLORS["gaze_point"]
OBJECT_CENTER_COLOR = SEMANTIC_COLORS["object_center"]


@dataclass(frozen=True)
class SceneViewerData:
    sequence_id: str
    frames: pd.DataFrame
    objects: pd.DataFrame
    hits: pd.DataFrame | None
    events: pd.DataFrame | None
    skeleton_summary: dict[str, Any] | None


def discover_scene_viewer_sequence_ids(reports_dir: str | Path) -> list[str]:
    """Return sequence ids that have gaze/head/object/skeleton CSVs."""

    root = Path(reports_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {root}")
    file_specs = [
        ("gaze", "gaze_samples.csv"),
        ("head", "head_samples.csv"),
        ("scene", "scene_object_boxes.csv"),
        ("skeleton", "skeleton_samples.csv"),
    ]
    id_sets = [
        {
            item.sequence_name
            for item in discover_sequence_files(root, layer, filename)
        }
        for layer, filename in file_specs
    ]
    sequence_ids = sorted(set.intersection(*id_sets))
    if not sequence_ids:
        raise ValueError(f"No complete scene-viewer sequence sets found in: {root}")
    return sequence_ids


def load_scene_viewer_data(
    reports_dir: str | Path,
    sequence_id: str,
) -> SceneViewerData:
    """Load gaze/head/skeleton rows plus Scene object boxes for one sequence."""

    root = Path(reports_dir).expanduser()
    gaze = _read_required_csv(
        find_sequence_file(root, sequence_id, "gaze", "gaze_samples.csv")
    )
    head = _read_required_csv(
        find_sequence_file(root, sequence_id, "head", "head_samples.csv")
    )
    skeleton = _read_required_csv(
        find_sequence_file(
            root,
            sequence_id,
            "skeleton",
            "skeleton_samples.csv",
        )
    )
    objects = _read_required_csv(
        find_sequence_file(
            root,
            sequence_id,
            "scene",
            "scene_object_boxes.csv",
        )
    )
    hits_path = find_optional_sequence_file(
        root,
        sequence_id,
        "scene",
        "gaze_object_hits.csv",
    )
    hits = pd.read_csv(hits_path) if hits_path is not None else None
    events_path = find_optional_sequence_file(
        root,
        sequence_id,
        "events",
        "scene_gaze_frame_labels.csv",
    )
    events = pd.read_csv(events_path) if events_path is not None else None
    skeleton_summary = _read_json_if_exists(
        find_optional_sequence_file(
            root,
            sequence_id,
            "skeleton",
            "skeleton_summary.json",
        )
    )

    frames = gaze.merge(
        head,
        on="query_timestamp_ns",
        how="inner",
        suffixes=("_gaze", "_head"),
    ).merge(
        skeleton,
        on="query_timestamp_ns",
        how="left",
        suffixes=("", "_skeleton"),
    )
    if frames.empty:
        raise ValueError(f"No aligned gaze/head/skeleton rows found: {sequence_id}")
    frames = frames.sort_values("query_timestamp_ns").reset_index(drop=True)
    frames["viewer_frame_index"] = np.arange(len(frames), dtype=int)
    return SceneViewerData(
        sequence_id=sequence_id,
        frames=frames,
        objects=objects,
        hits=hits,
        events=events,
        skeleton_summary=skeleton_summary,
    )


def build_scene_object_gaze_figure(
    data: SceneViewerData,
    start_frame: int = 0,
    end_frame: int | None = 120,
    stride: int = 10,
    focus_frame: int | None = None,
    show_static_objects: bool = True,
    show_dynamic_objects: bool = True,
    show_object_centers: bool = False,
    object_opacity: float = 0.22,
    dynamic_object_opacity: float = 0.75,
    max_static_objects: int | None = None,
    category_filter: str = "",
    exclude_category_filter: str = "shelter",
    show_skeleton: bool = True,
    show_skeleton_trajectory: bool = True,
    show_head_trajectory: bool = True,
    show_gaze_rays: bool = True,
    show_gaze_points: bool = True,
    show_object_hits: bool = True,
    show_hit_object: bool = True,
    gaze_ray_length_m: float = 1.0,
    gaze_scale_mode: str = "fixed",
) -> Any:
    """Build one Plotly 3D figure for a selected Scene-frame window."""

    import plotly.graph_objects as go

    window = slice_frame_window(data.frames, start_frame, end_frame, stride)
    focus_index = _focus_index(
        frames=data.frames,
        start_frame=start_frame,
        end_frame=end_frame,
        focus_frame=focus_frame,
    )
    focus_row = data.frames.iloc[focus_index]
    focus_timestamp_ns = int(focus_row["query_timestamp_ns"])
    categories = _parse_category_filter(category_filter)
    excluded_categories = _parse_category_filter(exclude_category_filter)
    object_rows = select_object_rows(
        data.objects,
        focus_timestamp_ns=focus_timestamp_ns,
        show_static=show_static_objects,
        show_dynamic=show_dynamic_objects,
        categories=categories,
        excluded_categories=excluded_categories,
        max_static_objects=max_static_objects,
    )
    focus_hit = _focus_hit_row(data.hits, focus_timestamp_ns)
    hit_object_rows = _hit_object_rows(data.objects, focus_hit)

    fig = go.Figure()
    plotted_points: list[np.ndarray] = []

    static_objects = object_rows[object_rows["timestamp_ns"] == -1]
    dynamic_objects = object_rows[object_rows["timestamp_ns"] != -1]
    if not static_objects.empty:
        fig.add_trace(
            _box_lines_trace(
                static_objects,
                name="static object boxes",
                color=STATIC_BOX_COLOR,
                opacity=object_opacity,
            )
        )
        plotted_points.append(_object_corner_points(static_objects))
    if not dynamic_objects.empty:
        fig.add_trace(
            _box_lines_trace(
                dynamic_objects,
                name="dynamic object boxes",
                color=DYNAMIC_BOX_COLOR,
                opacity=dynamic_object_opacity,
            )
        )
        plotted_points.append(_object_corner_points(dynamic_objects))
    if show_object_centers and not object_rows.empty:
        fig.add_trace(_object_center_trace(object_rows))
        plotted_points.append(
            object_rows[["scene_t_x_m", "scene_t_y_m", "scene_t_z_m"]].to_numpy(
                dtype=np.float64
            )
        )
    if show_hit_object and not hit_object_rows.empty:
        fig.add_trace(
            _box_lines_trace(
                hit_object_rows,
                name="current ray-box hit outline",
                color=HIT_BOX_COLOR,
                opacity=0.92,
                width=5,
            )
        )
        plotted_points.append(_object_corner_points(hit_object_rows))

    if show_head_trajectory:
        head_points = _finite_xyz(
            window,
            ["head_origin_scene_x_m", "head_origin_scene_y_m", "head_origin_scene_z_m"],
        )
        if len(head_points) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=head_points[:, 0],
                    y=head_points[:, 1],
                    z=head_points[:, 2],
                    mode="lines+markers",
                    name="head/device trajectory",
                    line=dict(color=HEAD_TRAJECTORY_COLOR, width=5),
                    marker=dict(size=3, color=HEAD_TRAJECTORY_COLOR),
                )
            )
            plotted_points.append(head_points)

    if show_skeleton:
        skeleton_traces, skeleton_points = _skeleton_traces(
            data=data,
            row=focus_row,
        )
        for trace in skeleton_traces:
            fig.add_trace(trace)
        if skeleton_points is not None:
            plotted_points.append(skeleton_points)

    if show_skeleton_trajectory:
        root_points = _finite_xyz(
            window,
            [
                "root_joint_scene_x_m",
                "root_joint_scene_y_m",
                "root_joint_scene_z_m",
            ],
        )
        if len(root_points) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=root_points[:, 0],
                    y=root_points[:, 1],
                    z=root_points[:, 2],
                    mode="lines",
                    name="skeleton root trajectory",
                    line=dict(color=SKELETON_BONE_COLOR, width=4),
                )
            )
            plotted_points.append(root_points)

    if show_gaze_rays:
        ray_trace, ray_points = _gaze_ray_trace(
            window,
            length_m=gaze_ray_length_m,
            scale_mode=gaze_scale_mode,
        )
        if ray_trace is not None:
            fig.add_trace(ray_trace)
            plotted_points.append(ray_points)

    if show_gaze_points:
        gaze_points = _finite_xyz(
            window,
            [
                "gaze_point_scene_x_m",
                "gaze_point_scene_y_m",
                "gaze_point_scene_z_m",
            ],
        )
        if len(gaze_points) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=gaze_points[:, 0],
                    y=gaze_points[:, 1],
                    z=gaze_points[:, 2],
                    mode="lines+markers",
                    name="depth-defined gaze points",
                    line=dict(color=GAZE_POINT_COLOR, width=3),
                    marker=dict(size=3, color=GAZE_POINT_COLOR),
                )
            )
            plotted_points.append(gaze_points)

    if show_object_hits and focus_hit is not None:
        hit_trace, hit_points = _hit_trace(focus_hit)
        if hit_trace is not None:
            fig.add_trace(hit_trace)
            plotted_points.append(hit_points)

    _style_scene_figure(
        fig,
        title=(
            f"{data.sequence_id}: Scene objects, skeleton, and gaze "
            f"frames {start_frame}-{end_frame if end_frame is not None else len(data.frames)}"
        ),
        plotted_points=plotted_points,
    )
    return fig


def slice_frame_window(
    frame: pd.DataFrame,
    start_frame: int,
    end_frame: int | None,
    stride: int,
) -> pd.DataFrame:
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if end_frame is not None and end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")
    window = frame.iloc[start_frame:end_frame].copy()
    if window.empty:
        raise ValueError("Selected frame window is empty")
    return window.iloc[::stride].reset_index(drop=True)


def select_object_rows(
    object_frame: pd.DataFrame,
    focus_timestamp_ns: int,
    show_static: bool,
    show_dynamic: bool,
    categories: set[str] | None = None,
    excluded_categories: set[str] | None = None,
    max_static_objects: int | None = None,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    if show_static:
        static_rows = object_frame[object_frame["timestamp_ns"] == -1]
        static_rows = _filter_object_categories(static_rows, categories, excluded_categories)
        if max_static_objects is not None and max_static_objects > 0:
            static_rows = static_rows.head(max_static_objects)
        selected.append(static_rows)

    if show_dynamic:
        dynamic_rows = object_frame[object_frame["timestamp_ns"] != -1]
        dynamic_rows = _filter_object_categories(dynamic_rows, categories, excluded_categories)
        if not dynamic_rows.empty:
            timestamps = np.asarray(sorted(dynamic_rows["timestamp_ns"].unique()), dtype=np.int64)
            nearest_timestamp = int(
                timestamps[np.argmin(np.abs(timestamps - int(focus_timestamp_ns)))]
            )
            selected.append(dynamic_rows[dynamic_rows["timestamp_ns"] == nearest_timestamp])

    if not selected:
        return object_frame.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)


def _filter_object_categories(
    rows: pd.DataFrame,
    categories: set[str] | None,
    excluded_categories: set[str] | None,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    filtered = rows
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if excluded_categories:
        filtered = filtered[~filtered["category"].isin(excluded_categories)]
    return filtered


def _box_lines_trace(
    object_rows: pd.DataFrame,
    name: str,
    color: str,
    opacity: float,
    width: int = 2,
) -> Any:
    import plotly.graph_objects as go

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for _, row in object_rows.iterrows():
        corners = _row_corners(row)
        for first, second in BOX_EDGES:
            xs.extend([corners[first, 0], corners[second, 0], None])
            ys.extend([corners[first, 1], corners[second, 1], None])
            zs.extend([corners[first, 2], corners[second, 2], None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        opacity=opacity,
        hoverinfo="skip",
    )


def _object_center_trace(object_rows: pd.DataFrame) -> Any:
    import plotly.graph_objects as go

    hover = [
        f"{row.get('instance_name', '')}<br>{row.get('category', '')}<br>{row.get('object_uid', '')}"
        for _, row in object_rows.iterrows()
    ]
    return go.Scatter3d(
        x=object_rows["scene_t_x_m"],
        y=object_rows["scene_t_y_m"],
        z=object_rows["scene_t_z_m"],
        mode="markers",
        name="object centers",
        marker=dict(size=2.5, color=OBJECT_CENTER_COLOR),
        text=hover,
        hoverinfo="text",
    )


def _skeleton_traces(data: SceneViewerData, row: pd.Series) -> tuple[list[Any], np.ndarray | None]:
    import plotly.graph_objects as go

    summary = data.skeleton_summary
    if not summary:
        return [], None
    joint_labels = summary.get("joint_labels", [])
    joint_connections = summary.get("joint_connections", [])
    points = []
    for joint_index, label in enumerate(joint_labels):
        safe = _safe_joint_label(label)
        columns = [
            f"joint_{joint_index:02d}_{safe}_scene_x_m",
            f"joint_{joint_index:02d}_{safe}_scene_y_m",
            f"joint_{joint_index:02d}_{safe}_scene_z_m",
        ]
        if not all(column in row.index for column in columns):
            points.append((np.nan, np.nan, np.nan))
            continue
        points.append(tuple(float(row[column]) for column in columns))
    point_array = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(point_array).all(axis=1)
    if not finite.any():
        return [], None

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for first, second in joint_connections:
        if first >= len(point_array) or second >= len(point_array):
            continue
        if not (finite[first] and finite[second]):
            continue
        xs.extend([point_array[first, 0], point_array[second, 0], None])
        ys.extend([point_array[first, 1], point_array[second, 1], None])
        zs.extend([point_array[first, 2], point_array[second, 2], None])

    traces = [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name="skeleton bones",
            line=dict(color=SKELETON_BONE_COLOR, width=5),
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=point_array[finite, 0],
            y=point_array[finite, 1],
            z=point_array[finite, 2],
            mode="markers",
            name="skeleton joints",
            marker=dict(size=3.5, color=SKELETON_JOINT_COLOR),
            text=[joint_labels[index] for index in np.where(finite)[0]],
            hoverinfo="text",
        ),
    ]
    return traces, point_array[finite]


def _gaze_ray_trace(
    window: pd.DataFrame,
    length_m: float,
    scale_mode: str,
) -> tuple[Any | None, np.ndarray]:
    import plotly.graph_objects as go

    origin_columns = [
        "gaze_origin_scene_x_m",
        "gaze_origin_scene_y_m",
        "gaze_origin_scene_z_m",
    ]
    direction_columns = [
        "gaze_dir_scene_unit_x",
        "gaze_dir_scene_unit_y",
        "gaze_dir_scene_unit_z",
    ]
    if not all(column in window.columns for column in (*origin_columns, *direction_columns)):
        return None, np.zeros((0, 3), dtype=np.float64)

    origins_all = window[origin_columns].to_numpy(dtype=np.float64)
    dirs_all = window[direction_columns].to_numpy(dtype=np.float64)
    valid = np.isfinite(origins_all).all(axis=1) & np.isfinite(dirs_all).all(axis=1)
    origins = origins_all[valid]
    dirs = dirs_all[valid]
    if len(origins) == 0:
        return None, origins

    if scale_mode == "depth" and "depth_m" in window.columns:
        depth = window["depth_m"].to_numpy(dtype=np.float64)[valid]
        lengths = np.where(np.isfinite(depth) & (depth > 0), depth, length_m)
    else:
        lengths = np.full(len(origins), float(length_m), dtype=np.float64)

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    points: list[np.ndarray] = []
    for origin, direction, ray_length in zip(origins, dirs, lengths):
        if not np.isfinite(direction).all():
            continue
        end = origin + direction * float(ray_length)
        xs.extend([origin[0], end[0], None])
        ys.extend([origin[1], end[1], None])
        zs.extend([origin[2], end[2], None])
        points.extend([origin, end])
    if not points:
        return None, origins
    return (
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name="gaze rays",
            line=dict(color=GAZE_RAY_COLOR, width=4),
            opacity=0.65,
            hoverinfo="skip",
        ),
        np.vstack(points),
    )


def _focus_hit_row(hits: pd.DataFrame | None, focus_timestamp_ns: int) -> pd.Series | None:
    if hits is None or hits.empty or "query_timestamp_ns" not in hits.columns:
        return None
    timestamps = hits["query_timestamp_ns"].to_numpy(dtype=np.int64)
    index = int(np.argmin(np.abs(timestamps - int(focus_timestamp_ns))))
    return hits.iloc[index]


def _hit_object_rows(objects: pd.DataFrame, hit_row: pd.Series | None) -> pd.DataFrame:
    if hit_row is None or not _row_bool(hit_row, "object_hit"):
        return objects.iloc[0:0].copy()
    object_uid = _normalize_uid(hit_row.get("hit_object_uid"))
    if object_uid is None:
        return objects.iloc[0:0].copy()
    object_rows = objects[objects["object_uid"].map(_normalize_uid) == object_uid]
    if object_rows.empty:
        return object_rows
    hit_object_timestamp = hit_row.get("hit_object_timestamp_ns")
    if pd.isna(hit_object_timestamp):
        return object_rows.iloc[0:0].copy()
    timestamp = int(hit_object_timestamp)
    if timestamp == -1:
        return object_rows[object_rows["timestamp_ns"] == -1]
    dynamic_rows = object_rows[object_rows["timestamp_ns"] != -1]
    if dynamic_rows.empty:
        return dynamic_rows
    timestamps = dynamic_rows["timestamp_ns"].to_numpy(dtype=np.int64)
    nearest = int(timestamps[np.argmin(np.abs(timestamps - timestamp))])
    return dynamic_rows[dynamic_rows["timestamp_ns"] == nearest]


def _hit_trace(hit_row: pd.Series) -> tuple[Any | None, np.ndarray]:
    import plotly.graph_objects as go

    if not _row_bool(hit_row, "object_hit"):
        return None, np.zeros((0, 3), dtype=np.float64)
    point = np.asarray(
        [
            _row_float(hit_row, "hit_x_m"),
            _row_float(hit_row, "hit_y_m"),
            _row_float(hit_row, "hit_z_m"),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(point).all():
        return None, np.zeros((0, 3), dtype=np.float64)
    hover = (
        f"{hit_row.get('hit_instance_name', '')}<br>"
        f"{hit_row.get('hit_category', '')}<br>"
        f"distance={_row_float(hit_row, 'hit_distance_m'):.3f} m"
    )
    trace = go.Scatter3d(
        x=[point[0]],
        y=[point[1]],
        z=[point[2]],
        mode="markers",
        name="current ray-box hit point",
        marker=dict(size=8, color=HIT_POINT_COLOR, symbol="diamond"),
        text=[hover],
        hoverinfo="text",
    )
    return trace, point.reshape(1, 3)


def _row_bool(row: pd.Series, column: str) -> bool:
    value = row.get(column)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _row_float(row: pd.Series, column: str) -> float:
    value = row.get(column)
    return float(value) if value is not None and not pd.isna(value) else float("nan")


def _normalize_uid(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value)
    if text.endswith(".0"):
        candidate = text[:-2]
        if candidate.isdigit():
            return candidate
    return text


def _style_scene_figure(
    fig: Any,
    title: str,
    plotted_points: Sequence[np.ndarray],
) -> None:
    ranges = _axis_ranges(plotted_points)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="Scene X [m]", range=ranges[0]),
            yaxis=dict(title="Scene Y [m]", range=ranges[1]),
            zaxis=dict(title="Scene Z [m]", range=ranges[2]),
            aspectmode="data",
            camera=dict(
                up=dict(x=0, y=1, z=0),
                eye=dict(x=1.55, y=0.9, z=1.55),
            ),
        ),
        margin=dict(l=0, r=0, t=48, b=0),
        height=760,
        legend=dict(x=0.01, y=0.99),
    )


def _axis_ranges(points_list: Sequence[np.ndarray]) -> list[list[float]]:
    finite_points = [
        points[np.isfinite(points).all(axis=1)]
        for points in points_list
        if points is not None and len(points) > 0
    ]
    if not finite_points:
        return [[-1, 1], [-1, 1], [-1, 1]]
    points = np.vstack(finite_points)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    spans = maxs - mins
    ranges: list[list[float]] = []
    for axis in range(3):
        span = float(spans[axis])
        if span <= 1e-6:
            center = float((mins[axis] + maxs[axis]) / 2.0)
            ranges.append([center - 0.5, center + 0.5])
            continue
        pad = max(span * 0.08, 0.15)
        ranges.append([float(mins[axis] - pad), float(maxs[axis] + pad)])
    return ranges


def _row_corners(row: pd.Series) -> np.ndarray:
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


def _object_corner_points(object_rows: pd.DataFrame) -> np.ndarray:
    return np.vstack([_row_corners(row) for _, row in object_rows.iterrows()])


def _finite_xyz(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not all(column in frame.columns for column in columns):
        return np.zeros((0, 3), dtype=np.float64)
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    return values[np.isfinite(values).all(axis=1)]


def _focus_index(
    frames: pd.DataFrame,
    start_frame: int,
    end_frame: int | None,
    focus_frame: int | None,
) -> int:
    if focus_frame is not None:
        return int(np.clip(focus_frame, 0, len(frames) - 1))
    if end_frame is None:
        end_frame = len(frames)
    return int(np.clip((start_frame + end_frame - 1) // 2, 0, len(frames) - 1))


def _parse_category_filter(value: str) -> set[str] | None:
    categories = {item.strip() for item in value.split(",") if item.strip()}
    return categories or None


def _safe_joint_label(label: str) -> str:
    import re

    safe = re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_")
    return safe or "joint"


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_optional_sequence_file(
    root: Path,
    sequence_id: str,
    layer: str,
    filename: str,
) -> Path | None:
    try:
        return find_sequence_file(root, sequence_id, layer, filename)
    except FileNotFoundError:
        return None
