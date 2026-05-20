"""Multi-view gaze/head/scene dashboard helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .scene_object_viewer import (
    SceneViewerData,
    build_scene_object_gaze_figure,
    load_scene_viewer_data,
    slice_frame_window,
)
from .viz_palette import (
    EVENT_FILL_COLORS,
    EVENT_LINE_COLORS,
    GAZE_TRACK_COLORS,
    NEUTRAL,
    SEMANTIC_COLORS,
)


@dataclass(frozen=True)
class GazeTrack:
    """One optional gaze track, e.g. GT or model prediction."""

    name: str
    frame: pd.DataFrame
    color: str


DEFAULT_TRACK_COLORS = GAZE_TRACK_COLORS
FOCUS_LINE = SEMANTIC_COLORS["head"]
HEAD_COLOR = SEMANTIC_COLORS["head"]
HIT_COLOR = SEMANTIC_COLORS["hit"]
HIT_GAP_COLOR = SEMANTIC_COLORS["hit_gap"]
MISS_COLOR = NEUTRAL["miss"]
IMAGE_BOUNDARY = SEMANTIC_COLORS["image_boundary"]
IMAGE_FILL = SEMANTIC_COLORS["image_fill"]

SCENE_LEGEND_NAMES = {
    "static object boxes",
    "dynamic object boxes",
    "current ray-box hit outline",
    "head/device trajectory",
    "skeleton bones",
    "gaze rays",
    "depth-defined gaze points",
    "current ray-box hit point",
}


def load_multiview_data(reports_dir: str | Path, sequence_id: str) -> SceneViewerData:
    """Load the same precomputed layers used by the scene viewer."""

    return load_scene_viewer_data(reports_dir, sequence_id)


def build_multiview_dashboard(
    data: SceneViewerData,
    start_frame: int = 0,
    end_frame: int | None = 180,
    stride: int = 5,
    focus_frame: int | None = None,
    prediction_tracks: Mapping[str, pd.DataFrame] | None = None,
    show_static_objects: bool = True,
    show_dynamic_objects: bool = True,
    max_static_objects: int | None = 120,
    category_filter: str = "",
    exclude_category_filter: str = "shelter",
    show_skeleton: bool = True,
    show_gaze_points: bool = True,
    show_hit_object: bool = True,
    show_object_hits: bool = True,
    gaze_ray_length_m: float = 1.0,
    gaze_scale_mode: str = "fixed",
) -> object:
    """Build a coordinated Plotly dashboard with temporal, image, and 3D views.

    Prediction tracks can be passed later as dataframes with the same gaze
    columns as `gaze_samples.csv`. If a track has `query_timestamp_ns`, it is
    aligned to the GT rows to inherit `viewer_frame_index`.
    """

    from plotly.subplots import make_subplots

    window = slice_frame_window(data.frames, start_frame, end_frame, stride)
    full_window = slice_frame_window(data.frames, start_frame, end_frame, 1)
    focus_index = _focus_index(data.frames, start_frame, end_frame, focus_frame)
    focus_row = data.frames.iloc[focus_index]
    focus_frame_index = int(focus_row["viewer_frame_index"])
    tracks = _build_tracks(data.frames, prediction_tracks)

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "scene", "colspan": 2}, None],
        ],
        row_heights=[0.26, 0.22, 0.52],
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        subplot_titles=(
            "A. Local eye-in-head gaze",
            "B. Motion magnitude and event context",
            "C. RGB image-space gaze",
            "D. Object-hit context",
            "E. 3D scene context",
        ),
    )

    events_window = _event_window(data.events, full_window)
    _add_event_background(fig, events_window, row=1, col=1)
    _add_yaw_pitch_view(fig, window, tracks, focus_frame_index, row=1, col=1)
    _add_event_background(fig, events_window, row=1, col=2)
    _add_motion_dynamics_view(fig, window, tracks, focus_frame_index, row=1, col=2)
    _add_image_space_view(fig, window, tracks, focus_row, row=2, col=1)
    _add_event_background(fig, events_window, row=2, col=2)
    _add_hit_context_view(fig, window, data.hits, events_window, focus_frame_index, row=2, col=2)

    scene_fig = build_scene_object_gaze_figure(
        data,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        focus_frame=focus_index,
        show_static_objects=show_static_objects,
        show_dynamic_objects=show_dynamic_objects,
        max_static_objects=max_static_objects,
        category_filter=category_filter,
        exclude_category_filter=exclude_category_filter,
        show_skeleton=show_skeleton,
        show_gaze_points=show_gaze_points,
        show_object_hits=show_object_hits,
        show_hit_object=show_hit_object,
        gaze_ray_length_m=gaze_ray_length_m,
        gaze_scale_mode=gaze_scale_mode,
    )
    for trace in scene_fig.data:
        trace.showlegend = trace.name in SCENE_LEGEND_NAMES
        if trace.showlegend:
            trace.legendgroup = f"scene::{trace.name}"
        fig.add_trace(trace, row=3, col=1)
    for trace in _current_direction_traces(
        focus_row,
        prediction_tracks=tracks[1:],
        gaze_ray_length_m=gaze_ray_length_m,
    ):
        trace.showlegend = True
        trace.legendgroup = f"scene::{trace.name}"
        fig.add_trace(trace, row=3, col=1)
    if scene_fig.layout.scene:
        scene_layout = scene_fig.layout.scene
        fig.update_scenes(
            xaxis=scene_layout.xaxis,
            yaxis=scene_layout.yaxis,
            zaxis=scene_layout.zaxis,
            aspectmode=scene_layout.aspectmode,
            camera=scene_layout.camera,
            row=3,
            col=1,
        )

    fig.update_xaxes(title_text="Frame", row=1, col=1)
    fig.update_yaxes(title_text="Angle [deg]", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=2)
    fig.update_yaxes(title_text="Velocity / speed", row=1, col=2)
    fig.update_xaxes(title_text="u [px]", row=2, col=1)
    fig.update_yaxes(title_text="v [px]", row=2, col=1)
    _set_image_axis_ranges(fig, focus_row, row=2, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=2)
    fig.update_yaxes(title_text="Event / distance [m]", row=2, col=2)
    fig.update_layout(
        title=dict(
            text=f"{data.sequence_id}: multi-view gaze dashboard",
            x=0.01,
            y=0.985,
            xanchor="left",
        ),
        template="plotly_white",
        height=1500,
        paper_bgcolor=NEUTRAL["paper"],
        plot_bgcolor=NEUTRAL["plot"],
        margin=dict(l=58, r=260, t=96, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.985,
            xanchor="left",
            x=1.015,
            font=dict(size=12),
            tracegroupgap=5,
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor=NEUTRAL["legend_border"],
            borderwidth=1,
            itemwidth=30,
        ),
        hovermode="closest",
        font=dict(color=NEUTRAL["text"], family="Arial"),
    )
    fig.update_annotations(font=dict(color=NEUTRAL["text"], size=14))
    fig.update_xaxes(
        showgrid=True,
        gridcolor=NEUTRAL["grid"],
        zeroline=False,
        linecolor=NEUTRAL["axis"],
        tickfont=dict(color=NEUTRAL["text"]),
        title_font=dict(color=NEUTRAL["text"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=NEUTRAL["grid"],
        zeroline=False,
        linecolor=NEUTRAL["axis"],
        tickfont=dict(color=NEUTRAL["text"]),
        title_font=dict(color=NEUTRAL["text"]),
    )
    fig.update_scenes(
        xaxis=dict(
            backgroundcolor=NEUTRAL["plot"],
            gridcolor=NEUTRAL["grid"],
            zerolinecolor=NEUTRAL["axis"],
        ),
        yaxis=dict(
            backgroundcolor=NEUTRAL["plot"],
            gridcolor=NEUTRAL["grid"],
            zerolinecolor=NEUTRAL["axis"],
        ),
        zaxis=dict(
            backgroundcolor=NEUTRAL["plot"],
            gridcolor=NEUTRAL["grid"],
            zerolinecolor=NEUTRAL["axis"],
        ),
    )
    return fig


def _build_tracks(
    gt_frame: pd.DataFrame,
    prediction_tracks: Mapping[str, pd.DataFrame] | None,
) -> list[GazeTrack]:
    tracks = [GazeTrack("GT", gt_frame, DEFAULT_TRACK_COLORS[0])]
    if not prediction_tracks:
        return tracks
    for index, (name, frame) in enumerate(prediction_tracks.items(), start=1):
        color = DEFAULT_TRACK_COLORS[index % len(DEFAULT_TRACK_COLORS)]
        tracks.append(GazeTrack(name, _align_prediction_frame(gt_frame, frame), color))
    return tracks


def _align_prediction_frame(gt_frame: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if "viewer_frame_index" in frame.columns:
        return frame
    if "query_timestamp_ns" not in frame.columns:
        aligned = frame.copy()
        aligned["viewer_frame_index"] = np.arange(len(aligned), dtype=int)
        return aligned
    columns = ["query_timestamp_ns", "viewer_frame_index"]
    return frame.merge(gt_frame[columns], on="query_timestamp_ns", how="left")


def _add_yaw_pitch_view(
    fig: object,
    window: pd.DataFrame,
    tracks: Sequence[GazeTrack],
    focus_frame_index: int,
    row: int,
    col: int,
) -> None:
    import plotly.graph_objects as go

    y_values_for_focus_line: list[float] = []
    for track in tracks:
        track_window = _track_window(track.frame, window)
        x = track_window["viewer_frame_index"]
        for column, suffix, dash in [
            ("yaw_rad", "yaw", "solid"),
            ("pitch_rad", "pitch", "dot"),
        ]:
            if column not in track_window.columns:
                continue
            y = np.rad2deg(track_window[column].to_numpy(dtype=np.float64))
            finite_y = y[np.isfinite(y)]
            y_values_for_focus_line.extend(float(value) for value in finite_y)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"{track.name} {suffix}",
                    line=dict(color=track.color, dash=dash, width=2),
                ),
                row=row,
                col=col,
            )
    if y_values_for_focus_line:
        y_min, y_max = min(y_values_for_focus_line), max(y_values_for_focus_line)
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        fig.add_trace(
            go.Scatter(
                x=[focus_frame_index, focus_frame_index],
                y=[y_min, y_max],
                mode="lines",
                name="focus frame",
                line=dict(color=FOCUS_LINE, width=1.5, dash="dash"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def _add_image_space_view(
    fig: object,
    window: pd.DataFrame,
    tracks: Sequence[GazeTrack],
    focus_row: pd.Series,
    row: int,
    col: int,
) -> None:
    import plotly.graph_objects as go

    focus_index = int(focus_row["viewer_frame_index"])
    width = _finite_scalar(focus_row.get("image_width_px"))
    height = _finite_scalar(focus_row.get("image_height_px"))
    if width is not None and height is not None:
        fig.add_trace(
            go.Scatter(
                x=[0, width, width, 0, 0],
                y=[0, 0, height, height, 0],
                mode="lines",
                name="image boundary",
                line=dict(color=IMAGE_BOUNDARY, width=1),
                fill="toself",
                fillcolor=IMAGE_FILL,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
    for track in tracks:
        track_window = _track_window(track.frame, window)
        if not {"gaze_u_px", "gaze_v_px"}.issubset(track_window.columns):
            continue
        fig.add_trace(
            go.Scatter(
                x=track_window["gaze_u_px"],
                y=track_window["gaze_v_px"],
                mode="lines+markers",
                name=f"{track.name} image",
                line=dict(color=track.color, width=2),
                marker=dict(color=track.color, size=5),
                showlegend=False,
                hovertemplate=(
                    f"{track.name}<br>"
                    "u=%{x:.1f}px<br>v=%{y:.1f}px<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        focus_track = track_window[track_window["viewer_frame_index"] == focus_index]
        if not focus_track.empty:
            focus = focus_track.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[focus.get("gaze_u_px")],
                    y=[focus.get("gaze_v_px")],
                    mode="markers",
                    name=f"{track.name} focus image",
                    marker=dict(color=track.color, size=11, symbol="x"),
                    showlegend=False,
                    hovertemplate=(
                        f"{track.name} focus<br>"
                        "u=%{x:.1f}px<br>v=%{y:.1f}px<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )


def _add_motion_dynamics_view(
    fig: object,
    window: pd.DataFrame,
    tracks: Sequence[GazeTrack],
    focus_frame_index: int,
    row: int,
    col: int,
) -> None:
    import plotly.graph_objects as go

    y_values: list[float] = []
    gt_window = _track_window(tracks[0].frame, window)
    for track in tracks:
        track_window = _track_window(track.frame, window)
        if track_window.empty:
            continue
        frame_index = track_window["viewer_frame_index"].to_numpy(dtype=np.float64)
        local_velocity = _cpf_local_gaze_velocity_deg_s(track_window)
        scene_velocity = _scene_gaze_velocity_deg_s(track_window)
        if _has_finite(local_velocity):
            y_values.extend(_finite_values(local_velocity))
            fig.add_trace(
                go.Scatter(
                    x=frame_index,
                    y=local_velocity,
                    mode="lines",
                    name=f"{track.name} CPF vel",
                    line=dict(color=track.color, width=2),
                ),
                row=row,
                col=col,
            )
        if _has_finite(scene_velocity):
            y_values.extend(_finite_values(scene_velocity))
            fig.add_trace(
                go.Scatter(
                    x=frame_index,
                    y=scene_velocity,
                    mode="lines",
                    name=f"{track.name} Scene vel",
                    line=dict(color=track.color, width=2, dash="dot"),
                ),
                row=row,
                col=col,
            )

    if "head_rotation_speed_deg_s" in gt_window.columns:
        head_speed = gt_window["head_rotation_speed_deg_s"].to_numpy(dtype=np.float64)
        if _has_finite(head_speed):
            y_values.extend(_finite_values(head_speed))
            fig.add_trace(
                go.Scatter(
                    x=gt_window["viewer_frame_index"],
                    y=head_speed,
                    mode="lines",
                    name="head rot speed",
                    line=dict(color=HEAD_COLOR, width=2),
                ),
                row=row,
                col=col,
            )
    _add_focus_line(fig, focus_frame_index, y_values, row=row, col=col)


def _add_hit_context_view(
    fig: object,
    window: pd.DataFrame,
    hits: pd.DataFrame | None,
    events_window: pd.DataFrame | None,
    focus_frame_index: int,
    row: int,
    col: int,
) -> None:
    import plotly.graph_objects as go

    _add_event_timeline(fig, events_window, row=row, col=col)
    if hits is None or hits.empty or "query_timestamp_ns" not in hits.columns:
        _add_empty_panel_annotation(fig, "No gaze_object_hits.csv loaded", row=row, col=col)
        return
    hit_window = hits.merge(
        window[["query_timestamp_ns", "viewer_frame_index"]],
        on="query_timestamp_ns",
        how="inner",
    )
    if hit_window.empty:
        _add_empty_panel_annotation(fig, "No hit rows in selected window", row=row, col=col)
        return

    frames = hit_window["viewer_frame_index"]
    y_values: list[float] = []
    if "hit_distance_m" in hit_window.columns:
        hit_distance = hit_window["hit_distance_m"].to_numpy(dtype=np.float64)
        y_values.extend(_finite_values(hit_distance))
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=hit_distance,
                mode="lines+markers",
                name="hit distance",
                line=dict(color=HIT_COLOR, width=2),
                marker=dict(
                    color=[
                        HIT_COLOR if _bool_value(value) else MISS_COLOR
                        for value in hit_window["object_hit"]
                    ],
                    size=5,
                ),
                legendgroup="object_hit",
                showlegend=True,
                hovertemplate="frame=%{x}<br>hit distance=%{y:.3f} m<extra></extra>",
            ),
            row=row,
            col=col,
        )
    if "gaze_point_to_hit_distance_m" in hit_window.columns:
        point_gap = hit_window["gaze_point_to_hit_distance_m"].to_numpy(dtype=np.float64)
        if _has_finite(point_gap):
            y_values.extend(_finite_values(point_gap))
            fig.add_trace(
                go.Scatter(
                    x=frames,
                    y=point_gap,
                    mode="lines",
                    name="depth point vs hit",
                    line=dict(color=HIT_GAP_COLOR, width=1.8, dash="dot"),
                    legendgroup="object_hit",
                    showlegend=True,
                    hovertemplate="frame=%{x}<br>depth-hit gap=%{y:.3f} m<extra></extra>",
                ),
                row=row,
                col=col,
            )
    _add_focus_line(fig, focus_frame_index, y_values or [0.0, 1.0], row=row, col=col)

    focus_rows = hit_window[hit_window["viewer_frame_index"] == focus_frame_index]
    if not focus_rows.empty:
        focus_hit = focus_rows.iloc[0]
        label = str(focus_hit.get("hit_category") or "no hit")
        if focus_hit.get("hit_instance_name") and not pd.isna(focus_hit.get("hit_instance_name")):
            label = f"{label}<br>{focus_hit.get('hit_instance_name')}"
        fig.add_annotation(
            text=label,
            xref=f"x{_axis_suffix(row, col)}",
            yref=f"y{_axis_suffix(row, col)}",
            x=focus_frame_index,
            y=max(y_values) if y_values else 1.0,
            showarrow=True,
            arrowhead=2,
            font=dict(size=10),
        )


def _event_window(events: pd.DataFrame | None, window: pd.DataFrame) -> pd.DataFrame | None:
    if events is None or events.empty or "query_timestamp_ns" not in events.columns:
        return None
    return events.merge(
        window[["query_timestamp_ns", "viewer_frame_index"]],
        on="query_timestamp_ns",
        how="inner",
    )


def _add_event_background(
    fig: object,
    events_window: pd.DataFrame | None,
    *,
    row: int,
    col: int,
) -> None:
    if events_window is None or events_window.empty:
        return
    label_column = "scene_event_label"
    if label_column not in events_window.columns:
        return
    suffix = _axis_suffix(row, col)
    xref = f"x{suffix}"
    yref = f"y{suffix} domain"
    for start, end, label in _event_runs(events_window):
        color = EVENT_FILL_COLORS.get(label, EVENT_FILL_COLORS["invalid"])
        fig.add_shape(
            type="rect",
            x0=start - 0.5,
            x1=end + 0.5,
            y0=0,
            y1=1,
            xref=xref,
            yref=yref,
            fillcolor=color,
            line_width=0,
            layer="below",
        )


def _add_event_timeline(
    fig: object,
    events_window: pd.DataFrame | None,
    *,
    row: int,
    col: int,
) -> None:
    import plotly.graph_objects as go

    if events_window is None or events_window.empty or "scene_event_label" not in events_window.columns:
        return
    labels = events_window["scene_event_label"].fillna("invalid")
    frame_indices = events_window["viewer_frame_index"]
    y_map = {"invalid": -0.12, "transition": 0.0, "fixation": 0.12}
    for label in ("fixation", "transition", "invalid"):
        mask = labels == label
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=frame_indices[mask],
                y=[y_map[label]] * int(mask.sum()),
                mode="markers",
                name=f"event: {label}",
                marker=dict(
                    color=EVENT_LINE_COLORS.get(label, EVENT_LINE_COLORS["invalid"]),
                    size=7,
                    symbol="square",
                ),
                showlegend=True,
                hovertemplate=f"frame=%{{x}}<br>event={label}<extra></extra>",
            ),
            row=row,
            col=col,
        )


def _event_runs(events_window: pd.DataFrame) -> list[tuple[int, int, str]]:
    if events_window.empty or "scene_event_label" not in events_window.columns:
        return []
    rows = events_window.sort_values("viewer_frame_index")
    runs: list[tuple[int, int, str]] = []
    current_label = str(rows.iloc[0]["scene_event_label"])
    start = int(rows.iloc[0]["viewer_frame_index"])
    prev = start
    for _, row in rows.iloc[1:].iterrows():
        frame = int(row["viewer_frame_index"])
        label = str(row["scene_event_label"])
        if label != current_label or frame != prev + 1:
            runs.append((start, prev, current_label))
            start = frame
            current_label = label
        prev = frame
    runs.append((start, prev, current_label))
    return runs


def _current_direction_traces(
    focus_row: pd.Series,
    *,
    prediction_tracks: Sequence[GazeTrack],
    gaze_ray_length_m: float,
) -> list[object]:
    import plotly.graph_objects as go

    traces: list[object] = []
    origin = _row_vector(
        focus_row,
        ["head_origin_scene_x_m", "head_origin_scene_y_m", "head_origin_scene_z_m"],
    )
    if origin is None:
        return traces
    head_dir = _row_vector(
        focus_row,
        [
            "head_forward_scene_unit_x",
            "head_forward_scene_unit_y",
            "head_forward_scene_unit_z",
        ],
    )
    if head_dir is not None:
        end = origin + head_dir * 0.45
        traces.append(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                name="current head forward",
                line=dict(color=HEAD_COLOR, width=8),
            )
        )
    for track in prediction_tracks:
        if "viewer_frame_index" not in track.frame.columns:
            continue
        matching = track.frame[
            track.frame["viewer_frame_index"] == int(focus_row["viewer_frame_index"])
        ]
        if matching.empty:
            continue
        pred_row = matching.iloc[0]
        pred_dir = _row_vector(
            pred_row,
            [
                "gaze_dir_scene_unit_x",
                "gaze_dir_scene_unit_y",
                "gaze_dir_scene_unit_z",
            ],
        )
        if pred_dir is None:
            continue
        end = origin + pred_dir * gaze_ray_length_m
        traces.append(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                name=f"{track.name} current scene ray",
                line=dict(color=track.color, width=7, dash="dash"),
            )
        )
    return traces


def _track_window(track_frame: pd.DataFrame, reference_window: pd.DataFrame) -> pd.DataFrame:
    if "viewer_frame_index" not in track_frame.columns:
        return track_frame.iloc[0:0].copy()
    start = int(reference_window["viewer_frame_index"].min())
    end = int(reference_window["viewer_frame_index"].max())
    return track_frame[
        (track_frame["viewer_frame_index"] >= start)
        & (track_frame["viewer_frame_index"] <= end)
    ].copy()


def _cpf_local_gaze_velocity_deg_s(frame: pd.DataFrame) -> np.ndarray:
    if not {"yaw_rad", "pitch_rad", "query_timestamp_ns"}.issubset(frame.columns):
        return np.full(len(frame), np.nan, dtype=np.float64)
    yaw = frame["yaw_rad"].to_numpy(dtype=np.float64)
    pitch = frame["pitch_rad"].to_numpy(dtype=np.float64)
    timestamps = frame["query_timestamp_ns"].to_numpy(dtype=np.float64) / 1e9
    velocity = np.full(len(frame), np.nan, dtype=np.float64)
    if len(frame) < 2:
        return velocity
    dt = np.diff(timestamps)
    delta = np.sqrt(np.diff(yaw) ** 2 + np.diff(pitch) ** 2)
    valid = np.isfinite(delta) & np.isfinite(dt) & (dt > 0)
    velocity[1:][valid] = np.rad2deg(delta[valid]) / dt[valid]
    return velocity


def _scene_gaze_velocity_deg_s(frame: pd.DataFrame) -> np.ndarray:
    columns = [
        "gaze_dir_scene_unit_x",
        "gaze_dir_scene_unit_y",
        "gaze_dir_scene_unit_z",
    ]
    if not set(columns + ["query_timestamp_ns"]).issubset(frame.columns):
        return np.full(len(frame), np.nan, dtype=np.float64)
    dirs = frame[columns].to_numpy(dtype=np.float64)
    timestamps = frame["query_timestamp_ns"].to_numpy(dtype=np.float64) / 1e9
    velocity = np.full(len(frame), np.nan, dtype=np.float64)
    if len(frame) < 2:
        return velocity
    prev = dirs[:-1]
    cur = dirs[1:]
    dt = np.diff(timestamps)
    valid = (
        np.isfinite(prev).all(axis=1)
        & np.isfinite(cur).all(axis=1)
        & np.isfinite(dt)
        & (dt > 0)
    )
    dots = np.einsum("ij,ij->i", prev, cur)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.rad2deg(np.arccos(dots))
    velocity[1:][valid] = angles[valid] / dt[valid]
    return velocity


def _add_focus_line(
    fig: object,
    focus_frame_index: int,
    y_values: Sequence[float],
    *,
    row: int,
    col: int,
) -> None:
    import plotly.graph_objects as go

    finite = [float(value) for value in y_values if np.isfinite(value)]
    if not finite:
        finite = [0.0, 1.0]
    y_min, y_max = min(finite), max(finite)
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    fig.add_trace(
        go.Scatter(
            x=[focus_frame_index, focus_frame_index],
            y=[y_min, y_max],
            mode="lines",
            name="focus frame",
            line=dict(color=FOCUS_LINE, width=1.4, dash="dash"),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _add_empty_panel_annotation(fig: object, text: str, *, row: int, col: int) -> None:
    suffix = _axis_suffix(row, col)
    fig.add_annotation(
        text=text,
        x=0.5,
        y=0.5,
        xref=f"x{suffix} domain",
        yref=f"y{suffix} domain",
        showarrow=False,
        font=dict(size=11, color="gray"),
    )


def _axis_suffix(row: int, col: int) -> str:
    index = (row - 1) * 2 + col
    return "" if index == 1 else str(index)


def _finite_values(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [float(value) for value in array[np.isfinite(array)]]


def _has_finite(values: Sequence[float]) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).any())


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    if value is None or pd.isna(value):
        return False
    return bool(value)


def _set_image_axis_ranges(fig: object, focus_row: pd.Series, row: int, col: int) -> None:
    width = _finite_scalar(focus_row.get("image_width_px"))
    height = _finite_scalar(focus_row.get("image_height_px"))
    if width is None or height is None:
        return
    suffix = _axis_suffix(row, col)
    fig.update_xaxes(
        range=[0, width],
        constrain="domain",
        row=row,
        col=col,
    )
    fig.update_yaxes(
        range=[height, 0],
        scaleanchor=f"x{suffix}",
        scaleratio=1,
        constrain="domain",
        row=row,
        col=col,
    )


def _row_vector(row: pd.Series, columns: Sequence[str]) -> np.ndarray | None:
    if not all(column in row.index for column in columns):
        return None
    values = np.asarray([row[column] for column in columns], dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    norm = np.linalg.norm(values)
    if norm <= 1e-12:
        return values
    return values / norm


def _finite_scalar(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


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
