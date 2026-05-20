"""Static gaze output visualization helpers.

This module owns reusable gaze visualization logic.  CLI scripts should stay
thin: parse arguments, locate input files, then call these functions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np

from adt_sandbox.gaze import (
    RGB_STREAM_ID,
    GazeSample,
    default_summary_json_path,
    get_rgb_image,
    project_scene_points_to_rgb,
    read_gaze_summary_json,
)


def generate_gaze_output_visualizations(
    *,
    gt_provider: Any,
    csv_path: Path,
    samples: list[GazeSample],
    sequence_name: str,
    output_root: Path,
    start_row: int,
    end_row: int | None,
    stride: int,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Generate the standard gaze diagnostic figures for one CSV row window.

    The function only consumes extracted gaze rows plus an ADT provider for RGB
    image lookup.  Keeping this outside the CLI makes the same visualizations
    reusable from notebooks and future model-output comparison scripts.
    """

    indexed_samples = list(enumerate(samples))
    window_pairs = slice_items(indexed_samples, start_row, end_row)
    viz_pairs = downsample_pairs(window_pairs, stride, include_last=True)
    viz_orders = [row_index for row_index, _ in viz_pairs]
    viz_samples = [sample for _, sample in viz_pairs]
    stream_id_value, make_upright = load_visualization_context(csv_path)

    resolved_run_name = run_name or default_run_name(start_row, end_row, stride, len(samples))
    output_dir = output_root / sequence_name / "visualizations" / resolved_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    write_scene_rays_plot(output_dir / "gaze_scene_rays.png", viz_samples)

    # The scanpath is projected into a single reference RGB frame.  This gives
    # an image-space view of the 3D gaze points without regenerating the full
    # extraction pipeline.
    scanpath = reference_scanpath_from_samples(
        gt_provider,
        viz_samples,
        viz_orders,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    write_reference_frame_scanpath_overlay(
        output_dir / "gaze_reference_frame_scanpath_overlay.png",
        scanpath,
    )
    write_reference_frame_scanpath_clean(
        output_dir / "gaze_reference_frame_scanpath_clean.png",
        scanpath,
    )

    overlay_dir = output_dir / "overlays"
    overlay_paths = write_overlay_frames(
        gt_provider,
        viz_pairs,
        overlay_dir,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    write_video_from_overlay_frames(output_dir / "gaze_overlay_video.mp4", overlay_paths)

    return {
        "output_dir": output_dir,
        "window_samples": len(window_pairs),
        "viz_samples": len(viz_samples),
        "make_upright": make_upright,
    }


def slice_items(items: list[Any], start_row: int, end_row: int | None) -> list[Any]:
    """Select a non-empty row window using Python slice semantics."""

    if start_row < 0:
        raise ValueError("start_row must be non-negative")
    if end_row is not None and end_row <= start_row:
        raise ValueError("end_row must be greater than start_row")
    selected = items[start_row:end_row]
    if not selected:
        raise ValueError("No rows selected; check --start-row and --end-row")
    return selected


def downsample_pairs(
    indexed_samples: list[tuple[int, GazeSample]],
    stride: int,
    include_last: bool,
) -> list[tuple[int, GazeSample]]:
    """Downsample selected rows while optionally preserving the window endpoint."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    selected = list(indexed_samples[::stride])
    if include_last and indexed_samples and selected[-1] != indexed_samples[-1]:
        selected.append(indexed_samples[-1])
    return selected


def load_visualization_context(csv_path: Path) -> tuple[str, bool]:
    """Read stream id and image orientation from the paired summary JSON."""

    summary_path = default_summary_json_path(csv_path)
    if not summary_path.exists():
        return RGB_STREAM_ID, True

    summary = read_gaze_summary_json(summary_path)
    stream_id_value = str(summary.get("stream_id", RGB_STREAM_ID))
    make_upright = summary.get("image_orientation", "upright") != "raw"
    return stream_id_value, make_upright


def default_run_name(start_row: int, end_row: int | None, stride: int, total_rows: int) -> str:
    """Build a stable output subdirectory name for a row window."""

    end_label = end_row if end_row is not None else total_rows
    return f"rows_{start_row}_{end_label}_stride_{stride}"


def write_image(path: os.PathLike[str] | str, image: np.ndarray) -> None:
    """Write an RGB image array to disk."""

    import imageio.v2 as imageio

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, image)


def save_overlay(
    path: os.PathLike[str] | str,
    image: Any,
    sample: GazeSample,
    image_dt_ns: int,
    make_upright: bool,
) -> None:
    """Save one diagnostic overlay image with gaze projection and timestamp info."""

    fig = render_overlay_figure(image, sample, image_dt_ns, make_upright)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    _pyplot().close(fig)


def render_overlay_figure(
    image: Any,
    sample: GazeSample,
    image_dt_ns: int,
    make_upright: bool,
) -> Any:
    """Render one RGB overlay figure for PNG export or video frames."""

    if make_upright:
        image = np.rot90(image, k=3)

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_axis_off()
    if sample.projection_valid and sample.gaze_u_px is not None and sample.gaze_v_px is not None:
        ax.scatter([sample.gaze_u_px], [sample.gaze_v_px], c="red", s=80)
        ax.plot(
            [sample.gaze_u_px - 20, sample.gaze_u_px + 20],
            [sample.gaze_v_px, sample.gaze_v_px],
            color="red",
            linewidth=2,
        )
        ax.plot(
            [sample.gaze_u_px, sample.gaze_u_px],
            [sample.gaze_v_px - 20, sample.gaze_v_px + 20],
            color="red",
            linewidth=2,
        )
    ax.set_title(
        f"t={sample.query_timestamp_ns} ns | gaze_dt={sample.gaze_dt_ns} ns | "
        f"image_dt={image_dt_ns} ns\n{sample.validation_notes}",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def write_scene_rays_plot(path: os.PathLike[str] | str, samples: Sequence[GazeSample]) -> None:
    """Write metric 3D gaze rays in ADT Scene coordinates."""

    rays = [
        sample
        for sample in samples
        if sample.gaze_origin_scene_x_m is not None and sample.gaze_point_scene_x_m is not None
    ]
    if not rays:
        return

    plt = _pyplot()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    origins = []
    points = []
    for sample in rays:
        origin = np.array(
            [
                sample.gaze_origin_scene_x_m,
                sample.gaze_origin_scene_y_m,
                sample.gaze_origin_scene_z_m,
            ],
            dtype=float,
        )
        point = np.array(
            [
                sample.gaze_point_scene_x_m,
                sample.gaze_point_scene_y_m,
                sample.gaze_point_scene_z_m,
            ],
            dtype=float,
        )
        origins.append(origin)
        points.append(point)
        ax.plot(
            [origin[0], point[0]],
            [origin[1], point[1]],
            [origin[2], point[2]],
            color="red",
            alpha=0.35,
        )

    origins_array = np.vstack(origins)
    points_array = np.vstack(points)
    order_values = np.arange(len(origins_array))
    if len(origins_array) > 1:
        ax.plot(
            origins_array[:, 0],
            origins_array[:, 1],
            origins_array[:, 2],
            color="black",
            linewidth=1.2,
            alpha=0.55,
            label="CPF origin trajectory",
        )
    scatter_origins = ax.scatter(
        origins_array[:, 0],
        origins_array[:, 1],
        origins_array[:, 2],
        c=order_values,
        cmap="viridis",
        s=18,
        label="CPF origin, time order",
    )
    ax.scatter(
        [origins_array[0, 0]],
        [origins_array[0, 1]],
        [origins_array[0, 2]],
        marker="o",
        color="lime",
        s=55,
        edgecolors="black",
        linewidths=0.6,
        label="start origin",
    )
    ax.scatter(
        [origins_array[-1, 0]],
        [origins_array[-1, 1]],
        [origins_array[-1, 2]],
        marker="X",
        color="yellow",
        s=70,
        edgecolors="black",
        linewidths=0.6,
        label="end origin",
    )
    ax.scatter(
        points_array[:, 0],
        points_array[:, 1],
        points_array[:, 2],
        color="red",
        s=14,
        label="gaze point",
    )
    set_axes_equal_3d(ax, np.vstack([origins_array, points_array]))
    ax.set_xlabel("Scene X [m]")
    ax.set_ylabel("Scene Y [m]")
    ax.set_zlabel("Scene Z [m]")
    ax.set_title("Gaze rays in ADT Scene frame (equal metric scale)")
    ax.legend(loc="upper left", fontsize=7)
    fig.colorbar(scatter_origins, ax=ax, label="sample order", shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_reference_frame_scanpath_overlay(path: os.PathLike[str] | str, scanpath: dict[str, Any]) -> None:
    """Write reference-frame scanpath over the reference RGB image."""

    image = scanpath["image"]
    xs = scanpath["xs"]
    ys = scanpath["ys"]
    orders = scanpath["orders"]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.plot(xs, ys, color="white", linewidth=2, alpha=0.75)
    ax.plot(xs, ys, color="black", linewidth=1, alpha=0.8)
    scatter = ax.scatter(
        xs,
        ys,
        c=orders,
        cmap="viridis",
        s=45,
        edgecolors="white",
        linewidths=0.6,
    )
    if len(xs) <= 25:
        for order, u_px, v_px in zip(orders, xs, ys, strict=True):
            ax.text(
                u_px + 4,
                v_px + 4,
                str(order),
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
            )
    if orders[-1] == scanpath["reference_order"]:
        ax.scatter(
            [xs[-1]],
            [ys[-1]],
            marker="*",
            c="yellow",
            s=120,
            edgecolors="black",
            linewidths=0.7,
        )

    ax.set_axis_off()
    ax.set_title(
        "Reference-frame gaze scanpath overlay "
        f"(ref sample={scanpath['reference_order']}, "
        f"in_image={len(xs)}/{scanpath['frame_count']})",
        fontsize=9,
    )
    fig.colorbar(scatter, ax=ax, label="sample order")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_reference_frame_scanpath_clean(path: os.PathLike[str] | str, scanpath: dict[str, Any]) -> None:
    """Write a zoomed clean pixel-coordinate view of the reference-frame scanpath."""

    xs = scanpath["xs"]
    ys = scanpath["ys"]
    orders = scanpath["orders"]
    width = scanpath["image_width"]
    height = scanpath["image_height"]

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f8f8f8")
    ax.plot(xs, ys, color="#333333", linewidth=1.2, alpha=0.65)
    scatter = ax.scatter(
        xs,
        ys,
        c=orders,
        cmap="viridis",
        s=50,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.scatter(
        [xs[0]],
        [ys[0]],
        marker="o",
        color="lime",
        s=90,
        edgecolors="black",
        linewidths=0.7,
        label="start",
    )
    ax.scatter(
        [xs[-1]],
        [ys[-1]],
        marker="X",
        color="yellow",
        s=110,
        edgecolors="black",
        linewidths=0.7,
        label="end",
    )
    if len(xs) <= 25:
        for order, u_px, v_px in zip(orders, xs, ys, strict=True):
            ax.text(u_px + 4, v_px + 4, str(order), fontsize=7, color="#222222")

    x_min, x_max, y_min, y_max = zoomed_pixel_limits(xs, ys, width, height)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.set_xlabel("reference RGB u [px]")
    ax.set_ylabel("reference RGB v [px]")
    ax.set_title(
        "Reference-frame gaze scanpath clean zoom "
        f"(ref sample={scanpath['reference_order']}, "
        f"in_image={len(xs)}/{scanpath['frame_count']})",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(scatter, ax=ax, label="sample order")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def reference_scanpath_from_samples(
    gt_provider: Any,
    samples: list[GazeSample],
    orders: list[int],
    stream_id_value: str,
    make_upright: bool,
) -> dict[str, Any]:
    """Project 3D gaze points from a window into the last RGB frame."""

    if not samples:
        raise ValueError("No samples available for reference-frame scanpath")

    scene_points = []
    for sample in samples:
        if (
            sample.gaze_point_scene_x_m is None
            or sample.gaze_point_scene_y_m is None
            or sample.gaze_point_scene_z_m is None
        ):
            scene_points.append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))
        else:
            scene_points.append(
                np.array(
                    [
                        sample.gaze_point_scene_x_m,
                        sample.gaze_point_scene_y_m,
                        sample.gaze_point_scene_z_m,
                    ],
                    dtype=np.float64,
                )
            )

    reference_timestamp_ns = samples[-1].query_timestamp_ns
    projections, image_size = project_scene_points_to_rgb(
        gt_provider,
        scene_points,
        reference_timestamp_ns,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    reference_image_with_dt = get_rgb_image(
        gt_provider,
        reference_timestamp_ns,
        stream_id_value=stream_id_value,
    )
    if not reference_image_with_dt.is_valid():
        raise ValueError(f"Reference RGB image is invalid at {reference_timestamp_ns}")

    image = reference_image_with_dt.data().to_numpy_array()
    if make_upright:
        image = np.rot90(image, k=3)

    width, height = image_size
    xs = []
    ys = []
    filtered_orders = []
    for order, projection in zip(orders, projections, strict=True):
        if projection is None:
            continue
        u_px = float(projection[0])
        v_px = float(projection[1])
        if 0 <= u_px < width and 0 <= v_px < height:
            xs.append(u_px)
            ys.append(v_px)
            filtered_orders.append(order)

    if not xs:
        raise ValueError("No reference-frame scanpath points are inside the reference image")

    return {
        "image": image,
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "xs": xs,
        "ys": ys,
        "orders": filtered_orders,
        "reference_order": orders[-1],
        "frame_count": len(samples),
    }


def write_overlay_frames(
    gt_provider: Any,
    indexed_samples: list[tuple[int, GazeSample]],
    output_dir: Path,
    stream_id_value: str,
    make_upright: bool,
) -> list[Path]:
    """Render one RGB overlay per selected sample."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for row_index, sample in indexed_samples:
        image_with_dt = get_rgb_image(
            gt_provider,
            sample.query_timestamp_ns,
            stream_id_value=stream_id_value,
        )
        if not image_with_dt.is_valid():
            continue

        overlay_path = output_dir / f"overlay_row{row_index:04d}_{sample.query_timestamp_ns}.png"
        save_overlay(
            overlay_path,
            image_with_dt.data().to_numpy_array(),
            sample,
            int(image_with_dt.dt_ns()),
            make_upright=make_upright,
        )
        output_paths.append(overlay_path)
    return output_paths


def write_video_from_overlay_frames(path: Path, overlay_paths: list[Path], fps: float = 10.0) -> None:
    """Pack rendered overlay PNGs into a lightweight diagnostic video."""

    import imageio.v2 as imageio

    if not overlay_paths:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for overlay_path in overlay_paths:
            image = np.asarray(imageio.imread(overlay_path))
            writer.append_data(image[:, :, :3])


def zoomed_pixel_limits(
    xs: Sequence[float],
    ys: Sequence[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Return clipped pixel limits padded around a scanpath."""

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    pad = max(x_range, y_range, 1.0) * 0.35
    pad = max(pad, 50.0)
    return (
        max(0.0, x_min - pad),
        min(float(width), x_max + pad),
        max(0.0, y_min - pad),
        min(float(height), y_max + pad),
    )


def set_axes_equal_3d(ax: Any, points: np.ndarray) -> None:
    """Set 3D plot limits so one unit has the same visual length on all axes."""

    centers = points.mean(axis=0)
    ranges = np.ptp(points, axis=0)
    radius = max(float(ranges.max()) / 2.0, 0.1)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _pyplot() -> Any:
    """Return a non-interactive pyplot module safe for sandboxed runs."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt
