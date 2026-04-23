#!/usr/bin/env python
"""Extract reusable ADT gaze samples and frame assets for one sequence.

This script is the runnable version of docs/tutorial_gaze_feature_extraction.md:
select RGB timestamps, query gaze/pose, write a flat CSV, and save the RGB
frames, per-frame gaze overlays, and manifest needed for offline visualization.

zh-CN:
这个脚本负责“一次性提取”。它会打开 ADT provider，在选定的 RGB timestamp
窗口上查询 gaze、pose 和 RGB frame，然后保存：
- gaze CSV：后续分析和质量检查用。
- clean RGB frames：后续离线可视化用。
- per-frame overlay frames：每一帧自己的 gaze 投影检查图。
- manifest.json：记录 CSV、图片路径、reference-frame projection 等元数据。

后续如果只是想改变 scanpath、scene_rays 或 overlay video 的抽稀/窗口参数，
不要重新运行本脚本，改用 scripts/visualize_gaze_outputs.py 读取已有输出。

Example:
    python scripts/extract_gaze_samples.py \
      Apartment_release_decoration_skeleton_seq131_M1292 \
      --start-index 900 \
      --end-index 905 \
      --stride 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINDOW_FRAMES = 30
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import (  # noqa: E402
    RGB_STREAM_ID,
    GazeSample,
    extract_gaze_sample,
    get_rgb_image,
    get_rgb_timestamps_ns,
    project_scene_points_to_rgb,
    select_timestamps,
)
from adt_sandbox.providers import create_adt_providers  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        type=Path,
        help="ADT sequence id resolved under ADT_DATA_ROOT, or an absolute sequence path.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Step between RGB frame timestamps. For 30 fps RGB, stride=30 is about 1 Hz.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting RGB timestamp index before applying stride.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help=(
            "Exclusive ending RGB timestamp index before applying stride. "
            "Default is start-index + 30 unless an end offset is set."
        ),
    )
    parser.add_argument(
        "--start-offset-s",
        type=float,
        default=None,
        help=(
            "Start time in seconds, relative to the first RGB timestamp after annotation "
            "range filtering. Applied before index selection."
        ),
    )
    parser.add_argument(
        "--end-offset-s",
        type=float,
        default=None,
        help=(
            "Exclusive end time in seconds, relative to the first RGB timestamp after "
            "annotation range filtering. Applied before index selection."
        ),
    )
    parser.add_argument(
        "--stream-id",
        default=RGB_STREAM_ID,
        help="Project Aria stream id for RGB. Default is 214-1.",
    )
    parser.add_argument(
        "--max-dt-ms",
        type=float,
        default=20.0,
        help="Flag gaze samples whose nearest timestamp differs by more than this value.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to outputs/reports/<sequence>_gaze_samples.csv.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "gaze",
        help="Directory for reusable RGB frames, overlay frames, manifest, and figures.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip reusable RGB/overlay frame outputs and manifest; only write CSV.",
    )
    parser.add_argument(
        "--raw-image-orientation",
        action="store_true",
        help="Keep RGB images in raw sensor orientation instead of rotating them upright.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    providers = create_adt_providers(args.sequence, skeleton_flag=True)
    sequence_name = providers.sequence_path.name
    output_csv = args.output_csv or (
        REPO_ROOT / "outputs" / "reports" / f"{sequence_name}_gaze_samples.csv"
    )
    figures_dir = args.figures_dir / sequence_name

    # Use RGB frames as anchors because they make projection errors easy to inspect.
    timestamps_ns = get_rgb_timestamps_ns(providers.gt_provider, args.stream_id)
    timestamps_ns = restrict_to_provider_time_range(providers.gt_provider, timestamps_ns)
    timestamps_ns = restrict_to_time_offsets(
        timestamps_ns,
        start_offset_s=args.start_offset_s,
        end_offset_s=args.end_offset_s,
    )
    end_index = args.end_index
    if end_index is None and args.end_offset_s is None:
        end_index = args.start_index + DEFAULT_WINDOW_FRAMES
    selected_timestamps = select_timestamps(
        timestamps_ns,
        stride=args.stride,
        start_index=args.start_index,
        end_index=end_index,
    )
    max_dt_ns = int(args.max_dt_ms * 1e6)
    make_upright = not args.raw_image_orientation

    # Each row keeps both raw gaze values and derived diagnostic fields.
    samples = [
        extract_gaze_sample(
            providers.gt_provider,
            timestamp_ns,
            stream_id_value=args.stream_id,
            max_dt_ns=max_dt_ns,
            make_upright=make_upright,
        )
        for timestamp_ns in selected_timestamps
    ]

    # Write the full sample list to CSV for later analysis and filtering.
    write_samples_csv(output_csv, samples)
    if not args.no_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        frame_manifest = write_rgb_and_overlay_frames(
            providers.gt_provider,
            samples,
            figures_dir,
            stream_id_value=args.stream_id,
            make_upright=make_upright,
        )
        write_manifest_json(
            figures_dir / "manifest.json",
            sequence_name=sequence_name,
            sequence_path=providers.sequence_path,
            provider_mode=providers.provider_mode,
            csv_path=output_csv,
            stream_id_value=args.stream_id,
            make_upright=make_upright,
            samples=samples,
            frame_manifest=frame_manifest,
        )

    print_summary(
        sequence_name,
        providers.sequence_path,
        providers.provider_mode,
        output_csv,
        figures_dir,
        samples,
        args.no_plots,
        make_upright,
    )


def restrict_to_provider_time_range(gt_provider: Any, timestamps_ns: list[int]) -> list[int]:
    """Keep RGB timestamps inside the gaze/pose annotation range when available."""
    # zh-CN: 如果 provider 暴露了注释的时间范围，就把 RGB 时间戳限制在这个范围内，
    # 避免后续对齐分析被无效的时间戳干扰。

    # The raw provider exposes the overlap of gaze and trajectory CSVs. Official
    # providers may not expose these helpers, so leave their timestamp list as is.
    # zh-CN: ADT 的原始数据提供者会暴露注释时间范围的接口，但官方 provider 可能没有，所以如果没有相关接口就直接返回不做限制。
    if not hasattr(gt_provider, "get_start_time_ns") or not hasattr(gt_provider, "get_end_time_ns"):
        return timestamps_ns

    start_ns = int(gt_provider.get_start_time_ns())
    end_ns = int(gt_provider.get_end_time_ns())
    filtered = [timestamp for timestamp in timestamps_ns if start_ns <= timestamp <= end_ns]
    if not filtered:
        raise ValueError(
            "No RGB timestamps overlap the provider annotation time range: "
            f"rgb={len(timestamps_ns)} annotation=[{start_ns}, {end_ns}]"
        )
    return filtered


def restrict_to_time_offsets(
    timestamps_ns: list[int],
    start_offset_s: float | None,
    end_offset_s: float | None,
) -> list[int]:
    """Keep RGB timestamps inside a relative time window.

    zh-CN:
    这里的 offset 是相对于当前可用 RGB 时间轴第一个 timestamp 的秒数，不是 ADT
    的绝对 nanosecond timestamp。这样调试时可以直接写
    `--start-offset-s 30 --end-offset-s 32`，表示从这段 sequence 的第 30 秒到
    第 32 秒。
    """

    if not timestamps_ns:
        raise ValueError("No RGB timestamps available")
    if start_offset_s is None and end_offset_s is None:
        return timestamps_ns
    if start_offset_s is not None and start_offset_s < 0:
        raise ValueError("start_offset_s must be non-negative")
    if end_offset_s is not None and end_offset_s < 0:
        raise ValueError("end_offset_s must be non-negative")
    if (
        start_offset_s is not None
        and end_offset_s is not None
        and end_offset_s <= start_offset_s
    ):
        raise ValueError("end_offset_s must be greater than start_offset_s")

    base_timestamp_ns = timestamps_ns[0]
    start_ns = (
        base_timestamp_ns
        if start_offset_s is None
        else base_timestamp_ns + int(start_offset_s * 1e9)
    )
    end_ns = (
        timestamps_ns[-1] + 1
        if end_offset_s is None
        else base_timestamp_ns + int(end_offset_s * 1e9)
    )
    filtered = [timestamp for timestamp in timestamps_ns if start_ns <= timestamp < end_ns]
    if not filtered:
        raise ValueError(
            "No RGB timestamps selected by time offsets: "
            f"offset_s=[{start_offset_s}, {end_offset_s}) "
            f"available_duration_s={(timestamps_ns[-1] - timestamps_ns[0]) / 1e9:.3f}"
        )
    return filtered


def downsample_samples(
    samples: Sequence[GazeSample],
    stride: int,
    include_last: bool,
) -> list[GazeSample]:
    """Return every Nth sample for visualization without changing CSV output.

    zh-CN:
    这个函数只用于可视化抽稀。CSV 仍然保留所有 selected samples。
    include_last=True 时会强制保留窗口最后一帧，便于轨迹图包含事件窗口末尾。
    """

    if stride <= 0:
        raise ValueError("visualization stride must be positive")
    selected = list(samples[::stride])
    if include_last and samples and selected[-1] != samples[-1]:
        selected.append(samples[-1])
    return selected


def write_samples_csv(path: Path, samples: list[GazeSample]) -> None:
    """Write gaze samples to a CSV file for later analysis and filtering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [sample.as_csv_row() for sample in samples]
    if not rows:
        raise ValueError("No gaze samples to write")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_samples_csv(path: Path) -> list[GazeSample]:
    """Read a gaze samples CSV previously written by this script."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [gaze_sample_from_csv_row(row) for row in reader]


def gaze_sample_from_csv_row(row: dict[str, str]) -> GazeSample:
    """Convert one CSV row into a GazeSample with the original field types."""

    return GazeSample(
        query_timestamp_ns=csv_int(row["query_timestamp_ns"]),
        gaze_valid=csv_bool(row["gaze_valid"]),
        gaze_dt_ns=csv_optional_int(row["gaze_dt_ns"]),
        yaw_rad=csv_optional_float(row["yaw_rad"]),
        pitch_rad=csv_optional_float(row["pitch_rad"]),
        depth_m=csv_optional_float(row["depth_m"]),
        yaw_confidence_width_rad=csv_optional_float(row["yaw_confidence_width_rad"]),
        pitch_confidence_width_rad=csv_optional_float(row["pitch_confidence_width_rad"]),
        projection_valid=csv_bool(row["projection_valid"]),
        gaze_u_px=csv_optional_float(row["gaze_u_px"]),
        gaze_v_px=csv_optional_float(row["gaze_v_px"]),
        projection_in_image=csv_bool(row["projection_in_image"]),
        image_width_px=csv_optional_int(row["image_width_px"]),
        image_height_px=csv_optional_int(row["image_height_px"]),
        pose_valid=csv_bool(row["pose_valid"]),
        pose_dt_ns=csv_optional_int(row["pose_dt_ns"]),
        pose_quality_score=csv_optional_float(row["pose_quality_score"]),
        gaze_origin_scene_x_m=csv_optional_float(row["gaze_origin_scene_x_m"]),
        gaze_origin_scene_y_m=csv_optional_float(row["gaze_origin_scene_y_m"]),
        gaze_origin_scene_z_m=csv_optional_float(row["gaze_origin_scene_z_m"]),
        gaze_point_scene_x_m=csv_optional_float(row["gaze_point_scene_x_m"]),
        gaze_point_scene_y_m=csv_optional_float(row["gaze_point_scene_y_m"]),
        gaze_point_scene_z_m=csv_optional_float(row["gaze_point_scene_z_m"]),
        validation_notes=row["validation_notes"],
    )


def csv_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid boolean value in CSV: {value!r}")


def csv_int(value: str) -> int:
    return int(value)


def csv_optional_int(value: str) -> int | None:
    return int(value) if value else None


def csv_optional_float(value: str) -> float | None:
    return float(value) if value else None


def write_rgb_and_overlay_frames(
    gt_provider: Any,
    samples: list[GazeSample],
    figures_dir: Path,
    stream_id_value: str,
    make_upright: bool,
) -> list[dict[str, Any]]:
    """Write complete reusable RGB and overlay frames for later visualization.

    zh-CN:
    extract 阶段一次性保存完整 RGB frames 和 overlay frames。后续
    `visualize_gaze_outputs.py` 只读取这些中间结果和 CSV/manifest，不再打开
    ADT provider。
    """

    if not samples:
        return []

    rgb_dir = figures_dir / "rgb"
    overlay_dir = figures_dir / "overlays"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    reference_sample = samples[-1]
    reference_projections, reference_size = reference_projections_for_samples(
        gt_provider,
        samples,
        reference_sample,
        stream_id_value,
        make_upright,
    )

    frame_rows = []
    for index, sample in enumerate(samples):
        image_with_dt = get_rgb_image(gt_provider, sample.query_timestamp_ns, stream_id_value)
        if not image_with_dt.is_valid():
            continue

        image = image_with_dt.data().to_numpy_array()
        if make_upright:
            image_for_rgb = np.rot90(image, k=3)
        else:
            image_for_rgb = image

        rgb_path = rgb_dir / f"rgb_{index:04d}_{sample.query_timestamp_ns}.png"
        overlay_path = overlay_dir / f"overlay_{index:04d}_{sample.query_timestamp_ns}.png"
        write_image(rgb_path, image_for_rgb)
        save_overlay(overlay_path, image, sample, image_with_dt.dt_ns(), make_upright)

        projection = reference_projections[index]
        ref_u = float(projection[0]) if projection is not None else None
        ref_v = float(projection[1]) if projection is not None else None
        ref_in_image = bool(
            projection is not None
            and ref_u is not None
            and ref_v is not None
            and 0 <= ref_u < reference_size[0]
            and 0 <= ref_v < reference_size[1]
        )
        frame_rows.append(
            {
                "index": index,
                "timestamp_ns": sample.query_timestamp_ns,
                "rgb_path": str(rgb_path.relative_to(figures_dir)),
                "overlay_path": str(overlay_path.relative_to(figures_dir)),
                "image_dt_ns": int(image_with_dt.dt_ns()),
                "reference_u_px": ref_u,
                "reference_v_px": ref_v,
                "reference_projection_in_image": ref_in_image,
            }
        )

    return frame_rows


def reference_projections_for_samples(
    gt_provider: Any,
    samples: list[GazeSample],
    reference_sample: GazeSample,
    stream_id_value: str,
    make_upright: bool,
) -> tuple[list[np.ndarray | None], tuple[int, int]]:
    """Project all Scene-frame gaze points into the extraction reference frame."""

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
    return project_scene_points_to_rgb(
        gt_provider,
        scene_points,
        reference_sample.query_timestamp_ns,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )


def write_manifest_json(
    path: Path,
    sequence_name: str,
    sequence_path: Path,
    provider_mode: str,
    csv_path: Path,
    stream_id_value: str,
    make_upright: bool,
    samples: list[GazeSample],
    frame_manifest: list[dict[str, Any]],
) -> None:
    """Write metadata that lets visualization run without an ADT provider."""

    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sequence_name": sequence_name,
        "sequence_path": str(sequence_path),
        "provider_mode": provider_mode,
        "csv_path": str(csv_path),
        "stream_id": stream_id_value,
        "image_orientation": "upright" if make_upright else "raw",
        "reference_timestamp_ns": samples[-1].query_timestamp_ns if samples else None,
        "reference_index": len(samples) - 1 if samples else None,
        "frames": frame_manifest,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an RGB image array to disk."""

    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)


def save_overlay(
    path: Path,
    image: Any,
    sample: GazeSample,
    image_dt_ns: int,
    make_upright: bool,
) -> None:
    """Save one diagnostic overlay image with gaze projection and timestamp info."""
    fig = render_overlay_figure(image, sample, image_dt_ns, make_upright)
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


def write_scene_rays_plot(path: Path, samples: list[GazeSample]) -> None:
    """Write metric 3D gaze rays in ADT Scene coordinates.

    The axes are forced to equal metric scale. Without that, Matplotlib stretches
    small-range axes and can make consecutive rays look much jumpier than they
    are in meters.

    zh-CN:
    这个图使用 ADT Scene/world frame，单位是米。这里强制 3D 坐标轴等比例显示；
    否则 Matplotlib 会把范围较小的轴拉满，连续几帧的 gaze rays 会看起来跳得
    比实际米制距离更夸张。时间方向通过 CPF/gaze origin 轨迹上的颜色渐变表示，
    不在 3D 图里逐点写数字，避免数字标签被误读成空间点。
    """

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


def write_reference_frame_scanpath_overlay(path: Path, scanpath: dict[str, Any]) -> None:
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


def write_reference_frame_scanpath_clean(path: Path, scanpath: dict[str, Any]) -> None:
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


def print_summary(
    sequence_name: str,
    sequence_path: Path,
    provider_mode: str,
    output_csv: Path,
    figures_dir: Path,
    samples: list[GazeSample],
    no_plots: bool,
    make_upright: bool,
) -> None:
    """Print a summary of the extracted gaze samples and output locations."""
    valid_gaze = sum(sample.gaze_valid for sample in samples)
    in_image = sum(sample.projection_in_image for sample in samples)
    ok = sum(sample.validation_notes == "ok" for sample in samples)
    print(f"sequence: {sequence_name}")
    print(f"sequence_path: {sequence_path}")
    print(f"provider_mode: {provider_mode}")
    print(f"image_orientation: {'upright' if make_upright else 'raw'}")
    print(f"samples: {len(samples)} gaze_valid={valid_gaze} projection_in_image={in_image} ok={ok}")
    if samples:
        first_timestamp_ns = samples[0].query_timestamp_ns
        last_timestamp_ns = samples[-1].query_timestamp_ns
        duration_s = (last_timestamp_ns - first_timestamp_ns) / 1e9
        print(
            "selected_timestamps_ns: "
            f"{first_timestamp_ns}..{last_timestamp_ns} duration_s={duration_s:.3f}"
        )
    print(f"csv: {output_csv}")
    if not no_plots:
        print(f"figures: {figures_dir}")


def _pyplot() -> Any:
    # Matplotlib may try to write config under the user's home; keep it sandbox-safe.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


if __name__ == "__main__":
    main()
