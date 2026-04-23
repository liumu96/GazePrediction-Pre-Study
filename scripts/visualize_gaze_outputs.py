#!/usr/bin/env python
"""Generate gaze visualizations from an extracted gaze CSV and a selected window.

Run `extract_gaze_samples.py` first. This script reads an existing CSV, selects
rows from it, then opens the ADT provider only for the chosen window so it can
render images and videos on demand.

zh-CN:
这个脚本负责“后处理可视化”：
- 先读取已经提取好的 `gaze_samples.csv`
- 再按 `start-row/end-row/stride` 选择一个 event/window
- 只为这个窗口重新查询 RGB image，生成：
  - `gaze_scene_rays.png`
  - `gaze_reference_frame_scanpath_overlay.png`
  - `gaze_reference_frame_scanpath_clean.png`
  - `gaze_overlay_video.mp4`
  - `overlays/overlay_*.png`

这样图片和视频就不再是默认主流程，而是围绕已有 CSV 的后处理步骤。

Example:
    python scripts/visualize_gaze_outputs.py \
      Apartment_release_decoration_skeleton_seq131_M1292 \
      --start-row 0 \
      --end-row 60 \
      --stride 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import (  # noqa: E402
    RGB_STREAM_ID,
    GazeSample,
    default_summary_json_path,
    get_rgb_image,
    project_scene_points_to_rgb,
    read_gaze_summary_json,
    read_samples_csv,
    save_overlay,
    write_reference_frame_scanpath_clean,
    write_reference_frame_scanpath_overlay,
    write_scene_rays_plot,
)
from adt_sandbox.providers import create_adt_providers  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        help="ADT sequence id resolved under ADT_DATA_ROOT, or an absolute sequence path.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Input CSV path. Defaults to outputs/reports/<sequence>_gaze_samples.csv.",
    )
    parser.add_argument("--start-row", type=int, default=0, help="Starting CSV row index.")
    parser.add_argument("--end-row", type=int, default=None, help="Exclusive ending CSV row index.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth selected row for all generated visualizations.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output subdirectory name under outputs/figures/gaze/<sequence>/visualizations/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.input_csv or (
        REPO_ROOT / "outputs" / "reports" / f"{Path(args.sequence).name}_gaze_samples.csv"
    )
    samples = read_samples_csv(csv_path)
    indexed_samples = list(enumerate(samples))
    window_pairs = slice_items(indexed_samples, args.start_row, args.end_row)
    viz_pairs = downsample_pairs(window_pairs, args.stride, include_last=True)
    viz_orders = [row_index for row_index, _ in viz_pairs]
    viz_samples = [sample for _, sample in viz_pairs]

    providers = create_adt_providers(args.sequence, skeleton_flag=True)
    stream_id_value, make_upright = load_visualization_context(csv_path)
    run_name = args.run_name or default_run_name(args.start_row, args.end_row, args.stride, len(samples))
    output_dir = (
        REPO_ROOT
        / "outputs"
        / "figures"
        / "gaze"
        / providers.sequence_path.name
        / "visualizations"
        / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    write_scene_rays_plot(output_dir / "gaze_scene_rays.png", viz_samples)
    scanpath = reference_scanpath_from_samples(
        providers.gt_provider,
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
        providers.gt_provider,
        viz_pairs,
        overlay_dir,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    write_video_from_overlay_frames(output_dir / "gaze_overlay_video.mp4", overlay_paths)

    print(f"sequence: {providers.sequence_path.name}")
    print(f"sequence_path: {providers.sequence_path}")
    print(f"csv: {csv_path}")
    print(f"rows: {args.start_row}..{args.end_row if args.end_row is not None else len(samples)}")
    print(f"window_samples: {len(window_pairs)} viz_samples={len(viz_samples)}")
    print(f"image_orientation: {'upright' if make_upright else 'raw'}")
    print(f"figures: {output_dir}")


def slice_items(items: list[Any], start_row: int, end_row: int | None) -> list[Any]:
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
    if stride <= 0:
        raise ValueError("stride must be positive")
    selected = list(indexed_samples[::stride])
    if include_last and indexed_samples and selected[-1] != indexed_samples[-1]:
        selected.append(indexed_samples[-1])
    return selected


def load_visualization_context(csv_path: Path) -> tuple[str, bool]:
    """Read stream id and image orientation from the paired summary JSON when available."""

    summary_path = default_summary_json_path(csv_path)
    if not summary_path.exists():
        return RGB_STREAM_ID, True

    summary = read_gaze_summary_json(summary_path)
    stream_id_value = str(summary.get("stream_id", RGB_STREAM_ID))
    make_upright = summary.get("image_orientation", "upright") != "raw"
    return stream_id_value, make_upright


def default_run_name(start_row: int, end_row: int | None, stride: int, total_rows: int) -> str:
    end_label = end_row if end_row is not None else total_rows
    return f"rows_{start_row}_{end_label}_stride_{stride}"


def reference_scanpath_from_samples(
    gt_provider: Any,
    samples: list[GazeSample],
    orders: list[int],
    stream_id_value: str,
    make_upright: bool,
) -> dict[str, Any]:
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
    """Render per-frame overlays for the selected visualization window."""

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
    import imageio.v2 as imageio

    if not overlay_paths:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for overlay_path in overlay_paths:
            image = np.asarray(imageio.imread(overlay_path))
            writer.append_data(image[:, :, :3])


if __name__ == "__main__":
    main()
