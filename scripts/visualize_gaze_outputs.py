#!/usr/bin/env python
"""Regenerate gaze visualizations from extracted CSV and saved frames.

Run `extract_gaze_samples.py` first. This script does not open the ADT provider
and does not re-query gaze, pose, or RGB frames. It only reads the CSV,
`manifest.json`, and the saved `rgb_*.png` / `overlay_*.png` assets.

zh-CN:
这个脚本负责“离线重画”。它读取 extract 阶段已经保存的 CSV、manifest、
clean RGB frames 和 overlay frames，然后按新的 row window / stride 重新生成：
- gaze_scene_rays.png
- gaze_reference_frame_scanpath_overlay.png
- gaze_reference_frame_scanpath_clean.png
- gaze_overlay_video.mp4

注意：reference-frame scanpath 使用 extract 阶段 manifest 里保存的 reference
frame projection。也就是说，如果要换 reference frame，应该重新用
extract_gaze_samples.py 提取对应 event/window；如果只是改变显示抽稀或选择
CSV 行区间，运行本脚本即可。

Example:
    python scripts/visualize_gaze_outputs.py \
      Apartment_release_decoration_skeleton_seq131_M1292 \
      --start-row 0 \
      --end-row 60 \
      --stride 10 \
      --run-name stride_10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_gaze_samples import (  # noqa: E402
    downsample_samples,
    read_samples_csv,
    write_reference_frame_scanpath_clean,
    write_reference_frame_scanpath_overlay,
    write_scene_rays_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", help="ADT sequence id used to locate the default manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.json. Defaults to outputs/figures/gaze/<sequence>/manifest.json.",
    )
    parser.add_argument("--start-row", type=int, default=0, help="Starting CSV row index.")
    parser.add_argument("--end-row", type=int, default=None, help="Exclusive ending CSV row index.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth selected row for all regenerated visualizations.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output subdirectory under the extracted figure directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or (
        REPO_ROOT / "outputs" / "figures" / "gaze" / args.sequence / "manifest.json"
    )
    manifest = read_manifest(manifest_path)
    base_dir = manifest_path.parent
    csv_path = resolve_manifest_path(manifest["csv_path"])

    samples = read_samples_csv(csv_path)
    frame_rows = manifest["frames"]
    window_samples = slice_items(samples, args.start_row, args.end_row)
    window_frames = slice_items(frame_rows, args.start_row, args.end_row)
    viz_samples = downsample_samples(window_samples, args.stride, include_last=True)
    viz_frames = downsample_frames(window_frames, args.stride, include_last=True)

    run_name = args.run_name or f"stride_{args.stride}"
    output_dir = base_dir / "visualizations" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    write_scene_rays_plot(output_dir / "gaze_scene_rays.png", viz_samples)
    scanpath = reference_scanpath_from_manifest(base_dir, manifest, viz_frames)
    write_reference_frame_scanpath_overlay(
        output_dir / "gaze_reference_frame_scanpath_overlay.png",
        scanpath,
    )
    write_reference_frame_scanpath_clean(
        output_dir / "gaze_reference_frame_scanpath_clean.png",
        scanpath,
    )
    write_video_from_overlay_frames(
        output_dir / "gaze_overlay_video.mp4",
        [base_dir / frame["overlay_path"] for frame in viz_frames],
    )

    print(f"sequence: {manifest['sequence_name']}")
    print(f"manifest: {manifest_path}")
    print(f"csv: {csv_path}")
    print(f"rows: {args.start_row}..{args.end_row if args.end_row is not None else len(samples)}")
    print(f"window_samples: {len(window_samples)} viz_samples={len(viz_samples)}")
    print(f"figures: {output_dir}")


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run extract_gaze_samples.py first to create "
            "manifest.json and saved RGB/overlay frames."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_manifest_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def slice_items(items: list, start_row: int, end_row: int | None) -> list:
    if start_row < 0:
        raise ValueError("start_row must be non-negative")
    if end_row is not None and end_row <= start_row:
        raise ValueError("end_row must be greater than start_row")
    selected = items[start_row:end_row]
    if not selected:
        raise ValueError("No rows selected; check --start-row and --end-row")
    return selected


def downsample_frames(frames: list[dict[str, Any]], stride: int, include_last: bool) -> list[dict[str, Any]]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    selected = list(frames[::stride])
    if include_last and frames and selected[-1] != frames[-1]:
        selected.append(frames[-1])
    return selected


def reference_scanpath_from_manifest(
    base_dir: Path,
    manifest: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    points = [
        frame
        for frame in frames
        if frame["reference_projection_in_image"]
        and frame["reference_u_px"] is not None
        and frame["reference_v_px"] is not None
    ]
    if not points:
        raise ValueError("No reference-frame scanpath points are inside the reference image")

    reference_index = int(manifest["reference_index"])
    reference_frame = manifest["frames"][reference_index]
    reference_rgb = base_dir / reference_frame["rgb_path"]
    image = read_image(reference_rgb)
    return {
        "image": image,
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "xs": [float(frame["reference_u_px"]) for frame in points],
        "ys": [float(frame["reference_v_px"]) for frame in points],
        "orders": [int(frame["index"]) for frame in points],
        "reference_order": reference_index,
        "frame_count": len(frames),
    }


def read_image(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    if not path.exists():
        raise FileNotFoundError(f"Saved frame not found: {path}")
    return np.asarray(imageio.imread(path))


def write_video_from_overlay_frames(path: Path, overlay_paths: list[Path], fps: float = 10.0) -> None:
    import imageio.v2 as imageio

    if not overlay_paths:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for overlay_path in overlay_paths:
            image = read_image(overlay_path)
            writer.append_data(image[:, :, :3])


if __name__ == "__main__":
    main()
