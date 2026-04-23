#!/usr/bin/env python
"""Extract ADT gaze samples and a lightweight quality summary for one sequence.

This script is the runnable version of docs/tutorial_gaze_feature_extraction.md:
select RGB timestamps, query gaze/pose, and write a flat gaze CSV plus a small
JSON summary. It does not generate PNGs or videos.

zh-CN:
这个脚本现在只负责“轻量提取”：
- 生成 `gaze_samples.csv`，作为后续分析和可视化的核心数据。
- 生成一个轻量 `gaze_summary.json`，快速查看 validity、projection、depth、
  `dt` 等质量指标。

图片和视频不再是默认主流程的一部分；如果后面需要 scanpath、scene rays、
overlay frames 或 overlay video，改用 `scripts/visualize_gaze_outputs.py`
基于已有 CSV 和选中窗口生成。

Example:
    python scripts/extract_gaze_samples.py \
      Apartment_release_decoration_skeleton_seq131_M1292 \
      --start-index 900 \
      --end-index 905 \
      --stride 1
    python scripts/extract_gaze_samples.py Apartment_release_decoration_skeleton_seq131_M1292
    python scripts/extract_gaze_samples.py Apartment_release_decoration_skeleton_seq131_M1292 --stride 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import (  # noqa: E402
    RGB_STREAM_ID,
    default_summary_json_path,
    extract_gaze_sample,
    get_rgb_timestamps_ns,
    summarize_gaze_samples,
    select_timestamps,
    write_gaze_summary_json,
    write_samples_csv,
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
        default=1,
        help=(
            "Step between RGB frame timestamps. Default is 1, which keeps every "
            "available RGB timestamp. For 30 fps RGB, stride=30 is about 1 Hz."
        ),
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
            "Default is the end of the sequence."
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
        "--raw-image-orientation",
        action="store_true",
        help="Keep RGB-related projections in raw sensor orientation instead of upright.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    providers = create_adt_providers(args.sequence, skeleton_flag=True)
    sequence_name = providers.sequence_path.name
    output_csv = args.output_csv or (
        REPO_ROOT / "outputs" / "reports" / f"{sequence_name}_gaze_samples.csv"
    )
    summary_json = default_summary_json_path(output_csv)

    raw_rgb_timestamps_ns = get_rgb_timestamps_ns(providers.gt_provider, args.stream_id)
    annotation_filtered_timestamps_ns = restrict_to_provider_time_range(
        providers.gt_provider,
        raw_rgb_timestamps_ns,
    )
    offset_filtered_timestamps_ns = restrict_to_time_offsets(
        annotation_filtered_timestamps_ns,
        start_offset_s=args.start_offset_s,
        end_offset_s=args.end_offset_s,
    )
    selected_timestamps = select_timestamps(
        offset_filtered_timestamps_ns,
        stride=args.stride,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    max_dt_ns = int(args.max_dt_ms * 1e6)
    make_upright = not args.raw_image_orientation

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

    write_samples_csv(output_csv, samples)
    summary = summarize_gaze_samples(samples)
    summary.update(
        {
            "sequence_name": sequence_name,
            "sequence_path": str(providers.sequence_path),
            "provider_mode": providers.provider_mode,
            "stream_id": args.stream_id,
            "image_orientation": "upright" if make_upright else "raw",
            "output_csv": str(output_csv),
            "field_coordinate_frames": {
                "query_timestamp_ns": "device_time_ns",
                "gaze_dt_ns": "device_time_ns_delta",
                "pose_dt_ns": "device_time_ns_delta",
                "yaw_rad": "cpf_angle_rad",
                "pitch_rad": "cpf_angle_rad",
                "depth_m": "cpf_depth_m",
                "gaze_dir_cpf_unit_x": "cpf_unit_direction",
                "gaze_dir_cpf_unit_y": "cpf_unit_direction",
                "gaze_dir_cpf_unit_z": "cpf_unit_direction",
                "gaze_u_px": (
                    "rgb_image_plane_upright_px"
                    if make_upright
                    else "rgb_image_plane_raw_px"
                ),
                "gaze_v_px": (
                    "rgb_image_plane_upright_px"
                    if make_upright
                    else "rgb_image_plane_raw_px"
                ),
                "image_width_px": (
                    "upright_rgb_image_width_px"
                    if make_upright
                    else "raw_rgb_image_width_px"
                ),
                "image_height_px": (
                    "upright_rgb_image_height_px"
                    if make_upright
                    else "raw_rgb_image_height_px"
                ),
                "gaze_origin_scene_x_m": "adt_scene_frame_m",
                "gaze_origin_scene_y_m": "adt_scene_frame_m",
                "gaze_origin_scene_z_m": "adt_scene_frame_m",
                "gaze_point_scene_x_m": "adt_scene_frame_m",
                "gaze_point_scene_y_m": "adt_scene_frame_m",
                "gaze_point_scene_z_m": "adt_scene_frame_m",
                "gaze_dir_scene_unit_x": "adt_scene_frame_unit_direction",
                "gaze_dir_scene_unit_y": "adt_scene_frame_unit_direction",
                "gaze_dir_scene_unit_z": "adt_scene_frame_unit_direction",
            },
            "field_definitions": {
                "depth_m": (
                    "Distance from the CPF origin along the gaze ray, matching "
                    "projectaria_tools.core.mps.get_eyegaze_point_at_depth; not the "
                    "CPF z coordinate."
                ),
                "gaze_dir_cpf_unit_xyz": (
                    "Normalized gaze direction in CPF. Independent of depth_m."
                ),
                "gaze_point_scene_xyz": (
                    "Scene-frame gaze point obtained by transforming the CPF gaze point "
                    "constructed with the official depth semantics."
                ),
                "gaze_dir_scene_unit_xyz": (
                    "Normalized Scene-frame gaze direction. Independent of depth_m."
                ),
            },
            "source_counts": {
                "eye_gaze_csv_count": count_eye_gaze_rows(
                    providers.sequence_path / "eyegaze.csv"
                ),
                "raw_rgb_timestamp_count": len(raw_rgb_timestamps_ns),
                "annotation_filtered_rgb_timestamp_count": len(
                    annotation_filtered_timestamps_ns
                ),
                "offset_filtered_rgb_timestamp_count": len(offset_filtered_timestamps_ns),
                "selected_rgb_timestamp_count": len(selected_timestamps),
            },
            "source_time_ranges_ns": {
                "eye_gaze_csv": describe_eye_gaze_csv(
                    providers.sequence_path / "eyegaze.csv"
                ),
                "raw_rgb_timestamps": describe_timestamp_list(raw_rgb_timestamps_ns),
                "annotation_filtered_rgb_timestamps": describe_timestamp_list(
                    annotation_filtered_timestamps_ns
                ),
                "offset_filtered_rgb_timestamps": describe_timestamp_list(
                    offset_filtered_timestamps_ns
                ),
                "selected_rgb_timestamps": describe_timestamp_list(selected_timestamps),
                "provider_annotation_range": describe_provider_annotation_range(
                    providers.gt_provider
                ),
            },
            "selection": {
                "start_index": args.start_index,
                "end_index": args.end_index,
                "start_offset_s": args.start_offset_s,
                "end_offset_s": args.end_offset_s,
                "stride": args.stride,
                "max_dt_ms": args.max_dt_ms,
            },
        }
    )
    write_gaze_summary_json(summary_json, summary)
    print_summary(output_csv, summary_json, summary)


def restrict_to_provider_time_range(gt_provider: Any, timestamps_ns: list[int]) -> list[int]:
    """Keep RGB timestamps inside the gaze/pose annotation range when available."""

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


def count_eye_gaze_rows(path: Path) -> int | None:
    """Count `eyegaze.csv` data rows when the file is available."""

    stats = describe_eye_gaze_csv(path)
    return stats["count"]


def describe_eye_gaze_csv(path: Path) -> dict[str, int | float | None]:
    """Describe `eyegaze.csv` row count and timestamp range.

    zh-CN:
    这里直接读取 sequence 根目录下的 `eyegaze.csv`，把 provider 日志里常见的
    `Loaded #EyeGazes` 数字也落进 summary，避免之后只看 JSON 时还要反推。
    """

    if not path.exists():
        return {
            "count": None,
            "timestamp_start_ns": None,
            "timestamp_end_ns": None,
            "duration_s": None,
        }

    count = 0
    first_ns: int | None = None
    last_ns: int | None = None
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count += 1
            tracking_us = row.get("tracking_timestamp_us")
            if not tracking_us:
                continue
            timestamp_ns = int(tracking_us) * 1000
            if first_ns is None:
                first_ns = timestamp_ns
            last_ns = timestamp_ns

    return {
        "count": count,
        "timestamp_start_ns": first_ns,
        "timestamp_end_ns": last_ns,
        "duration_s": (
            (last_ns - first_ns) / 1e9
            if first_ns is not None and last_ns is not None
            else None
        ),
    }


def describe_timestamp_list(timestamps_ns: list[int]) -> dict[str, int | float | None]:
    """Describe a timestamp list with count and start/end/duration."""

    if not timestamps_ns:
        return {
            "count": 0,
            "timestamp_start_ns": None,
            "timestamp_end_ns": None,
            "duration_s": None,
        }
    return {
        "count": len(timestamps_ns),
        "timestamp_start_ns": timestamps_ns[0],
        "timestamp_end_ns": timestamps_ns[-1],
        "duration_s": (timestamps_ns[-1] - timestamps_ns[0]) / 1e9,
    }


def describe_provider_annotation_range(gt_provider: Any) -> dict[str, int | float | None]:
    """Return provider annotation start/end if the API exposes them."""

    if not hasattr(gt_provider, "get_start_time_ns") or not hasattr(gt_provider, "get_end_time_ns"):
        return {
            "timestamp_start_ns": None,
            "timestamp_end_ns": None,
            "duration_s": None,
        }
    start_ns = int(gt_provider.get_start_time_ns())
    end_ns = int(gt_provider.get_end_time_ns())
    return {
        "timestamp_start_ns": start_ns,
        "timestamp_end_ns": end_ns,
        "duration_s": (end_ns - start_ns) / 1e9,
    }


def print_summary(output_csv: Path, summary_json: Path, summary: dict[str, Any]) -> None:
    """Print a concise extraction summary."""

    print(f"sequence: {summary['sequence_name']}")
    print(f"sequence_path: {summary['sequence_path']}")
    print(f"provider_mode: {summary['provider_mode']}")
    print(f"image_orientation: {summary['image_orientation']}")
    print(
        "samples: "
        f"{summary['sample_count']} "
        f"gaze_valid={summary['gaze_valid_count']} "
        f"projection_in_image={summary['projection_in_image_count']} "
        f"ok={summary['ok_count']}"
    )
    print(
        "selected_timestamps_ns: "
        f"{summary['query_timestamp_start_ns']}..{summary['query_timestamp_end_ns']} "
        f"duration_s={summary['duration_s']:.3f}"
    )
    source_counts = summary["source_counts"]
    print(
        "source_counts: "
        f"eye_gaze_csv={source_counts['eye_gaze_csv_count']} "
        f"rgb_raw={source_counts['raw_rgb_timestamp_count']} "
        f"rgb_annotation_filtered={source_counts['annotation_filtered_rgb_timestamp_count']} "
        f"rgb_selected={source_counts['selected_rgb_timestamp_count']}"
    )
    if summary["validation_note_counts"]:
        print(f"validation_note_counts: {summary['validation_note_counts']}")
    print(f"csv: {output_csv}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
