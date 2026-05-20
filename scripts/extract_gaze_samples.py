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
overlay frames 或 overlay video，改用 `visualization/visualize_gaze_outputs.py`
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
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import RGB_STREAM_ID  # noqa: E402
from adt_sandbox.gaze_extraction import (  # noqa: E402
    GazeExtractionConfig,
    default_gaze_csv_path,
    extract_sequence_gaze,
)

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        type=Path,
        help="ADT sequence id resolved under ADT_DATA_ROOT, or an absolute sequence path.",
    ) # sequence 参数可以是一个相对路径（相对于 ADT_DATA_ROOT）或者一个绝对路径，指向要处理的 ADT 序列。
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Step between RGB frame timestamps. Default is 1, which keeps every "
            "available RGB timestamp. For 30 fps RGB, stride=30 is about 1 Hz."
        ),
    ) # stride 参数控制在处理 RGB 帧时的步长。默认值为 1，表示使用每一个可用的 RGB 时间戳。如果设置为 30，则表示每隔 30 个 RGB 帧（约 1 秒钟）选取一个时间戳。
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting RGB timestamp index before applying stride.",
    ) # start-index 参数指定在应用 stride 之前，RGB 时间戳的起始索引。默认值为 0，表示从第一个 RGB 时间戳开始处理。
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help=(
            "Exclusive ending RGB timestamp index before applying stride. "
            "Default is the end of the sequence."
        ),
    ) # end-index 参数指定在应用 stride 之前，RGB 时间戳的结束索引（不包括该索引）。默认值为序列的末尾。
    parser.add_argument(
        "--start-offset-s",
        type=float,
        default=None,
        help=(
            "Start time in seconds, relative to the first RGB timestamp after annotation "
            "range filtering. Applied before index selection."
        ),
    ) # start-offset-s 参数指定一个相对于第一个 RGB 时间戳的起始时间（以秒为单位）。这个偏移量在应用索引选择之前应用，用于进一步过滤时间范围。
    parser.add_argument(
        "--end-offset-s",
        type=float,
        default=None,
        help=(
            "Exclusive end time in seconds, relative to the first RGB timestamp after "
            "annotation range filtering. Applied before index selection."
        ),
    ) # end-offset-s 参数指定一个相对于第一个 RGB 时间戳的结束时间（以秒为单位）。这个偏移量在应用索引选择之前应用，用于进一步过滤时间范围。
    parser.add_argument(
        "--stream-id",
        default=RGB_STREAM_ID,
        help="Project Aria stream id for RGB. Default is 214-1.",
    ) # stream-id 参数指定用于 RGB 的 Project Aria 流 ID。默认值是 214-1，表示使用 RGB 流。
    parser.add_argument(
        "--max-dt-ms",
        type=float,
        default=20.0,
        help="Flag gaze samples whose nearest timestamp differs by more than this value.",
    ) # max-dt-ms 参数指定一个阈值（以毫秒为单位），用于标记那些最近的时间戳之间差异超过该值的 gaze 样本。这有助于识别时间对齐不良的样本。
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Overrides --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Reports root for organized output. Default is outputs/reports; "
            "writes sequences/<sequence>/gaze/gaze_samples.csv."
        ),
    )
    parser.add_argument(
        "--raw-image-orientation",
        action="store_true",
        help="Keep RGB-related projections in raw sensor orientation instead of upright.",
    ) # raw-image-orientation 参数是一个布尔标志，如果设置了这个标志，RGB 相关的投影将保持在原始传感器方向，而不是调整为竖直方向。这可能对于某些分析或可视化任务更有用。
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv
    if output_csv is None and args.output_dir is not None:
        output_csv = default_gaze_csv_path(args.sequence.name, output_dir=args.output_dir)
    result = extract_sequence_gaze(
        args.sequence,
        config=GazeExtractionConfig(
            stride=args.stride,
            start_index=args.start_index,
            end_index=args.end_index,
            start_offset_s=args.start_offset_s,
            end_offset_s=args.end_offset_s,
            stream_id=args.stream_id,
            max_dt_ms=args.max_dt_ms,
            raw_image_orientation=args.raw_image_orientation,
        ),
        output_csv=output_csv,
    )
    print_summary(result.output_csv, result.summary_json, result.summary)


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
