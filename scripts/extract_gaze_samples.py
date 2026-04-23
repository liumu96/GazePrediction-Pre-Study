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
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import RGB_STREAM_ID  # noqa: E402
from adt_sandbox.gaze_extraction import (  # noqa: E402
    GazeExtractionConfig,
    extract_sequence_gaze,
)

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
        output_csv=args.output_csv,
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
