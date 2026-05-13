#!/usr/bin/env python
"""Batch extract ADT gaze CSV + summary files without visualization.

This script reuses the same gaze-only extraction logic as
`scripts/extract_gaze_samples.py`, but runs it across multiple sequences.
It writes one `<sequence>_gaze_samples.csv` and one paired summary JSON per
sequence, then saves one batch-level summary CSV + JSON.

zh-CN:
这个脚本只做批量 gaze 数据提取，不生成任何图片或视频。
输出分两层：
- 每个 sequence 一份 `gaze_samples.csv` 和 `gaze_summary.json`
- 一份批量总表，方便快速查看哪些 sequence 成功、每条的 sample 数量和质量比例

默认行为：
- 如果命令行显式给了 sequence 列表，就处理这些 sequence
- 如果没有给 sequence，就扫描 `ADT_DATA_ROOT` 下所有本地 ADT sequence 目录

Example:
    python scripts/batch_extract_gaze_samples.py
    python scripts/batch_extract_gaze_samples.py \
      Apartment_release_decoration_skeleton_seq131_M1292 \
      Apartment_release_bedroom_seq114_M1292
    python scripts/batch_extract_gaze_samples.py --stride 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv, resolve_data_root  # noqa: E402
from adt_sandbox.gaze import RGB_STREAM_ID  # noqa: E402
from adt_sandbox.gaze_extraction import (  # noqa: E402
    GazeExtractionConfig,
    GazeExtractionResult,
    default_gaze_csv_path,
    discover_sequence_paths,
    extract_sequence_gaze,
)
from adt_sandbox.results import batch_dir  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequences",
        nargs="*",
        help=(
            "Sequence ids or absolute paths. If omitted, scan every ADT sequence "
            "directory under ADT_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory for per-sequence CSV/summary outputs and the batch summary.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Step between RGB frame timestamps. Default is 1.",
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
        help="Exclusive ending RGB timestamp index before applying stride.",
    )
    parser.add_argument(
        "--start-offset-s",
        type=float,
        default=None,
        help="Start time in seconds relative to the first annotation-filtered RGB timestamp.",
    )
    parser.add_argument(
        "--end-offset-s",
        type=float,
        default=None,
        help="Exclusive end time in seconds relative to the first annotation-filtered RGB timestamp.",
    )
    parser.add_argument(
        "--stream-id",
        default=RGB_STREAM_ID,
        help="Project Aria RGB stream id. Default is 214-1.",
    )
    parser.add_argument(
        "--max-dt-ms",
        type=float,
        default=20.0,
        help="Flag gaze samples whose nearest timestamp differs by more than this value.",
    )
    parser.add_argument(
        "--raw-image-orientation",
        action="store_true",
        help="Keep RGB-related projections in raw sensor orientation instead of upright.",
    )
    return parser.parse_args()


def resolve_batch_sequences(sequence_args: list[str]) -> list[str | Path]:
    """Return explicit sequences or discover all sequences under ADT_DATA_ROOT."""

    if sequence_args:
        return sequence_args

    data_root = resolve_data_root()
    if data_root is None:
        raise RuntimeError(
            "No sequences were provided and ADT_DATA_ROOT is not configured. "
            "Set ADT_DATA_ROOT in .env or pass explicit sequence ids/paths."
        )
    return discover_sequence_paths(data_root)


def batch_row_from_result(result: GazeExtractionResult) -> dict[str, Any]:
    """Flatten one sequence result into a stable batch summary row."""

    summary = result.summary
    source_counts = summary["source_counts"]
    return {
        "sequence_name": result.sequence_name,
        "status": "ok",
        "sequence_path": str(result.sequence_path),
        "provider_mode": result.provider_mode,
        "sample_count": summary["sample_count"],
        "gaze_valid_ratio": summary["gaze_valid_ratio"],
        "projection_in_image_ratio": summary["projection_in_image_ratio"],
        "depth_available_ratio": summary["depth_available_ratio"],
        "ok_ratio": summary["ok_ratio"],
        "gaze_dt_mean_ms": summary["gaze_dt_ms"]["mean"],
        "pose_dt_mean_ms": summary["pose_dt_ms"]["mean"],
        "raw_rgb_timestamp_count": source_counts["raw_rgb_timestamp_count"],
        "annotation_filtered_rgb_timestamp_count": source_counts[
            "annotation_filtered_rgb_timestamp_count"
        ],
        "selected_rgb_timestamp_count": source_counts["selected_rgb_timestamp_count"],
        "output_csv": str(result.output_csv),
        "summary_json": str(result.summary_json),
        "error": "",
    }


def batch_row_from_error(sequence: str | Path, error: Exception) -> dict[str, Any]:
    """Return a stable batch summary row for a failed sequence."""

    return {
        "sequence_name": Path(sequence).name,
        "status": "error",
        "sequence_path": str(sequence),
        "provider_mode": "",
        "sample_count": None,
        "gaze_valid_ratio": None,
        "projection_in_image_ratio": None,
        "depth_available_ratio": None,
        "ok_ratio": None,
        "gaze_dt_mean_ms": None,
        "pose_dt_mean_ms": None,
        "raw_rgb_timestamp_count": None,
        "annotation_filtered_rgb_timestamp_count": None,
        "selected_rgb_timestamp_count": None,
        "output_csv": "",
        "summary_json": "",
        "error": str(error),
    }


def write_batch_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the flat batch summary table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_batch_summary_json(
    path: Path,
    config: GazeExtractionConfig,
    rows: list[dict[str, Any]],
) -> None:
    """Write batch-level metadata plus the per-sequence flat rows."""

    summary = {
        "sequence_count": len(rows),
        "success_count": sum(row["status"] == "ok" for row in rows),
        "failure_count": sum(row["status"] == "error" for row in rows),
        "selection": {
            "start_index": config.start_index,
            "end_index": config.end_index,
            "start_offset_s": config.start_offset_s,
            "end_offset_s": config.end_offset_s,
            "stride": config.stride,
            "max_dt_ms": config.max_dt_ms,
            "image_orientation": "raw" if config.raw_image_orientation else "upright",
        },
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_sequence_result(index: int, total: int, row: dict[str, Any]) -> None:
    """Print one concise line per processed sequence."""

    if row["status"] == "ok":
        print(
            f"[{index}/{total}] {row['sequence_name']}: "
            f"samples={row['sample_count']} "
            f"ok_ratio={row['ok_ratio']:.3f} "
            f"projection_in_image_ratio={row['projection_in_image_ratio']:.3f}"
        )
        return

    print(f"[{index}/{total}] {row['sequence_name']}: error={row['error']}")


def main() -> None:
    args = parse_args()
    config = GazeExtractionConfig(
        stride=args.stride,
        start_index=args.start_index,
        end_index=args.end_index,
        start_offset_s=args.start_offset_s,
        end_offset_s=args.end_offset_s,
        stream_id=args.stream_id,
        max_dt_ms=args.max_dt_ms,
        raw_image_orientation=args.raw_image_orientation,
    )
    sequences = resolve_batch_sequences(args.sequences)
    output_dir = args.output_dir
    batch_rows: list[dict[str, Any]] = []

    print(f"sequences: {len(sequences)}")
    print(f"output_dir: {output_dir}")
    print(
        "image_orientation: "
        f"{'raw' if config.raw_image_orientation else 'upright'}"
    )

    for index, sequence in enumerate(sequences, start=1):
        try:
            sequence_name = Path(sequence).name
            output_csv = default_gaze_csv_path(sequence_name, output_dir=output_dir)
            result = extract_sequence_gaze(sequence, config=config, output_csv=output_csv)
            row = batch_row_from_result(result)
        except Exception as exc:  # noqa: BLE001
            row = batch_row_from_error(sequence, exc)
        batch_rows.append(row)
        print_sequence_result(index, len(sequences), row)

    batch_output_dir = batch_dir(output_dir)
    batch_csv = batch_output_dir / "batch_gaze_extract_summary.csv"
    batch_json = batch_output_dir / "batch_gaze_extract_summary.json"
    write_batch_summary_csv(batch_csv, batch_rows)
    write_batch_summary_json(batch_json, config, batch_rows)
    print(
        "batch_summary: "
        f"success={sum(row['status'] == 'ok' for row in batch_rows)} "
        f"failure={sum(row['status'] == 'error' for row in batch_rows)}"
    )
    print(f"batch_csv: {batch_csv}")
    print(f"batch_json: {batch_json}")


if __name__ == "__main__":
    main()
