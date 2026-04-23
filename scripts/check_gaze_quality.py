#!/usr/bin/env python
"""Summarize ADT gaze quality from previously extracted summary JSON files.

This script reads existing per-sequence `*_gaze_summary.json` files and writes:
- one flat CSV for quick spreadsheet-style inspection
- one aggregate JSON with sequence-level rankings and note counts

zh-CN:
这个脚本只读取已经落盘的 `gaze_summary.json`，不会重新打开 ADT provider。
适合在批量提取完成后，对整个目录做一次 sequence-level 质量体检。

Example:
    python scripts/check_gaze_quality.py
    python scripts/check_gaze_quality.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.gaze import read_gaze_summary_json  # noqa: E402

DEFAULT_RATIO_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help=(
            "Directory containing per-sequence *_gaze_summary.json files. "
            "Default is outputs/reports."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for gaze_quality_report.csv/json. Default is the same as "
            "--reports-dir."
        ),
    )
    return parser.parse_args()


def discover_summary_paths(reports_dir: Path) -> list[Path]:
    """Return all per-sequence gaze summary JSON files in one directory."""

    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {reports_dir}")
    if not reports_dir.is_dir():
        raise NotADirectoryError(f"Expected reports directory: {reports_dir}")

    summary_paths = sorted(reports_dir.glob("*_gaze_summary.json"))
    if not summary_paths:
        raise ValueError(f"No *_gaze_summary.json files found in: {reports_dir}")
    return summary_paths


def safe_nested(summary: dict[str, Any], key: str, nested: str) -> Any:
    """Return one nested value when present, otherwise None."""

    value = summary.get(key)
    if not isinstance(value, dict):
        return None
    return value.get(nested)


def top_validation_issue(summary: dict[str, Any]) -> tuple[str, int]:
    """Return the most common validation note for one sequence."""

    note_counts = summary.get("validation_note_counts", {})
    if not note_counts:
        return "ok", 0
    issue, count = max(note_counts.items(), key=lambda item: item[1])
    return str(issue), int(count)


def quality_flags(summary: dict[str, Any]) -> list[str]:
    """Return simple review flags for one sequence.

    zh-CN:
    这里不直接下结论说 sequence “坏”或“好”，只给出当前提取口径下值得复查的
    信号。阈值先固定在 0.95，后面如果你形成正式过滤规则，再单独收敛。
    """

    flags: list[str] = []
    if float(summary.get("gaze_valid_ratio", 0.0)) < DEFAULT_RATIO_THRESHOLD:
        flags.append("low_gaze_valid_ratio")
    if float(summary.get("projection_in_image_ratio", 0.0)) < DEFAULT_RATIO_THRESHOLD:
        flags.append("low_projection_in_image_ratio")
    if float(summary.get("depth_available_ratio", 0.0)) < DEFAULT_RATIO_THRESHOLD:
        flags.append("low_depth_available_ratio")
    if float(summary.get("ok_ratio", 0.0)) < DEFAULT_RATIO_THRESHOLD:
        flags.append("low_ok_ratio")
    return flags


def row_from_summary(summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten one per-sequence summary JSON into one CSV row."""

    top_issue, top_issue_count = top_validation_issue(summary)
    flags = quality_flags(summary)
    source_counts = summary.get("source_counts", {})
    return {
        "sequence_name": summary.get("sequence_name", summary_path.stem),
        "sample_count": summary.get("sample_count"),
        "gaze_valid_ratio": summary.get("gaze_valid_ratio"),
        "projection_in_image_ratio": summary.get("projection_in_image_ratio"),
        "depth_available_ratio": summary.get("depth_available_ratio"),
        "pose_valid_ratio": summary.get("pose_valid_ratio"),
        "ok_ratio": summary.get("ok_ratio"),
        "gaze_dt_mean_ms": safe_nested(summary, "gaze_dt_ms", "mean"),
        "gaze_dt_max_ms": safe_nested(summary, "gaze_dt_ms", "max"),
        "pose_dt_mean_ms": safe_nested(summary, "pose_dt_ms", "mean"),
        "pose_dt_max_ms": safe_nested(summary, "pose_dt_ms", "max"),
        "depth_mean_m": safe_nested(summary, "depth_m", "mean"),
        "depth_max_m": safe_nested(summary, "depth_m", "max"),
        "raw_rgb_timestamp_count": source_counts.get("raw_rgb_timestamp_count"),
        "annotation_filtered_rgb_timestamp_count": source_counts.get(
            "annotation_filtered_rgb_timestamp_count"
        ),
        "selected_rgb_timestamp_count": source_counts.get("selected_rgb_timestamp_count"),
        "top_validation_issue": top_issue,
        "top_validation_issue_count": top_issue_count,
        "quality_flags": ";".join(flags) if flags else "ok",
        "summary_json": str(summary_path),
        "output_csv": summary.get("output_csv", ""),
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one flat CSV report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_note_counts(summaries: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate validation note counts across all sequences."""

    counter: Counter[str] = Counter()
    for summary in summaries:
        counter.update(summary.get("validation_note_counts", {}))
    return dict(sorted(counter.items()))


def aggregate_counts(rows: list[dict[str, Any]], key: str) -> float | None:
    """Return the mean of one numeric row field when values are present."""

    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def ranked_sequences(
    rows: list[dict[str, Any]],
    key: str,
    ascending: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-N rows sorted by one quality metric."""

    ranked = [row for row in rows if row.get(key) is not None]
    ranked.sort(key=lambda row: float(row[key]), reverse=not ascending)
    return [
        {
            "sequence_name": row["sequence_name"],
            key: row[key],
            "quality_flags": row["quality_flags"],
            "top_validation_issue": row["top_validation_issue"],
        }
        for row in ranked[:limit]
    ]


def build_quality_report(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the aggregate JSON report."""

    flagged_rows = [row for row in rows if row["quality_flags"] != "ok"]
    return {
        "sequence_count": len(rows),
        "flagged_sequence_count": len(flagged_rows),
        "mean_ratios": {
            "gaze_valid_ratio": aggregate_counts(rows, "gaze_valid_ratio"),
            "projection_in_image_ratio": aggregate_counts(rows, "projection_in_image_ratio"),
            "depth_available_ratio": aggregate_counts(rows, "depth_available_ratio"),
            "pose_valid_ratio": aggregate_counts(rows, "pose_valid_ratio"),
            "ok_ratio": aggregate_counts(rows, "ok_ratio"),
        },
        "aggregate_validation_note_counts": aggregate_note_counts(summaries),
        "lowest_ok_ratio_sequences": ranked_sequences(rows, "ok_ratio", ascending=True),
        "lowest_projection_ratio_sequences": ranked_sequences(
            rows,
            "projection_in_image_ratio",
            ascending=True,
        ),
        "lowest_depth_ratio_sequences": ranked_sequences(
            rows,
            "depth_available_ratio",
            ascending=True,
        ),
        "highest_gaze_dt_mean_sequences": ranked_sequences(
            rows,
            "gaze_dt_mean_ms",
            ascending=False,
        ),
        "flagged_sequences": [
            {
                "sequence_name": row["sequence_name"],
                "quality_flags": row["quality_flags"],
                "top_validation_issue": row["top_validation_issue"],
                "top_validation_issue_count": row["top_validation_issue_count"],
            }
            for row in flagged_rows
        ],
    }


def print_report(rows: list[dict[str, Any]], report: dict[str, Any], csv_path: Path, json_path: Path) -> None:
    """Print a concise terminal summary."""

    print(f"sequences: {report['sequence_count']}")
    print(f"flagged_sequences: {report['flagged_sequence_count']}")
    mean_ratios = report["mean_ratios"]
    print(
        "mean_ratios: "
        f"gaze_valid={mean_ratios['gaze_valid_ratio']:.3f} "
        f"projection_in_image={mean_ratios['projection_in_image_ratio']:.3f} "
        f"depth_available={mean_ratios['depth_available_ratio']:.3f} "
        f"ok={mean_ratios['ok_ratio']:.3f}"
    )
    if report["lowest_ok_ratio_sequences"]:
        worst = report["lowest_ok_ratio_sequences"][0]
        print(
            "lowest_ok_ratio: "
            f"{worst['sequence_name']} "
            f"ok_ratio={float(worst['ok_ratio']):.3f} "
            f"issue={worst['top_validation_issue']}"
        )
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")


def main() -> None:
    args = parse_args()
    reports_dir = args.reports_dir
    output_dir = args.output_dir or reports_dir
    summary_paths = discover_summary_paths(reports_dir)
    summaries = [read_gaze_summary_json(path) for path in summary_paths]
    rows = [row_from_summary(path, summary) for path, summary in zip(summary_paths, summaries)]
    rows.sort(key=lambda row: (row["quality_flags"] == "ok", -(row["ok_ratio"] or 0.0)))

    report = build_quality_report(rows, summaries)
    csv_path = output_dir / "gaze_quality_report.csv"
    json_path = output_dir / "gaze_quality_report.json"
    write_rows_csv(csv_path, rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(rows, report, csv_path, json_path)


if __name__ == "__main__":
    main()
