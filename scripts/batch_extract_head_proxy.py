#!/usr/bin/env python
"""Batch extract head-proxy CSV + summary files aligned to gaze CSVs.

This script scans one reports directory for `*_gaze_samples.csv`, then writes
one `<sequence>_head_samples.csv` and one paired summary JSON per sequence.

zh-CN:
这一步仍然不走 skeleton，而是基于 device/CPF pose 构建 head proxy，并且直接
对齐到已有的 gaze CSV 时间戳。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.head import (  # noqa: E402
    add_temporal_head_context,
    default_head_csv_path,
    default_head_summary_json_path,
    extract_head_sample,
    summarize_head_samples,
    write_head_samples_csv,
    write_head_summary_json,
)
from adt_sandbox.providers import create_adt_providers  # noqa: E402
from adt_sandbox.results import discover_sequence_files  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory containing per-sequence *_gaze_samples.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-sequence head outputs. Default is the same as --reports-dir.",
    )
    return parser.parse_args()


def discover_gaze_csv_paths(reports_dir: Path) -> list[Path]:
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {reports_dir}")
    if not reports_dir.is_dir():
        raise NotADirectoryError(f"Expected reports directory: {reports_dir}")
    gaze_csv_paths = [
        item.path
        for item in discover_sequence_files(
            reports_dir,
            "gaze",
            "gaze_samples.csv",
        )
    ]
    if not gaze_csv_paths:
        raise ValueError(f"No gaze sample CSV files found in: {reports_dir}")
    return gaze_csv_paths


def sequence_name_from_gaze_csv(path: Path) -> str:
    return path.stem[: -len("_gaze_samples")] if path.stem.endswith("_gaze_samples") else path.stem


def main() -> None:
    args = parse_args()
    gaze_csv_paths = discover_gaze_csv_paths(args.reports_dir)
    output_dir = args.output_dir or args.reports_dir

    print(f"gaze_csv_files: {len(gaze_csv_paths)}")
    print(f"output_dir: {output_dir}")

    for index, gaze_csv in enumerate(gaze_csv_paths, start=1):
        sequence_name = sequence_name_from_gaze_csv(gaze_csv)
        providers = create_adt_providers(sequence_name, skeleton_flag=True)
        gaze_samples = read_samples_csv(gaze_csv)
        head_samples = [
            extract_head_sample(providers.gt_provider, sample.query_timestamp_ns)
            for sample in gaze_samples
        ]
        head_samples = add_temporal_head_context(head_samples)

        output_csv = default_head_csv_path(sequence_name, output_dir=output_dir)
        summary_json = default_head_summary_json_path(output_csv)
        write_head_samples_csv(output_csv, head_samples)

        summary = summarize_head_samples(head_samples)
        summary.update(
            {
                "sequence_name": sequence_name,
                "sequence_path": str(providers.sequence_path),
                "provider_mode": providers.provider_mode,
                "input_gaze_csv": str(gaze_csv),
                "output_csv": str(output_csv),
                "head_proxy_source": "device_pose_cpf",
                "head_feature_scope": "absolute_scene_pose_plus_relative_motion",
            }
        )
        write_head_summary_json(summary_json, summary)
        print(
            f"[{index}/{len(gaze_csv_paths)}] {sequence_name}: "
            f"samples={summary['sample_count']} "
            f"pose_valid={summary['pose_valid_ratio']:.3f} "
            f"temporal_context={summary['temporal_context_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
