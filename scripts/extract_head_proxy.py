#!/usr/bin/env python
"""Extract head-proxy features aligned to an existing ADT gaze CSV.

This script reads the query timestamps from an existing `gaze_samples.csv`,
queries the nearest ADT pose on those timestamps, and writes a `head_samples.csv`
plus a lightweight `head_summary.json`.

zh-CN:
当前 head feature 不从 skeleton 提 head，而是直接用 device/CPF pose 作为
head proxy。这样 head 和当前 gaze 的 CPF / Scene 链天然对齐。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.head import (  # noqa: E402
    HEAD_FIELD_COORDINATE_FRAMES,
    HEAD_FIELD_DEFINITIONS,
    default_head_csv_path,
    default_head_summary_json_path,
    extract_head_samples_at_timestamps,
    summarize_head_samples,
    write_head_samples_csv,
    write_head_summary_json,
)
from adt_sandbox.providers import create_adt_providers  # noqa: E402
from adt_sandbox.results import find_sequence_file  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        type=Path,
        help="ADT sequence id resolved under ADT_DATA_ROOT, or an absolute sequence path.",
    )
    parser.add_argument(
        "--input-gaze-csv",
        type=Path,
        default=None,
        help="Input gaze CSV path. Defaults to outputs/reports/sequences/<sequence>/gaze/gaze_samples.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output head CSV path. Overrides --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Reports root for organized output. Default is outputs/reports; "
            "writes sequences/<sequence>/head/head_samples.csv."
        ),
    )
    return parser.parse_args()


def default_gaze_csv_path(sequence: Path) -> Path:
    return find_sequence_file(
        REPO_ROOT / "outputs" / "reports",
        sequence.name,
        "gaze",
        "gaze_samples.csv",
    )


def main() -> None:
    args = parse_args()
    gaze_csv = args.input_gaze_csv or default_gaze_csv_path(args.sequence)
    gaze_samples = read_samples_csv(gaze_csv)

    providers = create_adt_providers(args.sequence, skeleton_flag=True)
    output_csv = args.output_csv or default_head_csv_path(
        providers.sequence_path.name,
        output_dir=args.output_dir,
    )
    summary_json = default_head_summary_json_path(output_csv)

    head_samples = extract_head_samples_at_timestamps(
        providers.gt_provider,
        [sample.query_timestamp_ns for sample in gaze_samples],
    )
    write_head_samples_csv(output_csv, head_samples)

    summary = summarize_head_samples(head_samples)
    summary.update(
        {
            "sequence_name": providers.sequence_path.name,
            "sequence_path": str(providers.sequence_path),
            "provider_mode": providers.provider_mode,
            "input_gaze_csv": str(gaze_csv),
            "output_csv": str(output_csv),
            "head_proxy_source": "device_pose_cpf",
            "field_coordinate_frames": HEAD_FIELD_COORDINATE_FRAMES,
            "field_definitions": HEAD_FIELD_DEFINITIONS,
        }
    )
    write_head_summary_json(summary_json, summary)
    print_summary(output_csv, summary_json, summary)


def print_summary(output_csv: Path, summary_json: Path, summary: dict[str, Any]) -> None:
    print(f"sequence: {summary['sequence_name']}")
    print(f"sequence_path: {summary['sequence_path']}")
    print(f"provider_mode: {summary['provider_mode']}")
    print(f"head_proxy_source: {summary['head_proxy_source']}")
    print(
        "samples: "
        f"{summary['sample_count']} "
        f"pose_valid={summary['pose_valid_count']} "
        f"temporal_context={summary['temporal_context_count']}"
    )
    print(
        "selected_timestamps_ns: "
        f"{summary['query_timestamp_start_ns']}..{summary['query_timestamp_end_ns']} "
        f"duration_s={summary['duration_s']:.3f}"
    )
    if summary["validation_note_counts"]:
        print(f"validation_note_counts: {summary['validation_note_counts']}")
    print(f"input_gaze_csv: {summary['input_gaze_csv']}")
    print(f"csv: {output_csv}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
