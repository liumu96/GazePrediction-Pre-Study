#!/usr/bin/env python
"""Compute per-frame CPF-local gaze dynamics from existing gaze/head CSV files.

This script does not reopen the ADT provider. It reads previously extracted:
- `<sequence>_gaze_samples.csv`
- `<sequence>_head_samples.csv`

and writes:
- `<sequence>_gaze_dynamics.csv`
- `<sequence>_gaze_dynamics_summary.json`

The output is a feature layer, not a fixation/saccade label layer. The CPF-local
velocity and dispersion values are useful diagnostics, but they do not define
whether the user is fixating a stable target in the scene.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.gaze_dynamics import (  # noqa: E402
    compute_gaze_dynamics_features,
    default_gaze_dynamics_csv_path,
    default_gaze_dynamics_summary_json_path,
    summarize_gaze_dynamics_features,
    write_gaze_dynamics_csv,
    write_summary_json,
)
from adt_sandbox.head import read_head_samples_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence_names",
        nargs="*",
        help="Optional sequence names. If omitted, process all sequences in --reports-dir.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory containing per-sequence gaze/head CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for gaze dynamics CSV/JSON outputs. Default is the same as --reports-dir.",
    )
    parser.add_argument(
        "--dispersion-window-frames",
        type=int,
        default=5,
        help="Centered CPF-gaze dispersion window in frames. Default is 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = args.reports_dir
    output_dir = args.output_dir or reports_dir
    sequence_names = (
        list(args.sequence_names)
        if args.sequence_names
        else discover_sequence_names(reports_dir)
    )

    batch_rows: list[dict[str, Any]] = []
    for index, sequence_name in enumerate(sequence_names, start=1):
        gaze_csv = reports_dir / f"{sequence_name}_gaze_samples.csv"
        head_csv = reports_dir / f"{sequence_name}_head_samples.csv"
        gaze_samples = read_samples_csv(gaze_csv)
        head_samples = read_head_samples_csv(head_csv)
        feature_rows = compute_gaze_dynamics_features(
            gaze_samples,
            head_samples,
            dispersion_window_frames=args.dispersion_window_frames,
        )

        output_csv = default_gaze_dynamics_csv_path(sequence_name, output_dir=output_dir)
        summary_json = default_gaze_dynamics_summary_json_path(output_csv)
        write_gaze_dynamics_csv(output_csv, feature_rows)

        summary = summarize_gaze_dynamics_features(feature_rows)
        summary.update(
            {
                "sequence_name": sequence_name,
                "input_gaze_csv": str(gaze_csv),
                "input_head_csv": str(head_csv),
                "output_csv": str(output_csv),
                "dispersion_window_frames": args.dispersion_window_frames,
                "method": {
                    "local_space": "CPF unit gaze direction",
                    "dispersion_definition": "max pairwise angular separation within centered window",
                    "velocity_definition": "consecutive CPF angular step / dt",
                    "head_context_source": "device_pose_cpf",
                    "label_definition": "none; this script intentionally does not emit fixation labels",
                },
            }
        )
        write_summary_json(summary_json, summary)

        batch_rows.append(
            {
                "sequence_name": sequence_name,
                "sample_count": summary["sample_count"],
                "dynamics_input_valid_ratio": summary["dynamics_input_valid_ratio"],
                "local_velocity_p95_deg_s": summary["local_velocity_deg_s"]["p95"],
                "local_velocity_p99_deg_s": summary["local_velocity_deg_s"]["p99"],
                "dispersion_p95_deg": summary["window_dispersion_deg"]["p95"],
                "dispersion_p99_deg": summary["window_dispersion_deg"]["p99"],
                "gaze_head_angle_p95_deg": summary["gaze_head_angle_deg"]["p95"],
                "head_translation_speed_p95_m_s": summary["head_translation_speed_m_s"]["p95"],
                "head_forward_angle_step_p95_deg": summary["head_forward_angle_step_deg"]["p95"],
                "output_csv": str(output_csv),
                "summary_json": str(summary_json),
            }
        )
        print(
            f"[{index}/{len(sequence_names)}] {sequence_name}: "
            f"samples={summary['sample_count']} "
            f"dynamics_input_valid={summary['dynamics_input_valid_ratio']:.3f} "
            f"vel_p95={summary['local_velocity_deg_s']['p95']:.3f} "
            f"disp_p95={summary['window_dispersion_deg']['p95']:.3f}"
        )

    batch_csv = output_dir / "batch_gaze_dynamics_summary.csv"
    write_batch_csv(batch_csv, batch_rows)
    print(f"batch_csv: {batch_csv}")


def discover_sequence_names(reports_dir: Path) -> list[str]:
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {reports_dir}")
    names: list[str] = []
    for gaze_csv in sorted(reports_dir.glob("*_gaze_samples.csv")):
        sequence_name = gaze_csv.stem[: -len("_gaze_samples")]
        head_csv = reports_dir / f"{sequence_name}_head_samples.csv"
        if head_csv.exists():
            names.append(sequence_name)
    if not names:
        raise ValueError(
            f"No paired *_gaze_samples.csv and *_head_samples.csv files found in: {reports_dir}"
        )
    return names


def write_batch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No batch rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
