#!/usr/bin/env python
"""Run SparseGaze-oriented head-utility diagnostics.

Inputs per sequence:
- `<sequence>_gaze_samples.csv`
- `<sequence>_head_samples.csv`
- `<sequence>_scene_gaze_event_features.csv`
- `<sequence>_scene_gaze_frame_labels.csv`

Outputs:
- `<sequence>_sparsegaze_head_utility_summary.csv`
- `<sequence>_sparsegaze_head_utility_lead_lag.csv`
- `batch_sparsegaze_head_utility_summary.csv`
- `batch_sparsegaze_head_utility_lead_lag.csv`
- `batch_sparsegaze_head_utility_aggregate.csv`
- `batch_sparsegaze_head_utility_lead_lag_aggregate.csv`
- `batch_sparsegaze_head_utility_report.json`
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

from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.head import read_head_samples_csv  # noqa: E402
from adt_sandbox.head_gaze_analysis import require_full_head_feature_schema  # noqa: E402
from adt_sandbox.scene_gaze_events import (  # noqa: E402
    read_scene_gaze_event_features_csv,
    read_scene_gaze_frame_labels_csv,
)
from adt_sandbox.sparsegaze_head_utility import (  # noqa: E402
    aggregate_lead_lag_rows,
    aggregate_summary_rows,
    build_lead_lag_rows,
    build_sparse_anchor_utility_rows,
    rows_to_dicts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence_names",
        nargs="*",
        help="Optional sequence names. If omitted, process all complete input sets in --reports-dir.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory containing extracted gaze/head/event CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Default is the same as --reports-dir.",
    )
    parser.add_argument(
        "--anchor-intervals",
        type=int,
        nargs="+",
        default=[2, 3, 5, 10, 15, 30],
        help="Sparse gaze anchor intervals in frames. Default: 2 3 5 10 15 30.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=5,
        help="Head history window length used by diagnostic feature sets. Default: 5.",
    )
    parser.add_argument(
        "--max-lag-frames",
        type=int,
        default=15,
        help="Maximum lag in frames for lead-lag correlation. Default: 15.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = args.reports_dir
    output_dir = args.output_dir or reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_names = (
        list(args.sequence_names)
        if args.sequence_names
        else discover_sequence_names(reports_dir)
    )

    all_summary_rows: list[dict[str, Any]] = []
    all_lead_lag_rows: list[dict[str, Any]] = []
    for index, sequence_name in enumerate(sequence_names, start=1):
        inputs = input_paths(reports_dir, sequence_name)
        gaze_samples = read_samples_csv(inputs["gaze_csv"])
        head_samples = read_head_samples_csv(inputs["head_csv"])
        require_full_head_feature_schema(head_samples)
        scene_event_features = read_scene_gaze_event_features_csv(
            inputs["scene_features_csv"]
        )
        scene_frame_labels = read_scene_gaze_frame_labels_csv(
            inputs["scene_labels_csv"]
        )

        summary_rows = build_sparse_anchor_utility_rows(
            sequence_name=sequence_name,
            gaze_samples=gaze_samples,
            head_samples=head_samples,
            scene_event_features=scene_event_features,
            scene_frame_labels=scene_frame_labels,
            anchor_interval_frames=args.anchor_intervals,
            history_frames=args.history_frames,
        )
        lead_lag_rows = build_lead_lag_rows(
            sequence_name=sequence_name,
            gaze_samples=gaze_samples,
            head_samples=head_samples,
            scene_event_features=scene_event_features,
            scene_frame_labels=scene_frame_labels,
            max_lag_frames=args.max_lag_frames,
        )

        summary_dicts = rows_to_dicts(summary_rows)
        lead_lag_dicts = rows_to_dicts(lead_lag_rows)
        write_csv(
            output_dir / f"{sequence_name}_sparsegaze_head_utility_summary.csv",
            summary_dicts,
        )
        write_csv(
            output_dir / f"{sequence_name}_sparsegaze_head_utility_lead_lag.csv",
            lead_lag_dicts,
        )
        all_summary_rows.extend(summary_dicts)
        all_lead_lag_rows.extend(lead_lag_dicts)

        key_row = key_sequence_row(summary_dicts)
        print(
            f"[{index}/{len(sequence_names)}] {sequence_name}: "
            f"hold_last_N10_scene_resid={format_optional(key_row.get('mean_scene_residual_deg'))} "
            f"head_history_r2={format_optional(key_row.get('ridge_scene_r2_head_history'))}"
        )

    summary_aggregate = aggregate_summary_rows_from_dicts(all_summary_rows)
    lead_lag_aggregate = aggregate_lead_lag_rows_from_dicts(all_lead_lag_rows)

    write_csv(output_dir / "batch_sparsegaze_head_utility_summary.csv", all_summary_rows)
    write_csv(
        output_dir / "batch_sparsegaze_head_utility_lead_lag.csv",
        all_lead_lag_rows,
    )
    write_csv(
        output_dir / "batch_sparsegaze_head_utility_aggregate.csv",
        summary_aggregate,
    )
    write_csv(
        output_dir / "batch_sparsegaze_head_utility_lead_lag_aggregate.csv",
        lead_lag_aggregate,
    )
    write_json(
        output_dir / "batch_sparsegaze_head_utility_report.json",
        {
            "sequence_count": len(sequence_names),
            "reports_dir": str(reports_dir),
            "output_dir": str(output_dir),
            "anchor_intervals": args.anchor_intervals,
            "history_frames": args.history_frames,
            "max_lag_frames": args.max_lag_frames,
            "method": {
                "sparse_anchor_residual": (
                    "simulate missing gaze between periodic anchors; compare "
                    "hold-last and linear interpolation baselines in CPF and Scene"
                ),
                "ridge_diagnostic": (
                    "blocked cross-validated ridge R2 for residual magnitude using "
                    "gap-only, current-head, and head-history feature sets"
                ),
                "lead_lag": (
                    "correlate head motion at frame t with gaze dynamics at t+k; "
                    "positive k means head is compared with future gaze"
                ),
                "event_conditioning": (
                    "summaries are split by scene-direction fixation/transition labels"
                ),
            },
            "summary_aggregate": summary_aggregate,
            "lead_lag_aggregate": lead_lag_aggregate,
        },
    )
    print(f"sequences: {len(sequence_names)}")
    print(f"summary_csv: {output_dir / 'batch_sparsegaze_head_utility_summary.csv'}")
    print(
        f"lead_lag_csv: {output_dir / 'batch_sparsegaze_head_utility_lead_lag.csv'}"
    )
    print(f"report_json: {output_dir / 'batch_sparsegaze_head_utility_report.json'}")


def discover_sequence_names(reports_dir: Path) -> list[str]:
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {reports_dir}")
    names: list[str] = []
    for gaze_csv in sorted(reports_dir.glob("*_gaze_samples.csv")):
        sequence_name = gaze_csv.stem[: -len("_gaze_samples")]
        if all(path.exists() for path in input_paths(reports_dir, sequence_name).values()):
            names.append(sequence_name)
    if not names:
        raise ValueError(f"No complete SparseGaze head-utility input sets found in: {reports_dir}")
    return names


def input_paths(reports_dir: Path, sequence_name: str) -> dict[str, Path]:
    return {
        "gaze_csv": reports_dir / f"{sequence_name}_gaze_samples.csv",
        "head_csv": reports_dir / f"{sequence_name}_head_samples.csv",
        "scene_features_csv": reports_dir
        / f"{sequence_name}_scene_gaze_event_features.csv",
        "scene_labels_csv": reports_dir / f"{sequence_name}_scene_gaze_frame_labels.csv",
    }


def key_sequence_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if (
            row["anchor_interval_frames"] == 10
            and row["baseline"] == "hold_last"
            and row["event_group"] == "all"
        ):
            return row
    return rows[0] if rows else {}


def aggregate_summary_rows_from_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    typed_rows = [
        _summary_row_from_dict(row)
        for row in rows
    ]
    return aggregate_summary_rows(typed_rows)


def aggregate_lead_lag_rows_from_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    typed_rows = [_lead_lag_row_from_dict(row) for row in rows]
    return aggregate_lead_lag_rows(typed_rows)


def _summary_row_from_dict(row: dict[str, Any]):
    from adt_sandbox.sparsegaze_head_utility import SparseGazeHeadUtilitySummaryRow

    return SparseGazeHeadUtilitySummaryRow(**row)


def _lead_lag_row_from_dict(row: dict[str, Any]):
    from adt_sandbox.sparsegaze_head_utility import SparseGazeLeadLagRow

    return SparseGazeLeadLagRow(**row)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def format_optional(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "nan"


if __name__ == "__main__":
    main()
