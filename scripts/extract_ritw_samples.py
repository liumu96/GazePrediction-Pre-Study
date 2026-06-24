#!/usr/bin/env python
"""Extract RITW MPS gaze/head features and optional SparseGaze caches.

The output mirrors the organized ADT sandbox layout:

    <output-dir>/sequences/<recording>/gaze/gaze_samples.csv
    <output-dir>/sequences/<recording>/head/head_samples.csv
    <output-dir>/sequences/<recording>/mps/mps_summary.json

By default this also writes SparseGaze feature caches to:

    /mnt/d/SparseGaze/feature_cache/ritw30/sparsegaze

Each cache file includes both the generic SparseGaze keys and the CPF cache keys
used by the sparsegaze_cpf_* loaders.
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

from adt_sandbox.ritw import (  # noqa: E402
    DEFAULT_RITW_CACHE_ROOT,
    DEFAULT_RITW_REPORTS_ROOT,
    DEFAULT_RITW_ROOT,
    RitwExtractionConfig,
    RitwExtractionResult,
    discover_ritw_recordings,
    extract_ritw_recording,
    write_sparsegaze_manifest,
)
from adt_sandbox.results import batch_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recordings",
        nargs="*",
        help=(
            "Recording ids such as recording_1029649168624048 or absolute paths. "
            "If omitted, scan all recordings under --ritw-root."
        ),
    )
    parser.add_argument("--ritw-root", type=Path, default=DEFAULT_RITW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RITW_REPORTS_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_RITW_CACHE_ROOT)
    parser.add_argument(
        "--gaze-kind",
        choices=["personalized", "general", "auto"],
        default="personalized",
        help="Use personalized gaze when available by default, falling back to general.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride over RITW eye-gaze rows. Default 2 converts 60 Hz gaze to RITW30.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument(
        "--max-pose-dt-ms",
        type=float,
        default=20.0,
        help="Mark nearest SLAM poses farther than this as invalid. Use -1 to disable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N recordings.")
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing later recordings when one recording fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recordings = resolve_recordings(args.recordings, args.ritw_root)
    if args.limit is not None:
        recordings = recordings[: int(args.limit)]
    if not recordings:
        raise RuntimeError("No RITW recordings selected")

    max_pose_dt_ms = None if args.max_pose_dt_ms is not None and args.max_pose_dt_ms < 0 else args.max_pose_dt_ms
    config = RitwExtractionConfig(
        gaze_kind=args.gaze_kind,
        stride=args.stride,
        start_index=args.start_index,
        end_index=args.end_index,
        max_pose_dt_ms=max_pose_dt_ms,
        write_feature_cache=not args.no_feature_cache,
        cache_root=args.cache_root,
    )

    results: list[RitwExtractionResult] = []
    failures: list[dict[str, Any]] = []
    for index, recording in enumerate(recordings, start=1):
        print(f"[ritw] {index}/{len(recordings)} {recording.name}", flush=True)
        try:
            result = extract_ritw_recording(recording, output_dir=args.output_dir, config=config)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append(
                {
                    "sequence_name": recording.name,
                    "sequence_path": str(recording),
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"[ritw] error {recording.name}: {exc}", flush=True)
            continue
        results.append(result)
        gaze_summary = result.summary["gaze"]
        print(
            "[ritw] ok "
            f"{result.sequence_name} split={result.split} "
            f"samples={gaze_summary['sample_count']} "
            f"ok_ratio={gaze_summary['ok_ratio']:.3f} "
            f"cache={result.cache_npz}",
            flush=True,
        )

    manifest_path = None
    if (
        any(result.cache_npz is not None for result in results)
        and not args.no_feature_cache
        and not args.no_manifest
    ):
        manifest_path = write_sparsegaze_manifest(results, cache_root=args.cache_root)
        print(f"[ritw] manifest={manifest_path}", flush=True)

    summary_csv, summary_json = write_batch_summary(
        results,
        failures,
        output_dir=args.output_dir,
        manifest_path=manifest_path,
    )
    print(f"[ritw] batch_summary_csv={summary_csv}", flush=True)
    print(f"[ritw] batch_summary_json={summary_json}", flush=True)


def resolve_recordings(recording_args: list[str], root: Path) -> list[Path]:
    root = root.expanduser()
    if not recording_args:
        return discover_ritw_recordings(root)

    resolved: list[Path] = []
    for item in recording_args:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = root / path
        resolved.append(path.resolve())
    return resolved


def write_batch_summary(
    results: list[RitwExtractionResult],
    failures: list[dict[str, Any]],
    *,
    output_dir: Path,
    manifest_path: Path | None,
) -> tuple[Path, Path]:
    out_dir = batch_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [batch_row_from_result(result) for result in results] + failures
    if not rows:
        raise RuntimeError("No successful or failed rows to summarize")

    csv_path = out_dir / "ritw_extract_summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "dataset": "ritw30",
        "successful_count": len(results),
        "failure_count": len(failures),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "rows": rows,
    }
    json_path = out_dir / "ritw_extract_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def batch_row_from_result(result: RitwExtractionResult) -> dict[str, Any]:
    gaze = result.summary["gaze"]
    head = result.summary["head"]
    return {
        "sequence_name": result.sequence_name,
        "sequence_path": str(result.sequence_path),
        "split": result.split,
        "status": "ok",
        "sample_count": gaze["sample_count"],
        "gaze_ok_ratio": gaze["ok_ratio"],
        "gaze_pose_valid_ratio": gaze["pose_valid_ratio"],
        "head_pose_valid_ratio": head["pose_valid_ratio"],
        "gaze_csv": str(result.output_gaze_csv),
        "head_csv": str(result.output_head_csv),
        "mps_summary_json": str(result.output_mps_summary_json),
        "cache_npz": str(result.cache_npz) if result.cache_npz is not None else "",
    }


if __name__ == "__main__":
    main()
