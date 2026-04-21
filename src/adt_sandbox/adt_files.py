"""Lightweight ADT sequence file inspection utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

KNOWN_ADT_FILES: dict[str, tuple[str, ...]] = {
    "metadata": (
        "metadata.json",
        "instances.json",
        "scene_objects.csv",
        "aria_trajectory.csv",
    ),
    "annotations": (
        "2d_bounding_box.csv",
        "2d_bounding_box_with_skeleton.csv",
        "3d_bounding_box.csv",
        "Skeleton_T.json",
        "skeleton_aria_association.json",
        "eyegaze.csv",
    ),
    "vrs": (
        "video.vrs",
        "synthetic_video.vrs",
        "depth_images.vrs",
        "depth_images_with_skeleton.vrs",
        "segmentations.vrs",
        "segmentations_with_skeleton.vrs",
    ),
    "mps": (
        "mps/eye_gaze/general_eye_gaze.csv",
        "mps/eye_gaze/summary.json",
    ),
}


def inspect_sequence(sequence_dir: str | Path) -> dict[str, Any]:
    """Return a lightweight summary of an ADT sequence directory."""

    root = Path(sequence_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Sequence directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root}")

    known_files = []
    for group, relative_paths in KNOWN_ADT_FILES.items():
        for relative_path in relative_paths:
            path = root / relative_path
            known_files.append(
                {
                    "group": group,
                    "path": relative_path,
                    "exists": path.exists(),
                    "size": path.stat().st_size if path.exists() else None,
                }
            )

    csv_files = sorted(root.rglob("*.csv"))
    json_files = sorted([*root.rglob("*.json"), *root.rglob("*.jsonl")])
    vrs_files = sorted(root.rglob("*.vrs"))

    return {
        "root": str(root),
        "known_files": known_files,
        "csv_files": [_summarize_csv(path, root) for path in csv_files],
        "json_files": [_summarize_json(path, root) for path in json_files],
        "vrs_files": [_summarize_file(path, root) for path in vrs_files],
        "totals": {
            "csv": len(csv_files),
            "json": len(json_files),
            "vrs": len(vrs_files),
            "files": sum(1 for path in root.rglob("*") if path.is_file()),
        },
    }


def format_report(report: dict[str, Any]) -> str:
    """Format an `inspect_sequence` result for terminal output."""

    lines = [f"ADT sequence: {report['root']}", ""]

    lines.append("Known ADT files:")
    for item in report["known_files"]:
        status = "ok" if item["exists"] else "missing"
        size = f" ({_human_size(item['size'])})" if item["size"] is not None else ""
        lines.append(f"  [{status}] {item['group']}: {item['path']}{size}")

    lines.extend(["", "CSV files:"])
    if report["csv_files"]:
        for item in report["csv_files"]:
            rows = item["rows"]
            row_text = "unknown rows" if rows is None else f"{rows} rows"
            lines.append(f"  {item['path']}: {row_text}, {_human_size(item['size'])}")
    else:
        lines.append("  none")

    lines.extend(["", "JSON files:"])
    if report["json_files"]:
        for item in report["json_files"]:
            detail = item.get("detail") or "unreadable"
            lines.append(f"  {item['path']}: {detail}, {_human_size(item['size'])}")
    else:
        lines.append("  none")

    lines.extend(["", "VRS files:"])
    if report["vrs_files"]:
        for item in report["vrs_files"]:
            lines.append(f"  {item['path']}: {_human_size(item['size'])}")
    else:
        lines.append("  none")

    totals = report["totals"]
    lines.extend(
        [
            "",
            "Totals:",
            f"  files={totals['files']} csv={totals['csv']} json={totals['json']} vrs={totals['vrs']}",
        ]
    )
    return "\n".join(lines)


def _summarize_file(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
    }


def _summarize_csv(path: Path, root: Path) -> dict[str, Any]:
    rows = None
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header_seen = False
            count = 0
            for _ in reader:
                if not header_seen:
                    header_seen = True
                    continue
                count += 1
            rows = count
    except UnicodeDecodeError:
        rows = None

    summary = _summarize_file(path, root)
    summary["rows"] = rows
    return summary


def _summarize_json(path: Path, root: Path) -> dict[str, Any]:
    summary = _summarize_file(path, root)
    try:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                line_count = sum(1 for _ in handle)
            summary["detail"] = f"jsonl lines={line_count}"
            return summary

        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError):
        summary["detail"] = "parse error"
        return summary

    if isinstance(value, dict):
        keys = list(value)[:8]
        suffix = "..." if len(value) > len(keys) else ""
        summary["detail"] = f"object keys={keys}{suffix}"
    elif isinstance(value, list):
        summary["detail"] = f"array items={len(value)}"
    else:
        summary["detail"] = type(value).__name__
    return summary


def _human_size(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"
