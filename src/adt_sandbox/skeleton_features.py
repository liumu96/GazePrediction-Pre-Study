"""Skeleton feature extraction aligned to existing gaze timestamps."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SKELETON_JSON = "Skeleton_T.json"


@dataclass(frozen=True)
class SkeletonSample:
    query_timestamp_ns: int
    frame_index: int
    skeleton_valid: bool
    skeleton_dt_ns: int | None
    matched_skeleton_timestamp_ns: int | None
    root_joint_scene_x_m: float | None
    root_joint_scene_y_m: float | None
    root_joint_scene_z_m: float | None
    head_joint_scene_x_m: float | None
    head_joint_scene_y_m: float | None
    head_joint_scene_z_m: float | None
    joints_flat_scene_m: list[float | None]

    def as_csv_row(self, joint_labels: Sequence[str]) -> dict[str, Any]:
        row = asdict(self)
        joints_flat = row.pop("joints_flat_scene_m")
        for joint_index, label in enumerate(joint_labels):
            safe_label = safe_column_name(label)
            base = joint_index * 3
            row[f"joint_{joint_index:02d}_{safe_label}_scene_x_m"] = joints_flat[base]
            row[f"joint_{joint_index:02d}_{safe_label}_scene_y_m"] = joints_flat[base + 1]
            row[f"joint_{joint_index:02d}_{safe_label}_scene_z_m"] = joints_flat[base + 2]
        return row


@dataclass(frozen=True)
class SkeletonMetadata:
    joint_labels: list[str]
    joint_connections: list[tuple[int, int]]
    marker_labels: list[str]


def load_skeleton_metadata() -> SkeletonMetadata:
    """Load official ADT skeleton labels/connections from projectaria-tools."""

    try:
        from projectaria_tools.projects.adt import AriaDigitalTwinSkeletonProvider
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import projectaria_tools. Run this script in the `adt` conda "
            "environment, e.g. `conda run -n adt python scripts/extract_skeleton_samples.py ...`."
        ) from exc

    return SkeletonMetadata(
        joint_labels=list(AriaDigitalTwinSkeletonProvider.get_joint_labels()),
        joint_connections=[
            (int(first), int(second))
            for first, second in AriaDigitalTwinSkeletonProvider.get_joint_connections()
        ],
        marker_labels=list(AriaDigitalTwinSkeletonProvider.get_marker_labels()),
    )


def extract_skeleton_samples_at_timestamps(
    sequence_dir: str | Path,
    timestamps_ns: Sequence[int],
    max_dt_ns: int | None = 50_000_000,
) -> tuple[list[SkeletonSample], SkeletonMetadata]:
    """Extract skeleton joints at requested timestamps.

    The output joints are in ADT Scene/world frame, meters. If `max_dt_ns` is not
    None, samples whose nearest skeleton frame is farther away are marked invalid.
    """

    try:
        from projectaria_tools.projects.adt import AriaDigitalTwinSkeletonProvider
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import projectaria_tools. Run this script in the `adt` conda "
            "environment, e.g. `conda run -n adt python scripts/extract_skeleton_samples.py ...`."
        ) from exc

    root = Path(sequence_dir).expanduser().resolve()
    skeleton_json = root / SKELETON_JSON
    if not skeleton_json.exists():
        raise FileNotFoundError(f"Missing skeleton JSON: {skeleton_json}")

    metadata = load_skeleton_metadata()
    provider = AriaDigitalTwinSkeletonProvider(str(skeleton_json))

    samples: list[SkeletonSample] = []
    for frame_index, timestamp_ns in enumerate(timestamps_ns):
        skeleton_with_dt = provider.get_skeleton_by_timestamp_ns(int(timestamp_ns))
        dt_ns = int(skeleton_with_dt.dt_ns()) if skeleton_with_dt.is_valid() else None
        skeleton_valid = bool(skeleton_with_dt.is_valid())
        if max_dt_ns is not None and dt_ns is not None and abs(dt_ns) > max_dt_ns:
            skeleton_valid = False

        joints_flat: list[float | None]
        root_joint = (None, None, None)
        head_joint = (None, None, None)
        if skeleton_valid:
            frame = skeleton_with_dt.data()
            joints = [np.asarray(joint, dtype=np.float64) for joint in frame.joints]
            joints_flat = [
                _finite_or_none(float(value))
                for joint in joints
                for value in joint[:3]
            ]
            root_joint = _joint_xyz(joints, metadata.joint_labels, "Skeleton")
            head_joint = _joint_xyz(joints, metadata.joint_labels, "Head")
        else:
            joints_flat = [None] * (len(metadata.joint_labels) * 3)

        samples.append(
            SkeletonSample(
                query_timestamp_ns=int(timestamp_ns),
                frame_index=frame_index,
                skeleton_valid=skeleton_valid,
                skeleton_dt_ns=dt_ns,
                matched_skeleton_timestamp_ns=(
                    int(timestamp_ns) + dt_ns if dt_ns is not None else None
                ),
                root_joint_scene_x_m=root_joint[0],
                root_joint_scene_y_m=root_joint[1],
                root_joint_scene_z_m=root_joint[2],
                head_joint_scene_x_m=head_joint[0],
                head_joint_scene_y_m=head_joint[1],
                head_joint_scene_z_m=head_joint[2],
                joints_flat_scene_m=joints_flat,
            )
        )
    return samples, metadata


def write_skeleton_samples_csv(
    path: str | Path,
    samples: Sequence[SkeletonSample],
    metadata: SkeletonMetadata,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [sample.as_csv_row(metadata.joint_labels) for sample in samples]
    if not materialized:
        raise ValueError("No skeleton samples to write")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def summarize_skeleton_samples(
    samples: Sequence[SkeletonSample],
    metadata: SkeletonMetadata,
) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.skeleton_valid]
    dt_values = [
        abs(sample.skeleton_dt_ns)
        for sample in samples
        if sample.skeleton_dt_ns is not None
    ]
    return {
        "sample_count": len(samples),
        "valid_count": len(valid),
        "valid_ratio": len(valid) / len(samples) if samples else None,
        "joint_count": len(metadata.joint_labels),
        "marker_count": len(metadata.marker_labels),
        "joint_labels": metadata.joint_labels,
        "joint_connections": metadata.joint_connections,
        "marker_labels": metadata.marker_labels,
        "abs_dt_ns": describe_numbers(dt_values),
    }


def default_skeleton_samples_csv_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[2] / "outputs" / "reports"
    )
    return base_dir / f"{sequence_name}_skeleton_samples.csv"


def default_skeleton_summary_json_path(csv_path: str | Path) -> Path:
    output_path = Path(csv_path)
    stem = output_path.stem
    if stem.endswith("_skeleton_samples"):
        stem = stem[: -len("_skeleton_samples")] + "_skeleton_summary"
    return output_path.with_name(f"{stem}.json")


def safe_column_name(label: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
    return safe or "joint"


def _joint_xyz(
    joints: Sequence[np.ndarray],
    labels: Sequence[str],
    label: str,
) -> tuple[float | None, float | None, float | None]:
    try:
        index = labels.index(label)
    except ValueError:
        return (None, None, None)
    if index >= len(joints):
        return (None, None, None)
    return tuple(_finite_or_none(float(value)) for value in joints[index][:3])  # type: ignore[return-value]


def _finite_or_none(value: float) -> float | None:
    if np.isfinite(value):
        return float(value)
    return None


def describe_numbers(values: Sequence[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
