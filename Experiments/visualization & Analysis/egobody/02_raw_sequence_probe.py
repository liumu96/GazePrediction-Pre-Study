#!/usr/bin/env python
"""Probe and prepare EgoBody raw RGB/depth assets for one sequence.

This script is intentionally data-layout oriented. It answers whether a local
EgoBody root contains the raw streams needed for ADT-like visualization:

- egocentric RGB frames from HoloLens2 (`egocentric_color`)
- egocentric depth recordings (`egocentric_depth`)
- synchronized Kinect RGB-D (`kinect_color`, `kinect_depth`)
- scene mesh and calibration files
- egocentric gaze CSV (`egocentric_gaze`)

When `egocentric_color` and `egocentric_gaze` are both available, the script
also builds a first-person frame manifest and projects GT gaze points into the
PV image plane using the per-frame PV-to-world transform.

Example:

    python "Experiments/visualization & Analysis/egobody/02_raw_sequence_probe.py" \
      recording_20210907_S03_S04_01 \
      --raw-root /mnt/d/Pose2Gaze-EgoBody
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_RAW_ROOT = Path("/mnt/d/Pose2Gaze-EgoBody")
DEFAULT_CACHE_ROOT = Path("/mnt/d/sparsegaze")
DEFAULT_OUTPUT_ROOT = Path("outputs/figures/egobody_raw_probe")

GAZE_AVAILABLE_COL = 851
GAZE_ORIGIN_SLICE = slice(852, 856)
GAZE_DIRECTION_SLICE = slice(856, 860)
GAZE_DISTANCE_COL = 860


@dataclass
class PvRecord:
    timestamp: int
    frame_id: int | None
    fx: float
    fy: float
    pv_to_world: np.ndarray
    image_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", help="EgoBody recording name.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-gaze-dt", type=float, default=500_000.0, help="Max timestamp gap for PV-gaze matching.")
    parser.add_argument("--render-overlays", type=int, default=3, help="Number of first-person gaze overlay PNGs to render.")
    return parser.parse_args()


def is_real_file(path: Path) -> bool:
    return path.is_file() and not path.name.endswith(":Zone.Identifier")


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def frame_id_from_name(path: Path) -> int | None:
    match = re.search(r"frame[_-](\d+)", path.name)
    return int(match.group(1)) if match else None


def timestamp_from_name(path: Path) -> int | None:
    match = re.match(r"(\d+)_frame[_-]\d+", path.name)
    return int(match.group(1)) if match else None


def list_files(path: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not path.exists():
        return []
    out: list[Path] = []
    for pattern in patterns:
        out.extend(p for p in path.rglob(pattern) if is_real_file(p))
    return sorted(set(out))


def sequence_subdir(raw_root: Path, modality: str, sequence: str) -> Path:
    return raw_root / modality / sequence


def find_capture_dirs(raw_root: Path, modality: str, sequence: str) -> list[Path]:
    base = sequence_subdir(raw_root, modality, sequence)
    if not base.exists():
        return []
    return sorted(path for path in base.iterdir() if path.is_dir())


def find_gaze_csv(raw_root: Path, sequence: str) -> Path | None:
    base = sequence_subdir(raw_root, "egocentric_gaze", sequence)
    matches = list_files(base, ("*_head_hand_eye.csv",))
    return matches[0] if matches else None


def find_pv_txt(raw_root: Path, sequence: str) -> Path | None:
    base = sequence_subdir(raw_root, "egocentric_color", sequence)
    matches = list_files(base, ("*_pv.txt",))
    return matches[0] if matches else None


def find_pv_images(raw_root: Path, sequence: str) -> list[Path]:
    base = sequence_subdir(raw_root, "egocentric_color", sequence)
    return list_files(base, ("*.jpg", "*.jpeg", "*.png"))


def find_egocentric_depth_files(raw_root: Path, sequence: str) -> list[Path]:
    base = sequence_subdir(raw_root, "egocentric_depth", sequence)
    return list_files(base, ("*.png", "*.tiff", "*.tif", "*.pgm", "*.npy", "*.bin"))


def parse_pv_txt(path: Path) -> tuple[dict[str, float], list[PvRecord]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty PV txt: {path}")

    intrinsics_raw = ast.literal_eval(lines[0])
    if len(intrinsics_raw) != 4:
        raise ValueError(f"Expected four PV intrinsics in first line of {path}")
    cx, cy, width, height = [float(x) for x in intrinsics_raw]
    intrinsics = {"cx": cx, "cy": cy, "width": width, "height": height}

    records: list[PvRecord] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 19:
            continue
        timestamp = int(float(parts[0]))
        fx = float(parts[1])
        fy = float(parts[2])
        transform = np.asarray([float(x) for x in parts[3:19]], dtype=np.float64).reshape(4, 4)
        records.append(PvRecord(timestamp=timestamp, frame_id=None, fx=fx, fy=fy, pv_to_world=transform, image_path=None))
    return intrinsics, records


def attach_pv_images(records: list[PvRecord], image_paths: list[Path]) -> None:
    by_timestamp = {timestamp_from_name(path): path for path in image_paths if timestamp_from_name(path) is not None}
    by_frame_id = {frame_id_from_name(path): path for path in image_paths if frame_id_from_name(path) is not None}
    sorted_images = sorted(image_paths)
    for idx, record in enumerate(records):
        image_path = by_timestamp.get(record.timestamp)
        if image_path is None and idx < len(sorted_images):
            image_path = sorted_images[idx]
        record.image_path = image_path
        if image_path is not None:
            record.frame_id = frame_id_from_name(image_path)
        elif record.frame_id is None and idx in by_frame_id:
            record.frame_id = idx


def load_gaze_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] <= GAZE_DISTANCE_COL:
        raise ValueError(f"Unexpected gaze CSV column count {data.shape[1]} in {path}")
    return data


def gaze_point_from_row(row: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    available = bool(row[GAZE_AVAILABLE_COL] == 1)
    origin = np.asarray(row[GAZE_ORIGIN_SLICE], dtype=np.float64)
    direction = np.asarray(row[GAZE_DIRECTION_SLICE], dtype=np.float64)
    norm = np.linalg.norm(direction[:3])
    if norm > 1e-12:
        direction[:3] = direction[:3] / norm
    distance = float(row[GAZE_DISTANCE_COL])
    if not np.isfinite(distance) or distance <= 0:
        distance = 1.0
    point = origin + direction * distance
    return point[:3], origin[:3], direction[:3], distance, available


def nearest_indices(targets: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(candidates)
    sorted_candidates = candidates[order]
    pos = np.searchsorted(sorted_candidates, targets)
    pos0 = np.clip(pos - 1, 0, len(sorted_candidates) - 1)
    pos1 = np.clip(pos, 0, len(sorted_candidates) - 1)
    choose1 = np.abs(sorted_candidates[pos1] - targets) < np.abs(sorted_candidates[pos0] - targets)
    nearest_pos = np.where(choose1, pos1, pos0)
    indices = order[nearest_pos]
    deltas = candidates[indices] - targets
    return indices, deltas


def project_world_to_pv(point_world: np.ndarray, pv_to_world: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    point_h = np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
    world_to_pv = np.linalg.inv(pv_to_world)
    point_pv = world_to_pv @ point_h
    z = float(point_pv[2])
    if not np.isfinite(z) or abs(z) < 1e-12:
        return math.nan, math.nan, z
    u = float(fx * point_pv[0] / z + cx)
    v = float(fy * point_pv[1] / z + cy)
    return u, v, z


def build_egocentric_projection_manifest(
    *,
    raw_root: Path,
    sequence: str,
    output_dir: Path,
    max_gaze_dt: float,
) -> tuple[Path | None, list[dict[str, Any]]]:
    pv_txt = find_pv_txt(raw_root, sequence)
    gaze_csv = find_gaze_csv(raw_root, sequence)
    if pv_txt is None or gaze_csv is None:
        return None, []

    intrinsics, pv_records = parse_pv_txt(pv_txt)
    attach_pv_images(pv_records, find_pv_images(raw_root, sequence))
    gaze = load_gaze_csv(gaze_csv)
    gaze_timestamps = gaze[:, 0].astype(np.float64)
    pv_timestamps = np.asarray([record.timestamp for record in pv_records], dtype=np.float64)
    matched_indices, deltas = nearest_indices(pv_timestamps, gaze_timestamps)

    rows: list[dict[str, Any]] = []
    for record, gaze_idx, delta in zip(pv_records, matched_indices, deltas):
        gaze_row = gaze[int(gaze_idx)]
        point_world, origin_world, direction_world, distance, gaze_available = gaze_point_from_row(gaze_row)
        u, v, z = project_world_to_pv(
            point_world,
            record.pv_to_world,
            record.fx,
            record.fy,
            intrinsics["cx"],
            intrinsics["cy"],
        )
        in_image = (
            np.isfinite(u)
            and np.isfinite(v)
            and 0 <= u < intrinsics["width"]
            and 0 <= v < intrinsics["height"]
            and z > 0
            and abs(delta) <= max_gaze_dt
            and gaze_available
        )
        rows.append(
            {
                "frame_id": "" if record.frame_id is None else int(record.frame_id),
                "pv_timestamp": int(record.timestamp),
                "gaze_timestamp": int(gaze_timestamps[int(gaze_idx)]),
                "gaze_dt": float(delta),
                "image_path": "" if record.image_path is None else str(record.image_path),
                "fx": float(record.fx),
                "fy": float(record.fy),
                "cx": float(intrinsics["cx"]),
                "cy": float(intrinsics["cy"]),
                "width": int(intrinsics["width"]),
                "height": int(intrinsics["height"]),
                "gaze_available": bool(gaze_available),
                "gaze_distance_m": float(distance),
                "gaze_origin_world_x": float(origin_world[0]),
                "gaze_origin_world_y": float(origin_world[1]),
                "gaze_origin_world_z": float(origin_world[2]),
                "gaze_dir_world_x": float(direction_world[0]),
                "gaze_dir_world_y": float(direction_world[1]),
                "gaze_dir_world_z": float(direction_world[2]),
                "gaze_point_world_x": float(point_world[0]),
                "gaze_point_world_y": float(point_world[1]),
                "gaze_point_world_z": float(point_world[2]),
                "gaze_u_px": float(u),
                "gaze_v_px": float(v),
                "gaze_pv_z": float(z),
                "in_image": bool(in_image),
            }
        )

    output_path = output_dir / "egocentric_pv_gaze_projection.csv"
    write_csv(output_path, rows)
    return output_path, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_overlay_examples(output_dir: Path, rows: list[dict[str, Any]], count: int) -> list[str]:
    if count <= 0:
        return []
    rendered: list[str] = []
    valid_rows = [row for row in rows if row.get("in_image") and row.get("image_path")]
    for row in valid_rows[:count]:
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            continue
        image = plt.imread(image_path)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(image)
        ax.scatter([row["gaze_u_px"]], [row["gaze_v_px"]], marker="x", s=140, linewidths=3, color="yellow")
        ax.scatter([row["gaze_u_px"]], [row["gaze_v_px"]], marker="x", s=70, linewidths=1.5, color="black")
        ax.set_title(f"{image_path.name} | frame={row['frame_id']} | dt={row['gaze_dt']:.0f}")
        ax.axis("off")
        out = output_dir / f"egocentric_gaze_overlay_frame_{row['frame_id']}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        rendered.append(str(out))
    return rendered


def build_egocentric_depth_manifest(raw_root: Path, sequence: str, output_dir: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    depth_base = sequence_subdir(raw_root, "egocentric_depth", sequence)
    if not depth_base.exists():
        return None, []

    rows = []
    for path in find_egocentric_depth_files(raw_root, sequence):
        frame_id = frame_id_from_name(path)
        timestamp = timestamp_from_name(path)
        rows.append(
            {
                "relative_path": str(path.relative_to(depth_base)),
                "depth_path": str(path),
                "frame_id": "" if frame_id is None else int(frame_id),
                "timestamp": "" if timestamp is None else int(timestamp),
                "suffix": path.suffix.lower(),
            }
        )

    output_path = output_dir / "egocentric_depth_manifest.csv"
    write_csv(output_path, rows)
    return output_path, rows


def build_kinect_rgbd_manifest(raw_root: Path, sequence: str, output_dir: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    color_base = sequence_subdir(raw_root, "kinect_color", sequence)
    depth_base = sequence_subdir(raw_root, "kinect_depth", sequence)
    if not color_base.exists() and not depth_base.exists():
        return None, []

    def key_for(path: Path, base: Path) -> tuple[str, str, int | None]:
        stream_dir = str(path.parent.relative_to(base)) if path.parent != base else "."
        frame_id = frame_id_from_name(path)
        if frame_id is not None:
            return stream_dir, f"frame_{frame_id:06d}", frame_id
        timestamp = timestamp_from_name(path)
        if timestamp is not None:
            return stream_dir, f"timestamp_{timestamp}", None
        return stream_dir, path.stem, None

    color_files = list_files(color_base, ("*.jpg", "*.jpeg", "*.png"))
    depth_files = list_files(depth_base, ("*.png", "*.tiff", "*.tif", "*.npy"))
    color_by_key = {key_for(path, color_base)[:2]: (path, key_for(path, color_base)[2]) for path in color_files}
    depth_by_key = {key_for(path, depth_base)[:2]: (path, key_for(path, depth_base)[2]) for path in depth_files}

    rows: list[dict[str, Any]] = []
    for stream_dir, frame_key in sorted(set(color_by_key) | set(depth_by_key)):
        color_item = color_by_key.get((stream_dir, frame_key))
        depth_item = depth_by_key.get((stream_dir, frame_key))
        frame_id = color_item[1] if color_item is not None else depth_item[1] if depth_item is not None else None
        rows.append(
            {
                "stream_dir": stream_dir,
                "frame_key": frame_key,
                "frame_id": "" if frame_id is None else int(frame_id),
                "color_path": "" if color_item is None else str(color_item[0]),
                "depth_path": "" if depth_item is None else str(depth_item[0]),
                "has_color": color_item is not None,
                "has_depth": depth_item is not None,
            }
        )

    output_path = output_dir / "kinect_rgbd_manifest.csv"
    write_csv(output_path, rows)
    return output_path, rows


def count_gaze_rows(gaze_csv: Path | None) -> dict[str, Any]:
    if gaze_csv is None:
        return {"exists": False, "row_count": 0}
    data = load_gaze_csv(gaze_csv)
    available = data[:, GAZE_AVAILABLE_COL] == 1
    return {
        "exists": True,
        "path": str(gaze_csv),
        "row_count": int(len(data)),
        "column_count": int(data.shape[1]),
        "gaze_available_count": int(np.sum(available)),
        "timestamp_min": int(np.nanmin(data[:, 0])),
        "timestamp_max": int(np.nanmax(data[:, 0])),
    }


def summarize_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}

    arrays = []
    has_image_like = False
    has_depth_like = False
    data = np.load(path, allow_pickle=False)
    for key in data.files:
        arr = data[key]
        key_lower = key.lower()
        shape = tuple(int(x) for x in arr.shape)
        arrays.append({"key": key, "shape": shape, "dtype": str(arr.dtype)})
        has_image_like = has_image_like or key_lower in {"image", "images", "rgb", "color"} or "image" in key_lower
        has_depth_like = has_depth_like or key_lower in {"depth", "depths"} or "depth" in key_lower

    return {
        "exists": True,
        "path": str(path),
        "arrays": arrays,
        "has_image_like_array": bool(has_image_like),
        "has_depth_like_array": bool(has_depth_like),
    }


def inspect_sparsegaze_cache(cache_root: Path, sequence: str) -> dict[str, Any]:
    split_candidates = ("test", "train", "val")
    dataset_entries = []
    feature_entries = []
    for split in split_candidates:
        dataset_path = cache_root / "datasets" / "EgoBody" / split / f"{sequence}.npz"
        feature_path = cache_root / "feature_cache" / "egobody" / "sparsegaze" / split / f"{sequence}.npz"
        dataset_summary = summarize_npz(dataset_path)
        feature_summary = summarize_npz(feature_path)
        if dataset_summary["exists"]:
            dataset_summary["split"] = split
            dataset_entries.append(dataset_summary)
        if feature_summary["exists"]:
            feature_summary["split"] = split
            feature_entries.append(feature_summary)

    return {
        "cache_root": str(cache_root),
        "dataset_npz": dataset_entries,
        "feature_npz": feature_entries,
        "has_any_image_like_array": any(
            entry["has_image_like_array"] for entry in dataset_entries + feature_entries
        ),
        "has_any_depth_like_array": any(
            entry["has_depth_like_array"] for entry in dataset_entries + feature_entries
        ),
    }


def inventory(raw_root: Path, cache_root: Path, sequence: str) -> dict[str, Any]:
    calibration_dir = sequence_subdir(raw_root, "calibrations", sequence)
    calibration_files = list_files(calibration_dir, ("*.json",))
    pv_images = find_pv_images(raw_root, sequence)
    pv_txt = find_pv_txt(raw_root, sequence)
    depth_files = find_egocentric_depth_files(raw_root, sequence)
    gaze_csv = find_gaze_csv(raw_root, sequence)
    scene_mesh_files = list_files(raw_root / "scene_mesh", ("*.obj", "*.ply", "*.glb", "*.gltf", "*.stl"))

    report: dict[str, Any] = {
        "sequence": sequence,
        "raw_root": str(raw_root),
        "cache_root": str(cache_root),
        "modalities": {},
    }
    for modality in [
        "egocentric_color",
        "egocentric_depth",
        "egocentric_gaze",
        "kinect_color",
        "kinect_depth",
        "scene_mesh",
        "calibrations",
    ]:
        base = sequence_subdir(raw_root, modality, sequence) if modality != "scene_mesh" else raw_root / modality
        report["modalities"][modality] = {
            "path": str(base),
            "exists": base.exists(),
            "capture_dirs": [str(path) for path in find_capture_dirs(raw_root, modality, sequence)]
            if modality.startswith("egocentric")
            else [],
        }

    report["egocentric_color"] = {
        "pv_txt": "" if pv_txt is None else str(pv_txt),
        "pv_image_count": len(pv_images),
        "sample_images": [str(path) for path in pv_images[:5]],
    }
    report["egocentric_depth"] = {
        "file_count": len(depth_files),
        "sample_files": [str(path) for path in depth_files[:5]],
    }
    report["egocentric_gaze"] = count_gaze_rows(gaze_csv)
    report["scene_mesh"] = {
        "file_count": len(scene_mesh_files),
        "sample_files": [rel(path, raw_root) for path in scene_mesh_files[:10]],
    }
    report["calibrations"] = {
        "path": str(calibration_dir),
        "json_count": len(calibration_files),
        "files": [rel(path, raw_root) for path in calibration_files],
        "has_holo_to_kinect12": (calibration_dir / "cal_trans" / "holo_to_kinect12.json").exists()
        or (calibration_dir / "holo_to_kinect12.json").exists(),
    }
    report["sparsegaze_cache"] = inspect_sparsegaze_cache(cache_root, sequence)
    return report


def add_actionable_status(report: dict[str, Any], projection_rows: list[dict[str, Any]], kinect_rows: list[dict[str, Any]]) -> None:
    color_ok = report["egocentric_color"]["pv_image_count"] > 0 and bool(report["egocentric_color"]["pv_txt"])
    gaze_ok = bool(report["egocentric_gaze"].get("exists"))
    depth_ok = report["egocentric_depth"]["file_count"] > 0
    kinect_rgbd_ok = any(row["has_color"] and row["has_depth"] for row in kinect_rows)
    projection_ok = any(row.get("in_image") for row in projection_rows)
    cache_image_ok = bool(report["sparsegaze_cache"]["has_any_image_like_array"])
    cache_depth_ok = bool(report["sparsegaze_cache"]["has_any_depth_like_array"])

    missing = []
    if not color_ok:
        missing.append("egocentric_color/PV images and *_pv.txt")
    if not gaze_ok:
        missing.append("egocentric_gaze/*_head_hand_eye.csv")
    if not depth_ok:
        missing.append("egocentric_depth")
    if not kinect_rgbd_ok:
        missing.append("kinect_color + kinect_depth paired frames")

    report["adt_like_visualization_status"] = {
        "egocentric_image_overlay_possible": bool(color_ok and gaze_ok and projection_ok),
        "egocentric_depth_available": bool(depth_ok),
        "kinect_rgbd_available": bool(kinect_rgbd_ok),
        "sparsegaze_cache_has_image_like_array": cache_image_ok,
        "sparsegaze_cache_has_depth_like_array": cache_depth_ok,
        "missing_for_full_raw_visualization": missing,
        "note": (
            "ADT-like image/scene scanpath requires raw image/depth streams. "
            "The lightweight Pose2Gaze/SparseGaze caches are not enough by themselves."
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.sequence
    output_dir.mkdir(parents=True, exist_ok=True)

    report = inventory(args.raw_root, args.cache_root, args.sequence)
    projection_csv, projection_rows = build_egocentric_projection_manifest(
        raw_root=args.raw_root,
        sequence=args.sequence,
        output_dir=output_dir,
        max_gaze_dt=args.max_gaze_dt,
    )
    depth_csv, depth_rows = build_egocentric_depth_manifest(args.raw_root, args.sequence, output_dir)
    kinect_csv, kinect_rows = build_kinect_rgbd_manifest(args.raw_root, args.sequence, output_dir)
    overlays = render_overlay_examples(output_dir, projection_rows, args.render_overlays)

    report["outputs"] = {
        "egocentric_projection_csv": "" if projection_csv is None else str(projection_csv),
        "egocentric_projection_rows": len(projection_rows),
        "egocentric_projection_in_image_count": int(sum(bool(row.get("in_image")) for row in projection_rows)),
        "egocentric_depth_manifest_csv": "" if depth_csv is None else str(depth_csv),
        "egocentric_depth_rows": len(depth_rows),
        "kinect_rgbd_manifest_csv": "" if kinect_csv is None else str(kinect_csv),
        "kinect_rgbd_rows": len(kinect_rows),
        "overlay_examples": overlays,
    }
    add_actionable_status(report, projection_rows, kinect_rows)

    report_path = output_dir / "sequence_raw_availability.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = report["adt_like_visualization_status"]
    print(f"sequence: {args.sequence}")
    print(f"report: {report_path}")
    print(f"egocentric image overlay possible: {status['egocentric_image_overlay_possible']}")
    print(f"egocentric depth available: {status['egocentric_depth_available']}")
    print(f"kinect rgb-d available: {status['kinect_rgbd_available']}")
    print(f"sparsegaze cache has image-like array: {status['sparsegaze_cache_has_image_like_array']}")
    print(f"sparsegaze cache has depth-like array: {status['sparsegaze_cache_has_depth_like_array']}")
    if status["missing_for_full_raw_visualization"]:
        print("missing:")
        for item in status["missing_for_full_raw_visualization"]:
            print(f"  - {item}")
    if projection_csv is not None:
        print(f"egocentric projection csv: {projection_csv}")
    if kinect_csv is not None:
        print(f"kinect rgb-d manifest: {kinect_csv}")


if __name__ == "__main__":
    main()
