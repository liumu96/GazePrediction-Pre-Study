"""Ray-box gaze/object hit tests in ADT Scene frame."""

from __future__ import annotations

import csv
import json
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .gaze import GazeSample
from .results import sequence_file_path


@dataclass(frozen=True)
class GazeObjectHitRow:
    sequence_name: str
    sample_index: int
    query_timestamp_ns: int
    gaze_valid: bool
    ray_valid: bool
    candidate_box_count: int
    dynamic_object_timestamp_ns: int | None
    dynamic_object_dt_ns: int | None
    object_hit: bool
    hit_rank: int | None
    hit_object_uid: str | None
    hit_instance_name: str | None
    hit_category: str | None
    hit_motion_type: str | None
    hit_object_timestamp_ns: int | None
    hit_object_dt_ns: int | None
    hit_distance_m: float | None
    hit_x_m: float | None
    hit_y_m: float | None
    hit_z_m: float | None
    gaze_point_available: bool
    gaze_point_x_m: float | None
    gaze_point_y_m: float | None
    gaze_point_z_m: float | None
    gaze_point_inside_any_box: bool
    gaze_point_box_count: int
    gaze_point_object_uid: str | None
    gaze_point_category: str | None
    gaze_point_motion_type: str | None
    gaze_point_inside_hit_box: bool | None
    gaze_point_to_hit_distance_m: float | None

    def as_csv_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ObjectBox:
    object_uid: str
    timestamp_ns: int
    instance_name: str | None
    category: str | None
    motion_type: str | None
    translation: np.ndarray
    rotation: np.ndarray
    aabb: np.ndarray

    @property
    def center_scene(self) -> np.ndarray:
        return self.translation

    def ray_intersection_distance(
        self,
        origin_scene: np.ndarray,
        direction_scene: np.ndarray,
        *,
        min_distance_m: float,
        max_distance_m: float | None,
    ) -> float | None:
        origin_local = self.rotation.T @ (origin_scene - self.translation)
        direction_local = self.rotation.T @ direction_scene
        distance = _ray_aabb_intersection_distance(
            origin_local,
            direction_local,
            self.aabb,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
        )
        return distance

    def contains_point(self, point_scene: np.ndarray, *, epsilon_m: float = 1e-6) -> bool:
        point_local = self.rotation.T @ (point_scene - self.translation)
        xmin, xmax, ymin, ymax, zmin, zmax = self.aabb
        return bool(
            xmin - epsilon_m <= point_local[0] <= xmax + epsilon_m
            and ymin - epsilon_m <= point_local[1] <= ymax + epsilon_m
            and zmin - epsilon_m <= point_local[2] <= zmax + epsilon_m
        )


def compute_gaze_object_hit_rows(
    sequence_name: str,
    gaze_samples: Sequence[GazeSample],
    object_boxes_csv: str | Path,
    *,
    max_dynamic_dt_ns: int = 20_000_000,
    min_hit_distance_m: float = 0.05,
    max_hit_distance_m: float | None = 20.0,
    exclude_categories: Sequence[str] = (),
) -> list[GazeObjectHitRow]:
    """Compute first object-box ray hit for each gaze sample.

    Static boxes participate in every frame. Dynamic boxes are selected from the
    nearest object-pose timestamp if it is within `max_dynamic_dt_ns`.
    """

    static_boxes, dynamic_by_timestamp = read_object_boxes_csv(
        object_boxes_csv,
        exclude_categories=exclude_categories,
    )
    dynamic_timestamps = sorted(dynamic_by_timestamp)
    rows: list[GazeObjectHitRow] = []

    for sample_index, sample in enumerate(gaze_samples):
        nearest_ts, nearest_dt = _nearest_timestamp(
            dynamic_timestamps,
            sample.query_timestamp_ns,
            max_abs_dt_ns=max_dynamic_dt_ns,
        )
        dynamic_boxes = dynamic_by_timestamp.get(nearest_ts, []) if nearest_ts is not None else []
        candidate_boxes = [*static_boxes, *dynamic_boxes]
        row = _compute_one_hit_row(
            sequence_name=sequence_name,
            sample_index=sample_index,
            sample=sample,
            candidate_boxes=candidate_boxes,
            dynamic_object_timestamp_ns=nearest_ts,
            dynamic_object_dt_ns=nearest_dt,
            min_hit_distance_m=min_hit_distance_m,
            max_hit_distance_m=max_hit_distance_m,
        )
        rows.append(row)
    return rows


def read_object_boxes_csv(
    path: str | Path,
    *,
    exclude_categories: Sequence[str] = (),
) -> tuple[list[_ObjectBox], dict[int, list[_ObjectBox]]]:
    static_boxes: list[_ObjectBox] = []
    dynamic_by_timestamp: dict[int, list[_ObjectBox]] = {}
    excluded = {category.lower() for category in exclude_categories}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = (row.get("category") or "").lower()
            if category in excluded:
                continue
            box = _object_box_from_csv_row(row)
            if box is None:
                continue
            if box.timestamp_ns == -1:
                static_boxes.append(box)
            else:
                dynamic_by_timestamp.setdefault(box.timestamp_ns, []).append(box)
    return static_boxes, dynamic_by_timestamp


def write_gaze_object_hits_csv(
    path: str | Path,
    rows: Sequence[GazeObjectHitRow],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [row.as_csv_row() for row in rows]
    if not materialized:
        raise ValueError("No gaze-object hit rows to write")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def summarize_gaze_object_hits(rows: Sequence[GazeObjectHitRow]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "valid_ray_count": 0,
            "object_hit_count": 0,
            "object_hit_ratio": 0.0,
        }
    valid_ray_count = sum(row.ray_valid for row in rows)
    object_hit_rows = [row for row in rows if row.object_hit]
    point_available_rows = [row for row in rows if row.gaze_point_available]
    point_inside_rows = [row for row in rows if row.gaze_point_inside_any_box]
    point_hit_distances = [
        row.gaze_point_to_hit_distance_m
        for row in rows
        if row.gaze_point_to_hit_distance_m is not None
    ]
    return {
        "sample_count": len(rows),
        "valid_ray_count": valid_ray_count,
        "valid_ray_ratio": valid_ray_count / len(rows),
        "object_hit_count": len(object_hit_rows),
        "object_hit_ratio": len(object_hit_rows) / valid_ray_count if valid_ray_count else 0.0,
        "hit_category_counts": dict(sorted(Counter(row.hit_category or "unknown" for row in object_hit_rows).items())),
        "hit_motion_type_counts": dict(
            sorted(Counter(row.hit_motion_type or "unknown" for row in object_hit_rows).items())
        ),
        "gaze_point_available_count": len(point_available_rows),
        "gaze_point_available_ratio": len(point_available_rows) / len(rows),
        "gaze_point_inside_any_box_count": len(point_inside_rows),
        "gaze_point_inside_any_box_ratio": (
            len(point_inside_rows) / len(point_available_rows) if point_available_rows else 0.0
        ),
        "gaze_point_inside_hit_box_count": sum(row.gaze_point_inside_hit_box is True for row in rows),
        "ray_hit_and_gaze_point_available_count": len(point_hit_distances),
        "gaze_point_to_hit_distance_m": _describe_numbers(point_hit_distances),
        "mean_candidate_box_count": sum(row.candidate_box_count for row in rows) / len(rows),
    }


def default_gaze_object_hits_csv_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    return sequence_file_path(output_dir, sequence_name, "scene", "gaze_object_hits.csv")


def default_gaze_object_hits_summary_json_path(csv_path: str | Path) -> Path:
    output_path = Path(csv_path)
    stem = output_path.stem
    if stem == "gaze_object_hits":
        stem = "gaze_object_hits_summary"
    return output_path.with_name(f"{stem}.json")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _compute_one_hit_row(
    *,
    sequence_name: str,
    sample_index: int,
    sample: GazeSample,
    candidate_boxes: Sequence[_ObjectBox],
    dynamic_object_timestamp_ns: int | None,
    dynamic_object_dt_ns: int | None,
    min_hit_distance_m: float,
    max_hit_distance_m: float | None,
) -> GazeObjectHitRow:
    origin = _sample_origin_scene(sample)
    direction = _sample_direction_scene(sample)
    point = _sample_point_scene(sample)
    ray_valid = bool(sample.gaze_valid and origin is not None and direction is not None)

    hit_box: _ObjectBox | None = None
    hit_distance: float | None = None
    hit_point: np.ndarray | None = None
    if ray_valid:
        hit_box, hit_distance = _first_hit_box(
            candidate_boxes,
            origin,
            direction,
            min_hit_distance_m=min_hit_distance_m,
            max_hit_distance_m=max_hit_distance_m,
        )
        if hit_box is not None and hit_distance is not None:
            hit_point = origin + direction * hit_distance

    point_boxes: list[_ObjectBox] = []
    if point is not None:
        point_boxes = [box for box in candidate_boxes if box.contains_point(point)]
    point_box = _nearest_center_box(point, point_boxes) if point is not None else None
    point_inside_hit_box = None
    point_to_hit_distance = None
    if point is not None and hit_point is not None:
        point_to_hit_distance = float(np.linalg.norm(point - hit_point))
        point_inside_hit_box = bool(hit_box is not None and hit_box.contains_point(point))

    return GazeObjectHitRow(
        sequence_name=sequence_name,
        sample_index=sample_index,
        query_timestamp_ns=sample.query_timestamp_ns,
        gaze_valid=sample.gaze_valid,
        ray_valid=ray_valid,
        candidate_box_count=len(candidate_boxes),
        dynamic_object_timestamp_ns=dynamic_object_timestamp_ns,
        dynamic_object_dt_ns=dynamic_object_dt_ns,
        object_hit=hit_box is not None,
        hit_rank=1 if hit_box is not None else None,
        hit_object_uid=hit_box.object_uid if hit_box is not None else None,
        hit_instance_name=hit_box.instance_name if hit_box is not None else None,
        hit_category=hit_box.category if hit_box is not None else None,
        hit_motion_type=hit_box.motion_type if hit_box is not None else None,
        hit_object_timestamp_ns=hit_box.timestamp_ns if hit_box is not None else None,
        hit_object_dt_ns=(
            abs(sample.query_timestamp_ns - hit_box.timestamp_ns)
            if hit_box is not None and hit_box.timestamp_ns != -1
            else None
        ),
        hit_distance_m=hit_distance,
        hit_x_m=float(hit_point[0]) if hit_point is not None else None,
        hit_y_m=float(hit_point[1]) if hit_point is not None else None,
        hit_z_m=float(hit_point[2]) if hit_point is not None else None,
        gaze_point_available=point is not None,
        gaze_point_x_m=float(point[0]) if point is not None else None,
        gaze_point_y_m=float(point[1]) if point is not None else None,
        gaze_point_z_m=float(point[2]) if point is not None else None,
        gaze_point_inside_any_box=bool(point_boxes),
        gaze_point_box_count=len(point_boxes),
        gaze_point_object_uid=point_box.object_uid if point_box is not None else None,
        gaze_point_category=point_box.category if point_box is not None else None,
        gaze_point_motion_type=point_box.motion_type if point_box is not None else None,
        gaze_point_inside_hit_box=point_inside_hit_box,
        gaze_point_to_hit_distance_m=point_to_hit_distance,
    )


def _first_hit_box(
    boxes: Sequence[_ObjectBox],
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    min_hit_distance_m: float,
    max_hit_distance_m: float | None,
) -> tuple[_ObjectBox | None, float | None]:
    best_box = None
    best_distance = None
    for box in boxes:
        distance = box.ray_intersection_distance(
            origin,
            direction,
            min_distance_m=min_hit_distance_m,
            max_distance_m=max_hit_distance_m,
        )
        if distance is None:
            continue
        if best_distance is None or distance < best_distance:
            best_box = box
            best_distance = distance
    return best_box, best_distance


def _ray_aabb_intersection_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    aabb: np.ndarray,
    *,
    min_distance_m: float,
    max_distance_m: float | None,
) -> float | None:
    bounds = ((aabb[0], aabb[1]), (aabb[2], aabb[3]), (aabb[4], aabb[5]))
    t_min = -np.inf
    t_max = np.inf
    for axis, (lower, upper) in enumerate(bounds):
        d = direction[axis]
        o = origin[axis]
        if abs(d) < 1e-12:
            if o < lower or o > upper:
                return None
            continue
        t1 = (lower - o) / d
        t2 = (upper - o) / d
        near = min(t1, t2)
        far = max(t1, t2)
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return None
    if t_max < min_distance_m:
        return None
    # If the ray origin is already inside a large cuboid, the first physical
    # surface on the forward ray is the exit face, not `min_distance_m`.
    distance = t_max if t_min < min_distance_m else t_min
    if max_distance_m is not None and distance > max_distance_m:
        return None
    return float(distance)


def _object_box_from_csv_row(row: dict[str, str]) -> _ObjectBox | None:
    translation = np.asarray(
        [
            _optional_float(row["scene_t_x_m"]),
            _optional_float(row["scene_t_y_m"]),
            _optional_float(row["scene_t_z_m"]),
        ],
        dtype=np.float64,
    )
    quaternion = (
        _optional_float(row["scene_q_w"]),
        _optional_float(row["scene_q_x"]),
        _optional_float(row["scene_q_y"]),
        _optional_float(row["scene_q_z"]),
    )
    aabb = np.asarray(
        [
            _optional_float(row["bbox_local_xmin_m"]),
            _optional_float(row["bbox_local_xmax_m"]),
            _optional_float(row["bbox_local_ymin_m"]),
            _optional_float(row["bbox_local_ymax_m"]),
            _optional_float(row["bbox_local_zmin_m"]),
            _optional_float(row["bbox_local_zmax_m"]),
        ],
        dtype=np.float64,
    )
    if not (_finite_vector(translation) and _finite_vector(aabb)):
        return None
    rotation = _rotation_matrix_from_quaternion_wxyz(quaternion)
    if rotation is None:
        return None
    return _ObjectBox(
        object_uid=row["object_uid"],
        timestamp_ns=int(row["timestamp_ns"]),
        instance_name=row.get("instance_name") or None,
        category=row.get("category") or None,
        motion_type=row.get("motion_type") or None,
        translation=translation,
        rotation=rotation,
        aabb=aabb,
    )


def _sample_origin_scene(sample: GazeSample) -> np.ndarray | None:
    return _finite_array(
        [
            sample.gaze_origin_scene_x_m,
            sample.gaze_origin_scene_y_m,
            sample.gaze_origin_scene_z_m,
        ]
    )


def _sample_direction_scene(sample: GazeSample) -> np.ndarray | None:
    direction = _finite_array(
        [
            sample.gaze_dir_scene_unit_x,
            sample.gaze_dir_scene_unit_y,
            sample.gaze_dir_scene_unit_z,
        ]
    )
    if direction is None:
        return None
    norm = np.linalg.norm(direction)
    if norm <= 1e-12 or not isfinite(float(norm)):
        return None
    return direction / norm


def _sample_point_scene(sample: GazeSample) -> np.ndarray | None:
    return _finite_array(
        [
            sample.gaze_point_scene_x_m,
            sample.gaze_point_scene_y_m,
            sample.gaze_point_scene_z_m,
        ]
    )


def _nearest_timestamp(
    timestamps: Sequence[int],
    query_timestamp_ns: int,
    *,
    max_abs_dt_ns: int,
) -> tuple[int | None, int | None]:
    if not timestamps:
        return None, None
    index = bisect_left(timestamps, query_timestamp_ns)
    candidates = []
    if index < len(timestamps):
        candidates.append(timestamps[index])
    if index > 0:
        candidates.append(timestamps[index - 1])
    nearest = min(candidates, key=lambda ts: abs(ts - query_timestamp_ns))
    dt = abs(nearest - query_timestamp_ns)
    if dt > max_abs_dt_ns:
        return None, None
    return nearest, dt


def _nearest_center_box(point: np.ndarray, boxes: Sequence[_ObjectBox]) -> _ObjectBox | None:
    if not boxes:
        return None
    return min(boxes, key=lambda box: float(np.linalg.norm(point - box.center_scene)))


def _finite_array(values: Sequence[float | None]) -> np.ndarray | None:
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    if not _finite_vector(array):
        return None
    return array


def _finite_vector(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all())


def _optional_float(value: str | None) -> float:
    return float(value) if value not in (None, "") else float("nan")


def _rotation_matrix_from_quaternion_wxyz(
    quaternion: tuple[float, float, float, float],
) -> np.ndarray | None:
    w, x, y, z = quaternion
    norm = sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12 or not isfinite(norm):
        return None
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _describe_numbers(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
