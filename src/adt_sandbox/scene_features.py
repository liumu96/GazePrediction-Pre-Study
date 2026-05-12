"""Scene-level object and skeleton asset utilities for ADT sequences.

This module intentionally starts from files on disk instead of the official ADT
provider. Some local ADT exports can fail provider initialization with newer
`projectaria_tools` metadata expectations, while the scene CSV/JSON files remain
usable for feature extraction.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Iterable, Sequence


SCENE_OBJECTS_CSV = "scene_objects.csv"
OBJECT_3D_BOX_CSV = "3d_bounding_box.csv"
OBJECT_2D_BOX_CSV = "2d_bounding_box.csv"
INSTANCES_JSON = "instances.json"
SKELETON_JSON = "Skeleton_T.json"
SKELETON_ASSOCIATION_JSON = "skeleton_aria_association.json"
METADATA_JSON = "metadata.json"


@dataclass(frozen=True)
class SceneObjectBoxRow:
    sequence_name: str
    object_uid: str
    timestamp_ns: int
    instance_name: str | None
    prototype_name: str | None
    category: str | None
    motion_type: str | None
    instance_type: str | None
    rigidity: str | None
    scene_t_x_m: float
    scene_t_y_m: float
    scene_t_z_m: float
    scene_q_w: float
    scene_q_x: float
    scene_q_y: float
    scene_q_z: float
    bbox_local_xmin_m: float
    bbox_local_xmax_m: float
    bbox_local_ymin_m: float
    bbox_local_ymax_m: float
    bbox_local_zmin_m: float
    bbox_local_zmax_m: float
    bbox_size_x_m: float
    bbox_size_y_m: float
    bbox_size_z_m: float
    scene_corner_0_x_m: float
    scene_corner_0_y_m: float
    scene_corner_0_z_m: float
    scene_corner_1_x_m: float
    scene_corner_1_y_m: float
    scene_corner_1_z_m: float
    scene_corner_2_x_m: float
    scene_corner_2_y_m: float
    scene_corner_2_z_m: float
    scene_corner_3_x_m: float
    scene_corner_3_y_m: float
    scene_corner_3_z_m: float
    scene_corner_4_x_m: float
    scene_corner_4_y_m: float
    scene_corner_4_z_m: float
    scene_corner_5_x_m: float
    scene_corner_5_y_m: float
    scene_corner_5_z_m: float
    scene_corner_6_x_m: float
    scene_corner_6_y_m: float
    scene_corner_6_z_m: float
    scene_corner_7_x_m: float
    scene_corner_7_y_m: float
    scene_corner_7_z_m: float

    def as_csv_row(self) -> dict[str, Any]:
        return asdict(self)


def inspect_scene_assets(
    sequence_dir: str | Path,
    include_skeleton_json: bool = False,
) -> dict[str, Any]:
    """Inspect scene/object/skeleton assets in one ADT sequence directory."""

    root = _sequence_root(sequence_dir)
    metadata = _read_json_if_exists(root / METADATA_JSON)
    instances = _read_json_if_exists(root / INSTANCES_JSON)
    skeleton_association = _read_json_if_exists(root / SKELETON_ASSOCIATION_JSON)

    scene_objects = _inspect_scene_objects_csv(root / SCENE_OBJECTS_CSV)
    object_boxes_3d = _inspect_3d_boxes_csv(root / OBJECT_3D_BOX_CSV)
    object_boxes_2d = _inspect_2d_boxes_csv(root / OBJECT_2D_BOX_CSV)
    skeleton = _inspect_skeleton_json(root / SKELETON_JSON, include_skeleton_json)

    return {
        "sequence_name": root.name,
        "sequence_dir": str(root),
        "metadata": _metadata_summary(metadata),
        "instances": _instances_summary(instances),
        "scene_objects": scene_objects,
        "object_boxes_3d": object_boxes_3d,
        "object_boxes_2d": object_boxes_2d,
        "skeleton_association": _skeleton_association_summary(skeleton_association),
        "skeleton": skeleton,
        "recommended_priority": [
            "extract_scene_object_boxes",
            "compute_gaze_object_hits",
            "build_scene_object_gaze_viewer",
            "extract_skeleton_trajectory",
        ],
    }


def format_scene_asset_report(report: dict[str, Any]) -> str:
    """Return a readable text report for `inspect_scene_assets`."""

    lines = [
        f"ADT scene assets: {report['sequence_name']}",
        f"root: {report['sequence_dir']}",
        "",
        "Metadata:",
        f"  scene={report['metadata'].get('scene')} dataset={report['metadata'].get('dataset_name')} "
        f"version={report['metadata'].get('dataset_version')} skeletons={report['metadata'].get('num_skeletons')}",
        "",
        "Instances:",
        f"  total={report['instances']['instance_count']} objects={report['instances']['object_count']}",
        f"  categories={_format_top_counts(report['instances']['category_counts'])}",
        f"  motion_types={_format_top_counts(report['instances']['motion_type_counts'])}",
        "",
        "Object poses:",
        f"  rows={report['scene_objects']['row_count']} unique_objects={report['scene_objects']['unique_object_count']} "
        f"timestamps={report['scene_objects']['unique_timestamp_count']} static_rows={report['scene_objects']['static_row_count']}",
        "",
        "3D object boxes:",
        f"  rows={report['object_boxes_3d']['row_count']} unique_objects={report['object_boxes_3d']['unique_object_count']} "
        f"invalid_size_rows={report['object_boxes_3d']['invalid_size_row_count']}",
        "",
        "2D object boxes:",
        f"  rows={report['object_boxes_2d']['row_count']} streams={report['object_boxes_2d']['stream_ids']} "
        f"visible_rows={report['object_boxes_2d']['visible_row_count']}",
        "",
        "Skeleton:",
        f"  file_exists={report['skeleton']['file_exists']} size={report['skeleton']['size_bytes']} "
        f"frames={report['skeleton'].get('frame_count')} joints={report['skeleton'].get('joint_count')} "
        f"markers={report['skeleton'].get('marker_count')}",
        f"  association={report['skeleton_association'].get('skeletons')}",
        "",
        "Recommended order:",
    ]
    lines.extend(f"  {index}. {step}" for index, step in enumerate(report["recommended_priority"], start=1))
    return "\n".join(lines)


def extract_scene_object_box_rows(sequence_dir: str | Path) -> list[SceneObjectBoxRow]:
    """Join object pose, 3D local AABB, and instance metadata into scene boxes."""

    root = _sequence_root(sequence_dir)
    instances = _read_json_if_exists(root / INSTANCES_JSON) or {}
    box_by_object = _read_3d_box_by_object(root / OBJECT_3D_BOX_CSV)

    rows: list[SceneObjectBoxRow] = []
    scene_objects_path = root / SCENE_OBJECTS_CSV
    if not scene_objects_path.exists():
        raise FileNotFoundError(f"Missing scene objects CSV: {scene_objects_path}")

    with scene_objects_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            object_uid = raw["object_uid"]
            box = box_by_object.get(object_uid)
            if box is None:
                continue
            translation = (
                _float(raw["t_wo_x[m]"]),
                _float(raw["t_wo_y[m]"]),
                _float(raw["t_wo_z[m]"]),
            )
            quaternion = (
                _float(raw["q_wo_w"]),
                _float(raw["q_wo_x"]),
                _float(raw["q_wo_y"]),
                _float(raw["q_wo_z"]),
            )
            if not all(_is_finite(value) for value in (*translation, *quaternion)):
                continue

            corners = _transform_aabb_corners(
                translation=translation,
                quaternion_wxyz=quaternion,
                aabb=box,
            )
            if corners is None:
                continue

            instance = instances.get(object_uid, {})
            rows.append(
                _object_box_row(
                    sequence_name=root.name,
                    object_uid=object_uid,
                    timestamp_ns=_int(raw["timestamp[ns]"]),
                    instance=instance,
                    translation=translation,
                    quaternion=quaternion,
                    aabb=box,
                    corners=corners,
                )
            )
    return rows


def write_scene_object_boxes_csv(
    path: str | Path,
    rows: Sequence[SceneObjectBoxRow],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [row.as_csv_row() for row in rows]
    if not materialized:
        raise ValueError("No scene object box rows to write")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def summarize_scene_object_box_rows(rows: Sequence[SceneObjectBoxRow]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "unique_object_count": 0,
            "unique_timestamp_count": 0,
            "static_row_count": 0,
            "row_category_counts": {},
            "row_motion_type_counts": {},
            "object_category_counts": {},
            "object_motion_type_counts": {},
        }
    row_category_counts: dict[str, int] = {}
    row_motion_type_counts: dict[str, int] = {}
    object_category_by_uid: dict[str, str] = {}
    object_motion_type_by_uid: dict[str, str] = {}
    for row in rows:
        category = row.category or "unknown"
        motion_type = row.motion_type or "unknown"
        _increment(row_category_counts, category)
        _increment(row_motion_type_counts, motion_type)
        object_category_by_uid.setdefault(row.object_uid, category)
        object_motion_type_by_uid.setdefault(row.object_uid, motion_type)
    return {
        "row_count": len(rows),
        "unique_object_count": len({row.object_uid for row in rows}),
        "unique_timestamp_count": len({row.timestamp_ns for row in rows}),
        "static_row_count": sum(1 for row in rows if row.timestamp_ns == -1),
        "row_category_counts": dict(sorted(row_category_counts.items())),
        "row_motion_type_counts": dict(sorted(row_motion_type_counts.items())),
        "object_category_counts": _counts_from_values(object_category_by_uid.values()),
        "object_motion_type_counts": _counts_from_values(
            object_motion_type_by_uid.values()
        ),
    }


def default_scene_object_boxes_csv_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[2] / "outputs" / "reports"
    )
    return base_dir / f"{sequence_name}_scene_object_boxes.csv"


def default_scene_object_boxes_summary_json_path(csv_path: str | Path) -> Path:
    output_path = Path(csv_path)
    stem = output_path.stem
    if stem.endswith("_scene_object_boxes"):
        stem = stem[: -len("_scene_object_boxes")] + "_scene_object_boxes_summary"
    return output_path.with_name(f"{stem}.json")


def _sequence_root(sequence_dir: str | Path) -> Path:
    root = Path(sequence_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Sequence directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected sequence directory: {root}")
    return root


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metadata_summary(metadata: Any | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        "scene": metadata.get("scene"),
        "dataset_name": metadata.get("dataset_name"),
        "dataset_version": metadata.get("dataset_version"),
        "num_skeletons": metadata.get("num_skeletons"),
        "is_multi_person": metadata.get("is_multi_person"),
        "gt_time_domain": metadata.get("gt_time_domain"),
    }


def _instances_summary(instances: Any | None) -> dict[str, Any]:
    if not isinstance(instances, dict):
        return {
            "instance_count": 0,
            "object_count": 0,
            "category_counts": {},
            "motion_type_counts": {},
        }

    category_counts: dict[str, int] = {}
    motion_type_counts: dict[str, int] = {}
    object_count = 0
    for instance in instances.values():
        if not isinstance(instance, dict):
            continue
        if instance.get("instance_type") == "object":
            object_count += 1
        _increment(category_counts, str(instance.get("category") or "unknown"))
        _increment(motion_type_counts, str(instance.get("motion_type") or "unknown"))
    return {
        "instance_count": len(instances),
        "object_count": object_count,
        "category_counts": dict(sorted(category_counts.items())),
        "motion_type_counts": dict(sorted(motion_type_counts.items())),
    }


def _skeleton_association_summary(value: Any | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"skeletons": []}
    skeletons = []
    for item in value.get("SkeletonMetadata", []):
        if isinstance(item, dict):
            skeletons.append(
                {
                    "skeleton_id": item.get("SkeletonId"),
                    "skeleton_name": item.get("SkeletonName"),
                    "device_serial": item.get("AssociatedDeviceSerial"),
                }
            )
    return {"skeletons": skeletons}


def _inspect_scene_objects_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _missing_csv_summary()
    object_ids: set[str] = set()
    timestamps: set[int] = set()
    static_rows = 0
    rows = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            object_ids.add(row["object_uid"])
            timestamp = _int(row["timestamp[ns]"])
            timestamps.add(timestamp)
            if timestamp == -1:
                static_rows += 1
    return {
        "file_exists": True,
        "row_count": rows,
        "unique_object_count": len(object_ids),
        "unique_timestamp_count": len(timestamps),
        "static_row_count": static_rows,
    }


def _inspect_3d_boxes_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _missing_csv_summary() | {"invalid_size_row_count": 0}
    object_ids: set[str] = set()
    rows = 0
    invalid = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            object_ids.add(row["object_uid"])
            xmin = _float(row["p_local_obj_xmin[m]"])
            xmax = _float(row["p_local_obj_xmax[m]"])
            ymin = _float(row["p_local_obj_ymin[m]"])
            ymax = _float(row["p_local_obj_ymax[m]"])
            zmin = _float(row["p_local_obj_zmin[m]"])
            zmax = _float(row["p_local_obj_zmax[m]"])
            if xmax <= xmin or ymax <= ymin or zmax <= zmin:
                invalid += 1
    return {
        "file_exists": True,
        "row_count": rows,
        "unique_object_count": len(object_ids),
        "invalid_size_row_count": invalid,
    }


def _inspect_2d_boxes_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _missing_csv_summary() | {"stream_ids": [], "visible_row_count": 0}
    rows = 0
    visible = 0
    stream_ids: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            stream_ids.add(row["stream_id"])
            if _float(row["visibility_ratio[%]"]) > 0:
                visible += 1
    return {
        "file_exists": True,
        "row_count": rows,
        "stream_ids": sorted(stream_ids),
        "visible_row_count": visible,
    }


def _inspect_skeleton_json(path: Path, include_skeleton_json: bool) -> dict[str, Any]:
    if not path.exists():
        return {
            "file_exists": False,
            "size_bytes": None,
            "frame_count": None,
            "joint_count": None,
            "marker_count": None,
        }
    summary = {
        "file_exists": True,
        "size_bytes": path.stat().st_size,
        "frame_count": None,
        "joint_count": None,
        "marker_count": None,
    }
    if not include_skeleton_json:
        return summary
    data = _read_json_if_exists(path)
    frames = data.get("frames", []) if isinstance(data, dict) else []
    first = frames[0] if frames else {}
    summary.update(
        {
            "frame_count": len(frames),
            "joint_count": len(first.get("joints", [])) if isinstance(first, dict) else None,
            "marker_count": len(first.get("markers", [])) if isinstance(first, dict) else None,
            "dt_optitrack_minus_device_ns": data.get("dt_optitrack_minus_device_ns")
            if isinstance(data, dict)
            else None,
        }
    )
    return summary


def _read_3d_box_by_object(path: Path) -> dict[str, tuple[float, float, float, float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing 3D bounding box CSV: {path}")
    boxes: dict[str, tuple[float, float, float, float, float, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            aabb = (
                _float(row["p_local_obj_xmin[m]"]),
                _float(row["p_local_obj_xmax[m]"]),
                _float(row["p_local_obj_ymin[m]"]),
                _float(row["p_local_obj_ymax[m]"]),
                _float(row["p_local_obj_zmin[m]"]),
                _float(row["p_local_obj_zmax[m]"]),
            )
            if (
                aabb[1] > aabb[0]
                and aabb[3] > aabb[2]
                and aabb[5] > aabb[4]
                and all(_is_finite(value) for value in aabb)
            ):
                boxes[row["object_uid"]] = aabb
    return boxes


def _transform_aabb_corners(
    translation: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
    aabb: tuple[float, float, float, float, float, float],
) -> list[tuple[float, float, float]] | None:
    rotation = _rotation_matrix_from_quaternion_wxyz(quaternion_wxyz)
    if rotation is None:
        return None
    xmin, xmax, ymin, ymax, zmin, zmax = aabb
    corners_local = [
        (x, y, z)
        for x, y, z in product((xmin, xmax), (ymin, ymax), (zmin, zmax))
    ]
    corners_scene = []
    for x, y, z in corners_local:
        corners_scene.append(
            (
                translation[0] + rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z,
                translation[1] + rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z,
                translation[2] + rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z,
            )
        )
    return corners_scene


def _rotation_matrix_from_quaternion_wxyz(
    quaternion: tuple[float, float, float, float],
) -> list[list[float]] | None:
    w, x, y, z = quaternion
    norm = sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12 or not _is_finite(norm):
        return None
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _object_box_row(
    sequence_name: str,
    object_uid: str,
    timestamp_ns: int,
    instance: dict[str, Any],
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    aabb: tuple[float, float, float, float, float, float],
    corners: Sequence[tuple[float, float, float]],
) -> SceneObjectBoxRow:
    flattened_corners = [coordinate for corner in corners for coordinate in corner]
    return SceneObjectBoxRow(
        sequence_name=sequence_name,
        object_uid=object_uid,
        timestamp_ns=timestamp_ns,
        instance_name=instance.get("instance_name"),
        prototype_name=instance.get("prototype_name"),
        category=instance.get("category"),
        motion_type=instance.get("motion_type"),
        instance_type=instance.get("instance_type"),
        rigidity=instance.get("rigidity"),
        scene_t_x_m=translation[0],
        scene_t_y_m=translation[1],
        scene_t_z_m=translation[2],
        scene_q_w=quaternion[0],
        scene_q_x=quaternion[1],
        scene_q_y=quaternion[2],
        scene_q_z=quaternion[3],
        bbox_local_xmin_m=aabb[0],
        bbox_local_xmax_m=aabb[1],
        bbox_local_ymin_m=aabb[2],
        bbox_local_ymax_m=aabb[3],
        bbox_local_zmin_m=aabb[4],
        bbox_local_zmax_m=aabb[5],
        bbox_size_x_m=aabb[1] - aabb[0],
        bbox_size_y_m=aabb[3] - aabb[2],
        bbox_size_z_m=aabb[5] - aabb[4],
        scene_corner_0_x_m=flattened_corners[0],
        scene_corner_0_y_m=flattened_corners[1],
        scene_corner_0_z_m=flattened_corners[2],
        scene_corner_1_x_m=flattened_corners[3],
        scene_corner_1_y_m=flattened_corners[4],
        scene_corner_1_z_m=flattened_corners[5],
        scene_corner_2_x_m=flattened_corners[6],
        scene_corner_2_y_m=flattened_corners[7],
        scene_corner_2_z_m=flattened_corners[8],
        scene_corner_3_x_m=flattened_corners[9],
        scene_corner_3_y_m=flattened_corners[10],
        scene_corner_3_z_m=flattened_corners[11],
        scene_corner_4_x_m=flattened_corners[12],
        scene_corner_4_y_m=flattened_corners[13],
        scene_corner_4_z_m=flattened_corners[14],
        scene_corner_5_x_m=flattened_corners[15],
        scene_corner_5_y_m=flattened_corners[16],
        scene_corner_5_z_m=flattened_corners[17],
        scene_corner_6_x_m=flattened_corners[18],
        scene_corner_6_y_m=flattened_corners[19],
        scene_corner_6_z_m=flattened_corners[20],
        scene_corner_7_x_m=flattened_corners[21],
        scene_corner_7_y_m=flattened_corners[22],
        scene_corner_7_z_m=flattened_corners[23],
    )


def _missing_csv_summary() -> dict[str, Any]:
    return {
        "file_exists": False,
        "row_count": 0,
        "unique_object_count": 0,
        "unique_timestamp_count": 0,
        "static_row_count": 0,
    }


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _counts_from_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        _increment(counts, value)
    return dict(sorted(counts.items()))


def _format_top_counts(counts: dict[str, int], limit: int = 8) -> str:
    if not counts:
        return "none"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in ordered)


def _float(value: str) -> float:
    return float(value)


def _int(value: str) -> int:
    return int(float(value))


def _is_finite(value: float) -> bool:
    return isfinite(float(value))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
