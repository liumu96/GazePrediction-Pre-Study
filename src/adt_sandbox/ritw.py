"""Reading in the Wild extraction helpers.

RITW is Project Aria/MPS data rather than ADT data.  This module keeps the
output schema close to the ADT sandbox gaze/head layers while reading directly
from MPS eye-gaze and SLAM trajectory files.
"""

from __future__ import annotations

import csv
import gzip
import json
from bisect import bisect_left
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import atan, tan
from pathlib import Path
from typing import Any

import numpy as np

from .gaze import (
    GazeSample,
    default_summary_json_path,
    gaze_direction_cpf_unit,
    summarize_gaze_samples,
    write_gaze_summary_json,
    write_samples_csv,
)
from .head import (
    HEAD_FIELD_COORDINATE_FRAMES,
    HEAD_FIELD_DEFINITIONS,
    HeadSample,
    add_temporal_head_context,
    default_head_summary_json_path,
    summarize_head_samples,
    write_head_samples_csv,
    write_head_summary_json,
)
from .results import sequence_file_path


DEFAULT_RITW_ROOT = Path("/mnt/d/Reading_In_The_Wild")
DEFAULT_SPARSEGAZE_ROOT = Path("/mnt/d/SparseGaze")
DEFAULT_RITW_REPORTS_ROOT = DEFAULT_SPARSEGAZE_ROOT / "RITW-structured"
DEFAULT_RITW_CACHE_ROOT = DEFAULT_SPARSEGAZE_ROOT / "feature_cache" / "ritw30" / "sparsegaze"
IPD_METERS = 0.063


@dataclass(frozen=True)
class RitwExtractionConfig:
    gaze_kind: str = "personalized"
    stride: int = 2
    start_index: int = 0
    end_index: int | None = None
    max_pose_dt_ms: float | None = 20.0
    write_feature_cache: bool = True
    cache_root: Path = DEFAULT_RITW_CACHE_ROOT

    @property
    def max_pose_dt_ns(self) -> int | None:
        if self.max_pose_dt_ms is None:
            return None
        return int(self.max_pose_dt_ms * 1e6)


@dataclass(frozen=True)
class RitwExtractionResult:
    sequence_name: str
    sequence_path: Path
    split: str
    output_gaze_csv: Path
    output_head_csv: Path
    output_mps_summary_json: Path
    gaze_summary_json: Path
    head_summary_json: Path
    cache_npz: Path | None
    cache_meta_json: Path | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class _PoseSeries:
    timestamps_ns: np.ndarray
    translations_world_device: np.ndarray
    rotations_world_device: np.ndarray
    quality_scores: np.ndarray


@dataclass(frozen=True)
class _NearestPose:
    timestamp_ns: int
    dt_ns: int
    translation_world_device: np.ndarray
    rotation_world_device: np.ndarray
    quality_score: float


def discover_ritw_recordings(root: str | Path = DEFAULT_RITW_ROOT) -> list[Path]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"RITW root does not exist: {root_path}")
    recordings = [
        path
        for path in sorted(root_path.iterdir())
        if path.is_dir()
        and path.name.startswith("recording_")
        and (path / "metadata.json").exists()
        and (path / "recording.vrs").exists()
        and (path / "mps" / "eye_gaze").is_dir()
        and (path / "mps" / "slam" / "closed_loop_trajectory.csv").exists()
    ]
    if not recordings:
        raise ValueError(f"No RITW recording directories found under: {root_path}")
    return recordings


def extract_ritw_recording(
    recording: str | Path,
    *,
    output_dir: str | Path = DEFAULT_RITW_REPORTS_ROOT,
    config: RitwExtractionConfig | None = None,
) -> RitwExtractionResult:
    cfg = config or RitwExtractionConfig()
    recording_path = Path(recording).expanduser().resolve()
    if not recording_path.is_dir():
        raise NotADirectoryError(f"Expected RITW recording directory: {recording_path}")

    metadata = _read_json(recording_path / "metadata.json")
    sequence_name = recording_path.name
    split = str(metadata.get("split") or "unknown").strip().lower() or "unknown"
    gaze_csv = _select_gaze_csv(recording_path, cfg.gaze_kind)
    pose_series = _read_closed_loop_trajectory(recording_path / "mps" / "slam" / "closed_loop_trajectory.csv")
    transform_device_cpf = _read_transform_device_cpf(recording_path / "recording.vrs")

    gaze_samples, head_samples = _extract_aligned_samples(
        gaze_csv=gaze_csv,
        pose_series=pose_series,
        transform_device_cpf=transform_device_cpf,
        config=cfg,
    )

    out_root = Path(output_dir).expanduser()
    gaze_out = sequence_file_path(out_root, sequence_name, "gaze", "gaze_samples.csv")
    head_out = sequence_file_path(out_root, sequence_name, "head", "head_samples.csv")
    mps_summary_json = sequence_file_path(out_root, sequence_name, "mps", "mps_summary.json")
    gaze_summary_json = default_summary_json_path(gaze_out)
    head_summary_json = default_head_summary_json_path(head_out)

    write_samples_csv(gaze_out, gaze_samples)
    write_head_samples_csv(head_out, head_samples)

    gaze_summary = summarize_gaze_samples(gaze_samples)
    head_summary = summarize_head_samples(head_samples)
    mps_summary = summarize_ritw_mps(recording_path)
    source_summary = {
        "sequence_name": sequence_name,
        "sequence_path": str(recording_path),
        "provider_mode": "ritw_mps_csv",
        "dataset": "ritw30",
        "split": split,
        "gaze_kind": _gaze_kind_from_path(gaze_csv),
        "source_gaze_csv": str(gaze_csv),
        "source_trajectory_csv": str(recording_path / "mps" / "slam" / "closed_loop_trajectory.csv"),
        "source_vrs": str(recording_path / "recording.vrs"),
        "source_semidense_points": str(recording_path / "mps" / "slam" / "semidense_points.csv.gz"),
        "source_semidense_observations": str(recording_path / "mps" / "slam" / "semidense_observations.csv.gz"),
        "object_annotations": None,
        "object_annotation_notes": (
            "RITW recording directory does not contain ADT-style semantic object boxes; "
            "ego-blur boxes are anonymization metadata, not scene objects."
        ),
        "mps_summary_json": str(mps_summary_json),
        "selection": {
            "stride": cfg.stride,
            "start_index": cfg.start_index,
            "end_index": cfg.end_index,
            "max_pose_dt_ms": cfg.max_pose_dt_ms,
        },
    }
    gaze_summary.update(
        source_summary
        | {
            "field_coordinate_frames": _ritw_gaze_coordinate_frames(),
            "field_definitions": {
                "yaw_rad": "Combined CPF gaze yaw derived from new-format left/right yaw.",
                "pitch_rad": "Common CPF gaze pitch from the RITW MPS eye-gaze CSV.",
                "depth_m": "Distance from CPF origin to the vergence/intersection point.",
                "gaze_point_scene_xyz": "RITW SLAM world-frame gaze point.",
                "gaze_dir_scene_unit_xyz": "RITW SLAM world-frame unit gaze direction.",
            },
        }
    )
    head_summary.update(
        source_summary
        | {
            "field_coordinate_frames": _ritw_head_coordinate_frames(),
            "field_definitions": HEAD_FIELD_DEFINITIONS,
        }
    )
    write_gaze_summary_json(gaze_summary_json, gaze_summary)
    write_head_summary_json(head_summary_json, head_summary)
    mps_summary_json.parent.mkdir(parents=True, exist_ok=True)
    mps_summary_json.write_text(json.dumps(mps_summary, indent=2), encoding="utf-8")

    cache_npz = None
    cache_meta_json = None
    if cfg.write_feature_cache:
        try:
            cache_npz, cache_meta_json = write_sparsegaze_cache(
                sequence_name=sequence_name,
                split=split,
                gaze_samples=gaze_samples,
                head_samples=head_samples,
                source_csv=gaze_out,
                cache_root=cfg.cache_root,
                metadata=source_summary,
            )
        except ValueError as exc:
            source_summary["feature_cache_error"] = str(exc)

    return RitwExtractionResult(
        sequence_name=sequence_name,
        sequence_path=recording_path,
        split=split,
        output_gaze_csv=gaze_out,
        output_head_csv=head_out,
        output_mps_summary_json=mps_summary_json,
        gaze_summary_json=gaze_summary_json,
        head_summary_json=head_summary_json,
        cache_npz=cache_npz,
        cache_meta_json=cache_meta_json,
        summary={
            "gaze": gaze_summary,
            "head": head_summary,
            "source": source_summary,
        },
    )


def write_sparsegaze_manifest(
    results: Sequence[RitwExtractionResult],
    *,
    cache_root: str | Path = DEFAULT_RITW_CACHE_ROOT,
    sparsegaze_root: str | Path = DEFAULT_SPARSEGAZE_ROOT,
) -> Path:
    cache_root_path = Path(cache_root).expanduser()
    repo_root = Path(sparsegaze_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.cache_npz is None:
            continue
        row = {
            "sequence_id": result.sequence_name,
            "split": result.split,
            "n_frames": _cache_frame_count(result.cache_npz),
            "source_csv": str(result.output_gaze_csv.resolve()),
            "cache_npz": str(result.cache_npz.resolve()),
        }
        if result.cache_meta_json is not None:
            row["cache_meta"] = str(result.cache_meta_json.resolve())
        rows.append(row)

    if not rows:
        raise ValueError("No feature-cache rows to write")

    payload = {
        "version": "ritw_dual_sparsegaze_feature_cache_v1",
        "format": "ritw_dual_sparsegaze_feature_cache_v1",
        "dataset": "ritw30",
        "output_dir": str(cache_root_path.resolve()),
        "split_mode": "metadata_split",
        "groups": [
            "gaze_dir_core",
            "head_dir_core",
            "head_rotation_rel",
            "head_translation_rel",
        ],
        "total_dim": 12,
        "num_sequences_written": len(rows),
        "num_records_loaded": len(results),
        "num_records_skipped": len(results) - len(rows),
        "head_rotation_source": "R_world_prev_cpf.T @ R_world_current_cpf",
        "head_translation_source": "world-frame CPF origin delta",
        "cpf_cache_keys": [
            "timestamps_ns",
            "gaze_world",
            "gaze_cpf",
            "head_rotmat_world",
            "head_rotation_rel",
        ],
        "generic_cache_keys": [
            "frame_timestamps_ns",
            "targets_world_gaze",
            "feature_groups__gaze_dir_core",
            "feature_groups__head_dir_core",
            "feature_groups__head_rotation_rel",
            "feature_groups__head_translation_rel",
        ],
        "rows": rows,
    }
    manifest_path = cache_root_path / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def write_sparsegaze_cache(
    *,
    sequence_name: str,
    split: str,
    gaze_samples: Sequence[GazeSample],
    head_samples: Sequence[HeadSample],
    source_csv: str | Path,
    cache_root: str | Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    timestamps: list[int] = []
    gaze_world: list[list[float]] = []
    head_dir_world: list[list[float]] = []
    head_rotvec_rel: list[list[float]] = []
    head_translation_rel: list[list[float]] = []
    gaze_cpf: list[list[float]] = []
    head_rotmat_world: list[np.ndarray] = []

    for gaze, head in zip(gaze_samples, head_samples):
        gaze_dir = _optional_vector(
            gaze.gaze_dir_scene_unit_x,
            gaze.gaze_dir_scene_unit_y,
            gaze.gaze_dir_scene_unit_z,
        )
        gaze_dir_cpf = _optional_vector(
            gaze.gaze_dir_cpf_unit_x,
            gaze.gaze_dir_cpf_unit_y,
            gaze.gaze_dir_cpf_unit_z,
        )
        head_dir = _optional_vector(
            head.head_forward_scene_unit_x,
            head.head_forward_scene_unit_y,
            head.head_forward_scene_unit_z,
        )
        head_rotmat = _head_world_rotation(head)
        if gaze_dir is None or gaze_dir_cpf is None or head_dir is None or head_rotmat is None:
            continue

        rotmat = _head_relative_rotation(head)
        rotvec = (
            np.zeros(3, dtype=np.float64)
            if rotmat is None
            else _rotation_matrix_to_rotvec(rotmat)
        )
        translation = _optional_vector(
            head.translation_scene_dx_m,
            head.translation_scene_dy_m,
            head.translation_scene_dz_m,
        )
        if translation is None:
            translation = np.zeros(3, dtype=np.float64)

        timestamps.append(int(gaze.query_timestamp_ns))
        gaze_world.append(gaze_dir.astype(np.float64).tolist())
        gaze_cpf.append(gaze_dir_cpf.astype(np.float64).tolist())
        head_dir_world.append(head_dir.astype(np.float64).tolist())
        head_rotmat_world.append(head_rotmat.astype(np.float64))
        head_rotvec_rel.append(rotvec.astype(np.float64).tolist())
        head_translation_rel.append(translation.astype(np.float64).tolist())

    if not timestamps:
        raise ValueError(f"No valid SparseGaze cache frames for {sequence_name}")

    out_dir = Path(cache_root).expanduser() / split
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{sequence_name}.npz"
    meta_path = out_dir / f"{sequence_name}.meta.json"
    np.savez_compressed(
        npz_path,
        # Keys consumed directly by sparsegaze_cpf_* loaders.
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        gaze_world=np.asarray(gaze_world, dtype=np.float32),
        gaze_cpf=np.asarray(gaze_cpf, dtype=np.float32),
        head_rotmat_world=np.asarray(head_rotmat_world, dtype=np.float32),
        head_rotation_rel=np.asarray(head_rotvec_rel, dtype=np.float32),
        # Keys consumed by the generic SparseGaze multi-branch loader.
        frame_timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        targets_world_gaze=np.asarray(gaze_world, dtype=np.float32),
        feature_groups__gaze_dir_core=np.asarray(gaze_world, dtype=np.float32),
        feature_groups__head_dir_core=np.asarray(head_dir_world, dtype=np.float32),
        feature_groups__head_rotation_rel=np.asarray(head_rotvec_rel, dtype=np.float32),
        feature_groups__head_translation_rel=np.asarray(head_translation_rel, dtype=np.float32),
    )
    meta = {
        "sequence_id": sequence_name,
        "split": split,
        "source_csv": str(source_csv),
        "n_frames": len(timestamps),
        "metadata": metadata,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path, meta_path


def summarize_ritw_mps(recording_path: str | Path) -> dict[str, Any]:
    """Summarize RITW-only MPS sidecars without duplicating large raw files."""
    rec = Path(recording_path).expanduser().resolve()
    mps = rec / "mps"
    files = {
        "eye_gaze_general": mps / "eye_gaze" / "general_eye_gaze.csv",
        "eye_gaze_personalized": mps / "eye_gaze" / "personalized_eye_gaze.csv",
        "hand_tracking": mps / "hand_tracking" / "hand_tracking_frames.jsonl",
        "closed_loop_trajectory": mps / "slam" / "closed_loop_trajectory.csv",
        "open_loop_trajectory": mps / "slam" / "open_loop_trajectory.csv",
        "online_calibration": mps / "slam" / "online_calibration.jsonl",
        "semidense_points": mps / "slam" / "semidense_points.csv.gz",
        "semidense_observations": mps / "slam" / "semidense_observations.csv.gz",
        "anonymization_detections": mps
        / "ego_blur_anonymization_bounding_box"
        / "Anonymization_Detections",
    }
    object_hits = _find_object_like_mps_files(mps)
    semantic_hits = [
        item
        for item in object_hits
        if "ego_blur_anonymization_bounding_box" not in item["path"]
        and "Anonymization_Detections" not in item["path"]
    ]
    return {
        "dataset": "ritw30",
        "sequence_name": rec.name,
        "sequence_path": str(rec),
        "mps_path": str(mps),
        "available_files": {name: _file_info(path, rec) for name, path in files.items()},
        "eye_gaze": {
            "general": _csv_header_sample(files["eye_gaze_general"], compressed=False),
            "personalized": _csv_header_sample(files["eye_gaze_personalized"], compressed=False),
        },
        "hand_tracking": _jsonl_summary(files["hand_tracking"]),
        "scene": {
            "closed_loop_trajectory": _csv_header_sample(files["closed_loop_trajectory"], compressed=False),
            "open_loop_trajectory": _csv_header_sample(files["open_loop_trajectory"], compressed=False),
            "online_calibration": _jsonl_summary(files["online_calibration"]),
            "semidense_points": _csv_header_sample(files["semidense_points"], compressed=True),
            "semidense_observations": _csv_header_sample(files["semidense_observations"], compressed=True),
            "notes": (
                "Scene geometry is MPS SLAM trajectory plus semidense point cloud/observations; "
                "there are no ADT-style semantic scene object boxes in this recording."
            ),
        },
        "objects": {
            "semantic_object_annotations_found": bool(semantic_hits),
            "semantic_object_like_files": semantic_hits,
            "anonymization_detections": _anonymization_summary(files["anonymization_detections"]),
            "notes": (
                "The only object-like file found is the ego-blur anonymization detection file. "
                "Its categories are privacy classes such as face, not scene object labels."
            ),
        },
    }


def _extract_aligned_samples(
    *,
    gaze_csv: Path,
    pose_series: _PoseSeries,
    transform_device_cpf: np.ndarray,
    config: RitwExtractionConfig,
) -> tuple[list[GazeSample], list[HeadSample]]:
    raw_rows = list(_iter_gaze_rows(gaze_csv))
    selected_rows = raw_rows[config.start_index : config.end_index : config.stride]
    if not selected_rows:
        raise ValueError("No RITW gaze rows selected; check start/end/stride")

    gaze_samples: list[GazeSample] = []
    absolute_head_samples: list[HeadSample] = []
    for row in selected_rows:
        gaze_sample, head_sample = _extract_one_sample(
            row=row,
            pose_series=pose_series,
            transform_device_cpf=transform_device_cpf,
            max_pose_dt_ns=config.max_pose_dt_ns,
        )
        gaze_samples.append(gaze_sample)
        absolute_head_samples.append(head_sample)
    return gaze_samples, add_temporal_head_context(absolute_head_samples)


def _extract_one_sample(
    *,
    row: dict[str, str],
    pose_series: _PoseSeries,
    transform_device_cpf: np.ndarray,
    max_pose_dt_ns: int | None,
) -> tuple[GazeSample, HeadSample]:
    timestamp_ns = int(row["tracking_timestamp_us"]) * 1000
    notes: list[str] = []
    left_yaw = _float_or_nan(row.get("left_yaw_rads_cpf"))
    right_yaw = _float_or_nan(row.get("right_yaw_rads_cpf"))
    pitch = _float_or_nan(row.get("pitch_rads_cpf"))
    csv_depth = _float_or_nan(row.get("depth_m"))
    depth, yaw = _compute_depth_and_combined_yaw(left_yaw, right_yaw, pitch)
    if np.isfinite(csv_depth) and csv_depth > 0:
        depth = float(csv_depth)
    if not np.isfinite([yaw, pitch]).all():
        notes.append("yaw_or_pitch_not_finite")
    if not np.isfinite(depth) or depth <= 0:
        notes.append("depth_not_available")

    pose = _nearest_pose(pose_series, timestamp_ns)
    pose_valid = pose is not None
    if pose is not None and max_pose_dt_ns is not None and abs(pose.dt_ns) > max_pose_dt_ns:
        pose_valid = False
        notes.append("pose_dt_exceeds_threshold")

    gaze_dir_cpf = _gaze_direction_from_yaw_pitch(yaw, pitch)
    yaw_width, pitch_width = _confidence_widths_from_new_gaze(row)
    if not np.isfinite([yaw_width, pitch_width]).all():
        notes.append("confidence_width_not_finite")

    origin_world = None
    point_world = None
    gaze_dir_world = None
    transform_world_cpf = None
    if pose_valid and pose is not None:
        transform_world_device = _make_transform(pose.rotation_world_device, pose.translation_world_device)
        transform_world_cpf = transform_world_device @ transform_device_cpf
        if gaze_dir_cpf is not None and np.isfinite(depth) and depth > 0:
            gaze_point_cpf = gaze_dir_cpf * float(depth)
            origin_world = _transform_point(transform_world_cpf, np.zeros(3, dtype=np.float64))
            point_world = _transform_point(transform_world_cpf, gaze_point_cpf)
            gaze_dir_world = _normalize(point_world - origin_world)
        else:
            notes.append("scene_ray_unavailable")
    else:
        notes.append("pose_unavailable")

    gaze_sample = GazeSample(
        query_timestamp_ns=timestamp_ns,
        gaze_valid=not any(note in notes for note in ("yaw_or_pitch_not_finite",)),
        gaze_dt_ns=0,
        yaw_rad=_finite_or_none(yaw),
        pitch_rad=_finite_or_none(pitch),
        depth_m=_finite_or_none(depth),
        gaze_dir_cpf_unit_x=_component_or_none(gaze_dir_cpf, 0),
        gaze_dir_cpf_unit_y=_component_or_none(gaze_dir_cpf, 1),
        gaze_dir_cpf_unit_z=_component_or_none(gaze_dir_cpf, 2),
        yaw_confidence_width_rad=_finite_or_none(yaw_width),
        pitch_confidence_width_rad=_finite_or_none(pitch_width),
        projection_valid=False,
        gaze_u_px=None,
        gaze_v_px=None,
        projection_in_image=False,
        image_width_px=1408,
        image_height_px=1408,
        pose_valid=pose_valid,
        pose_dt_ns=pose.dt_ns if pose is not None else None,
        pose_quality_score=pose.quality_score if pose is not None else None,
        gaze_origin_scene_x_m=_component_or_none(origin_world, 0),
        gaze_origin_scene_y_m=_component_or_none(origin_world, 1),
        gaze_origin_scene_z_m=_component_or_none(origin_world, 2),
        gaze_point_scene_x_m=_component_or_none(point_world, 0),
        gaze_point_scene_y_m=_component_or_none(point_world, 1),
        gaze_point_scene_z_m=_component_or_none(point_world, 2),
        gaze_dir_scene_unit_x=_component_or_none(gaze_dir_world, 0),
        gaze_dir_scene_unit_y=_component_or_none(gaze_dir_world, 1),
        gaze_dir_scene_unit_z=_component_or_none(gaze_dir_world, 2),
        validation_notes=";".join(notes) if notes else "ok",
    )
    head_sample = _make_head_sample(
        timestamp_ns=timestamp_ns,
        pose=pose if pose_valid else None,
        transform_world_cpf=transform_world_cpf,
        notes="ok" if pose_valid else "pose_unavailable",
    )
    return gaze_sample, head_sample


def _make_head_sample(
    *,
    timestamp_ns: int,
    pose: _NearestPose | None,
    transform_world_cpf: np.ndarray | None,
    notes: str,
) -> HeadSample:
    if pose is None or transform_world_cpf is None:
        return HeadSample(
            query_timestamp_ns=timestamp_ns,
            pose_valid=False,
            pose_dt_ns=None,
            pose_quality_score=None,
            head_origin_scene_x_m=None,
            head_origin_scene_y_m=None,
            head_origin_scene_z_m=None,
            head_right_scene_unit_x=None,
            head_right_scene_unit_y=None,
            head_right_scene_unit_z=None,
            head_up_scene_unit_x=None,
            head_up_scene_unit_y=None,
            head_up_scene_unit_z=None,
            head_forward_scene_unit_x=None,
            head_forward_scene_unit_y=None,
            head_forward_scene_unit_z=None,
            head_rot_scene_r00=None,
            head_rot_scene_r01=None,
            head_rot_scene_r02=None,
            head_rot_scene_r10=None,
            head_rot_scene_r11=None,
            head_rot_scene_r12=None,
            head_rot_scene_r20=None,
            head_rot_scene_r21=None,
            head_rot_scene_r22=None,
            dt_from_prev_s=None,
            translation_scene_dx_m=None,
            translation_scene_dy_m=None,
            translation_scene_dz_m=None,
            translation_prev_head_dx_m=None,
            translation_prev_head_dy_m=None,
            translation_prev_head_dz_m=None,
            origin_step_m=None,
            head_translation_speed_m_s=None,
            relative_rot_prev_to_cur_r00=None,
            relative_rot_prev_to_cur_r01=None,
            relative_rot_prev_to_cur_r02=None,
            relative_rot_prev_to_cur_r10=None,
            relative_rot_prev_to_cur_r11=None,
            relative_rot_prev_to_cur_r12=None,
            relative_rot_prev_to_cur_r20=None,
            relative_rot_prev_to_cur_r21=None,
            relative_rot_prev_to_cur_r22=None,
            head_forward_angle_step_deg=None,
            head_rotation_angle_step_deg=None,
            head_rotation_speed_deg_s=None,
            validation_notes=notes,
        )

    origin = transform_world_cpf[:3, 3]
    rotation = transform_world_cpf[:3, :3]
    right = rotation[:, 0]
    up = rotation[:, 1]
    forward = rotation[:, 2]
    return HeadSample(
        query_timestamp_ns=timestamp_ns,
        pose_valid=True,
        pose_dt_ns=pose.dt_ns,
        pose_quality_score=pose.quality_score,
        head_origin_scene_x_m=_component_or_none(origin, 0),
        head_origin_scene_y_m=_component_or_none(origin, 1),
        head_origin_scene_z_m=_component_or_none(origin, 2),
        head_right_scene_unit_x=_component_or_none(right, 0),
        head_right_scene_unit_y=_component_or_none(right, 1),
        head_right_scene_unit_z=_component_or_none(right, 2),
        head_up_scene_unit_x=_component_or_none(up, 0),
        head_up_scene_unit_y=_component_or_none(up, 1),
        head_up_scene_unit_z=_component_or_none(up, 2),
        head_forward_scene_unit_x=_component_or_none(forward, 0),
        head_forward_scene_unit_y=_component_or_none(forward, 1),
        head_forward_scene_unit_z=_component_or_none(forward, 2),
        head_rot_scene_r00=_component_or_none(rotation, (0, 0)),
        head_rot_scene_r01=_component_or_none(rotation, (0, 1)),
        head_rot_scene_r02=_component_or_none(rotation, (0, 2)),
        head_rot_scene_r10=_component_or_none(rotation, (1, 0)),
        head_rot_scene_r11=_component_or_none(rotation, (1, 1)),
        head_rot_scene_r12=_component_or_none(rotation, (1, 2)),
        head_rot_scene_r20=_component_or_none(rotation, (2, 0)),
        head_rot_scene_r21=_component_or_none(rotation, (2, 1)),
        head_rot_scene_r22=_component_or_none(rotation, (2, 2)),
        dt_from_prev_s=None,
        translation_scene_dx_m=None,
        translation_scene_dy_m=None,
        translation_scene_dz_m=None,
        translation_prev_head_dx_m=None,
        translation_prev_head_dy_m=None,
        translation_prev_head_dz_m=None,
        origin_step_m=None,
        head_translation_speed_m_s=None,
        relative_rot_prev_to_cur_r00=None,
        relative_rot_prev_to_cur_r01=None,
        relative_rot_prev_to_cur_r02=None,
        relative_rot_prev_to_cur_r10=None,
        relative_rot_prev_to_cur_r11=None,
        relative_rot_prev_to_cur_r12=None,
        relative_rot_prev_to_cur_r20=None,
        relative_rot_prev_to_cur_r21=None,
        relative_rot_prev_to_cur_r22=None,
        head_forward_angle_step_deg=None,
        head_rotation_angle_step_deg=None,
        head_rotation_speed_deg_s=None,
        validation_notes=notes,
    )


def _select_gaze_csv(recording_path: Path, gaze_kind: str) -> Path:
    eye_dir = recording_path / "mps" / "eye_gaze"
    candidates = []
    kind = gaze_kind.strip().lower()
    if kind in {"personalized", "auto"}:
        candidates.append(eye_dir / "personalized_eye_gaze.csv")
    if kind in {"general", "auto", "personalized"}:
        candidates.append(eye_dir / "general_eye_gaze.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No usable eye-gaze CSV found under: {eye_dir}")


def _iter_gaze_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("tracking_timestamp_us"):
                yield row


def _read_closed_loop_trajectory(path: Path) -> _PoseSeries:
    timestamps: list[int] = []
    translations: list[list[float]] = []
    rotations: list[np.ndarray] = []
    qualities: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamps.append(int(row["tracking_timestamp_us"]) * 1000)
            translations.append(
                [
                    float(row["tx_world_device"]),
                    float(row["ty_world_device"]),
                    float(row["tz_world_device"]),
                ]
            )
            rotations.append(
                _quat_xyzw_to_rotation(
                    float(row["qx_world_device"]),
                    float(row["qy_world_device"]),
                    float(row["qz_world_device"]),
                    float(row["qw_world_device"]),
                )
            )
            qualities.append(_float_or_nan(row.get("quality_score")))
    return _PoseSeries(
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        translations_world_device=np.asarray(translations, dtype=np.float64),
        rotations_world_device=np.asarray(rotations, dtype=np.float64),
        quality_scores=np.asarray(qualities, dtype=np.float64),
    )


def _read_transform_device_cpf(vrs_path: Path) -> np.ndarray:
    from projectaria_tools.core import data_provider

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    transform = provider.get_device_calibration().get_transform_device_cpf()
    return np.asarray(transform.matrix(), dtype=np.float64)


def _nearest_pose(series: _PoseSeries, timestamp_ns: int) -> _NearestPose | None:
    if len(series.timestamps_ns) == 0:
        return None
    idx = bisect_left(series.timestamps_ns, timestamp_ns)
    candidates = []
    if idx < len(series.timestamps_ns):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None
    best_idx = min(candidates, key=lambda i: abs(int(series.timestamps_ns[i]) - timestamp_ns))
    pose_ts = int(series.timestamps_ns[best_idx])
    return _NearestPose(
        timestamp_ns=pose_ts,
        dt_ns=pose_ts - timestamp_ns,
        translation_world_device=series.translations_world_device[best_idx],
        rotation_world_device=series.rotations_world_device[best_idx],
        quality_score=float(series.quality_scores[best_idx]),
    )


def _compute_depth_and_combined_yaw(
    left_yaw: float,
    right_yaw: float,
    pitch: float,
) -> tuple[float, float]:
    if not np.isfinite([left_yaw, right_yaw, pitch]).all():
        return float("nan"), float("nan")
    denom = tan(right_yaw) - tan(left_yaw)
    if abs(denom) < 1e-12:
        return float("nan"), float("nan")
    x = (IPD_METERS / 2.0) * (tan(left_yaw) + tan(right_yaw)) / denom
    z = IPD_METERS / denom
    y = z * tan(pitch)
    point = np.asarray([x, y, z], dtype=np.float64)
    depth = float(np.linalg.norm(point))
    yaw = float(atan(x / z)) if z != 0 else float("nan")
    return depth, yaw


def _gaze_direction_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray | None:
    if not np.isfinite([yaw, pitch]).all():
        return None

    class _EyeGaze:
        pass

    eye_gaze = _EyeGaze()
    eye_gaze.yaw = float(yaw)
    eye_gaze.pitch = float(pitch)
    return gaze_direction_cpf_unit(eye_gaze)


def _confidence_widths_from_new_gaze(row: dict[str, str]) -> tuple[float, float]:
    left_low = _float_or_nan(row.get("left_yaw_low_rads_cpf"))
    left_high = _float_or_nan(row.get("left_yaw_high_rads_cpf"))
    right_low = _float_or_nan(row.get("right_yaw_low_rads_cpf"))
    right_high = _float_or_nan(row.get("right_yaw_high_rads_cpf"))
    pitch_low = _float_or_nan(row.get("pitch_low_rads_cpf"))
    pitch_high = _float_or_nan(row.get("pitch_high_rads_cpf"))
    yaw_candidates = np.asarray([left_high - left_low, right_high - right_low], dtype=np.float64)
    yaw_width = (
        float(np.nanmax(yaw_candidates))
        if np.isfinite(yaw_candidates).any()
        else float("nan")
    )
    pitch_width = pitch_high - pitch_low
    return yaw_width, float(pitch_width) if np.isfinite(pitch_width) else float("nan")


def _quat_xyzw_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 0 or not np.isfinite(quat).all():
        return np.full((3, 3), np.nan, dtype=np.float64)
    x, y, z, w = quat / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    hom = np.ones(4, dtype=np.float64)
    hom[:3] = point
    return (transform @ hom)[:3]


def _rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return axis / axis_norm * angle


def _head_relative_rotation(head: HeadSample) -> np.ndarray | None:
    values = [
        head.relative_rot_prev_to_cur_r00,
        head.relative_rot_prev_to_cur_r01,
        head.relative_rot_prev_to_cur_r02,
        head.relative_rot_prev_to_cur_r10,
        head.relative_rot_prev_to_cur_r11,
        head.relative_rot_prev_to_cur_r12,
        head.relative_rot_prev_to_cur_r20,
        head.relative_rot_prev_to_cur_r21,
        head.relative_rot_prev_to_cur_r22,
    ]
    if any(value is None for value in values):
        return None
    matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
    return matrix if np.isfinite(matrix).all() else None


def _head_world_rotation(head: HeadSample) -> np.ndarray | None:
    values = [
        head.head_rot_scene_r00,
        head.head_rot_scene_r01,
        head.head_rot_scene_r02,
        head.head_rot_scene_r10,
        head.head_rot_scene_r11,
        head.head_rot_scene_r12,
        head.head_rot_scene_r20,
        head.head_rot_scene_r21,
        head.head_rot_scene_r22,
    ]
    if any(value is None for value in values):
        return None
    matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
    return matrix if np.isfinite(matrix).all() else None


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    if arr.size != 3 or not np.isfinite(arr).all():
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return None
    return arr / norm


def _optional_vector(*values: float | None) -> np.ndarray | None:
    if any(value is None for value in values):
        return None
    arr = np.asarray(values, dtype=np.float64)
    return arr if np.isfinite(arr).all() else None


def _component_or_none(array: np.ndarray | None, index: int | tuple[int, int]) -> float | None:
    if array is None:
        return None
    value = array[index]
    return _finite_or_none(float(value))


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _float_or_nan(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_info(path: Path, root: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        rel_path = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        rel_path = str(path)
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "relative_path": rel_path,
        "size_bytes": stat.st_size,
    }


def _csv_header_sample(path: Path, *, compressed: bool, sample_rows: int = 2) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    opener = gzip.open if compressed else open
    rows: list[dict[str, str]] = []
    with opener(path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for _, row in zip(range(sample_rows), reader):
            rows.append(dict(row))
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "fieldnames": fieldnames,
        "sample_rows": rows,
        "row_count": None,
        "row_count_notes": "not counted during extraction to avoid re-reading large MPS files",
    }


def _jsonl_summary(path: Path, sample_rows: int = 2) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    samples: list[dict[str, Any]] = []
    top_level_keys: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _, line in zip(range(sample_rows), handle):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                top_level_keys.update(payload.keys())
                samples.append(payload)
            else:
                samples.append({"value": payload})
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "top_level_keys": sorted(top_level_keys),
        "sample_rows": samples,
        "row_count": None,
        "row_count_notes": "not counted during extraction to avoid re-reading large MPS files",
    }


def _anonymization_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories: Counter[str] = Counter()
    frame_count = 0
    detection_count = 0
    sample_entries: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        frame_count = len(payload)
        for timestamp, detections in payload.items():
            if not isinstance(detections, list):
                continue
            detection_count += len(detections)
            for detection in detections:
                if isinstance(detection, dict):
                    category = str(detection.get("category") or "unknown")
                    categories[category] += 1
            if len(sample_entries) < 3:
                sample_entries.append({"timestamp": timestamp, "detections": detections[:3]})
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "frame_count": frame_count,
        "detection_count": detection_count,
        "categories": dict(sorted(categories.items())),
        "sample_entries": sample_entries,
        "is_semantic_scene_object_annotation": False,
    }


def _find_object_like_mps_files(mps_path: Path) -> list[dict[str, Any]]:
    if not mps_path.exists():
        return []
    terms = (
        "object",
        "obj",
        "bbox",
        "bounding",
        "box",
        "detect",
        "seg",
        "semantic",
        "scene",
        "annotation",
        "label",
    )
    hits: list[dict[str, Any]] = []
    for path in sorted(mps_path.rglob("*")):
        if any(term in path.name.lower() for term in terms):
            try:
                rel_path = str(path.relative_to(mps_path))
            except ValueError:
                rel_path = str(path)
            hits.append(
                {
                    "path": rel_path,
                    "is_file": path.is_file(),
                    "size_bytes": path.stat().st_size if path.is_file() else None,
                }
            )
    return hits


def _gaze_kind_from_path(path: Path) -> str:
    if path.name.startswith("personalized"):
        return "personalized"
    if path.name.startswith("general"):
        return "general"
    return path.stem


def _ritw_gaze_coordinate_frames() -> dict[str, str]:
    return {
        "query_timestamp_ns": "device_time_ns",
        "gaze_dt_ns": "device_time_ns_delta",
        "pose_dt_ns": "device_time_ns_delta",
        "yaw_rad": "cpf_angle_rad",
        "pitch_rad": "cpf_angle_rad",
        "depth_m": "cpf_ray_distance_m",
        "gaze_dir_cpf_unit_x": "cpf_unit_direction",
        "gaze_dir_cpf_unit_y": "cpf_unit_direction",
        "gaze_dir_cpf_unit_z": "cpf_unit_direction",
        "gaze_origin_scene_x_m": "ritw_slam_world_frame_m",
        "gaze_origin_scene_y_m": "ritw_slam_world_frame_m",
        "gaze_origin_scene_z_m": "ritw_slam_world_frame_m",
        "gaze_point_scene_x_m": "ritw_slam_world_frame_m",
        "gaze_point_scene_y_m": "ritw_slam_world_frame_m",
        "gaze_point_scene_z_m": "ritw_slam_world_frame_m",
        "gaze_dir_scene_unit_x": "ritw_slam_world_frame_unit_direction",
        "gaze_dir_scene_unit_y": "ritw_slam_world_frame_unit_direction",
        "gaze_dir_scene_unit_z": "ritw_slam_world_frame_unit_direction",
    }


def _ritw_head_coordinate_frames() -> dict[str, str]:
    frames = dict(HEAD_FIELD_COORDINATE_FRAMES)
    for key, value in list(frames.items()):
        frames[key] = value.replace("adt_scene_frame", "ritw_slam_world_frame")
    return frames


def _rel_or_abs(path: str | Path, root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _cache_frame_count(path: Path) -> int:
    with np.load(path) as payload:
        return int(len(payload["frame_timestamps_ns"]))
