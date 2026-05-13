"""Reusable helpers for sequence-level and batch ADT gaze extraction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gaze import (
    RGB_STREAM_ID,
    default_summary_json_path,
    extract_gaze_sample,
    get_rgb_timestamps_ns,
    select_timestamps,
    summarize_gaze_samples,
    write_gaze_summary_json,
    write_samples_csv,
)
from .providers import create_adt_providers
from .results import sequence_file_path


@dataclass(frozen=True)
class GazeExtractionConfig:
    """Configuration shared by single-sequence and batch gaze extraction."""

    stride: int = 1
    start_index: int = 0
    end_index: int | None = None
    start_offset_s: float | None = None
    end_offset_s: float | None = None
    stream_id: str = RGB_STREAM_ID
    max_dt_ms: float | None = 20.0
    raw_image_orientation: bool = False

    @property
    def make_upright(self) -> bool:
        """Return whether RGB-related outputs should use upright orientation."""

        return not self.raw_image_orientation

    @property
    def max_dt_ns(self) -> int | None:
        """Return the allowed timestamp delta in nanoseconds when configured."""

        if self.max_dt_ms is None:
            return None
        return int(self.max_dt_ms * 1e6)


@dataclass(frozen=True)
class GazeExtractionResult:
    """Result metadata for one extracted sequence."""

    sequence_name: str
    sequence_path: Path
    provider_mode: str
    output_csv: Path
    summary_json: Path
    summary: dict[str, Any]


def default_gaze_csv_path(
    sequence_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    """Return the default per-sequence gaze CSV path."""

    return sequence_file_path(output_dir, sequence_name, "gaze", "gaze_samples.csv")


def extract_sequence_gaze(
    sequence: str | Path,
    config: GazeExtractionConfig,
    output_csv: str | Path | None = None,
) -> GazeExtractionResult:
    """Extract gaze CSV + summary for one sequence.

    zh-CN:
    这是当前仓库里“只做 gaze 数据提取”的统一入口。单 sequence 脚本和批量脚本
    都调用这里，避免维护两套平行逻辑。
    """

    providers = create_adt_providers(sequence, skeleton_flag=True)
    sequence_name = providers.sequence_path.name
    csv_path = Path(output_csv) if output_csv is not None else default_gaze_csv_path(sequence_name)
    summary_json = default_summary_json_path(csv_path)

    raw_rgb_timestamps_ns = get_rgb_timestamps_ns(providers.gt_provider, config.stream_id)
    annotation_filtered_timestamps_ns = restrict_to_provider_time_range(
        providers.gt_provider,
        raw_rgb_timestamps_ns,
    )
    offset_filtered_timestamps_ns = restrict_to_time_offsets(
        annotation_filtered_timestamps_ns,
        start_offset_s=config.start_offset_s,
        end_offset_s=config.end_offset_s,
    )
    selected_timestamps = select_timestamps(
        offset_filtered_timestamps_ns,
        stride=config.stride,
        start_index=config.start_index,
        end_index=config.end_index,
    )

    samples = [
        extract_gaze_sample(
            providers.gt_provider,
            timestamp_ns,
            stream_id_value=config.stream_id,
            max_dt_ns=config.max_dt_ns,
            make_upright=config.make_upright,
        )
        for timestamp_ns in selected_timestamps
    ]

    write_samples_csv(csv_path, samples)
    summary = summarize_gaze_samples(samples)
    summary.update(
        {
            "sequence_name": sequence_name,
            "sequence_path": str(providers.sequence_path),
            "provider_mode": providers.provider_mode,
            "stream_id": config.stream_id,
            "image_orientation": "upright" if config.make_upright else "raw",
            "output_csv": str(csv_path),
            "field_coordinate_frames": {
                "query_timestamp_ns": "device_time_ns",
                "gaze_dt_ns": "device_time_ns_delta",
                "pose_dt_ns": "device_time_ns_delta",
                "yaw_rad": "cpf_angle_rad",
                "pitch_rad": "cpf_angle_rad",
                "depth_m": "cpf_depth_m",
                "gaze_dir_cpf_unit_x": "cpf_unit_direction",
                "gaze_dir_cpf_unit_y": "cpf_unit_direction",
                "gaze_dir_cpf_unit_z": "cpf_unit_direction",
                "gaze_u_px": (
                    "rgb_image_plane_upright_px"
                    if config.make_upright
                    else "rgb_image_plane_raw_px"
                ),
                "gaze_v_px": (
                    "rgb_image_plane_upright_px"
                    if config.make_upright
                    else "rgb_image_plane_raw_px"
                ),
                "image_width_px": (
                    "upright_rgb_image_width_px"
                    if config.make_upright
                    else "raw_rgb_image_width_px"
                ),
                "image_height_px": (
                    "upright_rgb_image_height_px"
                    if config.make_upright
                    else "raw_rgb_image_height_px"
                ),
                "gaze_origin_scene_x_m": "adt_scene_frame_m",
                "gaze_origin_scene_y_m": "adt_scene_frame_m",
                "gaze_origin_scene_z_m": "adt_scene_frame_m",
                "gaze_point_scene_x_m": "adt_scene_frame_m",
                "gaze_point_scene_y_m": "adt_scene_frame_m",
                "gaze_point_scene_z_m": "adt_scene_frame_m",
                "gaze_dir_scene_unit_x": "adt_scene_frame_unit_direction",
                "gaze_dir_scene_unit_y": "adt_scene_frame_unit_direction",
                "gaze_dir_scene_unit_z": "adt_scene_frame_unit_direction",
            },
            "field_definitions": {
                "depth_m": (
                    "Distance from the CPF origin along the gaze ray, matching "
                    "projectaria_tools.core.mps.get_eyegaze_point_at_depth; not the "
                    "CPF z coordinate."
                ),
                "gaze_dir_cpf_unit_xyz": (
                    "Normalized gaze direction in CPF. Independent of depth_m."
                ),
                "gaze_point_scene_xyz": (
                    "Scene-frame gaze point obtained by transforming the CPF gaze point "
                    "constructed with the official depth semantics."
                ),
                "gaze_dir_scene_unit_xyz": (
                    "Normalized Scene-frame gaze direction. Independent of depth_m."
                ),
            },
            "source_counts": {
                "eye_gaze_csv_count": count_eye_gaze_rows(
                    providers.sequence_path / "eyegaze.csv"
                ),
                "raw_rgb_timestamp_count": len(raw_rgb_timestamps_ns),
                "annotation_filtered_rgb_timestamp_count": len(
                    annotation_filtered_timestamps_ns
                ),
                "offset_filtered_rgb_timestamp_count": len(offset_filtered_timestamps_ns),
                "selected_rgb_timestamp_count": len(selected_timestamps),
            },
            "source_time_ranges_ns": {
                "eye_gaze_csv": describe_eye_gaze_csv(
                    providers.sequence_path / "eyegaze.csv"
                ),
                "raw_rgb_timestamps": describe_timestamp_list(raw_rgb_timestamps_ns),
                "annotation_filtered_rgb_timestamps": describe_timestamp_list(
                    annotation_filtered_timestamps_ns
                ),
                "offset_filtered_rgb_timestamps": describe_timestamp_list(
                    offset_filtered_timestamps_ns
                ),
                "selected_rgb_timestamps": describe_timestamp_list(selected_timestamps),
                "provider_annotation_range": describe_provider_annotation_range(
                    providers.gt_provider
                ),
            },
            "selection": {
                "start_index": config.start_index,
                "end_index": config.end_index,
                "start_offset_s": config.start_offset_s,
                "end_offset_s": config.end_offset_s,
                "stride": config.stride,
                "max_dt_ms": config.max_dt_ms,
            },
        }
    )
    write_gaze_summary_json(summary_json, summary)
    return GazeExtractionResult(
        sequence_name=sequence_name,
        sequence_path=providers.sequence_path,
        provider_mode=providers.provider_mode,
        output_csv=csv_path,
        summary_json=summary_json,
        summary=summary,
    )


def discover_sequence_paths(root: str | Path) -> list[Path]:
    """Return ADT sequence directories under one root.

    zh-CN:
    这里用很保守的规则识别 sequence：目录下同时有 `metadata.json`、`video.vrs`
    和 `eyegaze.csv`。这样批量扫描时不会把别的子目录误认为 sequence。
    """

    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"ADT data root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Expected ADT data root directory: {root_path}")

    sequence_paths = [
        path
        for path in sorted(root_path.iterdir())
        if path.is_dir()
        and (path / "metadata.json").exists()
        and (path / "video.vrs").exists()
        and (path / "eyegaze.csv").exists()
    ]
    if not sequence_paths:
        raise ValueError(f"No ADT sequence directories found under: {root_path}")
    return sequence_paths


def restrict_to_provider_time_range(gt_provider: Any, timestamps_ns: list[int]) -> list[int]:
    """Keep RGB timestamps inside the gaze/pose annotation range when available."""

    if not hasattr(gt_provider, "get_start_time_ns") or not hasattr(gt_provider, "get_end_time_ns"):
        return timestamps_ns

    start_ns = int(gt_provider.get_start_time_ns())
    end_ns = int(gt_provider.get_end_time_ns())
    filtered = [timestamp for timestamp in timestamps_ns if start_ns <= timestamp <= end_ns]
    if not filtered:
        raise ValueError(
            "No RGB timestamps overlap the provider annotation time range: "
            f"rgb={len(timestamps_ns)} annotation=[{start_ns}, {end_ns}]"
        )
    return filtered


def restrict_to_time_offsets(
    timestamps_ns: list[int],
    start_offset_s: float | None,
    end_offset_s: float | None,
) -> list[int]:
    """Keep RGB timestamps inside a relative time window."""

    if not timestamps_ns:
        raise ValueError("No RGB timestamps available")
    if start_offset_s is None and end_offset_s is None:
        return timestamps_ns
    if start_offset_s is not None and start_offset_s < 0:
        raise ValueError("start_offset_s must be non-negative")
    if end_offset_s is not None and end_offset_s < 0:
        raise ValueError("end_offset_s must be non-negative")
    if (
        start_offset_s is not None
        and end_offset_s is not None
        and end_offset_s <= start_offset_s
    ):
        raise ValueError("end_offset_s must be greater than start_offset_s")

    base_timestamp_ns = timestamps_ns[0]
    start_ns = (
        base_timestamp_ns
        if start_offset_s is None
        else base_timestamp_ns + int(start_offset_s * 1e9)
    )
    end_ns = (
        timestamps_ns[-1] + 1
        if end_offset_s is None
        else base_timestamp_ns + int(end_offset_s * 1e9)
    )
    filtered = [timestamp for timestamp in timestamps_ns if start_ns <= timestamp < end_ns]
    if not filtered:
        raise ValueError(
            "No RGB timestamps selected by time offsets: "
            f"offset_s=[{start_offset_s}, {end_offset_s}) "
            f"available_duration_s={(timestamps_ns[-1] - timestamps_ns[0]) / 1e9:.3f}"
        )
    return filtered


def count_eye_gaze_rows(path: Path) -> int | None:
    """Count `eyegaze.csv` data rows when the file is available."""

    stats = describe_eye_gaze_csv(path)
    return stats["count"]


def describe_eye_gaze_csv(path: Path) -> dict[str, int | float | None]:
    """Describe `eyegaze.csv` row count and timestamp range."""

    if not path.exists():
        return {
            "count": None,
            "timestamp_start_ns": None,
            "timestamp_end_ns": None,
            "duration_s": None,
        }

    count = 0
    first_ns: int | None = None
    last_ns: int | None = None
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count += 1
            tracking_us = row.get("tracking_timestamp_us")
            if not tracking_us:
                continue
            timestamp_ns = int(tracking_us) * 1000
            if first_ns is None:
                first_ns = timestamp_ns
            last_ns = timestamp_ns

    return {
        "count": count,
        "timestamp_start_ns": first_ns,
        "timestamp_end_ns": last_ns,
        "duration_s": (
            (last_ns - first_ns) / 1e9
            if first_ns is not None and last_ns is not None
            else None
        ),
    }


def describe_timestamp_list(timestamps_ns: list[int]) -> dict[str, int | float | None]:
    """Describe a timestamp list with count and start/end/duration."""

    if not timestamps_ns:
        return {
            "count": 0,
            "timestamp_start_ns": None,
            "timestamp_end_ns": None,
            "duration_s": None,
        }
    return {
        "count": len(timestamps_ns),
        "timestamp_start_ns": timestamps_ns[0],
        "timestamp_end_ns": timestamps_ns[-1],
        "duration_s": (timestamps_ns[-1] - timestamps_ns[0]) / 1e9,
    }


def describe_provider_annotation_range(gt_provider: Any) -> dict[str, int | float | None]:
    """Return provider annotation start/end if the API exposes them."""

    if not hasattr(gt_provider, "get_start_time_ns") or not hasattr(gt_provider, "get_end_time_ns"):
        return {
            "timestamp_start_ns": None,
            "timestamp_end_ns": None,
            "duration_s": None,
        }
    start_ns = int(gt_provider.get_start_time_ns())
    end_ns = int(gt_provider.get_end_time_ns())
    return {
        "timestamp_start_ns": start_ns,
        "timestamp_end_ns": end_ns,
        "duration_s": (end_ns - start_ns) / 1e9,
    }
