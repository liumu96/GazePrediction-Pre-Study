"""Gaze extraction, validation, projection, and visualization helpers.

The helpers here keep one timestamp query inspectable end to end: raw gaze in
CPF, nearest pose in Scene frame, projection to RGB pixels, and validity notes.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import tan
from pathlib import Path
from typing import Any

import numpy as np

# Project Aria ADT RGB camera stream id used by the official examples.
RGB_STREAM_ID = "214-1"


@dataclass(frozen=True)
class GazeSample:
    """One gaze query result plus derived projection and Scene-frame values."""

    query_timestamp_ns: int                         # Original query timestamp in device time, nanoseconds.
    gaze_valid: bool                                # Whether the gaze query returned valid data.
    gaze_dt_ns: int | None                          # Time difference between query and gaze data, nanoseconds.
    yaw_rad: float | None                           # Yaw angle in radians.
    pitch_rad: float | None                         # Pitch angle in radians.
    depth_m: float | None                           # Distance along the CPF gaze ray, meters.
    gaze_dir_cpf_unit_x: float | None               # Unit gaze direction X in CPF.
    gaze_dir_cpf_unit_y: float | None               # Unit gaze direction Y in CPF.
    gaze_dir_cpf_unit_z: float | None               # Unit gaze direction Z in CPF.
    yaw_confidence_width_rad: float | None          # Yaw confidence interval width in radians. 
    pitch_confidence_width_rad: float | None        # Pitch confidence interval width in radians.
    projection_valid: bool                          # Whether projection to RGB image plane succeeded.
    gaze_u_px: float | None                         # Gaze U coordinate in RGB image pixels.
    gaze_v_px: float | None                         # Gaze V coordinate in RGB image pixels.
    projection_in_image: bool                       # Whether the projected gaze point is within the RGB image bounds.
    image_width_px: int | None                      # RGB image width in pixels.
    image_height_px: int | None                     # RGB image height in pixels.
    pose_valid: bool                                # Whether a valid pose was found near the gaze timestamp.
    pose_dt_ns: int | None                          # Time difference between gaze query and nearest pose, nanoseconds.
    pose_quality_score: float | None                # Quality score of the nearest pose, if available.
    gaze_origin_scene_x_m: float | None             # Gaze origin X coordinate in ADT Scene frame, meters.
    gaze_origin_scene_y_m: float | None             # Gaze origin Y coordinate in ADT Scene frame, meters.
    gaze_origin_scene_z_m: float | None             # Gaze origin Z coordinate in ADT Scene frame, meters.
    gaze_point_scene_x_m: float | None              # Gaze point X coordinate in ADT Scene frame, meters.
    gaze_point_scene_y_m: float | None              # Gaze point Y coordinate in ADT Scene frame, meters.
    gaze_point_scene_z_m: float | None              # Gaze point Z coordinate in ADT Scene frame, meters.
    gaze_dir_scene_unit_x: float | None             # Unit gaze direction X in ADT Scene frame.
    gaze_dir_scene_unit_y: float | None             # Unit gaze direction Y in ADT Scene frame.
    gaze_dir_scene_unit_z: float | None             # Unit gaze direction Z in ADT Scene frame.
    validation_notes: str                           # Semicolon-separated notes on data validity and projection results, or "ok" if no issues.

    def as_csv_row(self) -> dict[str, Any]:
        """Return a flat row suitable for csv.DictWriter."""

        return asdict(self)

    @property
    def gaze_dir_cpf_unit_xyz(self) -> np.ndarray | None:
        """Return the CPF unit gaze direction as a vector when available."""

        return _vector_from_optional_xyz(
            self.gaze_dir_cpf_unit_x,
            self.gaze_dir_cpf_unit_y,
            self.gaze_dir_cpf_unit_z,
        )

    @property
    def gaze_dir_scene_unit_xyz(self) -> np.ndarray | None:
        """Return the Scene-frame unit gaze direction as a vector when available."""

        return _vector_from_optional_xyz(
            self.gaze_dir_scene_unit_x,
            self.gaze_dir_scene_unit_y,
            self.gaze_dir_scene_unit_z,
        )


def stream_id(stream_id_value: str = RGB_STREAM_ID) -> Any:
    """Create a Project Aria StreamId only when a query needs one.

    zh-CN:
    这个函数的目的是延迟创建 Project Aria 的 StreamId 对象，避免在导入本模块时
    就初始化 projectaria_tools 的具体类型。真正查询数据时再调用这个函数创建
    StreamId。
    """

    from projectaria_tools.core.stream_id import StreamId

    return StreamId(stream_id_value)


def get_rgb_timestamps_ns(gt_provider: Any, stream_id_value: str = RGB_STREAM_ID) -> list[int]:
    """Return RGB capture timestamps in device time, nanoseconds.

    zh-CN:
    这个函数使用 gt_provider 查询指定 stream id 的 RGB 捕获时间戳。返回值单位
    是纳秒，并且时间域是 device time；后续 gaze 和 pose 对齐也使用同一个时间域。
    """

    return list(gt_provider.get_aria_device_capture_timestamps_ns(stream_id(stream_id_value)))


def select_timestamps(
    timestamps_ns: list[int],
    stride: int,
    start_index: int = 0,
    end_index: int | None = None,
) -> list[int]:
    """Pick a stable timestamp window and optional stride.

    zh-CN:
    这个函数只做两件事：先按 start_index/end_index 选择 RGB timestamp
    区间，再按 stride 抽稀。extract 脚本的默认窗口大小在脚本层处理，避免
    这里再保留额外的数量限制概念。
    """

    if stride <= 0:
        raise ValueError("stride must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is not None and end_index <= start_index:
        raise ValueError("end_index must be greater than start_index")

    window = timestamps_ns[start_index:end_index]
    selected = window[::stride]
    if not selected:
        raise ValueError("No timestamps selected; check start/end interval and stride")
    return selected


def gaze_point_cpf(eye_gaze: Any, depth_m: float | None = None) -> np.ndarray:
    """Convert yaw/pitch/depth to a 3D gaze point in Central Pupil Frame.

    zh-CN:
    eye_gaze 里包含 yaw、pitch 和 depth。这里的 `depth_m` 按官方
    `projectaria_tools.core.mps.get_eyegaze_point_at_depth(...)` 的语义处理：
    它表示从 CPF origin 沿 gaze ray 前进的距离，不是 CPF 的 `z` 坐标值。
    如果传入 depth_m，就用 depth_m 覆盖 eye_gaze.depth。最终返回值与官方
    helper 对齐，后续 Scene-frame gaze point 和 RGB projection 也保持同一
    套 depth 定义。
    """

    depth = eye_gaze.depth if depth_m is None else depth_m
    from projectaria_tools.core import mps

    return np.asarray(
        mps.get_eyegaze_point_at_depth(eye_gaze.yaw, eye_gaze.pitch, depth),
        dtype=np.float64,
    ).reshape(3)


def gaze_direction_cpf_unit(eye_gaze: Any) -> np.ndarray | None:
    """Return the unit gaze direction in CPF.

    zh-CN:
    这里把 `yaw/pitch` 直接转换成 CPF 下的单位方向向量，因此不依赖 `depth_m`。
    这更适合 SparseGaze 一类 local-gaze modeling；即使 depth 缺失，只要
    yaw/pitch 有效，局部 gaze direction 仍然可以使用。
    """

    direction = np.array(
        [tan(eye_gaze.yaw), tan(eye_gaze.pitch), 1.0],
        dtype=np.float64,
    )
    return _normalize_vector(direction)


def confidence_widths(eye_gaze: Any) -> tuple[float, float]:
    """Return yaw/pitch confidence interval widths when fields are available.

    zh-CN:
    ADT gaze 提供 yaw_low、yaw_high、pitch_low、pitch_high，表示 yaw/pitch
    置信区间的上下界。这里分别计算 yaw_high - yaw_low 和
    pitch_high - pitch_low。width 越大，通常表示该角度估计越不确定。
    如果字段不存在，返回 nan 参与后续 validity check。
    """

    yaw_low = getattr(eye_gaze, "yaw_low", np.nan)
    yaw_high = getattr(eye_gaze, "yaw_high", np.nan)
    pitch_low = getattr(eye_gaze, "pitch_low", np.nan)
    pitch_high = getattr(eye_gaze, "pitch_high", np.nan)
    return float(yaw_high - yaw_low), float(pitch_high - pitch_low)


def get_rgb_image(gt_provider: Any, timestamp_ns: int, stream_id_value: str = RGB_STREAM_ID) -> Any:
    """Query the RGB image closest to a timestamp.

    zh-CN:
    这个函数查询与给定 timestamp_ns 最近的 RGB 图像。返回值仍然保留 dt_ns，
    所以 overlay 图里可以显示图像帧时间和 query timestamp 之间的偏差。
    """

    return gt_provider.get_aria_image_by_timestamp_ns(timestamp_ns, stream_id(stream_id_value))


def project_gaze_to_rgb(
    gt_provider: Any,
    eye_gaze: Any,
    stream_id_value: str = RGB_STREAM_ID,
    make_upright: bool = True,
) -> tuple[np.ndarray | None, tuple[int, int]]:
    """Project a gaze point to the RGB camera image plane.

    zh-CN:
    这里直接调用官方 `projectaria_tools.core.mps.utils.get_gaze_vector_reprojection`
    来获得 RGB 投影，不再维护本地复刻版本。这样 `depth_m` 语义、
    `make_upright` 的处理方式，以及投影结果都与官方 helper 保持一致。
    """

    rgb_stream_id = stream_id(stream_id_value)
    camera_calibration = gt_provider.get_aria_camera_calibration(rgb_stream_id)
    image_size = camera_calibration.get_image_size()
    width_height = (int(image_size[0]), int(image_size[1]))
    if eye_gaze.depth <= 0:
        return None, width_height

    from projectaria_tools.core import mps

    projection = mps.utils.get_gaze_vector_reprojection(
        eye_gaze=eye_gaze,
        stream_id_label=camera_calibration.get_label(),
        device_calibration=gt_provider.raw_data_provider_ptr().get_device_calibration(),
        camera_calibration=camera_calibration,
        depth_m=float(eye_gaze.depth),
        make_upright=make_upright,
    )
    if projection is None:
        return None, width_height
    return np.asarray(projection, dtype=np.float64).reshape(-1)[:2], width_height


def project_scene_points_to_rgb(
    gt_provider: Any,
    scene_points_m: list[np.ndarray],
    reference_timestamp_ns: int,
    stream_id_value: str = RGB_STREAM_ID,
    make_upright: bool = True,
) -> tuple[list[np.ndarray | None], tuple[int, int]]:
    """Project Scene-frame 3D points into one reference RGB frame.

    zh-CN:
    这个函数用于 reference-frame scanpath。输入是一组已经在 ADT Scene/world
    frame 下的 3D gaze points，再选择一个 reference RGB timestamp。函数会
    读取 reference timestamp 附近的 device pose 和 RGB camera calibration，
    把所有 Scene points 变换到 reference camera frame，再投影到同一张 RGB
    图像上。

    注意：这里投影的是 `gaze_point_scene_*`，所以它依赖 eye_gaze.depth 的
    可靠性。落在 reference camera 背后的点或投影失败的点会返回 None。
    """

    camera_calibration, transform_device_camera, width_height = _rgb_camera_context(
        gt_provider,
        stream_id_value,
        make_upright,
    )
    pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(reference_timestamp_ns)
    if not pose_with_dt.is_valid():
        raise ValueError(f"Reference pose is invalid at {reference_timestamp_ns}")

    transform_scene_camera = (
        pose_with_dt.data().transform_scene_device @ transform_device_camera
    )
    transform_camera_scene = transform_scene_camera.inverse()

    projections: list[np.ndarray | None] = []
    for scene_point in scene_points_m:
        point_scene = np.asarray(scene_point, dtype=np.float64)
        if point_scene.shape != (3,) or not np.isfinite(point_scene).all():
            projections.append(None)
            continue

        point_camera = np.asarray(transform_camera_scene @ point_scene, dtype=np.float64)
        if point_camera[2] <= 0:
            projections.append(None)
            continue

        projection = camera_calibration.project(point_camera)
        if projection is None:
            projections.append(None)
            continue
        projections.append(np.asarray(projection, dtype=np.float64).reshape(-1)[:2])

    return projections, width_height


def scene_gaze_ray(
    gt_provider: Any,
    eye_gaze: Any,
    timestamp_ns: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return gaze origin and gaze point in ADT Scene frame.

    zh-CN:
    这个函数把 gaze 从 CPF 坐标系转换到 ADT Scene/world frame。Scene frame 是
    ADT 中对齐 skeleton、object boxes、device trajectory 的统一 3D 坐标系。
    返回值是两个点：CPF origin 在 Scene frame 下的位置，以及 gaze point 在
    Scene frame 下的位置。两点连线就是用于 3D 可视化的 gaze ray。
    """

    # 这里用 gaze timestamp 查询 pose，保持时间对齐的一致性。ADT provider 会找
    # 最近的 pose，并返回对应的 dt_ns 和 pose 数据。
    pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
    if not pose_with_dt.is_valid() or eye_gaze.depth <= 0:
        return None

    aria_pose = pose_with_dt.data()
    device_calibration = gt_provider.raw_data_provider_ptr().get_device_calibration()
    # T_scene_cpf = T_scene_device @ T_device_cpf, 用来把 CPF 下的点变换到
    # ADT Scene/world 坐标系，方便画米制 3D ray。
    transform_scene_cpf = (
        aria_pose.transform_scene_device @ device_calibration.get_transform_device_cpf()
    )
    origin_scene = np.asarray(transform_scene_cpf @ [0.0, 0.0, 0.0], dtype=np.float64)
    point_scene = np.asarray(
        transform_scene_cpf @ gaze_point_cpf(eye_gaze),
        dtype=np.float64,
    )
    return origin_scene, point_scene


def scene_gaze_direction_unit(
    gt_provider: Any,
    eye_gaze: Any,
    timestamp_ns: int,
) -> np.ndarray | None:
    """Return the unit gaze direction in ADT Scene frame.

    zh-CN:
    这个方向向量是把 CPF 下的单位 gaze direction 变换到 Scene frame 后再归一化，
    因此它不依赖 `depth_m`。对后续把 local gaze 和 world gaze 分开建模非常有用。
    """

    transform_scene_cpf = _transform_scene_cpf(gt_provider, timestamp_ns)
    if transform_scene_cpf is None:
        return None

    direction_cpf = gaze_direction_cpf_unit(eye_gaze)
    if direction_cpf is None:
        return None

    origin_scene = np.asarray(transform_scene_cpf @ [0.0, 0.0, 0.0], dtype=np.float64)
    direction_point_scene = np.asarray(
        transform_scene_cpf @ direction_cpf,
        dtype=np.float64,
    )
    return _normalize_vector(direction_point_scene - origin_scene)


def extract_gaze_sample(
    gt_provider: Any,
    timestamp_ns: int,
    stream_id_value: str = RGB_STREAM_ID,
    max_dt_ns: int | None = None,
    make_upright: bool = True,
) -> GazeSample:
    """Extract and validate gaze for one timestamp.

    zh-CN:
    这个函数是 gaze tutorial 的核心入口。它在一个 query timestamp 上查询
    nearest gaze 和 nearest pose，然后计算：
    - 原始 gaze 字段：yaw、pitch、depth、confidence width。
    - 时间对齐质量：gaze_dt_ns、pose_dt_ns。
    - RGB 投影：gaze_u_px、gaze_v_px、projection_in_image。
    - 3D Scene-frame ray：gaze_origin_scene_* 和 gaze_point_scene_*。

    探索阶段不会直接丢弃问题样本，而是把问题写入 validation_notes，方便之后
    批量统计哪些情况应该成为正式 preprocessing 的过滤规则。
    """

    notes: list[str] = []
    gaze_with_dt = gt_provider.get_eyegaze_by_timestamp_ns(timestamp_ns)
    pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)

    # 如果 gaze 无效，直接返回 gaze_valid=False 的样本；pose 相关字段仍然保留，
    # 便于后续判断问题来自 gaze 本身还是时间窗口。
    if not gaze_with_dt.is_valid():
        return _invalid_sample(timestamp_ns, pose_with_dt, "gaze_query_invalid")

    eye_gaze = gaze_with_dt.data()
    gaze_dt_ns = int(gaze_with_dt.dt_ns())
    pose_valid = bool(pose_with_dt.is_valid())
    pose_dt_ns = int(pose_with_dt.dt_ns()) if pose_valid else None
    pose_quality = float(pose_with_dt.data().quality_score) if pose_valid else None

    # Exploration keeps flagged rows instead of dropping them immediately. The
    # later quality report can decide which notes should become reject rules.
    if max_dt_ns is not None and abs(gaze_dt_ns) > max_dt_ns:
        notes.append("gaze_dt_exceeds_threshold")
    if not np.isfinite([eye_gaze.yaw, eye_gaze.pitch]).all():
        notes.append("yaw_or_pitch_not_finite")
    if eye_gaze.depth <= 0:
        notes.append("depth_not_available")

    yaw_width, pitch_width = confidence_widths(eye_gaze)
    if not np.isfinite([yaw_width, pitch_width]).all():
        notes.append("confidence_width_not_finite")

    gaze_dir_cpf_unit = gaze_direction_cpf_unit(eye_gaze)

    projection, image_size = project_gaze_to_rgb(
        gt_provider,
        eye_gaze,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    width, height = image_size
    projection_valid = projection is not None
    gaze_u = float(projection[0]) if projection_valid else None
    gaze_v = float(projection[1]) if projection_valid else None
    in_image = bool(
        projection_valid
        and gaze_u is not None
        and gaze_v is not None
        and 0 <= gaze_u < width
        and 0 <= gaze_v < height
    )
    if not projection_valid:
        notes.append("projection_failed")
    elif not in_image:
        notes.append("projection_outside_image")

    ray = scene_gaze_ray(gt_provider, eye_gaze, timestamp_ns)
    gaze_dir_scene_unit = scene_gaze_direction_unit(gt_provider, eye_gaze, timestamp_ns)
    if ray is None:
        notes.append("scene_ray_unavailable")
        origin = point = np.array([np.nan, np.nan, np.nan])
    else:
        origin, point = ray

    return GazeSample(
        query_timestamp_ns=int(timestamp_ns),
        gaze_valid=True,
        gaze_dt_ns=gaze_dt_ns,
        yaw_rad=float(eye_gaze.yaw),
        pitch_rad=float(eye_gaze.pitch),
        depth_m=float(eye_gaze.depth),
        gaze_dir_cpf_unit_x=_finite_or_none(gaze_dir_cpf_unit[0]) if gaze_dir_cpf_unit is not None else None,
        gaze_dir_cpf_unit_y=_finite_or_none(gaze_dir_cpf_unit[1]) if gaze_dir_cpf_unit is not None else None,
        gaze_dir_cpf_unit_z=_finite_or_none(gaze_dir_cpf_unit[2]) if gaze_dir_cpf_unit is not None else None,
        yaw_confidence_width_rad=yaw_width,
        pitch_confidence_width_rad=pitch_width,
        projection_valid=projection_valid,
        gaze_u_px=gaze_u,
        gaze_v_px=gaze_v,
        projection_in_image=in_image,
        image_width_px=width,
        image_height_px=height,
        pose_valid=pose_valid,
        pose_dt_ns=pose_dt_ns,
        pose_quality_score=pose_quality,
        gaze_origin_scene_x_m=_finite_or_none(origin[0]),
        gaze_origin_scene_y_m=_finite_or_none(origin[1]),
        gaze_origin_scene_z_m=_finite_or_none(origin[2]),
        gaze_point_scene_x_m=_finite_or_none(point[0]),
        gaze_point_scene_y_m=_finite_or_none(point[1]),
        gaze_point_scene_z_m=_finite_or_none(point[2]),
        gaze_dir_scene_unit_x=_finite_or_none(gaze_dir_scene_unit[0]) if gaze_dir_scene_unit is not None else None,
        gaze_dir_scene_unit_y=_finite_or_none(gaze_dir_scene_unit[1]) if gaze_dir_scene_unit is not None else None,
        gaze_dir_scene_unit_z=_finite_or_none(gaze_dir_scene_unit[2]) if gaze_dir_scene_unit is not None else None,
        validation_notes=";".join(notes) if notes else "ok",
    )

# Private helpers for invalid rows and calibration transforms.

def _invalid_sample(timestamp_ns: int, pose_with_dt: Any, reason: str) -> GazeSample:
    pose_valid = bool(pose_with_dt.is_valid())
    pose_dt_ns = int(pose_with_dt.dt_ns()) if pose_valid else None
    pose_quality = float(pose_with_dt.data().quality_score) if pose_valid else None
    return GazeSample(
        query_timestamp_ns=int(timestamp_ns),
        gaze_valid=False,
        gaze_dt_ns=None,
        yaw_rad=None,
        pitch_rad=None,
        depth_m=None,
        gaze_dir_cpf_unit_x=None,
        gaze_dir_cpf_unit_y=None,
        gaze_dir_cpf_unit_z=None,
        yaw_confidence_width_rad=None,
        pitch_confidence_width_rad=None,
        projection_valid=False,
        gaze_u_px=None,
        gaze_v_px=None,
        projection_in_image=False,
        image_width_px=None,
        image_height_px=None,
        pose_valid=pose_valid,
        pose_dt_ns=pose_dt_ns,
        pose_quality_score=pose_quality,
        gaze_origin_scene_x_m=None,
        gaze_origin_scene_y_m=None,
        gaze_origin_scene_z_m=None,
        gaze_point_scene_x_m=None,
        gaze_point_scene_y_m=None,
        gaze_point_scene_z_m=None,
        gaze_dir_scene_unit_x=None,
        gaze_dir_scene_unit_y=None,
        gaze_dir_scene_unit_z=None,
        validation_notes=reason,
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _vector_from_optional_xyz(
    x_value: float | None,
    y_value: float | None,
    z_value: float | None,
) -> np.ndarray | None:
    if x_value is None or y_value is None or z_value is None:
        return None
    vector = np.asarray([x_value, y_value, z_value], dtype=np.float64)
    return vector if np.isfinite(vector).all() else None


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def _transform_scene_cpf(gt_provider: Any, timestamp_ns: int) -> Any | None:
    """Return `T_scene_cpf` when a valid pose exists near the timestamp."""

    pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
    if not pose_with_dt.is_valid():
        return None

    aria_pose = pose_with_dt.data()
    device_calibration = gt_provider.raw_data_provider_ptr().get_device_calibration()
    return aria_pose.transform_scene_device @ device_calibration.get_transform_device_cpf()


def _rgb_camera_context(
    gt_provider: Any,
    stream_id_value: str,
    make_upright: bool,
) -> tuple[Any, Any, tuple[int, int]]:
    rgb_stream_id = stream_id(stream_id_value)
    camera_calibration = gt_provider.get_aria_camera_calibration(rgb_stream_id)
    device_calibration = gt_provider.raw_data_provider_ptr().get_device_calibration()
    image_size = camera_calibration.get_image_size()
    transform_device_camera = device_calibration.get_transform_device_sensor(
        camera_calibration.get_label(),
        True,
    )
    if make_upright:
        transform_device_camera = transform_device_camera @ _camera_cw90_transform()
    return camera_calibration, transform_device_camera, (int(image_size[0]), int(image_size[1]))


def _camera_cw90_transform() -> Any:
    """Return the SE3 rotation used by the official upright gaze projection helper.

    zh-CN:
    Project Aria 官方 `get_gaze_vector_reprojection(..., make_upright=True)`
    也是把 RGB camera frame 绕前向 Z 轴顺时针旋转 90 度，再使用原 camera
    calibration 投影。这里保留同样写法，便于和官方 helper 对照。
    """
    from projectaria_tools.core.sophus import SE3

    return SE3.from_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
    )


def downsample_samples(
    samples: Sequence[GazeSample],
    stride: int,
    include_last: bool,
) -> list[GazeSample]:
    """Return every Nth sample for visualization without changing CSV output.

    zh-CN:
    这个函数只用于可视化抽稀。CSV 仍然保留所有 selected samples。
    include_last=True 时会强制保留窗口最后一帧，便于轨迹图包含事件窗口末尾。
    """

    if stride <= 0:
        raise ValueError("visualization stride must be positive")
    selected = list(samples[::stride])
    if include_last and samples and selected[-1] != samples[-1]:
        selected.append(samples[-1])
    return selected


def write_samples_csv(path: os.PathLike[str] | str, samples: Sequence[GazeSample]) -> None:
    """Write gaze samples to a CSV file for later analysis and filtering."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [sample.as_csv_row() for sample in samples]
    if not rows:
        raise ValueError("No gaze samples to write")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_samples_csv(path: os.PathLike[str] | str) -> list[GazeSample]:
    """Read a gaze samples CSV previously written by this module."""

    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [gaze_sample_from_csv_row(row) for row in reader]


def gaze_sample_from_csv_row(row: dict[str, str]) -> GazeSample:
    """Convert one CSV row into a GazeSample with the original field types."""

    return GazeSample(
        query_timestamp_ns=csv_int(row["query_timestamp_ns"]),
        gaze_valid=csv_bool(row["gaze_valid"]),
        gaze_dt_ns=csv_optional_int(row["gaze_dt_ns"]),
        yaw_rad=csv_optional_float(row["yaw_rad"]),
        pitch_rad=csv_optional_float(row["pitch_rad"]),
        depth_m=csv_optional_float(row["depth_m"]),
        gaze_dir_cpf_unit_x=csv_optional_float(row.get("gaze_dir_cpf_unit_x", "")),
        gaze_dir_cpf_unit_y=csv_optional_float(row.get("gaze_dir_cpf_unit_y", "")),
        gaze_dir_cpf_unit_z=csv_optional_float(row.get("gaze_dir_cpf_unit_z", "")),
        yaw_confidence_width_rad=csv_optional_float(row["yaw_confidence_width_rad"]),
        pitch_confidence_width_rad=csv_optional_float(row["pitch_confidence_width_rad"]),
        projection_valid=csv_bool(row["projection_valid"]),
        gaze_u_px=csv_optional_float(row["gaze_u_px"]),
        gaze_v_px=csv_optional_float(row["gaze_v_px"]),
        projection_in_image=csv_bool(row["projection_in_image"]),
        image_width_px=csv_optional_int(row["image_width_px"]),
        image_height_px=csv_optional_int(row["image_height_px"]),
        pose_valid=csv_bool(row["pose_valid"]),
        pose_dt_ns=csv_optional_int(row["pose_dt_ns"]),
        pose_quality_score=csv_optional_float(row["pose_quality_score"]),
        gaze_origin_scene_x_m=csv_optional_float(row["gaze_origin_scene_x_m"]),
        gaze_origin_scene_y_m=csv_optional_float(row["gaze_origin_scene_y_m"]),
        gaze_origin_scene_z_m=csv_optional_float(row["gaze_origin_scene_z_m"]),
        gaze_point_scene_x_m=csv_optional_float(row["gaze_point_scene_x_m"]),
        gaze_point_scene_y_m=csv_optional_float(row["gaze_point_scene_y_m"]),
        gaze_point_scene_z_m=csv_optional_float(row["gaze_point_scene_z_m"]),
        gaze_dir_scene_unit_x=csv_optional_float(row.get("gaze_dir_scene_unit_x", "")),
        gaze_dir_scene_unit_y=csv_optional_float(row.get("gaze_dir_scene_unit_y", "")),
        gaze_dir_scene_unit_z=csv_optional_float(row.get("gaze_dir_scene_unit_z", "")),
        validation_notes=row["validation_notes"],
    )


def csv_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid boolean value in CSV: {value!r}")


def csv_int(value: str) -> int:
    return int(value)


def csv_optional_int(value: str) -> int | None:
    return int(value) if value else None


def csv_optional_float(value: str) -> float | None:
    return float(value) if value else None


def default_summary_json_path(csv_path: os.PathLike[str] | str) -> Path:
    """Return the default summary JSON path paired with a gaze CSV."""

    csv_file = Path(csv_path)
    stem = csv_file.stem
    if stem.endswith("_gaze_samples"):
        stem = stem[: -len("_gaze_samples")] + "_gaze_summary"
    else:
        stem = f"{stem}_summary"
    return csv_file.with_name(f"{stem}.json")


def summarize_gaze_samples(samples: Sequence[GazeSample]) -> dict[str, Any]:
    """Compute a lightweight per-sequence quality summary from gaze samples.

    zh-CN:
    这个 summary 只依赖 CSV 里已经有的字段，不生成图片。它的作用是快速回答：
    当前窗口或 sequence 里 gaze 有多少有效、投影有多少落在图像内、常见问题是
    什么、`dt`/`depth` 大概在什么范围。
    """

    sample_count = len(samples)
    if sample_count == 0:
        raise ValueError("No gaze samples to summarize")

    valid_gaze_count = sum(sample.gaze_valid for sample in samples)
    projection_in_image_count = sum(sample.projection_in_image for sample in samples)
    ok_count = sum(sample.validation_notes == "ok" for sample in samples)
    pose_valid_count = sum(sample.pose_valid for sample in samples)
    depth_available_count = sum(
        sample.depth_m is not None and np.isfinite(sample.depth_m) and sample.depth_m > 0
        for sample in samples
    )

    note_counter: Counter[str] = Counter()
    for sample in samples:
        if sample.validation_notes == "ok":
            continue
        for note in sample.validation_notes.split(";"):
            if note:
                note_counter[note] += 1

    return {
        "sample_count": sample_count,
        "query_timestamp_start_ns": samples[0].query_timestamp_ns,
        "query_timestamp_end_ns": samples[-1].query_timestamp_ns,
        "duration_s": (samples[-1].query_timestamp_ns - samples[0].query_timestamp_ns) / 1e9,
        "gaze_valid_count": valid_gaze_count,
        "gaze_valid_ratio": valid_gaze_count / sample_count,
        "projection_in_image_count": projection_in_image_count,
        "projection_in_image_ratio": projection_in_image_count / sample_count,
        "pose_valid_count": pose_valid_count,
        "pose_valid_ratio": pose_valid_count / sample_count,
        "depth_available_count": depth_available_count,
        "depth_available_ratio": depth_available_count / sample_count,
        "ok_count": ok_count,
        "ok_ratio": ok_count / sample_count,
        "validation_note_counts": dict(sorted(note_counter.items())),
        "gaze_dt_ms": describe_optional_numbers(
            [
                sample.gaze_dt_ns / 1e6
                for sample in samples
                if sample.gaze_dt_ns is not None
            ]
        ),
        "pose_dt_ms": describe_optional_numbers(
            [
                sample.pose_dt_ns / 1e6
                for sample in samples
                if sample.pose_dt_ns is not None
            ]
        ),
        "depth_m": describe_optional_numbers(
            [sample.depth_m for sample in samples if sample.depth_m is not None]
        ),
        "yaw_confidence_width_rad": describe_optional_numbers(
            [
                sample.yaw_confidence_width_rad
                for sample in samples
                if sample.yaw_confidence_width_rad is not None
            ]
        ),
        "pitch_confidence_width_rad": describe_optional_numbers(
            [
                sample.pitch_confidence_width_rad
                for sample in samples
                if sample.pitch_confidence_width_rad is not None
            ]
        ),
    }


def describe_optional_numbers(values: Sequence[float | None]) -> dict[str, float | int | None]:
    """Return count/min/max/mean for finite numeric values."""

    finite_values = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if finite_values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(finite_values.size),
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
    }


def write_gaze_summary_json(path: os.PathLike[str] | str, summary: dict[str, Any]) -> None:
    """Write a lightweight gaze quality summary JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def read_gaze_summary_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Read a gaze quality summary JSON if it exists."""

    input_path = Path(path)
    return json.loads(input_path.read_text(encoding="utf-8"))


def write_image(path: os.PathLike[str] | str, image: np.ndarray) -> None:
    """Write an RGB image array to disk."""

    import imageio.v2 as imageio

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, image)


def save_overlay(
    path: os.PathLike[str] | str,
    image: Any,
    sample: GazeSample,
    image_dt_ns: int,
    make_upright: bool,
) -> None:
    """Save one diagnostic overlay image with gaze projection and timestamp info."""

    fig = render_overlay_figure(image, sample, image_dt_ns, make_upright)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    _pyplot().close(fig)


def render_overlay_figure(
    image: Any,
    sample: GazeSample,
    image_dt_ns: int,
    make_upright: bool,
) -> Any:
    """Render one RGB overlay figure for PNG export or video frames."""

    if make_upright:
        image = np.rot90(image, k=3)

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_axis_off()
    if sample.projection_valid and sample.gaze_u_px is not None and sample.gaze_v_px is not None:
        ax.scatter([sample.gaze_u_px], [sample.gaze_v_px], c="red", s=80)
        ax.plot(
            [sample.gaze_u_px - 20, sample.gaze_u_px + 20],
            [sample.gaze_v_px, sample.gaze_v_px],
            color="red",
            linewidth=2,
        )
        ax.plot(
            [sample.gaze_u_px, sample.gaze_u_px],
            [sample.gaze_v_px - 20, sample.gaze_v_px + 20],
            color="red",
            linewidth=2,
        )
    ax.set_title(
        f"t={sample.query_timestamp_ns} ns | gaze_dt={sample.gaze_dt_ns} ns | "
        f"image_dt={image_dt_ns} ns\n{sample.validation_notes}",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def write_scene_rays_plot(path: os.PathLike[str] | str, samples: Sequence[GazeSample]) -> None:
    """Write metric 3D gaze rays in ADT Scene coordinates.

    The axes are forced to equal metric scale. Without that, Matplotlib stretches
    small-range axes and can make consecutive rays look much jumpier than they
    are in meters.

    zh-CN:
    这个图使用 ADT Scene/world frame，单位是米。这里强制 3D 坐标轴等比例显示；
    否则 Matplotlib 会把范围较小的轴拉满，连续几帧的 gaze rays 会看起来跳得
    比实际米制距离更夸张。时间方向通过 CPF/gaze origin 轨迹上的颜色渐变表示，
    不在 3D 图里逐点写数字，避免数字标签被误读成空间点。
    """

    rays = [
        sample
        for sample in samples
        if sample.gaze_origin_scene_x_m is not None and sample.gaze_point_scene_x_m is not None
    ]
    if not rays:
        return

    plt = _pyplot()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    origins = []
    points = []
    for sample in rays:
        origin = np.array(
            [
                sample.gaze_origin_scene_x_m,
                sample.gaze_origin_scene_y_m,
                sample.gaze_origin_scene_z_m,
            ],
            dtype=float,
        )
        point = np.array(
            [
                sample.gaze_point_scene_x_m,
                sample.gaze_point_scene_y_m,
                sample.gaze_point_scene_z_m,
            ],
            dtype=float,
        )
        origins.append(origin)
        points.append(point)
        ax.plot(
            [origin[0], point[0]],
            [origin[1], point[1]],
            [origin[2], point[2]],
            color="red",
            alpha=0.35,
        )

    origins_array = np.vstack(origins)
    points_array = np.vstack(points)
    order_values = np.arange(len(origins_array))
    if len(origins_array) > 1:
        ax.plot(
            origins_array[:, 0],
            origins_array[:, 1],
            origins_array[:, 2],
            color="black",
            linewidth=1.2,
            alpha=0.55,
            label="CPF origin trajectory",
        )
    scatter_origins = ax.scatter(
        origins_array[:, 0],
        origins_array[:, 1],
        origins_array[:, 2],
        c=order_values,
        cmap="viridis",
        s=18,
        label="CPF origin, time order",
    )
    ax.scatter(
        [origins_array[0, 0]],
        [origins_array[0, 1]],
        [origins_array[0, 2]],
        marker="o",
        color="lime",
        s=55,
        edgecolors="black",
        linewidths=0.6,
        label="start origin",
    )
    ax.scatter(
        [origins_array[-1, 0]],
        [origins_array[-1, 1]],
        [origins_array[-1, 2]],
        marker="X",
        color="yellow",
        s=70,
        edgecolors="black",
        linewidths=0.6,
        label="end origin",
    )
    ax.scatter(
        points_array[:, 0],
        points_array[:, 1],
        points_array[:, 2],
        color="red",
        s=14,
        label="gaze point",
    )
    set_axes_equal_3d(ax, np.vstack([origins_array, points_array]))
    ax.set_xlabel("Scene X [m]")
    ax.set_ylabel("Scene Y [m]")
    ax.set_zlabel("Scene Z [m]")
    ax.set_title("Gaze rays in ADT Scene frame (equal metric scale)")
    ax.legend(loc="upper left", fontsize=7)
    fig.colorbar(scatter_origins, ax=ax, label="sample order", shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_reference_frame_scanpath_overlay(path: os.PathLike[str] | str, scanpath: dict[str, Any]) -> None:
    """Write reference-frame scanpath over the reference RGB image."""

    image = scanpath["image"]
    xs = scanpath["xs"]
    ys = scanpath["ys"]
    orders = scanpath["orders"]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.plot(xs, ys, color="white", linewidth=2, alpha=0.75)
    ax.plot(xs, ys, color="black", linewidth=1, alpha=0.8)
    scatter = ax.scatter(
        xs,
        ys,
        c=orders,
        cmap="viridis",
        s=45,
        edgecolors="white",
        linewidths=0.6,
    )
    if len(xs) <= 25:
        for order, u_px, v_px in zip(orders, xs, ys, strict=True):
            ax.text(
                u_px + 4,
                v_px + 4,
                str(order),
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
            )
    if orders[-1] == scanpath["reference_order"]:
        ax.scatter(
            [xs[-1]],
            [ys[-1]],
            marker="*",
            c="yellow",
            s=120,
            edgecolors="black",
            linewidths=0.7,
        )

    ax.set_axis_off()
    ax.set_title(
        "Reference-frame gaze scanpath overlay "
        f"(ref sample={scanpath['reference_order']}, "
        f"in_image={len(xs)}/{scanpath['frame_count']})",
        fontsize=9,
    )
    fig.colorbar(scatter, ax=ax, label="sample order")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_reference_frame_scanpath_clean(path: os.PathLike[str] | str, scanpath: dict[str, Any]) -> None:
    """Write a zoomed clean pixel-coordinate view of the reference-frame scanpath."""

    xs = scanpath["xs"]
    ys = scanpath["ys"]
    orders = scanpath["orders"]
    width = scanpath["image_width"]
    height = scanpath["image_height"]

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f8f8f8")
    ax.plot(xs, ys, color="#333333", linewidth=1.2, alpha=0.65)
    scatter = ax.scatter(
        xs,
        ys,
        c=orders,
        cmap="viridis",
        s=50,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.scatter(
        [xs[0]],
        [ys[0]],
        marker="o",
        color="lime",
        s=90,
        edgecolors="black",
        linewidths=0.7,
        label="start",
    )
    ax.scatter(
        [xs[-1]],
        [ys[-1]],
        marker="X",
        color="yellow",
        s=110,
        edgecolors="black",
        linewidths=0.7,
        label="end",
    )
    if len(xs) <= 25:
        for order, u_px, v_px in zip(orders, xs, ys, strict=True):
            ax.text(u_px + 4, v_px + 4, str(order), fontsize=7, color="#222222")

    x_min, x_max, y_min, y_max = zoomed_pixel_limits(xs, ys, width, height)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.set_xlabel("reference RGB u [px]")
    ax.set_ylabel("reference RGB v [px]")
    ax.set_title(
        "Reference-frame gaze scanpath clean zoom "
        f"(ref sample={scanpath['reference_order']}, "
        f"in_image={len(xs)}/{scanpath['frame_count']})",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(scatter, ax=ax, label="sample order")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def zoomed_pixel_limits(
    xs: Sequence[float],
    ys: Sequence[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Return clipped pixel limits padded around a scanpath."""

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    pad = max(x_range, y_range, 1.0) * 0.35
    pad = max(pad, 50.0)
    return (
        max(0.0, x_min - pad),
        min(float(width), x_max + pad),
        max(0.0, y_min - pad),
        min(float(height), y_max + pad),
    )


def set_axes_equal_3d(ax: Any, points: np.ndarray) -> None:
    """Set 3D plot limits so one unit has the same visual length on all axes."""

    centers = points.mean(axis=0)
    ranges = np.ptp(points, axis=0)
    radius = max(float(ranges.max()) / 2.0, 0.1)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _pyplot() -> Any:
    """Return a non-interactive pyplot module safe for sandboxed runs."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt
