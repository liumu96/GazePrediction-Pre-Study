"""Gaze extraction, validation, projection, and visualization helpers.

The helpers here keep one timestamp query inspectable end to end: raw gaze in
CPF, nearest pose in Scene frame, projection to RGB pixels, and validity notes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import tan
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
    depth_m: float | None                           # Depth in meters.
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
    validation_notes: str                           # Semicolon-separated notes on data validity and projection results, or "ok" if no issues.

    def as_csv_row(self) -> dict[str, Any]:
        """Return a flat row suitable for csv.DictWriter."""

        return asdict(self)


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
    eye_gaze 里包含 yaw、pitch 和 depth。yaw/pitch 是 CPF 坐标系下相对前向
    Z 轴的角度偏移，depth 是沿 gaze ray 使用的深度。如果传入 depth_m，就用
    depth_m 覆盖 eye_gaze.depth。最终返回 CPF 坐标系下的 3D gaze point：
    [tan(yaw), tan(pitch), 1] * depth。
    """

    depth = eye_gaze.depth if depth_m is None else depth_m
    # In CPF, yaw and pitch are angular offsets from the forward Z axis.
    return (
        np.array([tan(eye_gaze.yaw), tan(eye_gaze.pitch), 1.0], dtype=np.float64)
        * depth
    )


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
    这个函数把 CPF 下的 gaze point 投影到 RGB camera 的图像平面。步骤是：
    1. 读取 RGB camera calibration 和 device calibration。
    2. 检查 depth 是否有效；depth <= 0 时无法构造 3D gaze point。
    3. 用 T_camera_cpf = inverse(T_device_camera) @ T_device_cpf，把 CPF 中的
       gaze point 变换到 camera frame。
    4. 调用 camera_calibration.project 得到 RGB 像素坐标。

    make_upright=True 时，projection 坐标对应顺时针旋转 90 度后的 upright RGB
    图像；显示 overlay 时图像也要同步旋转。
    """

    if eye_gaze.depth <= 0:
        _, _, width_height = _rgb_camera_context(
            gt_provider,
            stream_id_value,
            make_upright,
        )
        return None, width_height

    camera_calibration, transform_device_camera, width_height = _rgb_camera_context(
        gt_provider,
        stream_id_value,
        make_upright,
    )
    transform_camera_cpf = (
        transform_device_camera.inverse()
        @ gt_provider.raw_data_provider_ptr().get_device_calibration().get_transform_device_cpf()
    )
    # Camera calibration projection expects a 3D point in the camera frame.
    gaze_center_camera = transform_camera_cpf @ gaze_point_cpf(eye_gaze)
    projection = camera_calibration.project(gaze_center_camera)
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
        validation_notes=reason,
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


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
