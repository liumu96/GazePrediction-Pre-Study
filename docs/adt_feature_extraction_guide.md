# ADT Feature Extraction Guide / ADT 特征提取指南

这份文档整理“怎么从 ADT 原始数据里提取 feature”。中文解释优先，
但保留官方 API、字段名和坐标系英文名，方便对照 Project Aria 文档。

可复用代码应放在 `src/adt_sandbox/`，这里保留简洁 API map、坐标系说明、
validity checks 和可视化思路。

已经跑通的 gaze-first 实操笔记见
`docs/tutorial_gaze_feature_extraction.md`。

## Official References Checked / 已检查的官方参考

本指南基于本地 `external/projectaria_tools` 里的官方文档和源码：

- `external/projectaria_tools/website/docs/open_datasets/aria_digital_twin_dataset/data_loader.mdx`
- `external/projectaria_tools/website/docs/open_datasets/aria_digital_twin_dataset/data_format.mdx`
- `external/projectaria_tools/website/docs/open_datasets/aria_digital_twin_dataset/visualizers.mdx`
- `external/projectaria_tools/website/docs/data_formats/mps/mps_eye_gaze.mdx`
- `external/projectaria_tools/website/docs/data_formats/mps/slam/mps_trajectory.mdx`
- `external/projectaria_tools/website/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention.mdx`
- `external/projectaria_tools/website/docs/data_utilities/core_code_snippets/eye_gaze_code.mdx`
- `external/projectaria_tools/projects/AriaDigitalTwinDatasetTools/python/AriaDigitalTwinDatasetToolsPyBind.h`
- `external/projectaria_tools/projectaria_tools/utils/viewer_projects_adt.py`

## Provider Setup / 数据读取入口

ADT ground-truth 文件主要通过两个 provider 读取：

- `AriaDigitalTwinDataPathsProvider`：根据 sequence folder 找到所有文件路径。
- `AriaDigitalTwinDataProvider`：加载数据并提供 timestamp query APIs。

```python
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinDataProvider,
)

paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
data_paths = paths_provider.get_datapaths(skeleton_flag=True)
gt_provider = AriaDigitalTwinDataProvider(data_paths)
```

本仓库的推荐入口是：

```python
from adt_sandbox.providers import create_adt_providers

providers = create_adt_providers("Apartment_release_decoration_skeleton_seq131_M1292")
gt_provider = providers.gt_provider
print(providers.provider_mode)  # official_adt
```

`create_adt_providers()` 使用 `AriaDigitalTwinDataPathsProvider` 和
`AriaDigitalTwinDataProvider` 创建官方 provider。tutorial 主要用这些 API：

```python
gt_provider.get_aria_device_capture_timestamps_ns(rgb_stream_id)
gt_provider.get_aria_image_by_timestamp_ns(timestamp_ns, rgb_stream_id)
gt_provider.get_aria_camera_calibration(rgb_stream_id)
gt_provider.get_eyegaze_by_timestamp_ns(timestamp_ns)
gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
gt_provider.raw_data_provider_ptr()
```

常用 metadata / availability checks：

```python
paths_provider.get_scene_name()
paths_provider.get_device_serial_number()
paths_provider.get_num_skeletons()
paths_provider.is_multi_person()
paths_provider.get_concurrent_sequence_name()

gt_provider.has_aria_data()
gt_provider.has_aria_3d_poses()
gt_provider.has_eyegaze()
gt_provider.has_skeleton()
gt_provider.has_object_3d_boundingboxes()
gt_provider.has_instance_2d_boundingboxes()
gt_provider.has_depth_images()
gt_provider.has_segmentation_images()
gt_provider.has_synthetic_images()
gt_provider.has_instances_info()
gt_provider.has_mps()
```

多数 timestamp query 返回 `DataWithDt` wrapper。使用前必须检查
`is_valid()`，并记录 `dt_ns()`：

```python
result = gt_provider.get_eyegaze_by_timestamp_ns(timestamp_ns)
if result.is_valid():
    value = result.data()
    delta_ns = result.dt_ns()
```

`dt_ns()` 的含义是：returned data time - query time。它是后续对齐质量
判断的关键指标。

## Gaze / 视线特征

### API

ADT ground-truth gaze：

```python
eye_gaze_with_dt = gt_provider.get_eyegaze_by_timestamp_ns(timestamp_ns)
```

MPS general gaze 可通过 `gt_provider.mps_data_provider_ptr()` 或直接读取
`mps/eye_gaze/general_eye_gaze.csv`。当前优先从 ADT `eyegaze.csv` 开始，
因为它包含 ADT ground-truth depth。

### Fields / 字段

返回的 gaze 使用 Project Aria MPS 的 `EyeGaze` 类型：

- `yaw`：CPF frame 下的水平角，单位 radians
- `pitch`：CPF frame 下的垂直角，单位 radians
- `depth`：CPF frame 中 3D gaze point 的深度，单位 meters；0 表示没有深度
- `yaw_low`, `yaw_high`：yaw confidence interval
- `pitch_low`, `pitch_high`：pitch confidence interval
- `session_uid`：calibration/session id
- `vergence`：新版本 gaze 模型里的 left/right gaze fields

### Coordinate Frame / 坐标系

- gaze 的 `yaw`、`pitch`、`depth` 在 Central Pupil Frame (CPF) 下。
- CPF 原点在左右 eye boxes 的中点。
- 从佩戴者视角看，CPF X 轴向左，Y 轴向上，Z 轴向前。

CPF 中的 gaze point：

```python
import numpy as np
from math import tan

gaze_point_cpf = np.array(
    [tan(eye_gaze.yaw), tan(eye_gaze.pitch), 1.0],
    dtype=np.float64,
) * eye_gaze.depth
```

也可以用官方 helper：

```python
import projectaria_tools.core.mps as mps

gaze_point_cpf = mps.get_eyegaze_point_at_depth(
    eye_gaze.yaw,
    eye_gaze.pitch,
    depth_m,
)
```

### Projection To RGB / 投影到 RGB 图像

本仓库当前使用 `src/adt_sandbox/gaze.py` 中的手动投影 helper：

```python
from adt_sandbox.gaze import extract_gaze_sample

sample = extract_gaze_sample(gt_provider, timestamp_ns)
print(sample.gaze_u_px, sample.gaze_v_px, sample.projection_in_image)
```

默认 `make_upright=True`，输出坐标对应顺时针旋转 90 度后的 upright RGB 图像；
如果要和原始 VRS 图像方向对齐，显式传 `make_upright=False`。

核心计算：

```python
T_camera_cpf = inverse(T_device_camera) @ T_device_cpf
gaze_point_camera = T_camera_cpf @ gaze_point_cpf
pixel = camera_calibration.project(gaze_point_camera)
```

官方文档里也有 `get_gaze_vector_reprojection` helper。`adt` 环境中的
`projectaria-tools 2.x` 可以正常 import 这个 helper。当前 tutorial 仍保留
显式投影逻辑，是为了把 `T_camera_cpf`、CPF point 和 camera projection 这些
中间步骤暴露出来，方便学习和 debug；后续可以用官方 helper 对照验证结果：

```python
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

rgb_stream_id = StreamId("214-1")
camera_calib = gt_provider.get_aria_camera_calibration(rgb_stream_id)
raw_provider = gt_provider.raw_data_provider_ptr()
device_calib = raw_provider.get_device_calibration()

gaze_projection = get_gaze_vector_reprojection(
    eye_gaze,
    camera_calib.get_label(),
    device_calib,
    camera_calib,
    eye_gaze.depth,
)
```

### Scene-Frame Ray / 转成 Scene frame 中的 3D gaze ray

```python
aria_pose = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns).data()
transform_device_cpf = device_calib.get_transform_device_cpf()
transform_scene_cpf = aria_pose.transform_scene_device @ transform_device_cpf

gaze_origin_scene = transform_scene_cpf @ [0.0, 0.0, 0.0]
gaze_point_scene = transform_scene_cpf @ gaze_point_cpf
```

这里 `transform_scene_device` 是 `T_scene_device`，将 Device frame 中的点
变换到 Scene frame。`transform_device_cpf` 是 `T_device_cpf`。

官方 coordinate convention 文档和 ADT viewer 源码使用 `@` 组合 SE3 transforms。
本仓库按项目 `adt` 环境中的 `projectaria-tools 2.x` API 编写，也直接使用
`SE3 @ SE3` 执行上述矩阵逻辑。运行脚本前应确认 Python 来自
`/home/liumu/miniconda3/envs/adt/bin/python`，不要使用 base 环境里的旧版
`projectaria-tools`。

### Validity Checks / 有效性检查

- `eye_gaze_with_dt.is_valid()`
- `abs(eye_gaze_with_dt.dt_ns())` 是否小于当前任务的阈值
- `yaw`、`pitch` 是否 finite
- 需要 3D gaze point 时，`eye_gaze.depth > 0`
- confidence width 是否异常：
  `yaw_high - yaw_low`、`pitch_high - pitch_low`
- RGB projection 是否非空，并且落在 image bounds 内
- Scene frame 中的 gaze ray 是否相对可见物体合理

### Visualizations / 可视化

- 2D gaze point overlay：在 RGB frame 上画 gaze point
- Reference-frame scanpath：把短事件窗口内的 Scene-frame gaze points 统一
  重投影到最后一个选中 RGB frame，输出 overlay 和 clean zoom 两种 view，
  适合做 event analysis 的视觉对比
- Overlay video：将抽稀后的 RGB overlay frames 合成视频，用于和
  Scene-frame rays 中的用户移动趋势对照
- Fixation scanpath：将 gaze ray 投到稳定参考系，例如 Scene plane、object mesh
  或 object-local frame，再在该参考系上连接 fixation/gaze points
- 3D gaze ray：在 Scene frame 中画从 CPF origin 出发的 ray
- Rerun timeline：同时看 RGB、gaze projection、device pose、skeleton、
  object boxes

## Aria Device Pose / Aria 设备位姿

### API

```python
from projectaria_tools.projects.adt import (
    get_interpolated_aria_3d_pose_at_timestamp_ns,
)

pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
pose_interp = get_interpolated_aria_3d_pose_at_timestamp_ns(
    gt_provider,
    timestamp_ns,
)
```

### Fields / 字段

- `transform_scene_device`：`T_scene_device`
- `device_linear_velocity`
- `device_rotational_velocity`
- `gravity_world`
- `graph_uid`
- `quality_score`

### Coordinate Frame / 坐标系

- ADT ground-truth pose 在 Scene/world frame 下。
- `T_scene_device` 将 Device frame 中的点变换到 Scene frame。

### Checks / 检查

- `pose_with_dt.is_valid()`
- `abs(pose_with_dt.dt_ns())` 是否小于阈值
- `quality_score` 是否存在且合理
- nearest pose 与 interpolated pose 是否在短时间间隔内一致

### Visualizations / 可视化

- Scene frame 中的 device trajectory
- device pose + RGB camera frustum
- 与 gaze ray、skeleton、object boxes 一起可视化

## Skeleton Pose / 人体骨架

### API

```python
skeleton_ids = gt_provider.get_skeleton_ids()
skeleton_with_dt = gt_provider.get_skeleton_by_timestamp_ns(
    timestamp_ns,
    skeleton_ids[0],
)

skeleton_provider = gt_provider.get_skeleton_provider(skeleton_ids[0])
joint_labels = skeleton_provider.get_joint_labels()
joint_connections = skeleton_provider.get_joint_connections()
marker_labels = skeleton_provider.get_marker_labels()
```

### Coordinate Frame / 坐标系

- ADT skeleton markers 和 joints 都在 Scene frame 下。

### Checks / 检查

- `skeleton_with_dt.is_valid()`
- `abs(skeleton_with_dt.dt_ns())` 是否小于阈值
- joint count 是否符合 labels
- marker position `[0, 0, 0]` 表示 occluded/missing，需要当作缺失值

### Visualizations / 可视化

- 3D joints + limb connections
- skeleton 与 gaze ray、object boxes 放在同一 Scene frame
- skeleton 2D boxes：
  `get_skeleton_2d_boundingboxes_by_timestamp_ns`

## Objects And Bounding Boxes / 物体与 bounding boxes

### Instance Metadata / 实例元数据

```python
instance_ids = gt_provider.get_instance_ids()
object_ids = gt_provider.get_object_ids()
info = gt_provider.get_instance_info_by_id(object_ids[0])
```

常用字段：

- `id`
- `name`
- `prototype_name`
- `category`
- `instance_type`
- `motion_type`
- `rigidity_type`
- `rotational_symmetry`
- `canonical_pose`

### 3D Boxes / 3D 物体框

```python
bboxes3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
    timestamp_ns
)
bbox3d = bboxes3d_with_dt.data()[object_id]
transform_scene_object = bbox3d.transform_scene_object
aabb = bbox3d.aabb
```

- `transform_scene_object` 是 `T_scene_object`
- `aabb` 是 object local frame 下的 `[xmin, xmax, ymin, ymax, zmin, zmax]`

### 2D Boxes / 2D 图像框

```python
from projectaria_tools.core.stream_id import StreamId

rgb_stream_id = StreamId("214-1")
bboxes2d_with_dt = gt_provider.get_object_2d_boundingboxes_by_timestamp_ns(
    timestamp_ns,
    rgb_stream_id,
)
bbox2d = bboxes2d_with_dt.data()[object_id]
box_range = bbox2d.box_range
visibility_ratio = bbox2d.visibility_ratio
```

### Helper APIs / 辅助 API

```python
from projectaria_tools.projects.adt import (
    bbox2d_to_image_coordinates,
    bbox2d_to_image_line_coordinates,
    bbox3d_to_coordinates,
    bbox3d_to_line_coordinates,
    get_interpolated_object_3d_boundingboxes_at_timestamp_ns,
)
```

### Checks / 检查

- provider result 是否 `is_valid()`
- `abs(dt_ns())` 是否小于阈值
- object id 是否存在于 instance metadata
- 2D `visibility_ratio > 0` 才说明有可见部分
- 3D AABB 尺寸是否为正
- 3D box 投影到 RGB 后，是否和 2D box 大致一致

### Visualizations / 可视化

- RGB frame 上画 2D boxes
- Scene frame 中画 3D boxes
- dynamic object trajectories
- 如果有 object library，可以用 Rerun 加载 object meshes

## Images, Depth, And Segmentation / 图像、深度和分割

### RGB Timestamps / RGB 时间戳

```python
from projectaria_tools.core.stream_id import StreamId

rgb_stream_id = StreamId("214-1")
timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(rgb_stream_id)
```

### Image Queries / 图像查询

```python
image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(timestamp_ns, rgb_stream_id)
depth_with_dt = gt_provider.get_depth_image_by_timestamp_ns(timestamp_ns, rgb_stream_id)
seg_with_dt = gt_provider.get_segmentation_image_by_timestamp_ns(timestamp_ns, rgb_stream_id)
synth_with_dt = gt_provider.get_synthetic_image_by_timestamp_ns(timestamp_ns, rgb_stream_id)
```

### Data Conversion / 转 numpy 或可视化图像

```python
image_np = image_with_dt.data().to_numpy_array()
depth_np = depth_with_dt.data().to_numpy_array()
seg_np = seg_with_dt.data().to_numpy_array()
seg_vis = seg_with_dt.data().get_visualizable()
depth_vis = depth_with_dt.data().get_visualizable()
```

### Depth Semantics / 深度语义

- ADT depth images 存的是 camera Z-axis depth，单位是 millimeters。
- 它不是沿 pixel ray 的距离，这一点和一些合成数据集的 depth 定义不同。

### Checks / 检查

- query result 是否 `is_valid()`
- image shape 是否和 calibration image size 一致
- segmentation ids 是否能对应到 `instances.json`
- depth values 在有效区域是否为正

## Alignment Policy / 对齐策略

ADT provider 默认按 closest timestamp query。每次 query 返回的 `dt_ns()`
就是被选中的数据时间与 query timestamp 的差值。

建议流程：

- 每个任务选择一个 anchor stream
- 用 anchor timestamp query 其它 streams
- 保存每个 stream 的 `dt_ns()`
- Aria pose 和 3D object boxes 优先使用官方 interpolation helpers
- required stream invalid 或 `dt_ns()` 超阈值时，reject 或 flag sample

常见 anchor choices：

- RGB frame timestamps：用于 image overlay 和 visual debugging
- gaze timestamps：用于 gaze-centered feature tables
- fixed-rate timestamps：用于 sequence-level summaries

multi-person 或 concurrent sequences 中，可用 device-time/timecode conversion：

```python
timecode_ns = gt_provider.get_timecode_from_device_time_ns(device_time_ns)
device_time_ns = gt_provider.get_device_time_from_timecode_ns(timecode_ns)
```

## Visualization Tools / 可视化工具

官方 Python visualizer：

```bash
viewer_projects_adt --sequence_path /mnt/d/Pose2Gaze-ADT/<sequence_id>
```

官方 Rerun viewer 会记录：

- RGB image
- device pose and trajectory
- gaze projection and 3D gaze ray
- skeletons
- 3D object boxes and optional object meshes

本仓库的本地可视化目标：

- gaze CSV 和小型 JSON/CSV summaries 放 `outputs/reports/`
- 可复用 RGB frames、overlay frames、scanpath、scene_rays 等可视化输出放
  `outputs/figures/`
- 大型 Rerun `.rrd` 文件放 repo 外或 ignored outputs

## First Local Implementation Target / 第一个本地实现目标

已经先从 gaze 开始，因为它覆盖了最容易踩坑的部分：

- provider loading
- timestamp query 和 `dt_ns()`
- CPF coordinate frame
- camera calibration and projection
- Scene-frame transform
- 2D 与 3D visualization

已实现文件：

- `src/adt_sandbox/providers.py`
- `src/adt_sandbox/gaze.py`
- `scripts/extract_gaze_samples.py`
- `scripts/visualize_gaze_outputs.py`
- `docs/tutorial_gaze_feature_extraction.md`

下一步文件：

- `scripts/check_gaze_quality.py`
- `notebooks/01_gaze_feature_extraction.ipynb`
