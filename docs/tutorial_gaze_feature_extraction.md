# Tutorial 01: Gaze Feature Extraction / 视线特征提取

这份笔记记录当前仓库里已经实现并验证过的 gaze-first workflow。目标不是
一次性做完整 preprocessing，而是先回答：

- gaze 从哪里读？
- 它在什么坐标系下？
- 怎么判断一条 gaze sample 是否可用？
- 怎么投影到 RGB 图像？
- 怎么转成 Scene frame 下的 3D gaze ray？
- 怎么快速生成可检查的 CSV、quality summary 和按需可视化？

当前实现对应：

- `src/adt_sandbox/providers.py`
- `src/adt_sandbox/gaze.py`
- `scripts/extract_gaze_samples.py`

## Local Data Compatibility / 本地数据兼容性

你的数据来自 Pose2Gaze 已下载的 ADT 子集：

```bash
ADT_DATA_ROOT=/mnt/d/Pose2Gaze-ADT
```

实测当前 sequence metadata 是 `dataset_name=ADT_2023`、
`dataset_version=2.0`。在项目 `adt` 环境的 `projectaria-tools 2.x` 中，
官方 ADT provider 可以读取这份 Pose2Gaze-ADT 数据，因此本仓库默认使用：

```text
provider_mode=official_adt
```

它通过官方 provider 读取：

- `video.vrs`：RGB image timestamps、image data、camera calibration
- `eyegaze.csv`：ADT ground-truth gaze
- `aria_trajectory.csv`：Aria device pose
- `metadata.json`：sequence metadata

## Run The Tutorial Script / 运行脚本

先确认 `.env` 里有：

```bash
ADT_DATA_ROOT=/mnt/d/Pose2Gaze-ADT
```

从仓库根目录运行一个短窗口：

```bash
python scripts/extract_gaze_samples.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --start-index 900 \
  --end-index 905 \
  --stride 1
```

参数含义：

- `sequence`：sequence id，脚本会在 `ADT_DATA_ROOT` 下解析。
- `--start-index/--end-index`：在当前可用 RGB timestamps 上选择帧索引区间。
  `end-index` 是右开边界；例如 `--start-index 900 --end-index 905 --stride 1`
  会选择第 900、901、902、903、904 这 5 帧。
- `--start-offset-s/--end-offset-s`：按 sequence 内相对秒数选择时间区间。
  offset 是相对于 annotation range filtering 后第一个可用 RGB timestamp，
  例如 `--start-offset-s 30 --end-offset-s 32` 表示从当前 sequence 可用时间轴
  的第 30 秒到第 32 秒。
- `--stride`：RGB frame timestamp 的步长。RGB 约 30 fps 时，
  默认 `--stride 1`，会保留每个可用 RGB timestamp；`--stride 30` 约等于
  1 Hz；`--stride 300` 约等于 10 秒一个点。
- 如果没有指定 `--end-index` 或 `--end-offset-s`，脚本会从 `start-index`
  一直处理到当前 sequence 可用 RGB 时间轴的末尾。
- 默认输出 upright RGB 图像和对应的 upright gaze projection；如果需要保留
  Aria 传感器原始方向，使用 `--raw-image-orientation`。

一次已验证运行的摘要：

```text
sequence: Apartment_release_decoration_skeleton_seq131_M1292
provider_mode: official_adt
image_orientation: upright
samples: 5 gaze_valid=5 projection_in_image=5 ok=5
selected_timestamps_ns: 267877629193250..267877762503625 duration_s=0.133
csv: outputs/reports/Apartment_release_decoration_skeleton_seq131_M1292_gaze_samples.csv
summary_json: outputs/reports/Apartment_release_decoration_skeleton_seq131_M1292_gaze_summary.json
```

`outputs/` 下的生成内容会被 git 忽略。

## Re-visualize Existing CSV / 复用已有 CSV 重新可视化

如果已经用 `extract_gaze_samples.py` 生成过 CSV，后面想围绕一个 event/window
生成 scanpath、scene rays、overlay frames 和 overlay video，使用
`scripts/visualize_gaze_outputs.py`。它会先读取已有 CSV，再只为当前窗口打开
ADT provider 查询所需的 RGB image。

```bash
python scripts/visualize_gaze_outputs.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --start-row 0 \
  --end-row 60 \
  --stride 10 \
  --run-name stride_10
```

常用参数：

- `--start-row/--end-row`：从 CSV 中选一个行区间作为 event/window。
- `--stride`：统一抽稀 scanpath、scene_rays、overlay frames 和 overlay video。
- `--run-name`：把这次可视化写入单独子目录，方便比较不同参数。

## What The Script Does / 脚本流程

1. 解析 sequence path：

```python
providers = create_adt_providers(sequence)
```

默认返回官方 ADT provider。tutorial 主要用这些查询接口：

```python
gt_provider.get_aria_device_capture_timestamps_ns(rgb_stream_id)
gt_provider.get_aria_image_by_timestamp_ns(timestamp_ns, rgb_stream_id)
gt_provider.get_aria_camera_calibration(rgb_stream_id)
gt_provider.get_eyegaze_by_timestamp_ns(timestamp_ns)
gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
gt_provider.raw_data_provider_ptr()
```

2. 读取 RGB capture timestamps，并限制到 gaze/pose annotation 时间范围内。

3. 对每个 selected timestamp 查询 nearest gaze 和 nearest pose。

4. 计算 gaze validity、projection、Scene-frame ray。

5. 写出 CSV 和轻量 `gaze_summary.json`。
6. 需要图片或视频时，再用 `scripts/visualize_gaze_outputs.py` 读取 CSV，
   只对当前选中窗口生成 scene_rays、scanpath、overlay frames 和 overlay video。

## Gaze Fields / Gaze 字段

当前优先使用 sequence 根目录下的 `eyegaze.csv`，而不是
`mps/eye_gaze/general_eye_gaze.csv`，因为 ADT ground-truth gaze 中有
`depth_m`：

```text
tracking_timestamp_us
yaw_rads_cpf
pitch_rads_cpf
depth_m
yaw_low_rads_cpf
pitch_low_rads_cpf
yaw_high_rads_cpf
pitch_high_rads_cpf
```

核心字段：

- `tracking_timestamp_us`：device capture time，微秒；代码里转成 ns。
- `yaw_rads_cpf`、`pitch_rads_cpf`：CPF frame 下的 gaze angle。
- `depth_m`：从 CPF origin 沿 gaze ray 前进的距离，不是 CPF 的 `z` 坐标；
  `<= 0` 不能构造 3D gaze point。
- `yaw_low/high`、`pitch_low/high`：confidence interval。

## Coordinate Frames / 坐标系

gaze 原始角度在 Central Pupil Frame (CPF) 下：

- CPF 原点在左右 eye boxes 的中点。
- 从佩戴者视角看，CPF X 向左，Y 向上，Z 向前。

CPF 中的 3D gaze point：

```python
from projectaria_tools.core import mps

gaze_point_cpf = mps.get_eyegaze_point_at_depth(
    eye_gaze.yaw,
    eye_gaze.pitch,
    eye_gaze.depth,
)
```

这里要特别注意：`depth_m` 采用的是官方 helper 的语义，表示“沿 gaze ray 的
距离”。因此 `gaze_point_cpf` 与 `gaze_dir_cpf_unit * depth_m` 同方向，但不等于
简单的 `[tan(yaw), tan(pitch), 1] * depth_m`。

CPF 下的 unit gaze direction：

```python
gaze_dir_cpf_unit = normalize(
    np.array([tan(eye_gaze.yaw), tan(eye_gaze.pitch), 1.0], dtype=np.float64)
)
```

投影到 RGB 时使用：

```text
pixel = mps.utils.get_gaze_vector_reprojection(
    eye_gaze=eye_gaze,
    stream_id_label=camera_calibration.get_label(),
    device_calibration=device_calibration,
    camera_calibration=camera_calibration,
    depth_m=eye_gaze.depth,
    make_upright=make_upright,
)
```

当前仓库直接调用官方 helper，不再自己维护一份本地复刻的投影链。

转成 Scene frame 时使用：

```text
T_scene_cpf = T_scene_device @ T_device_cpf
gaze_origin_scene = T_scene_cpf @ [0, 0, 0]
gaze_point_scene = T_scene_cpf @ gaze_point_cpf
gaze_dir_scene_unit = normalize((T_scene_cpf @ gaze_dir_cpf_unit) - gaze_origin_scene)
```

和 HAGI / HAGI++ 的关系：

- HAGI / HAGI++ 论文把 gaze 表示成 `(pitch, yaw)` 序列，并把 head movement
  表示成以 eye tracker 为参考的相对运动，而不是 world-frame gaze。
- HAGI++ 明确写到 gaze 和 head 都是在 `tracker-centric coordinate system`
  下表示；因此这类模型学习的是 `eye-in-head / eye-in-tracker` 的局部 gaze，
  再结合 head pose 才能还原成 world/scene 下的 gaze ray。
- 对当前 ADT CSV 来说，`yaw_rad/pitch_rad/depth_m` 更具体地说是在 `CPF`
  下；CPF 是刚性附着在眼镜/头部上的局部坐标系，所以从建模语义上属于
  tracker-centric / head-centric，而不是 world-centric。

注意：本项目按 `adt` 环境中的 `projectaria-tools 2.x` API 编写。官方文档和
本地 `external/projectaria_tools` 源码使用
`transform_a_c = transform_a_b @ transform_b_c` 这类写法；ADT viewer 里也有
`T_scene_cpf = aria_3d_pose.transform_scene_device @ transform_device_cpf`。
因此本仓库也直接使用 `SE3 @ SE3` 做 transform composition，不再为 base 环境里
旧版 `projectaria-tools` 增加兼容层。运行脚本前应确认 Python 来自
`/home/liumu/miniconda3/envs/adt/bin/python`。

## Validity Checks / 有效性检查

`scripts/extract_gaze_samples.py` 每条 sample 会输出：

- `gaze_valid`：gaze query 是否成功。
- `gaze_dt_ns`：nearest gaze timestamp - query timestamp。
- `pose_valid`：pose query 是否成功。
- `pose_dt_ns`：nearest pose timestamp - query timestamp。
- `depth_m`：是否 `> 0`。
- `yaw_confidence_width_rad`、`pitch_confidence_width_rad`：
  confidence interval width。
- `projection_valid`：能否投影到 RGB camera plane。
- `projection_in_image`：投影点是否落在 RGB image bounds 内。
- `validation_notes`：汇总问题；没有问题时为 `ok`。

常见 `validation_notes`：

- `gaze_dt_exceeds_threshold`
- `yaw_or_pitch_not_finite`
- `depth_not_available`
- `confidence_width_not_finite`
- `projection_failed`
- `projection_outside_image`
- `scene_ray_unavailable`

当前默认阈值：

```bash
--max-dt-ms 20.0
```

这表示 nearest gaze timestamp 与 query timestamp 的差值超过 20 ms 时会被标记，
但不会直接丢弃。探索阶段先保留 row，后续做 feature table 时再决定 reject
policy。

## Output CSV / 输出表

CSV 默认写到：

```text
outputs/reports/<sequence_id>_gaze_samples.csv
```

重要列：

- `query_timestamp_ns`
- `yaw_rad`, `pitch_rad`, `depth_m`
- `gaze_dir_cpf_unit_x/y/z`
- `gaze_u_px`, `gaze_v_px`
- `image_width_px`, `image_height_px`
- `gaze_origin_scene_x_m/y_m/z_m`
- `gaze_point_scene_x_m/y_m/z_m`
- `gaze_dir_scene_unit_x/y/z`
- `pose_quality_score`
- `validation_notes`

如果 `projection_in_image=True`，`gaze_u_px/gaze_v_px` 可以直接用于对应帧的
RGB overlay。若要做 3D 分析，使用 Scene-frame origin/point 构造 ray。

轻量 summary JSON 默认写到：

```text
outputs/reports/<sequence_id>_gaze_summary.json
```

其中会包含：

- `gaze_valid_ratio`
- `projection_in_image_ratio`
- `pose_valid_ratio`
- `depth_available_ratio`
- `validation_note_counts`
- `gaze_dt_ms` / `pose_dt_ms` / `depth_m` 的 count/min/max/mean
- `field_coordinate_frames`：当前 CSV 关键字段所在坐标系，例如
  `yaw_rad/pitch_rad/depth_m` 和 `gaze_dir_cpf_unit_*` 在 CPF 下，
  `gaze_u_px/gaze_v_px` 在 RGB image plane，
  `gaze_origin_scene_* / gaze_point_scene_* / gaze_dir_scene_unit_*`
  在 ADT Scene frame 下
- `field_definitions`：关键字段的语义说明，尤其是 `depth_m` 表示“沿 gaze ray 的
  距离”，不是 CPF 的 `z` 坐标
- `source_counts`：`eyegaze.csv` 行数、原始 RGB timestamp 数、annotation
  过滤后的 RGB 数、offset 过滤后的 RGB 数、最终 selected RGB 数
- `source_time_ranges_ns`：上述各阶段的时间范围，以及 provider annotation 的
  `start/end`，用于解释为什么某个 sequence 会出现
  `Loaded #EyeGazes != samples`

## Visualization / 可视化解释

RGB overlay：

- 红点表示 projected gaze point。
- 红色十字只是 debug marker，不代表 uncertainty ellipse。
- title 中的 `gaze_dt` 和 `image_dt` 用来检查 timestamp 对齐。
- 默认 overlay 会把 RGB 图像顺时针旋转 90 度显示为 upright；gaze projection
  也按同一方向计算，因此不需要偏头看。

3D Scene-frame rays：

- CPF origin 在 Scene frame 中的位置使用颜色渐变表示 sample order。
- 绿色圆点是当前窗口的起始 origin，黄色 X 是结束 origin。
- 红线从 CPF origin 指向 gaze point，红点是 gaze point。
- 这是 sanity-check 图，不等同于完整 scene rendering。

Reference-frame scanpath：

- reference frame 默认是当前选中可视化窗口的最后一个 RGB frame。
- 每个 sample 先使用 `gaze_point_scene_*` 表示 Scene/world frame 中的 3D
  gaze point，再被统一投影到 reference frame 的 RGB camera 上。
- `gaze_reference_frame_scanpath_overlay.png` 把 scanpath 画在 reference RGB
  背景上，用于判断 gaze 是否落在合理物体或区域。
- `gaze_reference_frame_scanpath_clean.png` 去掉 RGB 背景，并自动 zoom 到
  scanpath 附近，用于更清楚地看轨迹形状、方向和跳变。
- 它仍然依赖 `depth_m` 的可靠性；如果早期 gaze point 在最后一帧相机背后、
  在画面外，或对应的真实物体已经移动，图上会缺点或产生解释偏差。

## Current Limitations / 当前限制

- 目前只做 gaze-first extraction 和离线可视化，还没有进入批量质量报告。
- pose 使用 nearest timestamp，还没有实现 interpolation comparison。
- reference-frame scanpath 适合短时间、受控事件窗口的可视化对比，但还不是
  最终 fixation feature；正式分析仍应考虑 Scene/object/mesh 等稳定参考系。
- 3D 图没有叠加 mesh、object boxes、skeleton，因此只能看 ray 的粗略方向。
- 当前没有直接比较 ADT `eyegaze.csv` 与 MPS `general_eye_gaze.csv`。

## Next Steps / 下一步

建议按这个顺序继续：

1. 用同一脚本多跑几个 sequence，统计 `validation_notes` 分布。
2. 增加 `scripts/check_gaze_quality.py`，批量输出每个 sequence 的
   valid ratio、projection ratio、depth coverage、dt distribution。
3. 写 `notebooks/01_gaze_feature_extraction.ipynb`，读取 CSV、summary 和按需
   生成的 figures，做交互式检查。
4. 为 fixation 分析设计真正的 scanpath 表示：先把 gaze ray 投到稳定参考系，
   例如 Scene frame 中的墙面/桌面、object mesh、3D bounding box，或某个物体的
   local coordinate frame，再在该参考系上连接 fixation/gaze points。
5. 在 event analysis 阶段补充 device/CPF origin 的上下文图，例如
   `origin_xyz_vs_time`、`origin_xy_topdown`、`origin_xz_sideview`，
   用来区分 gaze 变化和 ego-motion 的贡献；这类图先不作为当前默认输出。
6. 扩展 pose interpolation，再把 skeleton/object boxes 叠加到同一
   Scene-frame visualization。
