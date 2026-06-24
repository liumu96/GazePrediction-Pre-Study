# Gaze Event Analysis Notes / Gaze Event 分析笔记

这份文档记录当前对 gaze event 的最新结论。旧版 CPF-based `IDT / IVT /
combined` fixation pipeline 已经从主线移除；当前主线新增第一版
scene-direction event。

## 1. 当前结论

CPF-local velocity / dispersion 有用，但 CPF-thresholded fixation labels 不适合作为
最终 event label。

原因：

- CPF 表示 eye-in-device / eye-in-head。
- CPF gaze 稳定只说明眼睛相对设备稳定。
- 如果同时 head rotation 很大，scene gaze ray 可能仍然在扫过世界。
- 因此 CPF fixation 与直观的 scene/object fixation 语义不一致。

所以当前仓库保留：

- CPF-local gaze dynamics features
- head dynamics features
- head-gaze continuous relationship analysis
- scene-direction event labels

移除：

- CPF-based `IDT / IVT / combined` fixation labels
- fixation policy comparison
- CPF fixation viewer

## 2. 保留的 CPF dynamics

保留这些量，因为它们对分析 eye/head 分工有价值：

- `local_angle_step_deg`
- `local_velocity_deg_s`
- `delta_yaw_rad`
- `delta_pitch_rad`
- `window_dispersion_deg`

它们回答：

```text
眼睛相对 device/head 的局部运动有多大？
```

但它们不回答：

```text
用户是否在 scene/object 层面稳定看一个目标？
```

## 3. 当前脚本

计算 CPF-local gaze dynamics：

```bash
python scripts/compute_gaze_dynamics_features.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured
```

输出：

- `*_gaze_dynamics.csv`
- `*_gaze_dynamics_summary.json`
- `batch_gaze_dynamics_summary.csv`

这些是 feature outputs，不是 event labels。

## 4. Scene-direction event / 场景方向事件检测

### 4.1 Event Definition / 事件定义

中文定义：

当前使用的 event label 定义为 **Scene/world 坐标系下的 gaze direction 稳定性**。
它回答的问题是：

```text
用户视线在世界方向上是否稳定？
```

因此这里的 `fixation / transition` 是 scene-direction label，不是人工标注的
object-level fixation，也不是 CPF 坐标系下的 eye-in-head fixation。

English definition:

The current event label is defined by **gaze-direction stability in the
Scene/world coordinate frame**. It asks:

```text
Is the user's gaze direction stable in the world?
```

Therefore, `fixation / transition` here means a scene-direction label. It is not
a manually annotated object-level fixation label, and it is not a CPF-local
eye-in-head fixation label.

### 4.2 Why Not Directly Use a Standard Detector? / 为什么不直接套标准算法？

中文说明：

标准 gaze event detector 通常可以分为 velocity-based、dispersion-based 或混合方法，
例如 `I-VT`、`I-DT` 及其变体。这些方法本身是有价值的，但不能直接解决当前 ADT
分析里的语义问题：

- 如果在 CPF / eye-in-head 坐标系中运行，检测到的是眼睛相对头部或设备是否稳定。
- 在头部快速旋转时，CPF gaze 可能很稳定，但 world gaze ray 仍然在扫过场景。
- 如果在 image plane 中运行，结果会受到相机运动、投影和可见区域变化影响。
- 这些标准 detector 通常不会直接给出 object-level fixation；object-level 仍需要
  object hit、object identity 和持续时间约束。

所以当前做法不是放弃标准思想，而是把常见的 velocity / dispersion 判据应用到
Scene/world gaze direction 上。更准确地说，它是：

```text
an adapted rule-based scene-direction event labeling procedure
```

English explanation:

Standard gaze event detectors are commonly velocity-based, dispersion-based, or
hybrid methods, such as `I-VT`, `I-DT`, and their variants. These methods are
useful, but applying them directly does not resolve the semantic issue in the
current ADT setting:

- In the CPF / eye-in-head coordinate frame, they detect whether the eyes are
  stable relative to the head or device.
- During fast head rotation, CPF gaze can remain stable while the world-frame
  gaze ray still sweeps across the scene.
- In the image plane, labels can be affected by camera motion, projection, and
  changing visibility.
- These standard detectors do not by themselves produce object-level fixation
  labels; object-level labels still require object hits, object identity, and
  duration constraints.

The current method therefore keeps the standard velocity / dispersion idea, but
applies it to Scene/world gaze direction. It should be described as:

```text
an adapted rule-based scene-direction event labeling procedure
```

### 4.3 Current Implementation / 当前实现

代码入口：

```bash
python scripts/detect_scene_gaze_events.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured
```

当前状态：`/mnt/d/SparseGaze/ADT-Gaze-structured` 已经完成全量 scene-direction event
导出，包含逐帧 feature、逐帧 label、event segments 和 batch summary。

实现步骤：

1. 读取每一帧 GT gaze sample 里的 `gaze_dir_scene_unit_xyz`。
2. 在 Scene/world 坐标系中计算逐帧 gaze dynamics：

- `scene_angle_step_deg`
- `scene_velocity_deg_s`
- `scene_window_dispersion_deg`

其中：

- `scene_angle_step_deg` 是当前帧和上一帧 Scene gaze direction 的夹角。
- `scene_velocity_deg_s` 是 `scene_angle_step_deg / dt`。
- `scene_window_dispersion_deg` 是当前帧附近 centered window 内的 Scene gaze
  direction dispersion，默认 `dispersion_window_frames = 5`。

3. 使用 velocity + dispersion 双阈值生成 preliminary frame label。

默认 fixation candidate 判据：

```text
scene_velocity_deg_s <= 40
and scene_window_dispersion_deg <= 2.5
```

如果 gaze 无效，label 为 `invalid`；如果不满足 fixation candidate 条件，label 为
`transition`。

4. 对连续 fixation run 做最短时长过滤。

默认最短 fixation 时长：

```text
min_fixation_duration_ms = 133
```

因此最终 fixation 必须同时满足：

```text
scene_velocity_deg_s <= 40 deg/s
scene_window_dispersion_deg <= 2.5 deg
segment_duration_ms >= 133 ms
```

短于该阈值的 fixation run 会被降级为 `transition`。

输出：

- `*_scene_gaze_event_features.csv`
- `*_scene_gaze_frame_labels.csv`
- `*_scene_gaze_event_segments.csv`
- `*_scene_gaze_event_summary.json`
- `batch_scene_gaze_event_summary.csv`

窗口级可视化：

```bash
python visualization/visualize_scene_gaze_events.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured \
  --start-frame 0 \
  --end-frame 600
```

这张图不重新计算 label，只读取 scene event CSV，展示最终 label、scene
angular velocity 和 scene angular dispersion 的时间关系。它适合检查：

- 某段为什么被标为 `fixation` 或 `transition`
- velocity / dispersion 阈值附近是否存在边界样本
- 后续如果调阈值，label 是否按预期变化

### 4.4 Paper Wording / 论文写法建议

论文中建议使用：

```text
rule-based scene-direction event labels
```

或：

```text
pseudo event labels derived from world-frame gaze-direction stability
```

避免写成：

```text
ground-truth fixation labels
standard fixation detector
object-level fixation annotations
```

Recommended English wording:

```text
We derive rule-based scene-direction fixation/transition labels from
world-frame gaze directions. These labels are used for diagnostic and
event-conditioned analysis, rather than as ground-truth object-level fixation
annotations.
```

真正的 object-level event 需要在此基础上再结合 ray-box hit、hit object id、
object consistency 和 duration 等条件。

## 5. Object-level event 下一步怎么做

如果后面继续做更直观的 object/target fixation，应该在 scene-direction event
之上再引入：

- image projection stability
- object hit / scene intersection stability

推荐输出命名：

- `*_object_gaze_events.csv`
- `*_object_fixation_frame_labels.csv`
- `*_object_fixation_segments.csv`

这样可以明确区分：

```text
CPF dynamics = local eye/device motion diagnostics
scene-direction event = world-direction gaze stability
object fixation = object/task-level gaze event
```

## 6. 对现有分析的影响

之前基于 CPF fixation/non-fixation 的结果不再作为主结论使用。head-gaze
relationship 报告也应只保留 continuous dynamics、head rotation strata 和 lagged
diagnostic，不再使用 CPF fixation labels 做分层。
