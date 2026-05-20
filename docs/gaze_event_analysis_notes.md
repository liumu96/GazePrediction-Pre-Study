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

## 4. Scene-direction event

第一版 scene-direction event 已实现：

```bash
python scripts/detect_scene_gaze_events.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured
```

当前状态：`/mnt/d/SparseGaze/ADT-Gaze-structured` 已经完成全量 scene-direction event
导出，包含逐帧 feature、逐帧 label、event segments 和 batch summary。

它基于 `gaze_dir_scene_unit_xyz` 计算：

- `scene_angle_step_deg`
- `scene_velocity_deg_s`
- `scene_window_dispersion_deg`

默认 fixation 判据：

```text
scene_velocity_deg_s <= 40
and scene_window_dispersion_deg <= 2.5
and segment_duration_ms >= 133
```

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

当前已经完成至少一个窗口级 timeline 抽查。

这一步回答：

```text
用户视线在世界方向上是否稳定？
```

它仍然不是 object-level fixation，因为它还没有用 object hit / scene
intersection。

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
