# Gaze Event Analysis Notes / Gaze Event 分析笔记

这份文档记录当前对 gaze event 的最新结论。旧版 CPF-based `IDT / IVT /
combined` fixation pipeline 已经从主线移除。

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
python scripts/compute_gaze_dynamics_features.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

输出：

- `*_gaze_dynamics.csv`
- `*_gaze_dynamics_summary.json`
- `batch_gaze_dynamics_summary.csv`

这些是 feature outputs，不是 event labels。

## 4. 真正的 event 下一步怎么做

如果后面继续做 fixation/event，应该另起一套 scene/world 语义的 pipeline。

候选定义：

- scene gaze angular velocity / dispersion
- image projection stability
- object hit / scene intersection stability

推荐输出命名：

- `*_scene_gaze_events.csv`
- `*_scene_fixation_frame_labels.csv`
- `*_scene_fixation_segments.csv`

这样可以明确区分：

```text
CPF dynamics = local eye/device motion diagnostics
scene fixation = world/object/task-level gaze event
```

## 5. 对现有分析的影响

之前基于 CPF fixation/non-fixation 的结果不再作为主结论使用。head-gaze
relationship 报告也应只保留 continuous dynamics、head rotation strata 和 lagged
diagnostic，不再使用 CPF fixation labels 做分层。
