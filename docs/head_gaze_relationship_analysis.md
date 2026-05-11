# Head-Gaze Relationship Analysis / Head-Gaze 关系分析

这份文档记录当前保留的 head-gaze 关系分析层。经过 review 后，仓库不再把
CPF-local velocity / dispersion 阈值结果当作最终 fixation label。当前保留的是
连续 dynamics feature 和 head-gaze 关系统计。

## 1. 当前定位

这一步是 **GT-only continuous relationship analysis**，不是 event detector，也不是
SparseGaze 模型实验。

输入层：

- gaze feature layer:
  - `src/adt_sandbox/gaze.py`
  - `*_gaze_samples.csv`
- head feature layer:
  - `src/adt_sandbox/head.py`
  - `*_head_samples.csv`
- CPF-local gaze dynamics layer:
  - `src/adt_sandbox/gaze_dynamics.py`
  - `*_gaze_dynamics.csv`

Head-gaze analysis 的目标是回答：

- Scene frame 里 head forward 和 gaze direction 的几何关系是什么
- CPF-local gaze velocity 和 head rotation / translation 是否相关
- CPF-local gaze 的移动方向是否和 head rotation direction 有关系
- head rotation 较大时，local gaze dynamics 是否也更强
- 当前 head motion 和下一帧 local gaze dynamics 是否存在弱时序关系

## 2. 为什么删除 CPF fixation labels

CPF gaze dynamics 描述的是：

```text
eye-in-device / eye-in-head movement
```

它不能直接定义：

```text
用户是否在 scene / object / task 层面稳定看一个目标
```

例如：

```text
CPF gaze stable + head rotation large
```

在 CPF 下会像 fixation，但 scene gaze ray 可能正在扫过世界。这不是直观意义上的
scene/object fixation。因此，旧的 CPF-based `IDT / IVT / combined fixation labels`
被移出主线。

保留的是：

- `local_velocity_deg_s`
- `local_angle_step_deg`
- `delta_yaw_rad`
- `delta_pitch_rad`
- `window_dispersion_deg`

这些是有用的连续特征，可以解释 local eye dynamics，但不再被阈值化成主事件标签。

## 3. 当前代码路径

### 3.1 计算 CPF-local gaze dynamics

```bash
python scripts/compute_gaze_dynamics_features.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

输出：

- `*_gaze_dynamics.csv`
- `*_gaze_dynamics_summary.json`
- `batch_gaze_dynamics_summary.csv`

这一步只输出 feature，不输出 fixation/saccade label。

### 3.2 分析 head-gaze relationship

```bash
python scripts/analyze_head_gaze_relationship.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

输出：

- `*_head_gaze_analysis_rows.csv`
- `*_head_gaze_analysis_summary.json`
- `batch_head_gaze_analysis_summary.csv`
- `batch_head_gaze_analysis_report.json`

这一步会从 `gaze_samples.csv` 和 `head_samples.csv` 重新构建所需 dynamics，不依赖
`*_gaze_dynamics.csv` 是否已经存在。

### 3.3 生成报告

```bash
python scripts/report_head_gaze_relationship.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

输出：

- `docs/head_gaze_relationship_report.md`
- `outputs/figures/head_gaze_relationship/*.png`

报告只解释 continuous dynamics，不再包含 CPF fixation/non-fixation 分层。

### 3.4 分析 Scene-level head-gaze relationship

```bash
python scripts/analyze_scene_head_gaze_relationship.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
python scripts/report_scene_head_gaze_relationship.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

输出：

- `*_scene_head_gaze_analysis_rows.csv`
- `*_scene_head_gaze_analysis_summary.json`
- `batch_scene_head_gaze_analysis_summary.csv`
- `batch_scene_head_gaze_analysis_report.json`
- `docs/scene_head_gaze_relationship_report.md`

这一步读取 scene-direction event label，回答最终 Scene/world gaze dynamics 和
head motion 的关系。它和 CPF-local report 并列存在：CPF report 解释
eye-in-head dynamics；Scene-level report 解释 final world-gaze dynamics。

## 4. 当前保留的核心量

### Scene geometry

- `gaze_head_angle_deg`
- `gaze_dir_scene_unit_*`
- `head_forward_scene_unit_*`

用于描述 head forward 和 gaze direction 的瞬时几何偏差。

### CPF-local gaze dynamics

- `local_angle_step_deg = angle(g_t-1^cpf, g_t^cpf)`
- `local_velocity_deg_s = local_angle_step_deg / dt`
- `delta_yaw_rad`
- `delta_pitch_rad`
- `window_dispersion_deg`

用于描述眼睛相对 device/head 的局部运动。

### Head dynamics

- `head_rotation_angle_step_deg`
- `head_rotation_speed_deg_s`
- `head_rotvec_prev_head_*`
- `head_translation_speed_m_s`
- `translation_scene_*`
- `translation_prev_head_*`

用于描述 device/head 在 scene 中的整体运动。

### Directional head-gaze relation

- `head_rotvec_prev_head_*` 是 `R_{t-1}^{-1} R_t` 的 axis-angle vector，
  表达在上一帧 head/CPF frame 下，保留 head rotation 的有符号方向。
- `delta_yaw_rad`、`delta_pitch_rad` 是 CPF gaze angle 的有符号相邻帧变化。
- 当前分析额外计算：
  - `corr(signed delta yaw, head rotvec Y)`
  - `corr(signed delta pitch, head rotvec X)`
  - `corr(|delta yaw|, |head rotvec Y|)`
  - `corr(|delta pitch|, |head rotvec X|)`
  - `gaze_head_motion_alignment_2d`

这部分回答“gaze 是往哪个方向变”的问题，而不是只看 `local_velocity_deg_s`
这种速度大小。

## 5. 当前分析怎么解释

现在的结论应该只围绕 continuous relationship：

```text
head rotation 是否和 local gaze dynamics 有统计关系？
translation 是否弱于 rotation？
head rotation direction 是否解释 gaze direction？
high head-rotation windows 里 local gaze velocity 是否更高？
```

不再解释：

```text
CPF fixation 和 non-fixation 有什么差异
```

因为这个 label 与 scene/object fixation 语义不一致。

## 6. 当前报告结论

当前报告结论：

- head rotation 和 local gaze velocity 有稳定弱到中等相关，translation 很弱。
- high head-rotation group 的 local gaze velocity 明显更高，这是最稳定的结果。
- direction-level 结果比 speed-level 弱：有符号方向相关性较弱，但 component
  magnitude 仍有相关性。这说明 head rotation 更可靠地解释 gaze change 的大小，
  对“眼睛往哪个方向动”的解释力更弱。
- 2D alignment 显示既有同向也有反向/补偿式运动，因此后续模型不应该把 head
  motion 简化成单一 scalar speed；至少应保留方向分量或 rotation vector。

Scene-level report 的新增结论：

- Scene gaze velocity 和 head rotation speed 的相关性略高于 CPF-local velocity
  和 head rotation speed。
- CPF-local velocity 与 Scene velocity 高度相关，说明 rapid eye-in-head changes
  在当前数据中通常也对应 rapid world-gaze changes。
- Scene transition frames 中 Scene velocity、CPF velocity 和 head rotation 都高于
  scene fixation frames。
- high relative head-rotation frames 的 scene fixation fraction 明显更低，说明
  aggregate 上 head rotation 更常参与 world-gaze transition，而不是完全被眼动补偿。

## 7. Scene-direction event layer

第一版 scene-direction event 已经单独实现，基于 scene/world 方向语义：

- scene gaze angular velocity / dispersion

```bash
python scripts/detect_scene_gaze_events.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

输出与当前 CPF dynamics 明确分开：

- `*_scene_gaze_event_features.csv`
- `*_scene_gaze_frame_labels.csv`
- `*_scene_gaze_event_segments.csv`
- `*_scene_gaze_event_summary.json`

这一步仍然不是 object-level fixation；它回答的是 gaze direction 在世界中是否稳定。

## 8. 后续 object-level event

如果后面要更接近“看同一个物体/目标”，再引入：

- RGB projection stability
- object hit / scene intersection

这样可以避免把 local eye dynamics 和 world/object fixation 混在一起。
