# Gaze Event Analysis Notes / Gaze Event 分析笔记

这份文档记录当前关于 ADT gaze event analysis 的方法判断，避免这些决定只停留
在对话里。目标不是立刻写死最终算法，而是先把：

- event detection 应该在整条 sequence 上做，还是先人工选片段
- fixation / saccade 在 ADT 30 Hz 条件下能做到什么程度
- velocity / dispersion 应该在哪个坐标系下计算
- 第一版算法应该怎么参数化

这些前提写清楚。

## 1. 整体流程：先全序列 detection，再回头做复查

正式流程应该是：

```text
whole-sequence gaze timeline
-> event feature computation
-> event detection / segmentation
-> event table
-> event-level visualization and review
```

而不是：

```text
先人工选几个片段
-> 再在片段里做 detection
```

原因很直接：

- event 本质上是时间连续结构，边界会受切窗影响
- 如果先人工选窗，容易引入主观偏置
- 后续 sequence-level 统计、event distribution、downstream table 都做不干净

所以可视化窗口更适合放在 **event detection 之后**，作为复查和解释手段。

## 2. 用 `yaw/pitch` 做 dispersion 和 angular velocity，分别对应什么

如果在 local gaze 上做 event detection，最自然的两类基础特征就是：

1. 窗口内 `yaw/pitch` dispersion
2. 连续帧的 angular velocity / angular step

这基本对应经典眼动检测里的两条路线：

- dispersion-based：接近 `I-DT`
- velocity-based：接近 `I-VT`

更准确地说：

- 如果我们对固定长度或滑动窗口里的 `yaw/pitch` 范围、方差或角距离包络做阈值，
  这是 `I-DT` 风格
- 如果我们对相邻帧之间的 angular displacement / angular velocity 做阈值，
  这是 `I-VT` 风格

但当前项目里不建议机械照搬教科书名字，而是先明确你到底在阈值化什么。
后面实现里可以把这两类量都算出来，再决定第一版 detection 先用哪一条。

## 3. 30 Hz 的 ADT 能不能同时支持 fixation 和 saccade

能，但能力不对称。

### 3.1 fixation

30 Hz 对 fixation candidate detection 通常是够用的，因为 fixation：

- 持续时间更长
- 对极高时间分辨率的依赖没那么强
- 更适合用 dispersion / low-velocity / stability 做判定

所以第一版 event detection 可以明确支持：

- `fixation_candidate`
- `non_fixation` 或 `transition_candidate`

### 3.2 saccade

30 Hz 可以支持 **粗粒度 transition detection**，但不适合过度宣称高精度 saccade
眼动学分析。

原因：

- 30 Hz 一帧约 `33.3 ms`
- 很多 saccade 本身持续时间就只有几十毫秒
- 常见情况只会跨 1 到 3 帧

这意味着第一版可以做：

- “这里发生了快速 gaze shift / transition”
- fixation A 到 fixation B 之间有一个高速度段

但不适合直接承诺：

- 精确 onset / offset
- peak velocity
- microsaccade
- main-sequence 级别分析

因此文档和代码里更稳妥的命名是：

- `fixation_candidate`
- `transition_candidate`

而不是在第一版就强定义为严格的 `saccade`。

## 4. velocity / dispersion 应该在哪个坐标系下算

第一版建议：

- **主 event detection 空间：CPF / local gaze**
- **head motion：单独算 context feature**
- **Scene/world gaze：用于解释和下游分析，不先作为 event 主判据**

### 4.1 为什么先用 local gaze

当前导出的：

- `yaw_rad`
- `pitch_rad`
- `gaze_dir_cpf_unit_*`

本质上都是 CPF / local gaze 表示。对 event detection，优先在这里做有几个好处：

1. 语义更接近 `eye-in-head` 变化
2. 不会一开始就把 head motion 和 eye motion 混在一起
3. 和后续 SparseGaze-style local label 建模更一致

### 4.2 为什么不一开始就在 world 坐标系算

如果直接在 Scene/world gaze 上算 velocity / dispersion：

- head motion 会直接进入 gaze dynamics
- 很难区分“眼睛真的扫视了”还是“头动带来的 world ray 变化”
- event 解释会变脏

这会让你后面很难回答：

- 这是 local fixation？
- 还是 world-stabilized gaze？
- 还是 head compensation / VOR 一类现象？

### 4.3 head 要不要考虑

要，但不建议第一层就混进判定式。

更合理的做法是分两层：

#### A. local gaze event

基于 CPF gaze 做：

- fixation candidate
- transition candidate
- invalid

#### B. head / ego-motion context

单独计算：

- `origin_step_m`
- `head_forward_angle_step_deg`
- pose quality
- 可能的 `head_motion_low/medium/high`

最终 event table 里保留：

- `event_type`
- `head_motion_context`

这样后面既能研究 gaze event，也能分析这些 event 是否发生在明显的 head motion 中。

## 5. 第一版 detection 应该怎么选

当前更合理的是：

1. **先计算 event features**
2. **再定 detection policy**

不要还没看数据分布，就先把最终阈值写死。

### 5.1 推荐先算的基础特征

基于整条 sequence 逐帧计算：

- `dt_s`
- `delta_yaw_rad`
- `delta_pitch_rad`
- `local_angle_step_deg`
- `local_velocity_deg_s`
- `window_dispersion_deg`
- `gaze_valid`
- `projection_in_image`
- `depth_available`
- `validation_notes`

再额外算 head context：

- `origin_step_m`
- `head_forward_angle_step_deg`
- `pose_quality_score`

### 5.2 第一版 event 类别

建议先定成：

- `fixation_candidate`
- `transition_candidate`
- `invalid`

这样比一开始就输出强定义的 `fixation/saccade/pursuit` 更稳。

## 6. 阈值怎么处理

第一版不要把阈值写死在文档里，也不要现在就假设它们已经定了。

建议：

- **全部做成参数**
- 默认给一个可运行的 conservative 初值
- 但明确标记为“待数据校准”

也就是说，后面的脚本接口应该长这样：

```text
--min-fixation-duration-ms
--ivt-velocity-threshold-deg-s
--idt-dispersion-threshold-deg
--max-invalid-gap-frames
```

再通过当前 ADT 全量 CSV / summary 去看：

- `local_angle_step_deg`
- `local_velocity_deg_s`
- `window_dispersion_deg`

的分布，最后再收敛到正式阈值。

## 7. 第一版方法建议

第一版更推荐的实现顺序是：

1. `compute_gaze_event_features`
   - 对整条 sequence 输出 frame-level event features
2. `detect_gaze_events`
   - 读取这些 features，按参数做 event segmentation
3. `visualize_gaze_outputs.py`
   - 对选定 event 再做 overlay / scanpath / scene-ray 复查

其中 detection policy 可以同时保留两条候选：

- `I-VT` 风格：基于 `local_velocity_deg_s`
- `I-DT` 风格：基于 `window_dispersion_deg`

第一版优先做哪一个，不应该凭空拍定，而应该先看当前 ADT 数据分布。

## 8. 当前阶段最合理的结论

当前可以先明确这几条：

1. event detection 应该先在整条 sequence 上做
2. ADT 30 Hz 适合先做 fixation / transition，不适合过度承诺精细 saccade
3. 第一版主坐标系优先用 CPF / local gaze
4. head motion 先作为 context，而不是先并入 event 判据
5. 阈值一律参数化，等分布统计出来后再校准

这套判断比“先手动挑片段、再直接上固定阈值 fixation 算法”更稳，也更适合后续
和 SparseGaze、scene/object/task analysis 接起来。
