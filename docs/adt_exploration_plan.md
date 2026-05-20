# ADT Exploration Plan / ADT 探索计划

这份计划的目标是：高效理解如何从本地 Pose2Gaze ADT 子集
`/mnt/d/Pose2Gaze-ADT` 中提取、验证、对齐和可视化特征。

它不是让我们手动把官方 tutorial 一个个跑完，而是把官方 API 和示例整理成
本仓库里的小脚本、小工具和 notebook。这样后续查 gaze、pose、skeleton、
object、depth、segmentation 时，有一个稳定入口。

## Current Data / 当前数据

- 数据根目录：`ADT_DATA_ROOT=/mnt/d/Pose2Gaze-ADT`
- 存储位置：Windows D 盘，通过 WSL 的 `/mnt/d` 访问
- 当前子集：Pose2Gaze 已下载的 34 个 ADT sequences
- 仓库约定：不要把 ADT 原始数据复制进本 repo

## Principle / 工作原则

- 先理解官方 API 怎么提取数据，再写自己的 preprocessing。
- 每个 feature 都同时关注四件事：提取、validity、坐标系、可视化。
- 所有 timestamp query 都要记录 `dt_ns()`，不要只拿 `data()`。
- 坐标系相关内容必须写清楚：CPF、Device、Camera、Scene frame。
- 先用一个 sequence 做透，再扩展到全部 34 个 sequence。

## Phase 1: API And Feature Extraction Map / API 与特征提取地图

目标：建立一份本地参考，回答“每类 ADT 数据应该用哪个 API 取”。

主文档：

- `docs/adt_feature_extraction_guide.md`

要覆盖的 feature groups：

- sequence loading：`AriaDigitalTwinDataPathsProvider`,
  `AriaDigitalTwinDataProvider`
- gaze：ADT `eyegaze.csv`、MPS `mps/eye_gaze/general_eye_gaze.csv`、
  CPF frame、投影到 RGB camera、Scene frame 里的 3D gaze ray
- device pose：Aria trajectory、`T_scene_device`、`quality_score`、
  interpolation
- skeleton pose：joints、markers、joint labels、joint connections、Scene frame
- objects：instance metadata、3D object boxes、6DoF object poses、2D boxes、
  `visibility_ratio`
- image streams：RGB、depth、segmentation、synthetic images
- calibration：device calibration、camera calibration、CPF transform、
  camera projection

需要回答的问题：

- 每类 feature 对应哪个官方 API？
- 每类 feature 在哪个 coordinate frame 下？
- 用之前应该做哪些 validity checks？
- 最适合哪种可视化方式 debug？

## Phase 2: Gaze-First Extraction And Visualization / 先做 gaze 提取与可视化

目标：先把 gaze 这条线做透，因为它同时涉及 timestamp、CPF 坐标系、
camera projection、Scene transform 和可视化。

计划产物：

- `src/adt_sandbox/providers.py`：已实现，使用官方 ADT provider。
- `src/adt_sandbox/gaze.py`：已实现 gaze conversion、validation、
  RGB projection、Scene-frame ray helpers。
- `scripts/extract_gaze_samples.py`：已实现 gaze CSV 和轻量质量 summary 导出。
- `scripts/batch_extract_gaze_samples.py`：已实现批量 gaze-only 导出，不生成
  可视化。
- `scripts/check_gaze_quality.py`：已实现读取已有 summary 的批量质量汇总。
- `visualization/visualize_gaze_outputs.py`：已实现从已有 CSV 和选中窗口生成
  scanpath、scene_rays、overlay frames 和 overlay video。
- `docs/tutorial_gaze_feature_extraction.md`：已实现中文 tutorial 笔记。
- `notebooks/01_gaze_feature_extraction.ipynb`

Gaze validity checks：

- query result 是否 `is_valid()`
- `abs(dt_ns())` 是否在可接受范围内
- `yaw` 和 `pitch` 是否 finite
- 需要 3D point 时，`depth > 0`
- confidence interval width：
  `yaw_high - yaw_low`、`pitch_high - pitch_low`
- gaze projection 是否落在 RGB image bounds 内
- 可选：比较 ADT ground-truth gaze 与 MPS general gaze

Gaze visualizations：

- RGB frame 上的 2D gaze point overlay
- 短事件窗口内的 reference-frame scanpath visualization
- Scene frame 中的 3D gaze ray
- 可选：导出 Rerun `.rrd` 做交互式查看

需要回答的问题：

- ADT `eyegaze.csv` 在 RGB frame timestamps 上是否有效？
- gaze query 的 `dt_ns()` 分布如何？
- gaze projection 有多少比例会跑出 RGB image？
- gaze depth 是否稳定？
- 3D gaze ray 是否指向合理的可见物体区域？

## Phase 3: Pose, Skeleton, Objects, And Image Streams / 扩展到 pose、骨架、物体和图像流

目标：用和 gaze 相同的模式，提取并验证其它核心 feature。

计划产物：

- `src/adt_sandbox/pose.py`
- `src/adt_sandbox/objects.py`
- `notebooks/02_pose_skeleton_objects.ipynb`

Feature checks：

- Aria pose：`is_valid()`、`dt_ns()`、`quality_score`、transform sanity、
  interpolation comparison
- skeleton：`is_valid()`、`dt_ns()`、缺失 marker `[0,0,0]`、joint count、
  joint labels
- object boxes：`is_valid()`、`dt_ns()`、visible object count、
  `visibility_ratio`、object id metadata lookup
- images/depth/segmentation：timestamp matching、image dimensions、
  segmentation ids 是否存在于 `instances.json`

Visualizations：

- Scene frame 中的 Aria trajectory
- 3D skeleton joints 和 limb connections
- 3D object boxes 和 dynamic object trajectories
- RGB frame 上的 2D object boxes
- segmentation 和 depth image previews

## Phase 4: Alignment Policy / 时间对齐策略

目标：明确后续 Pose2Gaze-style samples 应该如何构建。

计划产物：

- `src/adt_sandbox/timestamps.py`
- `scripts/check_sequence_alignment.py`
- `notebooks/03_timestamp_alignment.ipynb`

候选 anchor timestamps：

- RGB frame timestamps：适合 image overlay 和 visual debugging
- gaze timestamps：适合 gaze-centered modeling
- fixed-rate timeline：适合 sequence-level summaries

Alignment rules：

- 默认使用 ADT provider APIs 的 closest-time query
- 每个被 query 的 stream 都必须记录 `dt_ns()`
- Aria pose 和 3D object boxes 优先尝试官方 interpolation helpers
- required stream 无效或 `dt_ns()` 超阈值时，reject 或 flag sample
- multi-person/concurrent sequences 才需要重点使用 timecode conversion

## Phase 5: Reusable Feature Tables / 可复用特征表

目标：当提取和对齐策略清楚以后，再写紧凑的 derived feature tables，
供分析或模型使用。

计划产物：

- `src/adt_sandbox/feature_table.py`
- `scripts/build_feature_table.py`

初始表粒度：

- 每行对应一个 selected gaze timestamp 或 RGB timestamp

候选列：

- sequence metadata
- gaze yaw、pitch、depth、confidence widths、validity flags
- projected gaze pixel location
- Scene frame 中的 gaze point 或 gaze ray
- Aria pose 和 pose quality
- nearest skeleton features
- visible object ids/classes 与 2D/3D box summaries
- 每个 stream 的 alignment deltas

输出约定：

- 小型 reports 放 `outputs/reports/`
- 大型 derived data 放 repo 外，例如 D 盘

## Phase 6: Gaze Dynamics and Head-Gaze Analysis / gaze dynamics 与 head-gaze 分析

目标：保留 CPF-local gaze dynamics 作为连续特征，并分析它和 head motion 的关系。
CPF-based fixation labels 已从主线移除；scene-direction event 已实现，object-level
event 后续单独设计。

主文档：

- `docs/gaze_event_analysis_notes.md`
- `docs/gaze_quality_report_notes.md`
- `docs/head_gaze_relationship_analysis.md`

当前原则：

- CPF-local velocity / dispersion 是有用的 dynamics feature
- CPF-thresholded fixation labels 不作为最终 event label
- `head.py` 是独立 head feature layer
- head-gaze analysis 不读取 CPF fixation labels
- scene-direction event 已单独实现
- object-level fixation 后续另起 pipeline

计划产物：

- `src/adt_sandbox/head.py`
- `src/adt_sandbox/gaze_dynamics.py`
- `src/adt_sandbox/scene_gaze_events.py`
- `src/adt_sandbox/head_gaze_analysis.py`
- `src/adt_sandbox/scene_head_gaze_analysis.py`
- `scripts/extract_head_proxy.py`
- `scripts/compute_gaze_dynamics_features.py`
- `scripts/detect_scene_gaze_events.py`
- `scripts/analyze_head_gaze_relationship.py`
- `scripts/analyze_scene_head_gaze_relationship.py`
- `notebooks/04_gaze_head_scene_viewer_interactive.ipynb`

当前状态：

- `head.py`、`extract_head_proxy.py`、`batch_extract_head_proxy.py` 已实现
- `head_samples.csv` 当前同时包含绝对 Scene pose 和相对运动特征
- `compute_gaze_dynamics_features.py` 已实现
- `detect_scene_gaze_events.py` 已实现：
  - 基于 `gaze_dir_scene_unit_xyz`
  - 输出 scene-direction `fixation` / `transition` / `invalid`
  - 阈值参数化，默认 `40 deg/s`、`2.5 deg`、`133 ms`
- `/mnt/d/SparseGaze/ADT-Gaze-structured` 已完成全量 scene-direction event 导出
- `visualize_scene_gaze_events.py` 已实现并完成窗口级 timeline 抽查
- `head_gaze_analysis.py` 与 `analyze_head_gaze_relationship.py` 已实现：
  - 从已有 gaze/head CSV 直接生成逐帧 joined table
  - 量化 Scene 几何关系、local 动态关系、head rotation strata
  - 以及 current head motion 对下一步 gaze change 的相关性
- `scene_head_gaze_analysis.py` 与 `analyze_scene_head_gaze_relationship.py` 已实现：
  - join scene event labels、Scene gaze dynamics、CPF-local dynamics 和 head motion
  - 比较 scene `fixation` / `transition` 下的 head-gaze dynamics
  - 生成 `docs/scene_head_gaze_relationship_report.md`
- `sparsegaze_head_utility.py` 与 `analyze_sparsegaze_head_utility.py` 已实现：
  - 模拟 sparse gaze anchors，比较 hold-last / linear interpolation residual
  - 分析 residual 是否能被 current head / head history 解释
  - 做 head motion 与 CPF/Scene gaze dynamics 的 lead-lag correlation
  - 生成 `docs/sparsegaze_head_utility_report.md`
- `scene_features.py`、`inspect_scene_assets.py` 与
  `extract_scene_object_boxes.py` 已开始实现 Scene feature layer：
  - 不依赖官方 provider，直接读取 `instances.json`、`scene_objects.csv`、
    `3d_bounding_box.csv` 和 skeleton files
  - 第一版导出 Scene-frame object boxes 和 8 个 3D corners
  - 详细路线见 `docs/scene_feature_extraction_plan.md`
- `skeleton_features.py`、`extract_skeleton_samples.py` 与
  `batch_extract_skeleton_samples.py` 已实现 gaze-aligned skeleton extraction：
  - 在 `adt` conda 环境下使用 `AriaDigitalTwinSkeletonProvider`
  - 直接读取 `Skeleton_T.json`
  - 输出 root/head joint 和全部 51 个 Scene-frame skeleton joints
  - 单序列验证 valid ratio = `1.000`

## Immediate Next Step / 下一步

gaze-first 路径已经先跑通。当前暂停模型分析，转向 Scene feature extraction。
更合理的顺序是：

1. 验证单 sequence 的 `*_scene_object_boxes.csv` 是否正确：
   - object count
   - bbox size
   - Scene-frame corners
   - 和 2D boxes / viewer 是否大体一致
2. 批量导出 object boxes。
3. 批量导出 skeleton samples，作为 3D viewer 的 body trajectory / 火柴人层。
4. 实现 gaze-object hit test：
   - GT gaze ray -> object box
   - hit object id / category / distance / no-hit
5. 做 3D scene viewer：
   - object boxes
   - body/head trajectory
   - skeleton joints / limb connections
   - GT gaze rays / gaze hit trajectory
6. 后续再把 SparseGaze prediction 接入，评估 predicted-vs-GT object hit
   agreement。
