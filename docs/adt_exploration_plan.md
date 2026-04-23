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
- `scripts/visualize_gaze_outputs.py`：已实现从已有 CSV 和选中窗口生成
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

## Immediate Next Step / 下一步

gaze-first 路径已经先跑通。当前下一步是把单 sequence tutorial 扩展成
批量质量检查：

1. 新增 `scripts/check_gaze_quality.py`：遍历多个 sequences，统计
   `validation_notes`、projection ratio、depth coverage、`gaze_dt_ns`
   分布和 pose quality。
2. 创建 `notebooks/01_gaze_feature_extraction.ipynb`：读取 tutorial 输出的
   CSV、summary 和按需生成的 figures，交互式检查 gaze projection、scanpath 和
   Scene-frame rays。
3. 如果 gaze quality report 稳定，再进入 Phase 3 的 pose、skeleton、
   object feature extraction。
