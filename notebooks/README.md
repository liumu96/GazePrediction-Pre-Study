# Notebooks / 交互式探索

notebooks 用于交互式探索和可视化验证。可复用的 parsing、inspection、
plotting helpers 应该放在 `src/adt_sandbox/`，不要把核心逻辑只留在
notebook cell 里。

计划 notebooks：

- `01_gaze_feature_extraction.ipynb`: inspect gaze extraction, validity,
  projection to RGB, scanpath visualization, and Scene-frame gaze rays。先把
  gaze 的提取、验证和 2D/3D 可视化跑通。
- `02_pose_skeleton_objects.ipynb`: inspect Aria pose, skeleton joints, object
  2D/3D boxes, and image/depth/segmentation streams。扩展到 pose、skeleton、
  object、depth 和 segmentation。
- `03_timestamp_alignment.ipynb`: compare gaze, skeleton, trajectory, and box
  timestamps before preprocessing。正式构建 feature table 前先理解
  timestamp alignment。

稳定结论要同步写进 `docs/`，不要只留在 notebook cells。当前 API map 在
`docs/adt_feature_extraction_guide.md`。
