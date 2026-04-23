# ADT Dataset Notes / ADT 数据集笔记

ADT sequence 通常包含：

- VRS streams：egocentric video、depth images、segmentations。
- CSV annotations：2D boxes、3D boxes、scene objects、trajectories、gaze。
- JSON metadata：sequence metadata、instance metadata、skeleton transforms、
  MPS summaries。
- MPS outputs：eye gaze 等 Machine Perception Services 结果。

有用参考：

- 官方 Project Aria ADT dataset documentation。
- 已安装 `projectaria-tools` package 中的 `projectaria_tools.projects.adt`。
- 可选本地源码 checkout：`external/projectaria_tools/`，用于查 examples 和实现细节。
- 本地提取指南：`docs/adt_feature_extraction_guide.md`。
- Gaze tutorial：`docs/tutorial_gaze_feature_extraction.md`。

当前项目使用 `adt` conda 环境中的 `projectaria-tools 2.x`。已验证
`AriaDigitalTwinDataPathsProvider` 和 `AriaDigitalTwinDataProvider` 可以读取
本地 Pose2Gaze-ADT sequence，因此默认使用官方 ADT provider。

当某个 dataset field、coordinate convention 或 timestamp alignment rule
通过实验确认后，把稳定结论记在这里。
