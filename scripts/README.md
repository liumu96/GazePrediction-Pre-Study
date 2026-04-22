# Scripts / 脚本

这里放命令行工具，用于数据检查、feature extraction、timestamp alignment
和可复现实验步骤。

除非脚本另有说明，都从仓库根目录运行。

当前脚本：

- `inspect_adt_sequence.py`: inspect one ADT sequence directory or one sequence
  id resolved under `ADT_DATA_ROOT`。用于快速检查一个 ADT sequence 的文件
  完整性和基本统计。

计划脚本：

- `extract_gaze_samples.py`: extract gaze at selected timestamps, validate it,
  project it to RGB, and optionally write debug visualizations。用于先把 gaze
  的提取、validity、投影和可视化跑通。
- `check_sequence_alignment.py`: summarize timestamp alignment between gaze,
  skeleton, trajectory, and object annotations。用于检查不同 stream 的
  `dt_ns()` 和对齐质量。
- `build_feature_table.py`: build compact downstream analysis/model tables once
  feature extraction and alignment are understood。在 feature extraction 和
  alignment 策略清楚之后再做。

完整路线见 `docs/adt_exploration_plan.md`，API map 见
`docs/adt_feature_extraction_guide.md`。
