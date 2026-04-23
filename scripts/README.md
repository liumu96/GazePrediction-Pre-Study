# Scripts / 脚本

这里放命令行工具，用于数据检查、feature extraction、timestamp alignment
和可复现实验步骤。

除非脚本另有说明，都从仓库根目录运行。

当前脚本：

- `inspect_adt_sequence.py`: inspect one ADT sequence directory or one sequence
  id resolved under `ADT_DATA_ROOT`。用于快速检查一个 ADT sequence 的文件
  完整性和基本统计。
- `extract_gaze_samples.py`: extract gaze at selected RGB timestamps, validate
  it, project it to RGB/reference frame, and save reusable CSV/RGB/overlay
  assets。用于先把 gaze 的提取、validity、投影和可视化中间结果跑通。
- `visualize_gaze_outputs.py`: regenerate visualizations from an existing gaze
  CSV/manifest/saved frames without re-extracting gaze。用于在已有中间结果上
  调整 row window、统一 `--stride`、scanpath、scene_rays 和 overlay video。

示例：

```bash
python scripts/extract_gaze_samples.py <sequence_id> --start-index 900 --end-index 905 --stride 1
```

也可以按 sequence 内的相对秒数选择时间窗口：

```bash
python scripts/extract_gaze_samples.py <sequence_id> --start-offset-s 30 --end-offset-s 32 --stride 1
```

如果已经有 CSV，只想重新调可视化参数：

```bash
python scripts/visualize_gaze_outputs.py <sequence_id> \
  --start-row 0 --end-row 60 \
  --stride 10 \
  --run-name stride_10
```

计划脚本：

- `check_sequence_alignment.py`: summarize timestamp alignment between gaze,
  skeleton, trajectory, and object annotations。用于检查不同 stream 的
  `dt_ns()` 和对齐质量。
- `check_gaze_quality.py`: batch summarize gaze validity, projection coverage,
  depth coverage, and timestamp deltas across sequences。用于把当前 tutorial
  变成批量质量报告。
- `build_feature_table.py`: build compact downstream analysis/model tables once
  feature extraction and alignment are understood。在 feature extraction 和
  alignment 策略清楚之后再做。

完整路线见 `docs/adt_exploration_plan.md`，API map 见
`docs/adt_feature_extraction_guide.md`，gaze tutorial 见
`docs/tutorial_gaze_feature_extraction.md`。
