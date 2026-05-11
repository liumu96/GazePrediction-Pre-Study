# Scripts / 脚本

这里放命令行工具，用于数据检查、feature extraction、timestamp alignment
和可复现实验步骤。

除非脚本另有说明，都从仓库根目录运行。

当前脚本：

- `inspect_adt_sequence.py`: inspect one ADT sequence directory or one sequence
  id resolved under `ADT_DATA_ROOT`。用于快速检查一个 ADT sequence 的文件
  完整性和基本统计。
- `extract_gaze_samples.py`: extract gaze at selected RGB timestamps, validate
  it, and write a gaze CSV plus a lightweight quality summary JSON。用于先把
  gaze 的核心数据和质量指标保存下来。
- `batch_extract_gaze_samples.py`: batch-run the same gaze-only extraction
  workflow across multiple sequences。用于批量生成每个 sequence 的
  `gaze_samples.csv`、`gaze_summary.json` 和一个批量总表。
- `extract_head_proxy.py`: extract a reusable head-feature table aligned to one
  existing `gaze_samples.csv`。第一版 head 使用 `device pose + CPF`，不从
  skeleton 提；输出现在是一层独立的 head feature layer，同时包含绝对
  Scene pose 和相对运动特征，而不只是 event context 摘要。
- `batch_extract_head_proxy.py`: batch-run head-proxy extraction over one reports
  directory of `gaze_samples.csv` files。
- `check_gaze_quality.py`: summarize the extracted `gaze_summary.json` files
  into one flat CSV and one aggregate JSON。用于在不重新打开 provider 的前提下，
  做 sequence-level gaze 质量体检。
- `compute_gaze_dynamics_features.py`: compute whole-sequence per-frame
  CPF-local gaze dynamics from existing `gaze_samples.csv` + `head_samples.csv`。
  这一层只算 local gaze velocity / dispersion 和 head context，不生成
  fixation/saccade labels。
- `detect_scene_gaze_events.py`: detect first scene-direction gaze events from
  existing `gaze_samples.csv`。这一层用 Scene-frame gaze direction 的 angular
  velocity / dispersion 生成 `fixation` / `transition` / `invalid` labels，
  和 CPF-local dynamics 分开保存。
- `visualize_scene_gaze_events.py`: visualize one scene event window from
  existing scene event CSV files。用于检查最终 label、Scene angular velocity
  和 Scene dispersion 的时间关系，不重新打开 ADT provider。
- `analyze_head_gaze_relationship.py`: build a per-frame joined head-gaze table
  plus sequence-level and batch-level statistics from existing gaze/head CSV
  exports。用于系统回答：
  - Scene 里 head 和 gaze 的夹角关系
  - local gaze dynamics 和 head motion 的关系
  - current head motion 对下一步 gaze change 的相关性
- `report_head_gaze_relationship.py`: turn the already generated batch analysis
  outputs into a readable markdown report with fixed tables and figures。适合把
  这一步结果沉淀成可引用的分析报告，而不是只看原始 CSV/JSON。
- `analyze_scene_head_gaze_relationship.py`: join Scene gaze events with
  gaze/head features。用于比较 Scene/world gaze dynamics、CPF-local dynamics
  和 head motion 的关系。
- `report_scene_head_gaze_relationship.py`: generate the Scene-level head-gaze
  report and figures from `batch_scene_head_gaze_analysis_summary.csv`。
- `visualize_gaze_outputs.py`: regenerate visualizations from an existing gaze
  CSV and a selected row window。它会读取已有 CSV，再只为当前窗口打开 ADT
  provider 生成 scanpath、scene_rays、overlay frames 和 overlay video。

示例：

```bash
python scripts/extract_gaze_samples.py <sequence_id> --start-index 900 --end-index 905 --stride 1
```

也可以按 sequence 内的相对秒数选择时间窗口：

```bash
python scripts/extract_gaze_samples.py <sequence_id> --start-offset-s 30 --end-offset-s 32 --stride 1
```

批量处理所有本地 sequence，只提取 gaze 数据：

```bash
python scripts/batch_extract_gaze_samples.py
```

也可以只处理显式给出的 sequence 列表：

```bash
python scripts/batch_extract_gaze_samples.py <sequence_id_1> <sequence_id_2> --stride 30
```

如果批量提取已经完成，只想汇总 sequence 质量：

```bash
python scripts/check_gaze_quality.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

如果要为 gaze dynamics 或 head-gaze 关系分析先提 head features：

```bash
python scripts/extract_head_proxy.py <sequence_id> --input-gaze-csv /mnt/d/SparseGaze/ADT-Gaze/<sequence_id>_gaze_samples.csv
```

如果要批量提 head proxy：

```bash
python scripts/batch_extract_head_proxy.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

如果要继续跑 whole-sequence CPF-local gaze dynamics features：

```bash
python scripts/compute_gaze_dynamics_features.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

如果要生成第一版 scene-direction event labels：

```bash
python scripts/detect_scene_gaze_events.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

默认阈值：

- `--velocity-threshold-deg-s 40`
- `--dispersion-threshold-deg 2.5`
- `--min-fixation-duration-ms 133`

输出：

- `*_scene_gaze_event_features.csv`
- `*_scene_gaze_frame_labels.csv`
- `*_scene_gaze_event_segments.csv`
- `*_scene_gaze_event_summary.json`
- `batch_scene_gaze_event_summary.csv`

如果要看某个 sequence / frame window 的 event timeline：

```bash
python scripts/visualize_scene_gaze_events.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --reports-dir /mnt/d/SparseGaze/ADT-Gaze \
  --start-frame 0 \
  --end-frame 600
```

输出默认写到 `outputs/figures/scene_gaze_events/`。

如果要系统分析 head-gaze 关系：

先确认 `head_samples.csv` 是新版 schema；如果 D 盘上还是旧导出，先重跑：

```bash
python scripts/batch_extract_head_proxy.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

然后再运行：

```bash
python scripts/analyze_head_gaze_relationship.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

脚本只依赖 gaze/head 基础层，生成 geometry / dynamics / temporal diagnostic
统计，不读取 CPF-based fixation labels。

如果 head-gaze 分析已经跑完，要把结果整理成报告和图表：

```bash
python scripts/report_head_gaze_relationship.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

如果要进一步分析 Scene/world gaze dynamics 和 head motion 的关系：

```bash
python scripts/analyze_scene_head_gaze_relationship.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
python scripts/report_scene_head_gaze_relationship.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

这一步读取 scene-direction event outputs，不替代 CPF head-gaze report，而是回答：
最终 world gaze 是否随 head motion 变化，以及 scene fixation / transition 下的
head-gaze dynamics 是否不同。

如果已经有 CSV，只想重新调可视化参数：

```bash
python scripts/visualize_gaze_outputs.py <sequence_id> \
  --start-row 0 --end-row 60 \
  --stride 10 \
  --run-name stride_10
```

后续计划脚本：

- `check_sequence_alignment.py`: summarize timestamp alignment between gaze,
  skeleton, trajectory, and object annotations。用于检查不同 stream 的
  `dt_ns()` 和对齐质量。
- `build_feature_table.py`: build compact downstream analysis/model tables once
  feature extraction and alignment are understood。在 feature extraction 和
  alignment 策略清楚之后再做。

完整路线见 `docs/adt_exploration_plan.md`，API map 见
`docs/adt_feature_extraction_guide.md`，gaze tutorial 见
`docs/tutorial_gaze_feature_extraction.md`，
head-gaze 分析说明见 `docs/head_gaze_relationship_analysis.md`。
