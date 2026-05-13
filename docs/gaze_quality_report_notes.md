# Gaze Quality Report Notes / gaze 质量检查结论

这份文档记录一次已经完成的批量 gaze 质量检查结果，避免结论只停留在命令行输出里。

## Run Context / 运行背景

- reports directory: `/mnt/d/SparseGaze/ADT-Gaze-structured`
- command:

```bash
python scripts/check_gaze_quality.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured
```

- sequence count: `34`

生成的结果文件：

- `/mnt/d/SparseGaze/ADT-Gaze-structured/batch/gaze_quality_report.csv`
- `/mnt/d/SparseGaze/ADT-Gaze-structured/batch/gaze_quality_report.json`

## High-Level Result / 总体结论

当前这批 ADT gaze 数据在现有提取口径下质量非常好，已经足够进入 event analysis。

更具体地说：

- `flagged_sequences = 0`
- 没有任何 sequence 触发当前脚本的 review flags
- 所有 sequence 都能直接进入下一阶段的 event feature computation

## Aggregate Numbers / 聚合数字

来自 `gaze_quality_report.json` 的全局均值：

- `gaze_valid_ratio = 1.0`
- `projection_in_image_ratio = 0.9999679472751135`
- `depth_available_ratio = 0.9999892970288552`
- `pose_valid_ratio = 1.0`
- `ok_ratio = 0.9996562042900033`

说明：

- 终端里显示的 `1.000` 是四舍五入结果
- 实际上并不是绝对零错误，但异常比例已经低到可以忽略不计

## Issue Counts / 异常计数

聚合 `validation_notes` 统计如下：

- `gaze_dt_exceeds_threshold: 29`
- `projection_failed: 3`
- `depth_not_available: 1`
- `scene_ray_unavailable: 1`

总样本量约为：

- `93,426`

如果把上面的 note 次数直接求和，得到：

- `34`

对应的粗略占比约为：

- `34 / 93426 = 0.000364`

注意：

- 这 `34` 只是 note 次数，不一定等于 `34` 个独立坏样本
- 同一帧理论上可能同时带多个 note

但无论如何，量级都非常小。

## Worst Sequences / 最低 ok_ratio 的 sequence

按 `ok_ratio` 排序，最靠后的几条是：

1. `Apartment_release_work_skeleton_seq133_M1292`
   - `ok_ratio = 0.9989010989010989`
   - top issue: `gaze_dt_exceeds_threshold`
2. `Apartment_release_work_skeleton_seq136_M1292`
   - `ok_ratio = 0.9989055089383436`
   - top issue: `gaze_dt_exceeds_threshold`
3. `Apartment_release_meal_skeleton_seq136_M1292`
   - `ok_ratio = 0.998906306963179`
   - top issue: `gaze_dt_exceeds_threshold`

这些 sequence 依然没有被标记为 `flagged`，因为问题只占极少数帧。

## What This Result Means / 这些结果说明了什么

这次质量报告支持下面几个判断：

1. 当前 gaze extraction pipeline 没有系统性错误  
   `gaze_valid`、`projection_in_image`、`depth_available` 都接近 1。

2. 当前 ADT 子集整体是稳定可用的  
   34 条 sequence 没有出现明显坏序列。

3. 当前阶段不需要再花太多精力排查 sequence-level 数据质量  
   质量这关基本已经过了。

## What It Does Not Yet Mean / 这些结果还没有说明什么

这份质量报告并不回答下面的问题：

1. `I-DT` 还是 `I-VT` 更适合当前 ADT 30 Hz 数据  
2. fixation / transition 的阈值应该设成多少  
3. head motion context 应该怎样进入 event analysis  
4. event detection 的最终定义应该是什么

也就是说：

- 它验证了“数据和提取流程可用”
- 但还没有验证“event detection policy 已经合理”

## Recommended Next Step / 下一步建议

基于当前结果，下一步应直接进入：

1. whole-sequence event feature computation
2. event detection policy tuning
3. event-level visualization review

而不是继续停留在 sequence-level 数据质量检查。

对应方法判断见：

- [docs/gaze_event_analysis_notes.md](./gaze_event_analysis_notes.md)
