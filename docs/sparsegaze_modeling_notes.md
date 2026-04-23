# SparseGaze Modeling Notes / SparseGaze 建模笔记

这份笔记不是 SparseGaze 仓库本身的实现文档，而是基于当前 ADT 探索得到的
建模结论整理。之所以先放在这个仓库里，是因为这些判断直接依赖：

- `docs/tutorial_gaze_feature_extraction.md`
- 当前导出的 ADT gaze CSV / summary
- 对 HAGI / HAGI++ 的坐标系和建模设定的阅读

目标是把下面几个问题沉淀下来，而不是只停留在对话里：

- HAGI / HAGI++ 里的 `pitch / yaw` 到底在哪个坐标系下？
- 这和我们当前 ADT 导出的 `yaw_rad / pitch_rad / depth_m` 是什么关系？
- SparseGaze 如果要在低频 eye tracker 条件下更好地利用 head 信息，label 应该
  设计成 local 还是 world？
- 如果模型 MAE 改善了，怎么证明它对 scene / object / task analysis 也真的有用？

## 1. HAGI / HAGI++ 的核心设定

根据 HAGI（UIST 2025）和 HAGI++（arXiv 2025）论文：

- gaze 被表示成时间序列 `(pitch, yaw)`
- head movement 被表示成相对于 eye tracker 的相对运动
- HAGI++ 明确说明 gaze 和 head 都是在 `tracker-centric coordinate system`
  下表示，而不是直接用 world-frame gaze

这意味着 HAGI / HAGI++ 主要学习的是：

- `eye-in-head`
- `eye-in-tracker`
- 或更一般地说，局部 gaze dynamics

而不是直接学习 scene/world 中的绝对 gaze ray。

## 2. 这和当前 ADT 导出的 gaze 有什么关系

当前仓库导出的 ADT gaze CSV 里：

- `yaw_rad`
- `pitch_rad`
- `depth_m`

都来自 ADT `eyegaze.csv`，更具体地说是在 `CPF (Central Pupil Frame)` 下：

- CPF 原点在左右 eye boxes 的中点
- CPF 是刚性附着在眼镜/头部上的局部坐标系
- 因此它从建模语义上属于 `tracker-centric / head-centric`

当前 CSV 里另外还有两类衍生表示：

- `gaze_u_px / gaze_v_px`：RGB image plane
- `gaze_origin_scene_* / gaze_point_scene_*`：ADT Scene frame

所以当前导出的数据天然已经同时覆盖了：

1. local gaze（CPF）
2. image-plane gaze
3. scene/world gaze

这对后续 SparseGaze 非常重要，因为我们不需要重新定义标签体系，只需要决定：

- 训练时预测哪一种表示
- 评估时看哪一种表示

## 3. SparseGaze 当前问题和 HAGI 不是完全同一个任务

如果 SparseGaze 之前用的是：

- gaze: world 坐标系
- head: world 坐标系

那么它和 HAGI / HAGI++ 的设定并不完全一致。

### 3.1 world-world 设定不是错

它当然可以工作，而且对“用户在场景里看哪里”这种问题是直接的。

### 3.2 但 world-world 会把两个因素混在一起

当 label 是 `world gaze` 时，它同时包含：

- head 在 world 里的朝向变化
- eye 相对 head 的偏转变化

这会导致：

- `head` 信息的收益一部分来自真正的 eye-head coordination
- 另一部分来自 world 坐标系下 head 和 gaze 的几何耦合

所以如果只做 world-world 建模，模型可以做，但解释会比较脏。

## 4. 对 SparseGaze 更干净的建模拆分

如果研究重点是：

- 低频 eye tracker 下怎么利用高频 head 信息补 gaze
- multi-modal gaze prediction 怎么建模更有效

更推荐把问题拆成两层：

### 4.1 层 A：预测 local gaze

例如预测：

- CPF 下的 `yaw / pitch`
- CPF / head frame 下的 unit gaze direction

### 4.2 层 B：组合回 world gaze

再用 head pose 把 local gaze 变换到 world / scene：

```text
g_world = R_world_head * g_local
```

或者在当前 ADT 里：

```text
T_scene_cpf * gaze_point_cpf
```

这样做的好处是：

- 更接近 HAGI / HAGI++ 的建模语义
- 可以把 eye motion 和 head motion 分开解释
- 最终仍然保留 world-frame gaze，方便 scene/object/task 分析
- 后续误差分析更清楚：到底是 local gaze 预测错了，还是 head pose / 组合过程带来的误差

## 5. 模型层面的改善 和 分析层面的改善 是两回事

这个区分必须明确。

### 5.1 模型层面的改善

这回答：

> 模型是不是更会补 gaze 了？

对应指标通常是：

- local angular error
- world angular error
- MAE
- velocity distribution
- temporal smoothness / continuity

### 5.2 分析层面的改善

这回答：

> 补出来的高频 gaze，对 scene / object / task analysis 到底有没有帮助？

这里不能只拿 MAE 代替。因为从预测值到下游分析之间还有多层变换：

```text
predicted local gaze
-> composed world gaze
-> scene/object/AOI attribution
-> fixation/event/task analysis
```

中间任何一层都可能放大或吞掉前一层的改善。

所以：

- `MAE` 改善，不自动等于 scene/object/task 分析改善
- downstream evaluation 必须单独设计

## 6. SparseGaze 应该怎么证明“真的更有用”

建议至少做三层评估，而不是只做 gaze reconstruction。

### 6.1 第一层：重建层

回答：

> 预测的 gaze 数值更准了吗？

建议指标：

- local angular error（CPF / head frame）
- world angular error（scene/world）
- velocity profile
- temporal continuity

### 6.2 第二层：行为层

回答：

> 预测结果能不能更稳定地恢复 gaze event 结构？

建议指标：

- fixation detection F1
- fixation center / duration error
- saccade onset / offset error
- event boundary F1
- scanpath similarity

### 6.3 第三层：语义层

回答：

> 预测结果能不能更好地支持 scene/object/task 解释？

建议指标：

- object hit / AOI attribution accuracy
- dwell time per object / AOI
- object transition consistency
- task / phase / event inference accuracy

## 7. 对当前研究最值得做的实验对照

为了把 local 和 world 的问题讲清楚，建议至少做这三组实验：

### A. `world -> world`

这是之前 SparseGaze 的路线。

优点：

- 最接近 scene/task 目标

缺点：

- head 的收益更难解释

### B. `local -> local`

更接近 HAGI / HAGI++ 的路线。

优点：

- 更能直接衡量 head 对 local eye prediction 的帮助

缺点：

- 不能直接回答“用户在场景里看哪里”

### C. `predict local, compose to world`

这是当前最推荐的路线。

优点：

- 训练目标更干净
- 与 HAGI / HAGI++ 的启发一致
- 仍然可以在 world / scene 下做最终评估

## 8. 对当前 ADT 导出流程的直接要求

为了支持上面的实验，当前 ADT CSV 最好同时保留两类标签：

### 8.1 local labels

- `yaw_rad`
- `pitch_rad`
- `depth_m`（沿 CPF gaze ray 的距离，不是 CPF `z` 坐标）
- `gaze_dir_cpf_unit_x/y/z`

### 8.2 world labels

- `gaze_origin_scene_*`
- `gaze_point_scene_*`
- `gaze_dir_scene_unit_x/y/z`

这样后面可以很快切换：

- 用 local label 训练
- 用 world label 评估
- 或直接比较 world-world 和 local-local 两条路线

## 9. 当前更合理的研究叙述

如果之后要写论文或开题，建议把 SparseGaze 讲成两句话，而不是一句：

1. SparseGaze 在低频 eye tracker 条件下，能够更好地利用 head 信息恢复高频 gaze。
2. 这种恢复不仅降低了 gaze reconstruction error，还改善了基于 gaze 的
   scene / object / task analysis。

第二句不能只靠 MAE 证明，必须有 downstream evaluation。

## 10. 下一步建议

按优先级排序：

1. 在当前 ADT 导出中补 `gaze_dir_cpf_unit_xyz` 和 `gaze_dir_scene_unit_xyz`
   这两类 unit direction。当前仓库已经实现为扁平 CSV 列：
   `gaze_dir_cpf_unit_x/y/z` 和 `gaze_dir_scene_unit_x/y/z`，同时在
   `GazeSample` 上提供 `gaze_dir_cpf_unit_xyz` /
   `gaze_dir_scene_unit_xyz` 便捷属性。 √
2. 在 event analysis 阶段优先建立两类 downstream evaluation：
   - fixation / event
   - object / AOI attribution
3. 在 SparseGaze 仓库里明确区分：
   - training target
   - reporting target
   - downstream analysis target

## References / 参考

- HAGI: Head-Assisted Gaze Imputation for Mobile Eye Trackers. UIST 2025.
  https://www.collaborative-ai.org/publications/jiao25_uist.pdf
- HAGI++: Head-Assisted Gaze Imputation and Generation. arXiv 2025.
  https://www.collaborative-ai.org/publications/jiao25_arxiv.pdf
- 当前 ADT 提取说明：
  - `docs/tutorial_gaze_feature_extraction.md`
  - `outputs/reports/*_gaze_summary.json`
