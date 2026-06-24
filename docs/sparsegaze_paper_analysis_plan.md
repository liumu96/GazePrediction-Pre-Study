# SparseGaze Paper Analysis Plan

这份文档沉淀当前 SparseGaze 收尾阶段的 paper 分析计划。目标不是罗列所有可能指标，而是把论文需要支撑的 claim、已有证据、待补实验、预期结果、图表形式和参考论文对应起来。

## 1. 当前 Paper Story

SparseGaze 的核心问题是：在低频 mobile eye tracker 条件下，如何利用 sparse gaze anchors 和高频 head motion 恢复缺失帧的高频 gaze。

更合适的论文叙事是两层：

1. SparseGaze 在 missing-frame gaze reconstruction 上，比简单 temporal baselines 和相关 head-assisted imputation 方法更合理地恢复 gaze dynamics。
2. 这种恢复不仅降低 direction error，还能更好地保留 gaze event、scanpath、gaze point 和 scene/object-level attention analysis。

当前结果已经能支撑的问题设定：

- Sparse gaze anchors 变稀疏后，missing-frame error 明显上升。
- Transition frames 明显比 fixation frames 难。
- Head rotation 比 head translation 更有用。
- Head history 比 current head 更适合用于 sparse-gaze residual correction。
- Anchor-gap position 是必须报告的关键维度：error 会随离前一个 anchor 越远而上升。

当前最大风险：

- 当前 deployable `SparseGaze rollout` 在 6Hz common-frame 结果上还没有稳定超过 HAGI++。
- `rollout_linear` / `rollout_pchip` 略好，但更像 interpolation/repair variant。
- `rollout_gt` 是 oracle-like upper bound，不能作为主 deployable 结果。

因此，如果最终 rollout 不能稳定超过 HAGI++，论文 claim 应该从 "outperforming SOTA" 调整为 "diagnosing and improving sparse-gaze residual repair with head motion"，并把贡献重点放在任务定义、分析框架、gap/event-aware evaluation 和 scene-level utility 上。

## 2. Claim-Evidence Map

| Claim | 当前证据 | 缺口 | 需要的图/表 |
| --- | --- | --- | --- |
| Sparse gaze recovery error grows inside anchor gaps. | 已实现 anchor-gap position analysis。6Hz 下 rollout 从 gap position 0.2 的约 1.58 deg 增至 0.8 的约 4.51 deg。 | 需要确认多频率和更多模型配置。 | Anchor-gap curve，event-conditioned gap curve。 |
| Transition frames are the main failure mode. | 已有 event-conditioned evaluation。Transition 比 fixation 高约 3 deg。 | 需要结合 gap position 和 scanpath behavior。 | Fixation/transition bar，gap x event curve，timeline case study。 |
| Head motion is useful mainly as residual correction signal. | `sparsegaze_head_utility_report.md`：head-history R2 明显高于 gap-only，transition 上更明显。 | 需要最终模型 ablation 证明，而不只是 diagnostic ridge R2。 | Ablation table，R2 diagnostic figure，lead-lag curve。 |
| Local gaze prediction plus world/scene composition gives cleaner interpretation. | `sparsegaze_modeling_notes.md` 已分析 local/world 语义。 | 需要实验对比 local-local、world-world、local-to-world。 | Local vs scene angular error table，coordinate-frame ablation。 |
| SparseGaze is useful for scene/object analysis, not only direction MAE. | 目前只有方向和 event 证据。 | 需要 gaze point、scanpath、3D view、object/AOI analysis。 | Image gaze point overlay，3D ray view，fixation object-hit agreement。 |

## 3. 已完成或已有基础的分析

### 3.1 Overall Missing-Frame Evaluation

当前位置：

- `Experiments/sparsegaze_evaluation/REPORT.md`
- `Experiments/sparsegaze_evaluation/overall_evaluation.py`
- `analysis/analyze_prediction_results.py`

已有结论：

- 6Hz common-frame comparison 下，deployable `SparseGaze rollout` 平均还未稳定超过 HAGI++。
- `rollout_linear` / `rollout_pchip` 略优于 HAGI++，但 margin 小。
- `rollout_gt` 最好，但应解释为 oracle-like upper bound。

Paper 用法：

- 主表必须区分 deployable methods 和 oracle-like upper bound。
- 报 sequence-macro MAE、frame-weighted MAE、median、p90。
- 使用 common frames，避免 coverage 差异污染模型对比。

### 3.2 Head Utility Diagnostic

当前位置：

- `docs/sparsegaze_head_utility_report.md`
- `scripts/report_sparsegaze_head_utility.py`

已有结论：

- Head rotation 明显比 translation 有用。
- Head-history R2 高于 gap-only 和 current-head。
- Head utility 在 transition / long gap 下更明显。
- Lead-lag peak 约为 -2 frames，说明在 ADT 30Hz 下 head 与 gaze dynamics 近同步或 gaze 略领先，不支持简单 "head leads gaze" claim。

Paper 用法：

- 作为 model design motivation，不直接当最终模型结果。
- 支撑 residual correction formulation：

```text
predicted_gaze_t = sparse_gaze_baseline_t + f(head_history, gaze_history, gap, event/context)
```

### 3.3 Anchor-Gap Position Analysis

当前位置：

- `analysis/analyze_anchor_gap_position.py`
- `outputs/analysis/anchor_gap_position/summary.md`
- `outputs/analysis/anchor_gap_position_all_hz/summary.md`

当前命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib-sparsegaze python analysis/analyze_anchor_gap_position.py
MPLCONFIGDIR=/tmp/matplotlib-sparsegaze python analysis/analyze_anchor_gap_position.py --all-hz --output-dir outputs/analysis/anchor_gap_position_all_hz
```

已有结论：

- 6Hz 下 error 随 normalized gap position 单调上升。
- Transition 曲线整体远高于 fixation。
- 当前 deployable rollout 与 HAGI++ 接近，但没有稳定优势。
- `rollout_gt` 在 gap positions 上 consistently lower，说明 repair/alignment 上界仍有空间。

Paper 用法：

- 这是必须放主文的图。
- 最好用 6Hz 主图，all-Hz 放 appendix 或 supplement。

## 4. 必做 TODO

### TODO 1: Final Main Result Table

目的：

证明最终 SparseGaze deployable model 在 missing-frame gaze reconstruction 上的整体表现。

怎么做：

- 固定最终 model directory。
- 使用 common frames。
- 比较 HAGI++、hold-last、linear、pchip、SparseGaze rollout、SparseGaze variants。
- 排除 anchor frames，只评估 `eval_mask & ~anchor_mask`。
- 同时报告 sequence-macro 和 frame-weighted metrics。

指标：

- Angular MAE。
- Median angular error。
- p90 angular error。
- Sequence win count。
- Coverage / retained ratio。

期望结果：

- 理想：deployable SparseGaze rollout 在 sequence-macro MAE 和 p90 上稳定优于 HAGI++。
- 可接受：overall 接近 HAGI++，但在 transition / long-gap / high-head-motion 子集上明显优于 HAGI++。
- 风险：只靠 `rollout_linear` / `rollout_gt` 赢。若如此，主 claim 需要降低。

图表：

- Table 1: Main comparison。
- Per-sequence heatmap。
- Delta vs HAGI++ scatter/box。

### TODO 2: Anchor-Gap Position Curves

目的：

证明 SparseGaze 任务的核心困难来自 anchor interval 内部，而不是简单 overall average。

怎么做：

- 使用 `analysis/analyze_anchor_gap_position.py`。
- 先跑 6Hz common frames。
- 再跑 all-Hz 检查频率敏感性。
- 保留 HAGI++，但给 HAGI++ 贴 SparseGaze 的 anchor schedule，只比较 shared missing frames。

指标：

- Sequence-macro MAE by normalized gap position。
- p90 by normalized gap position。
- Event-conditioned gap curves。

期望结果：

- Error 随 gap position 单调上升。
- SparseGaze 应在 gap 后半段优于 baselines，尤其 transition frames。
- 如果 SparseGaze 只在靠近 anchor 的位置好，说明 rollout propagation 仍弱。

图表：

- Fig. Anchor-gap error curve。
- Fig. Anchor-gap x event curve。

### TODO 3: Model Ablation Table

目的：

回答 reviewer 最可能问的问题：head、history、residual、local target 各自贡献是什么。

怎么做：

至少比较：

- Gaze-only。
- Current head。
- Head history。
- Rotation-only。
- Rotation + translation。
- Residual correction vs direct prediction。
- Local prediction vs world prediction。
- Local prediction + world/scene composition。

指标：

- Overall missing-frame MAE。
- Fixation MAE。
- Transition MAE。
- Long-gap MAE，例如 normalized position >= 0.6。
- p90。

期望结果：

- Head history > current head。
- Rotation-only 接近或优于 rotation+translation，除非 translation 提供明确增益。
- Residual correction 在 long-gap 或 transition 上更有优势。
- Local-to-world 有更清晰的解释，并且 scene/world error 不下降太多。

图表：

- Table 2: Ablation。
- Optional grouped bar: transition / long-gap subset。

### TODO 4: Gaze Point Evaluation

目的：

补足 direction error 之外的 projected gaze quality，回答用户真正看向哪里。

怎么做：

分两层做：

1. Image-space gaze point。
2. Scene/3D projected gaze point。

Image-space 指标：

- Pixel L2 error。
- Normalized pixel error，除以 image diagonal。
- Valid projection ratio。
- Fixation-only pixel error。
- Transition-only pixel error。

Scene/3D 指标：

- Fixed-depth projected point distance。
- GT-depth projected point distance，必须标注为 diagnostic。
- Ray/AOI/object hit agreement。

期望结果：

- SparseGaze 的 projected point 更少 lag / overshoot。
- Transition 段比 baseline 更接近 GT trajectory。
- Fixation frames 上 object/AOI attribution 更稳定。

注意：

- 如果 3D point 使用 GT depth，不能 claim 模型预测了 true 3D fixation point。
- 3D Euclidean error 不应取代 angular error，因为同样角度误差会随 depth 放大。

图表：

- Image-space gaze point overlay。
- Pixel error over time with event ribbon。
- 3D ray / projected endpoint view。

### TODO 5: Scanpath Evaluation

目的：

证明 SparseGaze 恢复的不是独立帧，而是 temporal viewing behavior。

怎么做：

优先做 reviewer 容易理解的简单指标，不先上复杂 DTW。

指标：

- Scanpath length error。
- Step angular velocity profile。
- Velocity distribution distance。
- Fixation center displacement。
- Fixation duration error。
- Event boundary F1。
- Transition onset/offset timing error。

期望结果：

- SparseGaze 比 hold-last 更少 stair-step。
- 比 linear/pchip 更少 oversmoothing 或 transition delay。
- Transition-heavy windows 中更贴近 GT scanpath。

图表：

- Qualitative scanpath overlay：GT / SparseGaze / HAGI++ / linear / pchip。
- Temporal error curve + sparse anchors + event ribbon。
- Velocity distribution comparison。

### TODO 6: Scene/Object-Level Downstream Evaluation

目的：

证明 gaze reconstruction 对下游 scene understanding 有意义。

怎么做：

- 先只做 fixation frames。
- 使用 object boxes / AOI hits。
- 优先 category-level agreement，再做 instance-level。

指标：

- Fixation-only object hit agreement。
- Category-level AOI accuracy。
- Dwell time per object/AOI error。
- Object transition consistency。

期望结果：

- SparseGaze 在 fixation object attribution 上优于 baselines。
- Dwell time error 更低。
- Transition object hits 不作为主结果，因为噪声较大。

图表：

- Object/AOI agreement bar。
- Category confusion matrix。
- Dwell time error plot。

### TODO 7: 3D Scene View

目的：

给 paper 一个 scene-level qualitative evidence，展示方向误差如何影响真实场景解释。

怎么做：

- 选 transition-heavy 和 object-rich sequence。
- 在 scene 中画 GT ray、SparseGaze ray、HAGI++ ray、linear/pchip ray。
- 可显示 fixed-depth 或 GT-depth projected endpoint，但需要清楚标注。

期望结果：

- SparseGaze ray 更贴近 GT ray。
- 在 transition 期间不明显滞后。
- 能直观看出 angular improvement 是否真的对应 scene-level improvement。

图表：

- 3D ray comparison。
- 3D projected scanpath。
- Failure case：angular error 不大但 object attribution 错，或 angular error 大导致 scene point 明显漂移。

### TODO 8: Frequency Sensitivity

目的：

证明 SparseGaze 不是只在 6Hz 上有效，而是对不同低频 gaze rate 有稳定行为。

怎么做：

- 跑 6Hz / 10Hz / 15Hz。
- 如果有 3Hz 或更低频，也应加入。
- 对每个 frequency 报 overall、event、gap-position。

指标：

- Missing-frame MAE by frequency。
- p90 by frequency。
- Long-gap subset MAE。

期望结果：

- 频率越低，所有方法误差越大。
- SparseGaze 相对 baselines 的优势应在更低频或更长 gap 上更明显。

图表：

- Frequency curve。
- Frequency x event table。

## 5. Figure Plan

建议主文图表顺序：

1. Fig. 1: SparseGaze task and pipeline。显示 sparse gaze anchors、head history、residual correction、local-to-world composition。
2. Fig. 2: Head utility diagnostic。Residual vs anchor interval、head-history R2、lead-lag curve。
3. Table 1: Main missing-frame result。HAGI++、interpolation、SparseGaze variants、oracle-like upper bound。
4. Fig. 3: Anchor-gap position curve。主文必须有。
5. Fig. 4: Event-conditioned gap curve。Fixation vs transition。
6. Table 2: Ablation。Gaze-only、current head、head history、residual、local/world。
7. Fig. 5: Gaze point / scanpath qualitative comparison。
8. Fig. 6: 3D scene view or object/AOI downstream result。

Appendix:

- Per-sequence heatmap。
- All-Hz anchor-gap curves。
- Yaw/pitch residual cloud。
- Coverage table。
- Additional qualitative cases。

## 6. Reference Papers and How They Support the Story

### HAGI / HAGI++

Use for:

- Missing gaze imputation framing。
- Head-assisted gaze imputation baseline。
- Tracker-centric / local gaze representation motivation。

Relevant claims:

- SparseGaze should compare against HAGI++ on common frames。
- Local gaze representation is a defensible target。
- Head information should be treated as auxiliary signal, not head-only gaze generation.

### SGaze: A Data-Driven Eye-Head Coordination Model for Realtime Gaze Prediction

Use for:

- Eye-head coordination motivation。
- Relationship between gaze positions and head angular velocity。
- Latency / lead-lag discussion。

Relevant claims:

- Head motion contains information about gaze dynamics。
- But lead-lag should be empirically checked rather than assumed。

### Pose2Gaze

Use for:

- Gaze behavior generation from pose / body motion。
- Eye-body coordination as a model design motivation。

Relevant claims:

- Body/head motion can help predict gaze behavior。
- SparseGaze is narrower: imputation from sparse gaze anchors plus head, not full gaze generation from pose.

### Gaze-in-the-Wild

Use for:

- Natural head-free gaze behavior analysis。
- Eye-head coordination and gaze event classification。

Relevant claims:

- Event-conditioned analysis is necessary。
- Fixation and transition/saccade-like frames should not be averaged blindly。

### Feature-space / encoding-model style ridge diagnostics

Use for:

- Justifying head utility diagnostic as feature-space incremental value analysis。
- Explaining gap-only vs current-head vs head-history R2。

Relevant claims:

- Diagnostic R2 supports feature usefulness。
- It does not prove the final neural architecture is optimal。

## 7. Current Interpretation Rules

Do:

- Report missing-frame metrics, not anchor-frame metrics, as the main result。
- Use common frames for HAGI++ comparisons。
- Treat `rollout_gt` as upper bound。
- Separate fixation and transition。
- Separate direction, projected gaze point, scanpath, and semantic/object analyses。
- Be explicit when 3D points use fixed depth or GT depth。

Do not:

- Claim 3D fixation point prediction if only gaze direction is predicted。
- Use pixel error as the primary metric without projection-validity discussion。
- Claim head leads gaze unless lead-lag results support it。
- Claim downstream scene/object improvement from angular MAE alone。
- Present oracle-like repair as deployable SparseGaze.

## 8. Immediate Next Actions

- [ ] Pick final model directories for paper comparison。
- [ ] Re-run main overall evaluation with final deployable model and baselines。
- [ ] Re-run anchor-gap analysis for final model, all selected frequencies。
- [ ] Build ablation table: gaze-only, current-head, head-history, residual, local/world。
- [ ] Add image-space gaze point metrics and qualitative scanpath overlay。
- [ ] Add one 3D scene view case study。
- [ ] Add fixation-only object/AOI agreement if scene assets are reliable。
- [ ] Decide final paper claim based on whether deployable rollout beats HAGI++ overall or only on transition/long-gap subsets。
