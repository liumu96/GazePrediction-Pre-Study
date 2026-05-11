# SparseGaze Head Utility Report

## 中文速读结论

这份报告的结论只限定在当前 ADT 30Hz、GT gaze/head diagnostic 上，还不是
SparseGaze 模型本身的最终实验结论。

先说明几个术语：

- `gaze anchor`：稀疏 eye tracker 真正观测到的 gaze 点。例如 N=10 表示每
  10 帧保留 1 个 GT gaze，其余 9 帧当作缺失 gaze。
- `hold-last baseline`：最简单的补全方法，缺失帧直接沿用上一个 gaze anchor。
- `linear interpolation baseline`：离线上界 baseline，用前后两个 anchors 做插值。
- `residual`：baseline 预测和 GT gaze 之间的角度误差。这里的
  `Scene residual` 是世界/场景坐标系下的 gaze 方向误差，更接近“看向场景哪里”的
  误差。
- `gap-only R2`：只用“当前帧离上一个 anchor 有多远”解释 residual。
- `current-head R2`：在 gap 信息之外，加当前帧 head motion 后解释 residual。
- `head-history R2`：在 current head 之外，再加最近一段 head motion 和累计 head
  rotation 后解释 residual。
- `transition`：scene-direction event label 里的非 fixation / 高动态片段。
- `lead-lag peak = -2 frames`：head 和 gaze 的相关性最高时，gaze dynamics 比
  head rotation 早约 2 帧出现；它不是说 head 领先 gaze。

当前比较稳的判断是：`head rotation` 是有用的，但它更像是
`sparse-gaze baseline residual correction signal`，而不是“只用当前 head 就能强预测
未来 gaze”的信号。

这里的 `correction` 不是说模型只预测一个事后修正量，也不是否定把 head history
作为模型输入。它的含义是：模型不应该被理解成“从 head 直接生成 gaze”，而应该
被理解成“先有一个由 sparse gaze anchors 提供的 gaze baseline / temporal prior，
再用 head history 去解释并修正 baseline 在缺失帧上的误差”。等价的建模形式可以是：

```text
predicted_gaze_t = sparse_gaze_baseline_t + f(head_history, gaze_history, gap, event/context)
```

其中 `f(...)` 可以显式预测 residual，也可以隐式地通过 neural model 学到这种修正。
这和已有模型把 head history 作为 input 并不矛盾；当前结果反而支持 head history
比单帧 current head 更值得保留。

具体来说：

- head rotation 明显比 head translation 更有用。
- 当 gaze anchors 变稀疏时，简单 baseline 的 Scene residual 会快速变大：
  N=10 hold-last residual 是 `6.528 deg`，N=30 增到 `14.465 deg`。
- 加 head 特征后，Scene residual magnitude 的解释能力明显提升：
  N=10 hold-last 下，gap-only R2 是 `0.093`，current-head R2 是 `0.346`，
  head-history R2 是 `0.379`。
- head 的价值主要出现在 transition / 高动态段。N=10 hold-last 下，
  fixation residual 是 `3.465 deg`，transition residual 是 `8.641 deg`；
  transition 下 head-history R2 也更高。
- lead-lag peak 在 `-2 frames`：Scene velocity vs head rotation 的 mean
  correlation 是 `0.405`，CPF-local velocity vs head rotation 是 `0.357`。
  这说明在 ADT 30Hz 下，gaze dynamics 更像先发生，head rotation 同步或略滞后，
  不是明确领先 gaze。

对 SparseGaze 的直接启发是：下一版模型/分析不应该只喂 current head motion。
更有依据的方向是加入 head history、cumulative head rotation、anchor-gap-aware
或 event-aware residual correction。当前结果不支持“只靠当前 head 就能强预测
gaze”的简单假设。

## Scope

This report evaluates whether head motion contains useful information for recovering high-frequency gaze between sparse gaze anchors. It is not a full model evaluation; it is a diagnostic step before changing the SparseGaze model.

- Input directory: `/mnt/d/SparseGaze/ADT-Gaze`
- Sequence count: 34
- Required inputs: `*_gaze_samples.csv`, `*_head_samples.csv`, `*_scene_gaze_event_features.csv`, `*_scene_gaze_frame_labels.csv`

## Method

For each sequence, gaze is sparsified by keeping one anchor every N frames (N = 2, 3, 5, 10, 15, 30). Non-anchor frames are reconstructed with two baselines: hold-last and linear interpolation. The residual is the angular distance between the baseline prediction and ground-truth gaze, computed in both CPF-local gaze direction and Scene gaze direction.

Head utility is measured in two ways. First, residual magnitude is correlated with current and cumulative head rotation. Second, a blocked cross-validated ridge diagnostic predicts residual magnitude from three feature sets: gap-only (position inside the anchor interval), current-head, and head-history. A useful head signal should improve R2 over gap-only, especially as the anchor interval grows.

Lead-lag curves compute corr(head motion at frame t, gaze dynamics at frame t+k). Positive lag means the current head signal is compared with future gaze dynamics; negative lag means gaze dynamics precedes head motion.

## Coordinate-Frame Audit

The CPF-related analyses do not compare CPF gaze direction with CPF head forward
direction. In CPF coordinates, head forward is constant by construction because
CPF/head is the local device frame. Therefore CPF results should be read as
`CPF-local gaze dynamics vs inter-frame head motion`, not as `CPF gaze vs CPF
head direction`.

Head rotation features are inter-frame motion features:
`head_rotation_speed_deg_s`, `head_rotation_angle_step_deg`, and relative
rotation vectors derived from `R_{t-1}^{-1} R_t` in the previous head/CPF frame.
These are dynamic even though the current-frame CPF forward axis is fixed.

`head_translation_speed_m_s` is a scalar origin-speed feature computed from
Scene/world displacement magnitude. It is useful as motion intensity, but it is
not a CPF translation direction. Directional local translation would require
`translation_prev_head_dxyz_m`, which this SparseGaze utility diagnostic does
not currently use.

## Method Rationale and Related Work

This analysis should be read as a model-design diagnostic, not as the final
SparseGaze model. The goal is to test whether head-motion features provide
incremental information beyond sparse gaze anchors.

The rationale comes from three related lines of work. First, eye-head and
eye-body coordination studies show that gaze is statistically coupled with
head/body motion. SGaze reports a linear relation between gaze positions and
head rotation angular velocities and analyzes eye/head latency. Pose2Gaze first
performs eye-body coordination analysis, then designs a gaze generation model
using head/full-body pose. Gaze-in-the-Wild shows that head-free natural gaze
behavior requires eye+head analysis and that head movement information helps
event classification.

Second, gaze imputation work is directly relevant to SparseGaze. HAGI and
HAGI++ address missing gaze data using head orientation as an auxiliary modality,
and compare against interpolation and generic time-series imputation baselines.
This supports framing SparseGaze as sparse gaze anchors plus auxiliary head
motion, rather than head-only gaze prediction.

Third, the ridge/R2 diagnostic follows the broader feature-space utility
analysis pattern used in encoding-model work: use a simple regularized model
and cross-validation to test whether adding a feature space improves explained
variance. Here the feature spaces are gap-only, current-head, and head-history;
the target is sparse-gaze baseline residual magnitude. Therefore, the R2 gain is
evidence that head motion contains usable residual information, not evidence
that ridge regression should be the final SparseGaze architecture.

Representative references:

- Hu et al., 2019, *SGaze: A Data-Driven Eye-Head Coordination Model for
  Realtime Gaze Prediction*, IEEE TVCG.
  https://www.collaborative-ai.org/publications/hu19_tvcg/
- Hu et al., 2024, *Pose2Gaze: Generating Realistic Human Gaze Behaviour from
  Full-body Poses using an Eye-body Coordination Model*, IEEE TVCG.
  https://arxiv.org/abs/2312.12042
- Kothari et al., 2020, *Gaze-in-Wild: A Dataset for Studying Eye and Head
  Coordination in Everyday Activities*, Scientific Reports.
  https://www.nature.com/articles/s41598-020-59251-5
- Jiao et al., 2025, *HAGI: Head-Assisted Gaze Imputation for Mobile Eye
  Trackers*, ACM UIST.
  https://www.hcics.simtech.uni-stuttgart.de/publications/jiao25_uist/
- Jiao et al., 2025, *HAGI++: Head-Assisted Gaze Imputation and Generation*,
  arXiv.
  https://www.collaborative-ai.org/publications/jiao25_arxiv/
- Dupré la Tour et al., 2022, *Feature-space Selection with Banded Ridge
  Regression*, NeuroImage.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9807218/

## Main Results

At a 10-frame anchor interval, hold-last Scene residual is 6.528 deg on average, while linear interpolation gives 3.458 deg. At a 30-frame interval, hold-last Scene residual rises to 14.465 deg. This tells us how quickly missing-gaze error grows as gaze becomes sparse.

For N=10 hold-last Scene residuals, gap-only R2 is 0.093, current-head R2 is 0.346, and head-history R2 is 0.379. The important quantity is not the absolute R2 alone, but whether current/head-history improves over gap-only. A small gain means head motion is only weakly explanatory for baseline residual magnitude under this diagnostic.

Event conditioning checks whether head is more useful outside stable viewing. For N=10 hold-last, fixation Scene residual is 3.465 deg and transition Scene residual is 8.641 deg. Head-history R2 is 0.209 for fixation and 0.372 for transition.

Lead-lag results show where head rotation aligns most strongly with gaze dynamics. For Scene velocity, the strongest mean correlation with head rotation appears at lag -2 frames (mean r=0.405). For CPF-local velocity, it appears at lag -2 frames (mean r=0.357). If the peak is near zero, head and gaze dynamics are mostly synchronous at 30 Hz; if it is positive, current head motion has more predictive relation to future gaze dynamics; if negative, gaze tends to lead head.

## Tables

### Table 1. Sparse-anchor residuals

This table measures how much error remains when gaze is removed between anchors. Scene residual is the more task-facing quantity; CPF residual isolates eye-in-head local gaze change.

| N | baseline | event | CPF residual deg | Scene residual deg | samples |
| --- | --- | --- | --- | --- | --- |
| 5 | hold_last | all | 3.427 | 3.585 | 2198 |
| 5 | linear_interp | all | 1.968 | 1.947 | 2196 |
| 10 | hold_last | all | 5.651 | 6.528 | 2473 |
| 10 | linear_interp | all | 3.379 | 3.458 | 2468 |
| 15 | hold_last | all | 7.182 | 8.858 | 2564 |
| 15 | linear_interp | all | 4.407 | 4.712 | 2556 |
| 30 | hold_last | all | 10.201 | 14.465 | 2656 |
| 30 | linear_interp | all | 6.814 | 8.039 | 2641 |

### Table 2. Head feature utility for residual magnitude

This table compares blocked-CV R2 for predicting residual magnitude. `gap` uses only position inside the sparse interval; `current` adds current head motion; `history` adds recent and cumulative head motion. The meaningful signal is the R2 gain from `gap` to `current` or `history`.

| N | baseline | event | gap R2 | current-head R2 | head-history R2 |
| --- | --- | --- | --- | --- | --- |
| 5 | hold_last | all | 0.051 | 0.231 | 0.282 |
| 5 | hold_last | fixation | 0.117 | 0.111 | 0.093 |
| 5 | hold_last | transition | 0.073 | 0.224 | 0.268 |
| 10 | hold_last | all | 0.093 | 0.346 | 0.379 |
| 10 | hold_last | fixation | 0.153 | 0.186 | 0.209 |
| 10 | hold_last | transition | 0.099 | 0.340 | 0.372 |
| 15 | hold_last | all | 0.116 | 0.369 | 0.433 |
| 15 | hold_last | fixation | 0.168 | 0.242 | 0.295 |
| 15 | hold_last | transition | 0.119 | 0.369 | 0.432 |
| 30 | hold_last | all | 0.116 | 0.335 | 0.526 |
| 30 | hold_last | fixation | 0.144 | 0.265 | 0.481 |
| 30 | hold_last | transition | 0.111 | 0.338 | 0.494 |

### Table 3. Strongest lead-lag correlations

This table asks whether head motion is synchronized with, leads, or lags gaze dynamics. Positive lag means head at frame t is compared with gaze at a future frame.

| target | head feature | best lag | mean corr | median corr |
| --- | --- | --- | --- | --- |
| scene_velocity_deg_s | head_rotation_speed_deg_s | -2 frames (mean r=0.405) | 0.405 | 0.407 |
| scene_velocity_deg_s | head_translation_speed_m_s | -2 frames (mean r=0.084) | 0.084 | 0.082 |
| cpf_local_velocity_deg_s | head_rotation_speed_deg_s | -2 frames (mean r=0.357) | 0.357 | 0.360 |
| cpf_local_velocity_deg_s | head_translation_speed_m_s | -3 frames (mean r=0.086) | 0.086 | 0.076 |
| scene_transition_indicator | head_rotation_speed_deg_s | -3 frames (mean r=0.302) | 0.302 | 0.296 |
| scene_transition_indicator | head_translation_speed_m_s | -1 frames (mean r=0.169) | 0.169 | 0.161 |

## Figures

### Figure 1. Residual vs sparse anchor interval

![Figure 1. Residual vs sparse anchor interval](../outputs/figures/sparsegaze_head_utility/residual_vs_anchor_interval.png)

### Figure 2. Scene residual R2 vs sparse anchor interval

![Figure 2. Scene residual R2 vs sparse anchor interval](../outputs/figures/sparsegaze_head_utility/scene_residual_r2_vs_anchor_interval.png)

### Figure 3. Event-conditioned Scene residual R2

![Figure 3. Event-conditioned Scene residual R2](../outputs/figures/sparsegaze_head_utility/event_conditioned_scene_r2.png)

### Figure 4. Lead-lag relation for head rotation

![Figure 4. Lead-lag relation for head rotation](../outputs/figures/sparsegaze_head_utility/lead_lag_head_rotation.png)

## Interpretation

This analysis is useful if it produces one of three outcomes. First, if head-history R2 consistently improves over gap-only, SparseGaze should explicitly model recent head trajectory rather than only current head. Second, if gain appears mainly in transition frames or long anchor intervals, head features should be event/gap-aware instead of uniformly weighted. Third, if lead-lag peaks are not near zero, the model should align head history with gaze targets using the observed temporal offset.

If these gains are small, that is also actionable: head motion may still help as a regularizer or context signal, but a large architecture change based only on head features would be weakly supported. The next step would then be model-output diagnostics: compare where SparseGaze errors grow relative to anchor gap, Scene event label, and head-motion regime.
