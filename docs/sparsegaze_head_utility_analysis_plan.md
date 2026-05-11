# SparseGaze Head Utility Analysis Plan

## Current Status

This plan has been implemented for ADT GT-only diagnostics.

Scripts:

- `src/adt_sandbox/sparsegaze_head_utility.py`
- `scripts/analyze_sparsegaze_head_utility.py`
- `scripts/report_sparsegaze_head_utility.py`

The full `/mnt/d/SparseGaze/ADT-Gaze` export has been processed. Main outputs:

- `/mnt/d/SparseGaze/ADT-Gaze/batch_sparsegaze_head_utility_summary.csv`
- `/mnt/d/SparseGaze/ADT-Gaze/batch_sparsegaze_head_utility_lead_lag.csv`
- `/mnt/d/SparseGaze/ADT-Gaze/batch_sparsegaze_head_utility_aggregate.csv`
- `/mnt/d/SparseGaze/ADT-Gaze/batch_sparsegaze_head_utility_lead_lag_aggregate.csv`
- `docs/sparsegaze_head_utility_report.md`
- `outputs/figures/sparsegaze_head_utility/*.png`

Current high-level result: head rotation is meaningfully more useful than head
translation. It explains sparse-anchor residuals better than a gap-only
baseline, especially for Scene residuals, transition frames, and larger anchor
intervals. Lead-lag peaks are currently at negative lag in ADT, meaning gaze
dynamics tend to precede head rotation by about 2 frames under this diagnostic.

Coordinate-frame audit:

- CPF head forward is constant by construction, because CPF/head is the local
  device frame.
- Therefore CPF analyses must not compare `gaze_dir_cpf` with `head_forward_cpf`.
- Current CPF analyses compare CPF-local gaze dynamics with inter-frame head
  motion: `relative_rot_prev_to_cur`, `head_rotation_speed_deg_s`, rotation
  vectors in the previous head/CPF frame, and cumulative head rotation.
- `head_translation_speed_m_s` is a scalar Scene-origin displacement speed. It
  can be used as motion intensity, but it is not a CPF translation direction.
  Local directional translation should use `translation_prev_head_*`.

## Method Rationale and Related Work

This analysis is not meant to claim that ridge regression is the final
SparseGaze model. It is a lightweight diagnostic to answer a narrower question:

```text
Does head motion provide incremental information beyond sparse gaze anchors?
```

The method combines three established ideas.

First, eye-head and eye-body coordination work shows that gaze is not independent
of head/body motion. SGaze reports a linear relationship between gaze positions
and head rotation angular velocities, and also observes latency between eye and
head movements. Pose2Gaze follows a similar research pattern: it first performs
a comprehensive eye-body coordination analysis, then designs a model based on
the observed correlations and time delays. Gaze-in-the-Wild also shows that
head-free gaze behavior requires eye+head analysis, and that head movement
information helps event classification under natural movement.

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

Second, gaze imputation work is directly relevant to SparseGaze. HAGI and
HAGI++ explicitly address missing gaze data and use head orientation as an
auxiliary signal. They compare against interpolation and generic time-series
imputation baselines, which supports our decision to frame SparseGaze as
"sparse anchors plus auxiliary head motion" rather than "head-only gaze
prediction".

Representative references:

- Jiao et al., 2025, *HAGI: Head-Assisted Gaze Imputation for Mobile Eye
  Trackers*, ACM UIST.
  https://www.hcics.simtech.uni-stuttgart.de/publications/jiao25_uist/
- Jiao et al., 2025, *HAGI++: Head-Assisted Gaze Imputation and Generation*,
  arXiv.
  https://www.collaborative-ai.org/publications/jiao25_arxiv/

Third, using a regularized linear model and cross-validated R2 to test feature
utility is common in feature-space / encoding-model analyses. The point is not
to maximize final task accuracy, but to test whether adding a feature space
improves explained variance under a controlled model. In this project, the
feature spaces are:

```text
gap-only
gap + current head
gap + current head + head history
```

The target is:

```text
sparse-gaze baseline residual magnitude
```

Representative reference:

- Dupré la Tour et al., 2022, *Feature-space Selection with Banded Ridge
  Regression*, NeuroImage.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9807218/

Therefore, the current diagnostic should be described as:

```text
We use a lightweight cross-validated residual diagnostic, inspired by
feature-space utility analysis, to test whether head-motion features provide
incremental information beyond sparse-gaze baselines.
```

It should not be described as:

```text
Ridge regression is the proposed SparseGaze model.
```

## 1. Goal

This analysis should answer a SparseGaze-specific question:

```text
When gaze is sparse, what usable information does head motion provide for
recovering missing high-frequency gaze?
```

This is different from the current head-gaze relationship reports. The existing
reports characterize GT head-gaze statistics in ADT:

- head rotation is more informative than translation
- larger head rotation is associated with larger CPF/Scene gaze velocity
- direction-level relationships are weaker than magnitude-level relationships
- high head rotation is less likely to be scene fixation

Those findings justify that head is relevant, but they do not yet decide how a
SparseGaze model should use head. The new analysis should move from correlation
description to prediction-oriented diagnostics.

## 2. Core Hypotheses

### H1. Head motion is useful mainly for residual correction

SparseGaze does not need to know whether raw head and raw gaze are correlated.
It needs to know whether head explains the error left by a sparse-gaze baseline.

Expected useful result:

```text
head features explain residuals between sparse-anchor baseline gaze and GT gaze
```

If this is true, head should be used as a correction signal. If false, head may
only provide weak context.

### H2. Head utility depends on sampling gap

Head may be unnecessary when gaze anchors are dense, but useful when the gap
between gaze anchors becomes larger.

Expected useful result:

```text
head residual explainability increases as anchor interval increases
```

This would directly support SparseGaze's low-frequency eye-tracker setting.

### H3. Head utility is event-dependent

Head may be useful in transition/high-motion frames, while fixation frames may
already be well handled by last-gaze or interpolation baselines.

Expected useful result:

```text
head features explain more residual variance in transition than fixation
```

This would motivate event-aware or motion-regime-aware modeling.

### H4. Head may be synchronous, leading, or lagging gaze

For prediction, it matters whether head motion comes before gaze change, after
gaze change, or only at the same time.

Expected useful result:

```text
corr(head_t, gaze_{t+k}) peaks at k > 0
```

where `k > 0` means current head motion is associated with future gaze dynamics.
If the peak is at `k < 0`, head is lagging gaze and is less useful for real-time
prediction.

### H5. Head history may be more useful than current head alone

The current SparseGaze-style input is sequential. Therefore the analysis should
compare current head, head history, and oracle future head.

Expected useful result:

```text
head history explains more residual than current head only
```

If oracle future head is much stronger than history, then head is informative
but not available causally, which explains weak model gains.

## 3. Data Inputs

For ADT, use the current derived files in `/mnt/d/SparseGaze/ADT-Gaze`:

- `*_gaze_samples.csv`
- `*_head_samples.csv`
- `*_scene_gaze_event_features.csv`
- `*_scene_gaze_frame_labels.csv`
- optionally `*_scene_head_gaze_analysis_rows.csv`

Key gaze representations:

- CPF/local:
  - `yaw_rad`
  - `pitch_rad`
  - `gaze_dir_cpf_unit_x/y/z`
- Scene/world:
  - `gaze_dir_scene_unit_x/y/z`
  - `scene_velocity_deg_s`
  - `scene_event_label`

Key head representations:

- absolute:
  - `head_forward_scene_unit_x/y/z`
  - `head_rot_scene_rij`
- relative:
  - `head_rotation_speed_deg_s`
  - `head_rotvec_prev_head_*`
  - `translation_prev_head_*`
  - `head_translation_speed_m_s`

## 4. Analysis A: Lead-Lag Predictability

### Question

Does head motion lead, lag, or synchronize with gaze dynamics?

### Method

For lag `k` in a symmetric window, for example `[-15, 15]` frames:

```text
corr(head_feature_t, gaze_feature_{t+k})
```

Run this for:

- CPF local gaze velocity
- Scene gaze velocity
- `abs(delta_yaw)`, `abs(delta_pitch)`
- signed `delta_yaw`, signed `delta_pitch`
- scene event transition indicator

Head features:

- `head_rotation_speed_deg_s`
- `head_rotvec_prev_head_x/y/z`
- `head_forward_angle_step_deg`
- `head_translation_speed_m_s`

Report:

- peak lag
- peak correlation
- whether peak is causal (`k > 0`), synchronous (`k = 0`), or lagging (`k < 0`)
- curves by sequence and batch summary

### Expected Interpretation

Useful outcomes:

- peak at `k > 0`: head can help predict future gaze dynamics
- peak at `k = 0`: head is useful as current context but less predictive
- peak at `k < 0`: head reacts after gaze; weak causal utility
- different peaks for fixation vs transition: motivates event-conditioned model

## 5. Analysis B: Sparse-Anchor Residual Explainability

### Question

When gaze anchors are sparse, can head explain the residual left by a simple
baseline?

### Sparse Sampling Setup

For anchor interval `N`, keep GT gaze every `N` frames:

```text
anchor frames: 0, N, 2N, ...
target frames: all non-anchor frames
```

Recommended intervals:

```text
N = 2, 3, 5, 10, 15, 30
```

These correspond roughly to:

- 15 Hz
- 10 Hz
- 6 Hz
- 3 Hz
- 2 Hz
- 1 Hz

### Baselines

Use simple model-free baselines:

1. hold-last-anchor:

```text
gaze_hat_t = gaze_anchor_prev
```

2. linear interpolation between anchors:

```text
gaze_hat_t = interp(gaze_anchor_prev, gaze_anchor_next)
```

For causal prediction, hold-last-anchor is the cleanest baseline. Linear
interpolation can be treated as an offline upper baseline.

### Residuals

Compute residual in both local and scene spaces:

```text
residual_angle_t = angle(gaze_hat_t, gaze_gt_t)
```

For direction:

```text
residual_yaw_t = yaw_gt_t - yaw_hat_t
residual_pitch_t = pitch_gt_t - pitch_hat_t
```

### Head Explainability

Fit lightweight diagnostic models, not full SparseGaze:

- linear regression / ridge regression
- optionally random forest or small MLP as nonlinear diagnostic

Feature groups:

1. gap only:
   - relative position inside anchor gap
   - time since last anchor
2. last gaze only:
   - last anchor gaze
   - local delta from previous anchors
3. current head:
   - current head rotation speed
   - current relative head rotation vector
4. head history:
   - head features over the previous `L` frames
   - summary stats: mean, max, cumulative rotation vector
5. oracle future head:
   - future head over `[t, t+L]`
   - only for upper-bound analysis, not deployable

Metrics:

- residual MAE reduction
- residual angular error reduction
- R² for residual magnitude
- cosine alignment for residual direction
- by-gap-position error curve

### Expected Interpretation

Useful outcomes:

- current/head history reduces residual error vs last-gaze baseline
- gain increases with larger `N`
- head history outperforms current head
- oracle future head much stronger than history means head is informative but
  temporally delayed

## 6. Analysis C: Event-Conditioned Head Utility

### Question

Does head help more during scene transition than scene fixation?

### Method

Repeat Analysis B with event stratification:

- target frame label: `fixation` / `transition`
- anchor label
- whether the gap crosses event boundary

Report for each event group:

- baseline residual
- head-enhanced residual
- head gain
- residual direction explainability

### Expected Interpretation

Useful outcomes:

- fixation:
  - low baseline error
  - small head gain
  - possible compensation cases
- transition:
  - high baseline error
  - larger head gain
  - head participates in world-gaze shift
- event boundary:
  - largest baseline error
  - hardest prediction region

This can justify event-aware training, sampling, or loss weighting.

## 7. Analysis D: Motion-Regime Utility

### Question

Does head help only when head motion is high?

### Method

Create both sequence-relative and absolute motion groups.

Sequence-relative groups:

```text
low/mid/high = per-sequence 33.3% / 66.7% head rotation speed percentiles
```

Absolute groups:

```text
low:  < 30 deg/s
mid:  30-90 deg/s
high: > 90 deg/s
```

For each group, compute:

- baseline residual
- head-enhanced residual
- head gain
- fixation / transition fraction

### Expected Interpretation

Useful outcomes:

- head gain concentrated in high-motion group
- high-motion group overlaps strongly with transition
- low-motion fixation may not need complex head modeling

This can motivate motion-regime gating.

## 8. Analysis E: Current Head vs Head History

### Question

Is current head enough, or does the model need head history?

### Method

Use the same residual diagnostic task and compare feature sets:

```text
F0: gap + last gaze
F1: F0 + current head
F2: F0 + head history
F3: F0 + head history + current head
F4: F0 + future head oracle
```

History windows:

```text
L = 3, 5, 10, 15 frames
```

Metrics:

- residual error reduction
- R²
- direction alignment
- event-conditioned gain

### Expected Interpretation

Useful outcomes:

- F2 > F1: history matters
- F1 ≈ F0: current head alone is weak
- F4 >> F2: head is informative but not available causally
- F2 improves transition more than fixation: event-aware model justified

## 9. Analysis F: Cross-Dataset Generality

### Question

Are the ADT conclusions dataset-specific or general?

### Candidate datasets

- ADT / Pose2Gaze-ADT
- Reading In the Wild
- EgoBody
- future SparseGaze datasets

### Required common fields

At minimum:

- gaze direction or yaw/pitch
- head pose / head orientation
- timestamps
- enough sampling rate to simulate sparse anchors

Optional:

- event labels
- scene/world coordinates
- object/AOI labels

### Cross-dataset table

For each dataset:

- head-gaze lead/lag peak
- head residual gain at each anchor interval
- current head vs head history gain
- event-conditioned gain if labels exist
- whether conclusions match ADT

### Expected Interpretation

Useful outcomes:

- consistent pattern across datasets:
  - stronger paper motivation
  - head utility is general
- different pattern by dataset/task:
  - head utility is task-dependent
  - model should adapt by domain or motion regime

## 10. Recommended Output Files

Suggested scripts:

- `scripts/analyze_sparsegaze_head_utility.py`
- `scripts/report_sparsegaze_head_utility.py`

Suggested core module:

- `src/adt_sandbox/sparsegaze_head_utility.py`

Suggested outputs:

- `*_sparsegaze_head_utility_rows.csv`
- `*_sparsegaze_head_utility_summary.json`
- `batch_sparsegaze_head_utility_summary.csv`
- `batch_sparsegaze_head_utility_report.json`
- `docs/sparsegaze_head_utility_report.md`
- `outputs/figures/sparsegaze_head_utility/*.png`

## 11. Implementation Order

### Step 1. ADT GT-only residual diagnostics

Implement:

- anchor interval simulation
- hold-last-anchor baseline
- residual computation in CPF and Scene
- residual summary by interval/event/head-motion group

This should not require SparseGaze model predictions.

### Step 2. Lead-lag analysis

Implement:

- lag curves for CPF and Scene dynamics
- event-conditioned lag curves
- peak lag summary

This is closest to the Pose2Gaze-style analysis.

### Step 3. Lightweight residual models

Implement:

- ridge regression for residual magnitude/direction
- feature group ablation:
  - gap + last gaze
  - current head
  - head history
  - future head oracle

This gives model-design evidence without training the full SparseGaze model.

### Step 4. Model-output diagnostics

Only after GT-only analysis is clear, use SparseGaze model outputs:

- compare model error to GT residual explainability
- check whether the model actually uses the head-utility regimes discovered in
  GT-only analysis

## 12. Expected Paper-Level Claims

Potential claims if supported:

1. Head motion is a regime-dependent cue for sparse gaze reconstruction.
2. Head rotation is more useful than translation, but speed alone is insufficient.
3. Head helps most when sparse-anchor baselines drift during transition or
   high-motion segments.
4. Current head alone may be weaker than head history, explaining why simple
   current-head features can underperform.
5. Event-aware or motion-regime-aware head conditioning is better motivated than
   uniformly injecting head features at every frame.

Claims to avoid unless proven:

- Head can directly predict gaze direction.
- Head always improves sparse gaze reconstruction.
- CPF-local event labels define scene/object fixation.
- ADT-only findings are universal across datasets.

## 13. Decision Criteria For Model Design

Use this analysis to decide:

- whether to keep head translation features
- whether to use head rotation speed or full rotation vector
- whether to use current head, head history, or both
- whether to train event-aware / motion-aware modules
- whether local-to-world modeling is better than direct world-gaze modeling
- which anchor intervals should be emphasized during training/evaluation

The analysis should produce actionable model-design decisions, not only
descriptive correlations.
