# SparseGaze Evaluation Report

This experiment folder keeps SparseGaze evaluation planning and code separate
from the general ADT sandbox utilities. The goal is to define a focused
evaluation story instead of adding unrelated metrics whenever a new derived
signal becomes available.

## Core Question

The central question is:

> Does SparseGaze recover missing high-frequency gaze better than baselines, and
> in which temporal/behavioral conditions does that improvement appear?

The evaluation should therefore prioritize missing-frame behavior, event
condition, and position inside sparse-anchor gaps. Qualitative image-space and
scene-space views are useful for explanation, but they should not replace the
direction-error metrics that directly match the model output.

## Current Model

Default SparseGaze evaluation root:

```text
/home/liumu/Github_Projects/SparseGaze/outputs/eval/adt/sparsegaze_cpf_forward_head_motion_residual_ss
```

This is the current residual model with per-sequence `sequence_predictions`
files needed for frame-level analysis. Another residual directory,
`sparsegaze_cpf_residual_update_ss`, currently appears to have summary JSONs
but not the same per-sequence NPZ files, so it is not the default target for
timeline notebooks.

## Dataset-Level Result Snapshot

The current all-sequence notebook reports 6 Hz common-frame results. Common
frames mean that a frame is evaluated only if all selected methods, including
HAGI++, have a valid row for that sequence/frame. This avoids mixing method
quality with coverage differences.

![Overall 6 Hz dashboard](outputs/overall/hz6_overall_dashboard.png)

![Per-sequence 6 Hz heatmap](outputs/overall/hz6_sequence_heatmap.png)

### 6 Hz Common-Frame Summary

Formatting convention: `gt-repair` is an oracle-like upper bound and is not
ranked as a normal method. **Bold** marks the best non-oracle method.

| method | sequence-macro MAE deg | frame-weighted MAE deg | median deg | p90 deg |
| --- | ---: | ---: | ---: | ---: |
| SparseGaze gt-repair (oracle upper bound) | 2.9291 | 2.9300 | 1.0511 | 8.2950 |
| SparseGaze linear | **3.0396** | **3.0406** | **1.1192** | 8.6065 |
| SparseGaze pchip | 3.0435 | 3.0445 | 1.1356 | 8.5930 |
| HAGI++ | 3.0949 | 3.0958 | 1.2030 | 8.7497 |
| SparseGaze rollout | 3.1124 | 3.1135 | 1.2994 | **8.4893** |

Coverage for this comparison:

| method | all-available frames | common frames | retained ratio |
| --- | ---: | ---: | ---: |
| HAGI++ | 20731 | 20731 | 1.0000 |
| SparseGaze gt-repair | 21803 | 20731 | 0.9508 |
| SparseGaze linear | 21803 | 20731 | 0.9508 |
| SparseGaze pchip | 21803 | 20731 | 0.9508 |
| SparseGaze rollout | 21803 | 20731 | 0.9508 |

### Interpretation

`SparseGaze gt-repair` should be read as an oracle-assisted upper-bound
variant, not as a deployable baseline. It wins all 10/10 sequences in the
current 6 Hz common-frame comparison, but this result mainly tells us how much
room exists if repair/alignment information is very strong.

For deployable comparison, separate the conclusion:

- The plain `SparseGaze rollout` is not yet better than HAGI++ on average in
  this snapshot.
- The interpolation-like repair variants, `linear` and `pchip`, are slightly
  better than HAGI++ but the margin is small.
- The gap between `gt-repair` and plain `rollout` suggests that the failure is
  not only the final gaze representation; interval repair or propagation
  quality is a major factor.

Event-conditioned results show the dominant failure mode:

The MAE columns use the same convention. The `transition - fixation` column is
diagnostic rather than a ranked score.

| method | fixation MAE deg | transition MAE deg | transition - fixation deg |
| --- | ---: | ---: | ---: |
| SparseGaze gt-repair (oracle upper bound) | 1.0863 | 4.1869 | 3.1006 |
| SparseGaze linear | **1.1505** | 4.3304 | 3.1799 |
| SparseGaze pchip | 1.1644 | **4.3273** | 3.1630 |
| HAGI++ | 1.2069 | 4.3810 | 3.1741 |
| SparseGaze rollout | 1.3018 | 4.3496 | 3.0478 |

Transition frames are roughly 3 degrees harder than fixation frames for every
method. Therefore the next quantitative analysis should focus on transition
segments and position inside sparse-anchor gaps, rather than adding more
overall averages.

## Evaluation Scope

Use only SparseGaze `eval_mask` frames for model-error curves and tables. In
the current rollout NPZ files, `eval_mask` excludes anchor frames:

```text
eval_mask & anchor_mask = 0
```

So these tables and plots are missing-frame evaluations, not anchor-frame
sanity checks. This avoids misleading near-zero anchor values and keeps the
metric aligned with the sparse-gaze recovery task.

## Primary Quantitative Metrics

### 1. Overall Missing-Frame Direction Error

Keep this as the primary score:

- angular MAE
- median angular error
- p90 angular error

This directly evaluates predicted gaze direction and remains the main result
for comparing models and baselines.

### 2. Event-Conditioned Direction Error

Report the same metrics separately for GT scene-gaze events:

- fixation
- transition

This answers whether the model is merely stable during fixations or also
handles rapid gaze movement. Transition behavior is especially important for
low-frequency gaze recovery.

### 3. Anchor-Gap Position Analysis

This should be the next main module. SparseGaze is about filling intervals
between sparse gaze anchors, so evaluate error as a function of:

- frames since previous anchor
- frames until next anchor
- normalized position inside the missing interval
- target frequency, e.g. 6/10/15 Hz

This is likely more important than adding more image-space metrics.

## Qualitative Analysis

### Image-Space Scanpath

Image-space scanpaths should be used primarily as qualitative figures. They are
good for showing whether a method lags, overshoots, smooths too much, or follows
GT through transitions.

Recommended case-study figure:

- GT image-space scanpath
- SparseGaze residual rollout
- simple interpolation baselines, e.g. linear/pchip
- optional HAGI++ if frame alignment is handled explicitly

Do not promote pixel MAE to a primary metric yet. Pixel error introduces camera
projection validity, field of view, and depth-related filtering issues. It may
become useful later, but it is not the clearest first-line evidence.

### Scene 3D Path

Scene 3D gaze path is useful for qualitative inspection only if it is clearly
described as direction projected to a chosen depth. If GT depth is reused, the
result is not a predicted 3D gaze point. It is a visual diagnostic of direction
error in scene context.

## Deferred Metrics

### Object Hit Agreement

Object hit agreement is promising but should be second-stage:

- fixation-only object hit agreement is meaningful
- transition object hit agreement is noisy and should be treated carefully
- category-level agreement may be more stable than instance-level agreement

This can connect gaze prediction to semantic attention, but it depends on
ray-box intersection approximations and depth/geometry assumptions.

### Complex Scanpath Distance

Avoid DTW, Fréchet distance, or other scanpath similarity metrics until a clear
failure mode requires them. They add complexity without directly answering the
current model-comparison question.

## Current Implementation Plan

1. Use `overall_evaluation.py` and
   `sparsegaze_overall_evaluation_viewer.ipynb` for the main all-sequence
   comparison. The default result filters to frames common to all selected
   methods within each sequence.
2. Keep the interactive single-sequence event viewer for debugging and case
   selection.
3. Add anchor-gap position curves as the next quantitative module.
4. Add image-space scanpath notebook as a qualitative figure generator.
5. Only then consider fixation object-hit agreement.

## HAGI++ Alignment Note

The HAGI++ ADT preprocessing intentionally drops the final frame because it
constructs head motion as:

```text
delta[t] = inv(T[t]) @ T[t + 1]
```

The final frame has no `T[t + 1]`. Therefore HAGI++ comparisons should use
common frames, while SparseGaze-only event analysis can use the available
SparseGaze/eval frames subject to the CPF conversion cache length.

HAGI++ sliding evaluation also has a warmup window. Early frames before the
first sliding prediction, typically around frame 149 for the current setup, do
not have HAGI++ rows. Use windows starting at or after the first HAGI++ frame
when a local HAGI++ comparison is needed.
