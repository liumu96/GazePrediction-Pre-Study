# Scene-Level Head-Gaze Relationship Report

Generated on 2026-05-11 from `batch_scene_head_gaze_analysis_summary.csv`.

## Scope

This report analyzes final Scene/world gaze dynamics, not only CPF-local eye-in-head dynamics. It joins scene gaze velocity, CPF-local gaze velocity, head motion, and scene-direction event labels to test whether head motion changes world gaze, compensates world gaze, or both.

## Research Questions

1. Is Scene gaze velocity related to head rotation?
2. Does this relation differ from CPF-local gaze velocity?
3. During scene fixation versus transition, how do head and CPF dynamics differ?
4. When head rotation is high, does Scene gaze remain stable or transition?

## Method

Scene velocity is computed from `gaze_dir_scene_unit_xyz` and the scene event labels produced by `detect_scene_gaze_events.py`. CPF-local velocity is recomputed from `gaze_dir_cpf_unit_xyz`. Head rotation speed comes from the refactored head proxy layer. Correlations are computed per sequence and then summarized across 34 sequences.

## Coverage

- Sequence count: `34`
- Mean valid analysis ratio: `1.000`
- Mean scene fixation frame fraction: `0.410`

## Results

### Table 1. Scene vs CPF Correlation Summary

Table 1 compares head-motion correlations in Scene and CPF spaces. This directly tests whether head rotation is more related to final world-gaze motion or to eye-in-head local motion.

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| corr(scene gaze velocity, head rotation speed) | 0.360 | 0.360 | 0.238 | 0.530 |
| corr(CPF local velocity, head rotation speed) | 0.322 | 0.331 | 0.215 | 0.425 |
| corr(scene gaze velocity, head translation speed) | 0.078 | 0.081 | -0.008 | 0.166 |
| corr(CPF local velocity, scene gaze velocity) | 0.956 | 0.957 | 0.923 | 0.977 |

Scene gaze velocity has a slightly stronger relation to head rotation than CPF-local velocity in this batch. The very high CPF/Scene velocity correlation indicates that many rapid eye-in-head changes also appear as rapid world-gaze changes, so compensation is not the dominant aggregate pattern under the current scene-event settings.

![Scene/CPF correlations](../outputs/figures/scene_head_gaze_relationship/scene_cpf_correlation_distributions.png)

### Table 2. Event-Conditioned Dynamics

Table 2 compares scene fixation and transition frames. This is the first use of the scene-direction event labels in the head-gaze analysis.

| Metric | Fixation Mean | Transition Mean |
|---|---:|---:|
| Scene gaze velocity (deg/s) | 11.123 | 74.391 |
| CPF local gaze velocity (deg/s) | 17.944 | 70.535 |
| Head rotation speed (deg/s) | 17.045 | 31.217 |

Transition frames have much larger Scene and CPF gaze velocities than fixation frames. Head rotation also increases during transitions, but the gap is smaller than the gaze-velocity gap.

![Event-conditioned dynamics](../outputs/figures/scene_head_gaze_relationship/event_conditioned_dynamics.png)

### Table 3. Head-Rotation Group Summary

Table 3 groups frames by sequence-specific head rotation speed percentiles. It tests what happens to Scene and CPF gaze dynamics as head motion becomes relatively larger within each sequence.

| Metric | Low | Mid | High |
|---|---:|---:|---:|
| Scene gaze velocity (deg/s) | 30.982 | 39.442 | 75.230 |
| CPF local gaze velocity (deg/s) | 31.378 | 41.222 | 74.498 |
| Scene fixation fraction | 0.555 | 0.435 | 0.239 |

As head rotation increases, both Scene and CPF gaze velocities increase and the fixation fraction drops. This supports a head-contributed transition pattern more strongly than a pure compensation pattern at the aggregate level.

![Gaze velocity by head group](../outputs/figures/scene_head_gaze_relationship/gaze_velocity_by_head_rotation_group.png)

![Fixation fraction by head group](../outputs/figures/scene_head_gaze_relationship/fixation_fraction_by_head_rotation_group.png)

## Findings

1. In Scene space, head rotation is positively related to final world-gaze velocity and is still much more informative than translation.
2. Scene and CPF gaze velocities are highly correlated, so rapid local eye motion often coincides with rapid world-gaze motion in this dataset.
3. Scene fixation frames have low Scene velocity by definition, but they also have lower CPF gaze velocity and lower head rotation than transition frames.
4. High relative head-rotation frames are less likely to be scene fixations, which suggests that head rotation often participates in world-gaze transitions rather than being fully compensated.

## Limitations

- Scene event labels are direction-level labels, not object-level fixation labels.
- Head rotation groups are sequence-relative percentile groups, not absolute physical thresholds.
- This is correlation and stratification analysis, not a prediction experiment.

