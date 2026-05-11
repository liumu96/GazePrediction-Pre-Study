# Head-Gaze Relationship Report

Generated on 2026-05-11 from `batch_head_gaze_analysis_summary.csv`.

## Scope

This report summarizes GT head-gaze relationship analysis over the extracted ADT sequences. It no longer uses CPF-derived fixation labels. The goal is to describe continuous relationships among scene gaze/head geometry, CPF-local gaze dynamics, and head motion before adding event-conditioned analysis.

## Research Questions

1. Does head rotation co-vary with CPF-local gaze dynamics within the same 30 Hz timestep?
2. Is head rotation more informative than head translation for local gaze dynamics?
3. Does head motion explain only gaze-change magnitude, or also gaze-change direction?
4. Do larger head-rotation regimes contain larger local gaze changes?

## Inputs

- `*_gaze_samples.csv`
- `*_head_samples.csv`
- `batch_head_gaze_analysis_summary.csv`
- `batch_head_gaze_analysis_report.json`

## Computation Method

CPF-local gaze dynamics are computed from `gaze_dir_cpf_unit_xyz`, while scene geometry uses `gaze_dir_scene_unit_xyz` and `head_forward_scene_unit`. The CPF dynamics are not converted into fixation labels in this report.

Core quantities:

- `local_angle_step_deg = angle(g_t-1^cpf, g_t^cpf)`
- `local_velocity_deg_s = local_angle_step_deg / dt`
- `window_dispersion_deg = max pairwise CPF angular separation in a centered window`
- `head_rotation_speed_deg_s = angle(R_{t-1}^{-1} R_t) / dt`
- `head_rotvec_prev_head_*`: signed axis-angle vector of `R_{t-1}^{-1} R_t`, expressed in the previous head/CPF frame
- `gaze_head_motion_alignment_2d = cosine([delta_yaw, delta_pitch], [head_rotvec_y, head_rotvec_x])`
- `head_translation_speed_m_s = ||p_t - p_{t-1}|| / dt`
- `gaze_head_angle_deg = angle(gaze_dir_scene_unit, head_forward_scene_unit)`

All correlations are Pearson correlations computed per sequence and then summarized across sequences. They measure linear association only, not causality or model performance.
The directional metrics use signed CPF angular deltas and signed relative head rotation vector components. Their signs are coordinate-frame conventions; the robust question is whether direction components carry additional structure beyond speed magnitude.

## Coverage

- Sequence count: `34`
- Mean dynamics-input valid ratio: `1.000`
- Positive current gaze/head-rotation correlations: `34/34`
- Current gaze/head-rotation correlation larger than translation correlation: `34/34`

## Results

### Table 1. Batch-Level Summary

Table 1 tests whether head rotation or head translation has the stronger relationship with CPF-local gaze velocity. This is a dynamics diagnostic, not a fixation analysis.

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Median gaze-head angle (deg) | 27.916 | 29.976 | 18.044 | 33.693 |
| corr(current local gaze vel, current head rot) | 0.322 | 0.331 | 0.215 | 0.425 |
| corr(current local gaze vel, current head trans) | 0.079 | 0.075 | -0.023 | 0.182 |

Head rotation is consistently more related to local gaze velocity than translation: mean correlation `0.322` vs `0.079`. The magnitude is still weak-to-moderate, so this should be read as evidence of useful dynamics context, not as evidence that head motion alone explains gaze.

### Figure 1. Current-Step Correlation Distributions

![Correlation distributions](../outputs/figures/head_gaze_relationship/correlation_distributions.png)

The per-sequence distribution confirms that the rotation result is stable rather than driven by a small number of sequences. Translation remains close to zero on average.

### Table 2. Directional Component Summary

Table 2 addresses the missing direction question. It compares signed gaze angular deltas with signed head rotation-vector components in the previous head frame, and also compares absolute component magnitudes. Signed correlations test whether motion directions co-vary. Absolute component correlations test whether larger component-wise head turns come with larger component-wise eye-in-head changes.

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| corr(signed delta yaw, head rotvec Y) | 0.079 | 0.077 | -0.027 | 0.180 |
| corr(signed delta pitch, head rotvec X) | -0.069 | -0.064 | -0.238 | 0.067 |
| corr(|delta yaw|, |head rotvec Y|) | 0.312 | 0.309 | 0.206 | 0.411 |
| corr(|delta pitch|, |head rotvec X|) | 0.268 | 0.256 | 0.151 | 0.378 |
| mean 2D gaze/head motion alignment | -0.067 | -0.068 | -0.162 | 0.030 |
| aligned frame fraction | 0.316 | 0.316 | 0.276 | 0.373 |
| opposed frame fraction | 0.391 | 0.392 | 0.339 | 0.463 |
| weak/orthogonal frame fraction | 0.293 | 0.295 | 0.254 | 0.323 |

The direction-level results should be read separately from the speed results. A strong speed correlation with weak signed component correlation means head motion is informative about how much local gaze changes, but less deterministic about which way the eyes move. Opposed alignment can indicate compensatory eye-in-head motion during head turns, while aligned motion can indicate co-moving eye/head behavior under the chosen CPF sign convention.

### Figure 2. Directional Component Correlations

![Directional component correlations](../outputs/figures/head_gaze_relationship/directional_component_correlations.png)

### Figure 3. 2D Motion Alignment Fractions

![Motion alignment fractions](../outputs/figures/head_gaze_relationship/motion_alignment_fractions.png)

### Table 3. Head-Rotation Speed Group Summary

Table 3 groups frames by head rotation speed within each sequence using the 33.3% and 66.7% sequence-specific percentiles. The table then summarizes the 34 per-sequence group means, so the statistical unit is a sequence-level group mean rather than a pooled frame.

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Low-rotation-group mean local gaze velocity (deg/s) | 31.378 | 32.276 | 23.107 | 39.484 |
| Mid-rotation-group mean local gaze velocity (deg/s) | 41.222 | 40.697 | 28.199 | 54.385 |
| High-rotation-group mean local gaze velocity (deg/s) | 74.498 | 74.444 | 49.281 | 104.663 |
| Low-rotation-group mean gaze-head angle (deg) | 27.524 | 29.849 | 18.234 | 32.372 |
| Mid-rotation-group mean gaze-head angle (deg) | 26.859 | 28.435 | 20.025 | 31.958 |
| High-rotation-group mean gaze-head angle (deg) | 27.408 | 27.942 | 21.876 | 31.601 |

Mean local gaze velocity rises from `31.378` to `41.222` and then to `74.498` deg/s across low/mid/high head-rotation groups. Because the groups are relative within each sequence, this result should be read as a within-sequence trend rather than an absolute physical threshold.

### Figure 4. Local Gaze Velocity by Head-Rotation Speed Group

![Local gaze velocity by head-rotation stratum](../outputs/figures/head_gaze_relationship/local_gaze_velocity_by_head_rotation_stratum.png)

High-head-rotation frames have larger local gaze velocity than low-head-rotation frames in `34/34` sequences. This is the strongest result in the report, but it remains a statistical relationship rather than a deterministic predictor.

### Table 4. Within-Sequence Effect Size Summary

Table 4 normalizes each sequence by its own mean local gaze velocity. This reduces the influence of sequences that are globally faster or slower and directly tests whether relatively higher head rotation corresponds to a relative increase in local gaze velocity.

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Low group / sequence mean | 0.644 | 0.638 | 0.495 | 0.844 |
| Mid group / sequence mean | 0.841 | 0.824 | 0.747 | 1.009 |
| High group / sequence mean | 1.515 | 1.514 | 1.371 | 1.659 |
| High-low delta (deg/s) | 43.120 | 42.701 | 18.943 | 70.466 |
| High-low delta / sequence mean | 0.872 | 0.869 | 0.527 | 1.101 |
| High / low ratio | 2.392 | 2.363 | 1.624 | 3.209 |

The normalized view supports the same conclusion without relying on absolute velocity scale: within a typical sequence, high-rotation frames have substantially larger local gaze velocity than low-rotation frames.

### Figure 5. Normalized Local Gaze Velocity by Head-Rotation Group

![Normalized local gaze velocity by head-rotation group](../outputs/figures/head_gaze_relationship/normalized_local_gaze_velocity_by_head_rotation_group.png)

### Figure 6. Current-vs-Next Rotation Correlation

The one-step lagged diagnostic asks whether current head rotation remains associated with next-frame local gaze velocity. It is included only as a temporal diagnostic, not as a SparseGaze model result.

![Current vs next rotation correlation](../outputs/figures/head_gaze_relationship/current_vs_next_rotation_correlation.png)

### Table 5. Auxiliary Next-Step Summary

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| corr(next local gaze vel, current head rot) | 0.276 | 0.284 | 0.197 | 0.355 |
| corr(next local gaze vel, current head trans) | 0.073 | 0.070 | -0.026 | 0.177 |

The lagged rotation relation remains positive but is weaker than the same-step relation. This can motivate later temporal modeling, but it is not itself a prediction experiment.

### Table 6. Sequences With Strongest Next-Step Head-Rotation Signal

| Sequence | corr(next local gaze vel, current head rot) |
|---|---:|
| Apartment_release_meal_skeleton_seq131_M1292 | 0.355 |
| Apartment_release_work_skeleton_seq136_M1292 | 0.338 |
| Apartment_release_work_skeleton_seq132_M1292 | 0.321 |
| Apartment_release_meal_skeleton_seq140_M1292 | 0.319 |
| Apartment_release_decoration_skeleton_seq140_M1292 | 0.317 |

## Findings

1. Head rotation is the useful head-motion family in this analysis; translation is weak.
2. The strongest relationship is in motion magnitude: larger head-rotation regimes contain larger CPF-local gaze velocities.
3. Directional metrics are necessary because speed-only analysis hides whether gaze moves with, against, or orthogonal to head rotation.
4. CPF-local velocity and dispersion are useful continuous diagnostics, but CPF-thresholded fixation labels are intentionally excluded because they do not define scene/object fixation.

## Limitations

- The report is based on correlation and stratification, not causal inference.
- Scene-direction event detection is implemented separately; object-level fixation detection is not implemented here.
- CPF dynamics should be used as auxiliary features, not as final event labels.

