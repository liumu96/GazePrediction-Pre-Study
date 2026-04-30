# Head-Gaze Relationship Report

Generated on 2026-04-30 from `batch_head_gaze_analysis_summary.csv`.

## Scope

This report summarizes GT head-gaze relationship analysis over the extracted ADT sequences. It no longer uses CPF-derived fixation labels. The goal is narrower and cleaner: describe continuous relationships among scene gaze/head geometry, CPF-local gaze dynamics, and head motion.

## Research Questions

1. Does head rotation co-vary with CPF-local gaze dynamics within the same 30 Hz timestep?
2. Is head rotation more informative than head translation for local gaze dynamics?
3. Do larger head-rotation regimes contain larger local gaze changes?

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
- `head_translation_speed_m_s = ||p_t - p_{t-1}|| / dt`
- `gaze_head_angle_deg = angle(gaze_dir_scene_unit, head_forward_scene_unit)`

All correlations are Pearson correlations computed per sequence and then summarized across sequences. They measure linear association only, not causality or model performance.

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

### Table 2. Head-Rotation Strata Summary

Table 2 stratifies frames by head rotation speed within each sequence. This asks a more direct question than correlation alone: when head rotation is larger, are local gaze changes also larger?

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Low-stratum mean local gaze velocity (deg/s) | 31.378 | 32.276 | 23.107 | 39.484 |
| Mid-stratum mean local gaze velocity (deg/s) | 41.222 | 40.697 | 28.199 | 54.385 |
| High-stratum mean local gaze velocity (deg/s) | 74.498 | 74.444 | 49.281 | 104.663 |
| Low-stratum mean gaze-head angle (deg) | 27.524 | 29.849 | 18.234 | 32.372 |
| Mid-stratum mean gaze-head angle (deg) | 26.859 | 28.435 | 20.025 | 31.958 |
| High-stratum mean gaze-head angle (deg) | 27.408 | 27.942 | 21.876 | 31.601 |

Mean local gaze velocity rises from `31.378` to `41.222` and then to `74.498` deg/s across low/mid/high head-rotation strata. The gaze-head angle changes much less, so the clearest relation is dynamic rather than static.

### Figure 2. Local Gaze Velocity by Head-Rotation Stratum

![Local gaze velocity by head-rotation stratum](../outputs/figures/head_gaze_relationship/local_gaze_velocity_by_head_rotation_stratum.png)

High-head-rotation frames have larger local gaze velocity than low-head-rotation frames in `34/34` sequences. This is the strongest result in the report, but it remains a statistical relationship rather than a deterministic predictor.

### Figure 3. Current-vs-Next Rotation Correlation

The one-step lagged diagnostic asks whether current head rotation remains associated with next-frame local gaze velocity. It is included only as a temporal diagnostic, not as a SparseGaze model result.

![Current vs next rotation correlation](../outputs/figures/head_gaze_relationship/current_vs_next_rotation_correlation.png)

### Table 3. Auxiliary Next-Step Summary

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| corr(next local gaze vel, current head rot) | 0.276 | 0.284 | 0.197 | 0.355 |
| corr(next local gaze vel, current head trans) | 0.073 | 0.070 | -0.026 | 0.177 |

The lagged rotation relation remains positive but is weaker than the same-step relation. This can motivate later temporal modeling, but it is not itself a prediction experiment.

### Table 4. Sequences With Strongest Next-Step Head-Rotation Signal

| Sequence | corr(next local gaze vel, current head rot) |
|---|---:|
| Apartment_release_meal_skeleton_seq131_M1292 | 0.355 |
| Apartment_release_work_skeleton_seq136_M1292 | 0.338 |
| Apartment_release_work_skeleton_seq132_M1292 | 0.321 |
| Apartment_release_meal_skeleton_seq140_M1292 | 0.319 |
| Apartment_release_decoration_skeleton_seq140_M1292 | 0.317 |

## Findings

1. Head rotation is the useful head-motion family in this analysis; translation is weak.
2. The relationship is weak-to-moderate, so head motion is context rather than a standalone gaze predictor.
3. CPF-local velocity and dispersion are useful continuous diagnostics, but CPF-thresholded fixation labels are intentionally excluded because they do not define scene/object fixation.

## Limitations

- The report is based on correlation and stratification, not causal inference.
- Scene/object-level fixation detection is not implemented here.
- CPF dynamics should be used as auxiliary features, not as final event labels.

