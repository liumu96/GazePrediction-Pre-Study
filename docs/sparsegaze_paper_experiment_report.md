# SparseGaze Paper Experiment Report

This report summarizes the current paper-facing SparseGaze experiments and links to the generated CSV/figure artifacts. It should be read together with `docs/sparsegaze_paper_analysis_plan.md`.

Generated artifacts:

- Main generated report: `outputs/analysis/paper_missing_results/REPORT.md`
- Main CSVs:
  - `outputs/analysis/paper_missing_results/overall_summary.csv`
  - `outputs/analysis/paper_missing_results/event_summary.csv`
  - `outputs/analysis/paper_missing_results/long_gap_summary.csv`
  - `outputs/analysis/paper_missing_results/frequency_summary.csv`
  - `outputs/analysis/paper_missing_results/scene_point_summary.csv`
  - `outputs/analysis/paper_missing_results/scanpath_summary.csv`
- Main figures:
  - `outputs/analysis/paper_missing_results/figures/overall_ablation.png`
  - `outputs/analysis/paper_missing_results/figures/event_ablation.png`
  - `outputs/analysis/paper_missing_results/figures/long_gap_ablation.png`
  - `outputs/analysis/paper_missing_results/figures/frequency_sensitivity.png`
  - `outputs/analysis/paper_missing_results/figures/scene_point_error.png`
  - `outputs/analysis/paper_missing_results/figures/scanpath_metrics.png`
- Anchor-gap report:
  - `outputs/analysis/anchor_gap_position/summary.md`
  - `outputs/analysis/anchor_gap_position_all_hz/summary.md`

## 1. Run Configuration

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-sparsegaze python analysis/analyze_paper_missing_results.py
```

Default compared methods:

- HAGI++
- `sparsegaze_cpf_gaze_only_ss`
- `sparsegaze_cpf_local_head_motion_ss`
- `sparsegaze_cpf_rotation_only_ss`
- `sparsegaze_cpf_rotation_translation_ss`
- `sparsegaze_cpf_forward_head_motion_ss`
- `sparsegaze_cpf_forward_head_motion_residual_ss`

Evaluation policy:

- Use evaluated missing frames only.
- Exclude sparse gaze anchor frames.
- Keep only frames inside valid anchor gaps.
- Use common frames shared by all selected methods.
- Use sequence-macro MAE as the primary paper-facing aggregate.
- Include 6Hz, 10Hz, and 15Hz target gaze rates.

Current run size:

- SparseGaze prediction files: 180
- Test sequences: 10
- Evaluated rows after method expansion: 356,692

## 2. Main Reconstruction Results

At 6Hz, HAGI++ is still slightly better in overall sequence-macro MAE, while SparseGaze residual is close and has a better p90 tail.

| Method | 6Hz sequence-macro MAE deg | Median deg | p90 deg |
| --- | ---: | ---: | ---: |
| HAGI++ | 3.095 | 1.203 | 8.747 |
| Residual | 3.113 | 1.299 | 8.494 |
| Forward head | 3.151 | 1.345 | 8.600 |
| Local head | 3.190 | 1.385 | 8.575 |
| Rotation+translation | 3.261 | 1.503 | 8.562 |
| Rotation-only | 3.305 | 1.525 | 8.671 |
| Gaze-only | 3.610 | 1.935 | 8.703 |

Interpretation:

- Head-assisted SparseGaze variants clearly improve over gaze-only.
- Residual is the best SparseGaze variant among the current deployable rollout results.
- The overall 6Hz result does not yet support a strong "SparseGaze beats HAGI++ overall" claim.
- The p90 result suggests residual correction may reduce some high-error tail cases, even when mean MAE is close.

## 3. Frequency Sensitivity

SparseGaze residual is consistently strongest among the SparseGaze variants and becomes better than HAGI++ at 10Hz and 15Hz in the current common-frame comparison.

| Method | 6Hz MAE | 10Hz MAE | 15Hz MAE |
| --- | ---: | ---: | ---: |
| HAGI++ | 3.095 | 2.154 | 1.518 |
| Residual | 3.113 | 2.002 | 1.417 |
| Forward head | 3.151 | 2.035 | 1.450 |
| Local head | 3.190 | 2.064 | 1.464 |
| Rotation+translation | 3.261 | 2.105 | 1.482 |
| Rotation-only | 3.305 | 2.122 | 1.487 |
| Gaze-only | 3.610 | 2.237 | 1.523 |

Interpretation:

- Error decreases as gaze rate increases.
- Head/residual variants consistently outperform gaze-only.
- The current weakest point is 6Hz, where HAGI++ remains slightly ahead in mean MAE.

## 4. Event-Conditioned Results

At 6Hz, transition frames are the main failure mode.

| Method | Fixation MAE deg | Transition MAE deg |
| --- | ---: | ---: |
| HAGI++ | 1.207 | 4.382 |
| Residual | 1.302 | 4.351 |
| Forward head | 1.326 | 4.398 |
| Local head | 1.375 | 4.429 |
| Rotation+translation | 1.439 | 4.507 |
| Rotation-only | 1.456 | 4.569 |
| Gaze-only | 1.853 | 4.809 |

Interpretation:

- Residual is slightly worse than HAGI++ on fixation but slightly better on transition at 6Hz.
- This supports a narrower claim: residual correction helps in high-dynamics frames.
- A paper claim should avoid saying the current rollout is uniformly better.

## 5. Long-Gap Results

Long gap is defined as normalized anchor-gap position >= 0.60.

| Method | 6Hz long-gap MAE deg | Median deg | p90 deg |
| --- | ---: | ---: | ---: |
| HAGI++ | 4.023 | 1.848 | 10.671 |
| Residual | 4.083 | 1.990 | 10.676 |
| Forward head | 4.145 | 1.999 | 10.885 |
| Local head | 4.191 | 2.048 | 10.760 |
| Rotation+translation | 4.281 | 2.201 | 10.840 |
| Rotation-only | 4.342 | 2.240 | 11.028 |
| Gaze-only | 4.715 | 2.759 | 11.278 |

Interpretation:

- Long-gap behavior confirms that gaze-only is weak.
- Residual is the best SparseGaze variant but still does not beat HAGI++ at 6Hz long-gap mean MAE.
- The current model still needs stronger interval propagation or repair if the paper needs a strong 6Hz claim.

## 6. Anchor-Gap Position Results

The dedicated anchor-gap analysis shows monotonic error growth inside the sparse anchor interval.

6Hz residual rollout:

- Gap position 0.2: 1.58 deg
- Gap position 0.4: 2.70 deg
- Gap position 0.6: 3.65 deg
- Gap position 0.8: 4.51 deg

Event-conditioned 6Hz residual rollout:

- Fixation: 0.58 -> 2.23 deg from gap position 0.2 to 0.8
- Transition: 2.26 -> 6.07 deg from gap position 0.2 to 0.8

Interpretation:

- Anchor-gap position should be a main paper figure.
- Transition and gap position should be analyzed jointly.
- The gap curve makes the failure mode clearer than a single overall MAE.

## 7. GT-Depth Scene Point Diagnostic

This diagnostic projects each predicted gaze direction to the GT gaze depth and compares the projected endpoint with the GT scene gaze point. It does not evaluate true depth prediction.

| Method | 6Hz mean scene point error m | 6Hz p90 m |
| --- | ---: | ---: |
| Residual | 0.096 | 0.239 |
| HAGI++ | 0.097 | 0.244 |
| Forward head | 0.098 | 0.242 |
| Local head | 0.100 | 0.249 |
| Rotation+translation | 0.102 | 0.254 |
| Rotation-only | 0.104 | 0.254 |
| Gaze-only | 0.116 | 0.288 |

Interpretation:

- The scene-point diagnostic mirrors angular error: residual is close to HAGI++ and slightly better in this GT-depth projected endpoint metric.
- This metric is useful as a scene-facing secondary result.
- It must be described as GT-depth projection, not 3D fixation prediction.

## 8. Scanpath Dynamics Diagnostic

Current scanpath metric uses contiguous evaluated missing-frame runs and summarizes:

- predicted path length / GT path length
- mean absolute step-magnitude error

At 6Hz:

| Method | Step magnitude MAE deg/frame | Pred/GT path length ratio |
| --- | ---: | ---: |
| Forward head | 1.231 | 0.410 |
| Residual | 1.232 | 0.414 |
| Local head | 1.238 | 0.406 |
| Rotation+translation | 1.257 | 0.382 |
| Rotation-only | 1.265 | 0.369 |
| Gaze-only | 1.299 | 0.457 |
| HAGI++ | 1.361 | 0.645 |

Interpretation:

- All SparseGaze rollout variants currently under-recover GT path length; they are smoother than GT.
- HAGI++ preserves more path length but has higher step-magnitude error in this diagnostic.
- This should be treated as a behavior diagnostic, not a primary metric.

## 9. Current Claim Status

Supported:

- Sparse-gaze recovery becomes harder inside anchor gaps.
- Transition frames are much harder than fixation frames.
- Head input improves over gaze-only.
- Residual correction is currently the strongest SparseGaze rollout variant.
- GT-depth scene point diagnostic is consistent with angular error.
- Current rollout variants are smoother than GT scanpaths.

Partially supported:

- SparseGaze residual helps transition frames at 6Hz compared with HAGI++.
- SparseGaze residual beats HAGI++ at 10Hz and 15Hz overall in the current common-frame setup.

Not yet supported:

- A strong claim that deployable SparseGaze beats HAGI++ overall at 6Hz.
- A strong claim that SparseGaze solves long-gap propagation better than HAGI++ at 6Hz.
- A true 3D fixation-point prediction claim.
- Object/AOI downstream improvement.

## 10. Remaining TODO

- [ ] Decide whether 6Hz is the main paper setting or whether 10Hz/15Hz are also central.
- [ ] Re-run these reports after selecting the final model checkpoint/directory.
- [ ] Add image-space gaze point evaluation if projection for predictions is made reliable.
- [ ] Add qualitative image-space scanpath overlay for one transition-heavy window.
- [ ] Add 3D scene ray visualization for GT / HAGI++ / residual / baseline.
- [ ] Add fixation-only object/AOI agreement if predicted rays can be robustly intersected with scene object boxes.
- [ ] Add per-sequence win/loss table for the final selected methods.
- [ ] Decide final paper claim based on whether the final deployable rollout can beat HAGI++ at 6Hz or only in transition/10Hz/15Hz subsets.
