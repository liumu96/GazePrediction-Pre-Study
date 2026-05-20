# Prediction Analysis

This folder contains reusable, non-notebook analysis code for comparing
SparseGaze-style model predictions against ground-truth gaze.

The current target is ADT SparseGaze eval output:

```text
/home/liumu/Github_Projects/SparseGaze/outputs/eval/adt/
  <model>/
    test/
      rollout/
        sequence_predictions/
          <sequence>/
            hz6_phase0.npz
            hz10_phase0.npz
            hz15_phase0.npz
```

The default analysis frequency is **6 Hz**, because that is the current
low-frequency setting of interest. The code can also scan all available
frequencies.

## What To Analyze

The first reusable analysis layer should answer these questions:

1. **Overall prediction error**
   - Mean and median 3D angular error between predicted and GT gaze directions.
   - Mean and median absolute yaw/pitch residuals for bias inspection.
   - Computed per sequence first, then averaged at model level.

2. **Anchor vs missing-frame behavior**
   - Anchor frames and missing/evaluated frames are separated using the NPZ
     `anchor_mask`.
   - This is important because SparseGaze is primarily about what happens away
     from ground-truth gaze anchors.
   - In rollout outputs, `eval_mask` usually selects missing frames only. The
     tables therefore report `n_anchor_total`, `n_eval_anchor`, and
     `n_eval_missing` separately. `missing_*` metrics are the main SparseGaze
     metrics; `anchor_*` metrics are an anchor-frame sanity check.

3. **Frequency sensitivity**
   - Default: 6 Hz.
   - Optional: compare 6/10/15 Hz or any available target rates via `--all-hz`.
   - This should show how model error changes as gaze observations become
     sparser.

4. **Model comparison**
   - The scanner supports multiple model directories under the same eval root.
   - The output tables include `model`, `split`, `eval_kind`, `target_hz`, and
     `phase`, so later models can be compared without changing the analysis
     logic.

5. **Scene-event-conditioned error**
   - If `/mnt/d/SparseGaze/ADT-Gaze-structured` contains
     `events/scene_gaze_frame_labels.csv`, the analysis attaches
     fixation/transition labels by frame index.
   - This lets us check whether model errors are concentrated in transition
     segments rather than fixation segments.

Later analysis can add object/box hit agreement, scene-path comparison, and
GT-vs-predicted visualization hooks. Those should build on the same per-frame
prediction table instead of re-parsing NPZ files in each notebook.

## Run

Default: all rollout NPZ predictions at 6 Hz under the ADT eval root.

```bash
python -m analysis.analyze_prediction_results
```

One model only:

```bash
python -m analysis.analyze_prediction_results \
  --model sparsegaze_cpf_rotation_only_ss
```

All available frequencies:

```bash
python -m analysis.analyze_prediction_results \
  --model sparsegaze_cpf_rotation_only_ss \
  --all-hz
```

Compare rollout variants:

```bash
python -m analysis.analyze_prediction_results \
  --model sparsegaze_cpf_rotation_only_ss \
  --eval-kind rollout \
  --eval-kind rollout_linear \
  --eval-kind rollout_pchip
```

Outputs are written to:

```text
outputs/analysis/prediction_results/
  analysis_config.json
  prediction_files.csv
  frame_summary.csv
  sequence_summary.csv
  model_summary.csv
  event_summary.csv
  summary.md
  figures/
    model_missing_error.png
    frequency_curve.png
    sequence_missing_error.png
    event_error.png
    error_distribution.png
    yaw_pitch_residual.png
```

## Output Tables

- `prediction_files.csv`: discovered NPZ files and parsed metadata.
- `sequence_summary.csv`: one row per model/sequence/frequency/phase. Includes
  overall evaluated-frame metrics, missing-frame metrics, and anchor-frame
  sanity metrics.
- `model_summary.csv`: sequence-level metrics averaged per
  model/eval/frequency/phase.
- `event_summary.csv`: one row per scene event label when event labels are
  available.
- `frame_summary.csv`: per-frame GT/prediction error table used by the plots.

## Output Figures

- `model_missing_error.png`: mean/median missing-frame angular error by model
  and eval mode.
- `frequency_curve.png`: missing-frame error across target gaze rates. This is
  only informative when running with `--all-hz`.
- `sequence_missing_error.png`: sequences ranked by missing-frame mean angular
  error.
- `event_error.png`: fixation vs transition error when scene-event labels are
  available.
- `error_distribution.png`: whole evaluated-frame angular error distribution.
- `yaw_pitch_residual.png`: residual direction in yaw/pitch space, useful for
  spotting directional bias.
- `summary.md`: compact Markdown report with key tables and embedded figures.

## Notes

- The NPZ `pred_xyz` and `gt_xyz` are treated as unit 3D gaze directions.
- The primary metric is angular error from the 3D dot product.
- Yaw/pitch residuals are diagnostic projections, not the primary metric.
- Current ADT SparseGaze NPZ files use negative z as forward for yaw/pitch
  conversion; this does not affect 3D angular error.
