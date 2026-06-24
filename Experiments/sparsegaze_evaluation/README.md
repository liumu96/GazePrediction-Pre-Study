# SparseGaze Evaluation Experiment

This folder contains the focused evaluation workspace for SparseGaze ADT
experiments. Keep evaluation reports, notebooks, and scripts here instead of
mixing them with general-purpose ADT sandbox utilities.

## Files

- `REPORT.md`: evaluation plan, metric priorities, and deferred ideas.
- `event_evaluation.py`: reusable loading, event-conditioned metric, and
  plotting helpers.
- `overall_evaluation.py`: dataset-level aggregation, common-frame filtering,
  summary tables, and aggregate figures.
- `sparsegaze_event_comparison_viewer.ipynb`: interactive single-sequence
  viewer for missing-frame error by GT scene-gaze event.
- `sparsegaze_overall_evaluation_viewer.ipynb`: cross-sequence notebook for
  overall MAE, event-conditioned MAE, per-sequence variation, and HAGI++ deltas.

## Default Inputs

```text
reports: /mnt/d/SparseGaze/ADT-Gaze-structured
model:   /home/liumu/Github_Projects/SparseGaze/outputs/eval/adt/sparsegaze_cpf_forward_head_motion_residual_ss
```

Override the model path with:

```bash
export SPARSEGAZE_ADT_EVAL_DIR=/path/to/model/eval/dir
```

## Quick Checks

```bash
python -m py_compile Experiments/sparsegaze_evaluation/event_evaluation.py
python -m py_compile Experiments/sparsegaze_evaluation/overall_evaluation.py
python Experiments/sparsegaze_evaluation/event_evaluation.py \
  Apartment_release_decoration_skeleton_seq133_M1292 \
  --fps 6 \
  --start-frame 149 \
  --max-frames 300
python Experiments/sparsegaze_evaluation/overall_evaluation.py --fps 6
```
