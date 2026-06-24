# EgoBody Visualization And Analysis

This folder contains EgoBody-specific visualization and analysis utilities.

## Raw RGB/Depth Probe

Use `02_raw_sequence_probe.py` before building ADT-like scanpath videos for
EgoBody. It checks whether one recording has the raw streams needed for
first-person image overlays and RGB-D/scene visualization:

- HoloLens PV RGB frames and `*_pv.txt` camera poses under `egocentric_color`
- HoloLens depth under `egocentric_depth`
- Kinect RGB-D frames under `kinect_color` and `kinect_depth`
- calibration files
- SparseGaze cache arrays that might already contain image/depth data

Example:

```bash
python "Experiments/visualization & Analysis/egobody/02_raw_sequence_probe.py" \
  recording_20210907_S03_S04_01 \
  --raw-root /mnt/d/Pose2Gaze-EgoBody \
  --cache-root /mnt/d/sparsegaze \
  --render-overlays 0
```

The current local `/mnt/d/Pose2Gaze-EgoBody` plus `/mnt/d/sparsegaze` cache has
gaze, head pose, body pose, and calibration, but no PV RGB, egocentric depth, or
Kinect RGB-D frames for this sequence. In that state, ADT-like image/scene
scanpath visualization is not possible; only direction/error diagnostics are
meaningful.

Outputs are written to:

```text
outputs/figures/egobody_raw_probe/<sequence>/
```

## Missing Gaze Direction Visualization

When raw RGB/depth is unavailable, use `01_missing_gaze_direction_compare.py`
for direction-level comparison against HAGI++:

```bash
python "Experiments/visualization & Analysis/egobody/01_missing_gaze_direction_compare.py" \
  recording_20210907_S03_S04_01 \
  --target-hz 6 \
  --phase 0 \
  --start-frame 149 \
  --end-frame 300 \
  --skip-3d \
  --output-root outputs/figures/egobody_missing_gaze_direction_only
```

This writes the per-frame error CSV, angular-error timeline, and CPF pitch/yaw
timeline to:

```text
outputs/figures/egobody_missing_gaze_direction_only/<sequence>/hz6_phase0_frames_149_300/
```
