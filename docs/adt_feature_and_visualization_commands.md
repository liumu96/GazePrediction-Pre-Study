# ADT Feature Extraction and Visualization Commands

This document is the command entry point for the current ADT sandbox pipeline.
It records how to regenerate the feature layers and how to inspect them with
scripts or notebooks.

Run commands from the repository root:

```bash
cd /home/liumu/Github_Projects/adt_dataset_sandbox
```

Recommended environment:

```bash
conda activate adt
```

Large outputs should stay outside the repository. Current working output root:

```bash
REPORTS_DIR=/mnt/d/SparseGaze/ADT-Gaze-structured
```

The older flat result directory was:

```text
/mnt/d/SparseGaze/ADT-Gaze
```

It has been copied non-destructively into the organized directory above. The
current code and notebooks use the organized directory; the old flat directory
is no longer required by this repository after the copy is verified.

ADT raw data is resolved from `ADT_DATA_ROOT` in `.env` or the shell
environment.

## Result Directory Layout

The organized layout groups files by sequence first:

```text
$REPORTS_DIR/
├── sequences/
│   └── <sequence_id>/
│       ├── gaze/
│       ├── head/
│       ├── dynamics/
│       ├── events/
│       ├── events_legacy/
│       ├── scene/
│       ├── skeleton/
│       └── analysis/
├── batch/
├── organization_manifest.csv
└── organization_manifest.json
```

This makes it possible to inspect one sequence by opening:

```text
$REPORTS_DIR/sequences/<sequence_id>/
```

To copy an existing flat directory into this structure:

```bash
python scripts/organize_flat_reports.py \
  /mnt/d/SparseGaze/ADT-Gaze \
  /mnt/d/SparseGaze/ADT-Gaze-structured
```

The command above is a dry-run. To actually copy:

```bash
python scripts/organize_flat_reports.py \
  /mnt/d/SparseGaze/ADT-Gaze \
  /mnt/d/SparseGaze/ADT-Gaze-structured \
  --execute
```

It copies files only; it does not delete or move the source files.

After validating the organized directory, the old flat directory is no longer
needed by the current code. Delete it only when you no longer need it as a
manual backup:

```text
/mnt/d/SparseGaze/ADT-Gaze
```

The repository does not keep runtime fallback logic for this old layout.

## Feature Layers

| Layer | Main script | Main output | Coordinate / meaning |
| --- | --- | --- | --- |
| Gaze samples | `batch_extract_gaze_samples.py` | `gaze/gaze_samples.csv` | CPF, RGB camera, Scene gaze features |
| Gaze quality | `check_gaze_quality.py` | `batch/gaze_quality_report.csv/json` | sequence-level validity |
| Head proxy | `batch_extract_head_proxy.py` | `head/head_samples.csv` | device/CPF pose in Scene plus relative motion |
| CPF gaze dynamics | `compute_gaze_dynamics_features.py` | `dynamics/gaze_dynamics.csv` | local eye-in-head velocity / dispersion |
| Scene gaze events | `detect_scene_gaze_events.py` | `events/scene_gaze_frame_labels.csv`, `events/scene_gaze_event_segments.csv` | world/Scene fixation vs transition labels |
| Scene object boxes | `batch_extract_scene_object_boxes.py` | `scene/scene_object_boxes.csv` | object cuboids in Scene frame |
| Gaze-object hits | `batch_compute_gaze_object_hits.py` | `scene/gaze_object_hits.csv` | Scene ray-box first hit and depth-point box check |
| Skeleton samples | `batch_extract_skeleton_samples.py` | `skeleton/skeleton_samples.csv` | skeleton joints in Scene frame |
| Head-gaze analysis | `analyze_*`, `report_*` | markdown reports + figures | diagnostic analysis, not base features |

Current scope:

- CPF dynamics are continuous local motion features, not final fixation labels.
- Scene gaze events are the current fixation/transition label layer.
- Object boxes are cuboids, not meshes or textures.
- Gaze-object hits use ray-box intersection against object cuboids. They are
  separate from `gaze_point_scene_xyz`, which is the ADT depth-defined gaze
  point.

## 0. Inspect One Sequence

Use this when checking whether a sequence has the expected ADT files:

```bash
python scripts/inspect_adt_sequence.py Apartment_release_decoration_skeleton_seq131_M1292
```

Inspect scene/object/skeleton assets without opening VRS data:

```bash
python scripts/inspect_scene_assets.py Apartment_release_decoration_skeleton_seq131_M1292
```

Optional skeleton JSON counting:

```bash
python scripts/inspect_scene_assets.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --include-skeleton-json
```

## 1. Extract Gaze Samples

Single sequence:

```bash
python scripts/extract_gaze_samples.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --output-dir "$REPORTS_DIR" \
  --stride 1
```

Batch all local ADT sequences under `ADT_DATA_ROOT`:

```bash
python scripts/batch_extract_gaze_samples.py \
  --output-dir "$REPORTS_DIR" \
  --stride 1
```

Useful windowed extraction for debugging:

```bash
python scripts/extract_gaze_samples.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --output-dir "$REPORTS_DIR" \
  --start-index 900 \
  --end-index 960 \
  --stride 1
```

Outputs:

- `sequences/<sequence>/gaze/gaze_samples.csv`
- `sequences/<sequence>/gaze/gaze_summary.json`
- `batch/batch_gaze_extract_summary.csv`
- `batch/batch_gaze_extract_summary.json`

Important fields:

- `yaw_rads_cpf`, `pitch_rads_cpf`: ADT eye gaze angles in CPF/head frame.
- `gaze_dir_cpf_unit_xyz`: unit gaze direction in CPF/head frame.
- `gaze_dir_scene_unit_xyz`: unit gaze direction in Scene/world frame.
- `gaze_origin_scene_xyz`: gaze/head/device origin in Scene frame.
- `gaze_point_scene_xyz`: ADT depth-defined gaze point in Scene frame.

## 2. Check Gaze Quality

After batch gaze extraction:

```bash
python scripts/check_gaze_quality.py --reports-dir "$REPORTS_DIR"
```

Outputs:

- `batch/gaze_quality_report.csv`
- `batch/gaze_quality_report.json`

This only reads existing `*_gaze_summary.json`; it does not reopen ADT
providers.

## 3. Extract Head Proxy Features

Single sequence:

```bash
python scripts/extract_head_proxy.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --input-gaze-csv "$REPORTS_DIR/sequences/Apartment_release_decoration_skeleton_seq131_M1292/gaze/gaze_samples.csv" \
  --output-dir "$REPORTS_DIR"
```

Batch:

```bash
python scripts/batch_extract_head_proxy.py --reports-dir "$REPORTS_DIR"
```

Outputs:

- `sequences/<sequence>/head/head_samples.csv`
- `sequences/<sequence>/head/head_summary.json`

Meaning:

- absolute head/device/CPF pose is represented in Scene frame.
- relative motion is computed between adjacent frames.
- `translation_scene_dxyz_m` is the adjacent-frame displacement in Scene axes.
- `translation_prev_head_dxyz_m` is the same displacement expressed in the
  previous head/CPF frame.
- `relative_rot_prev_to_cur_rij` is the adjacent-frame head rotation matrix.
- `head_rotation_speed_deg_s` and `head_translation_speed_m_s` are scalar
  motion magnitudes.

## 4. Compute CPF-Local Gaze Dynamics

This layer is useful for head-gaze relationship analysis. It is not the final
Scene fixation label layer.

```bash
python scripts/compute_gaze_dynamics_features.py --reports-dir "$REPORTS_DIR"
```

Outputs:

- `sequences/<sequence>/dynamics/gaze_dynamics.csv`
- `sequences/<sequence>/dynamics/gaze_dynamics_summary.json`
- `batch/batch_gaze_dynamics_summary.csv`

Meaning:

- velocity / dispersion are computed from CPF-local gaze direction.
- this describes eye-in-head movement, useful for asking how local gaze changes
  when the head rotates or translates.

## 5. Detect Scene-Direction Gaze Events

This is the current fixation/transition label pipeline.

```bash
python scripts/detect_scene_gaze_events.py --reports-dir "$REPORTS_DIR"
```

Default parameters:

- `--velocity-threshold-deg-s 40`
- `--dispersion-threshold-deg 2.5`
- `--dispersion-window-frames 5`
- `--min-fixation-duration-ms 133`

Explicit example:

```bash
python scripts/detect_scene_gaze_events.py \
  --reports-dir "$REPORTS_DIR" \
  --velocity-threshold-deg-s 40 \
  --dispersion-threshold-deg 2.5 \
  --dispersion-window-frames 5 \
  --min-fixation-duration-ms 133
```

Outputs:

- `sequences/<sequence>/events/scene_gaze_event_features.csv`
- `sequences/<sequence>/events/scene_gaze_frame_labels.csv`
- `sequences/<sequence>/events/scene_gaze_event_segments.csv`
- `sequences/<sequence>/events/scene_gaze_event_summary.json`
- `batch/batch_scene_gaze_event_summary.csv`

Meaning:

- event space is Scene-frame unit gaze direction.
- `fixation` means Scene gaze direction is stable under both velocity and
  dispersion thresholds for at least the minimum duration.
- `transition` covers non-fixation valid frames.
- `invalid` covers invalid or missing Scene gaze direction.

## 6. Extract Scene Object Boxes

Single sequence:

```bash
python scripts/extract_scene_object_boxes.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --output-dir "$REPORTS_DIR"
```

Batch:

```bash
python scripts/batch_extract_scene_object_boxes.py --output-dir "$REPORTS_DIR"
```

Outputs:

- `sequences/<sequence>/scene/scene_object_boxes.csv`
- `sequences/<sequence>/scene/scene_object_boxes_summary.json`
- `batch/batch_scene_object_boxes_summary.csv`
- `batch/batch_scene_object_boxes_report.json`

Meaning:

- joins `instances.json`, `scene_objects.csv`, and `3d_bounding_box.csv`.
- outputs object category, motion type, object pose, and eight Scene-frame box
  corners.
- static objects can be drawn once; dynamic objects need timestamp-specific
  rows.

## 7. Compute Gaze-Object Hits

This step answers a different question from `gaze_point_scene_xyz`.

- `gaze_point_scene_xyz` is the ADT depth-defined gaze point.
- `gaze_object_hits.csv` is the first intersection between the Scene-frame gaze
  ray and annotated object cuboids.
- By default, `shelter` is excluded because it is the room envelope and would
  otherwise dominate point-inside-box statistics.

Single sequence:

```bash
python scripts/compute_gaze_object_hits.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --reports-dir "$REPORTS_DIR"
```

Batch:

```bash
python scripts/batch_compute_gaze_object_hits.py --reports-dir "$REPORTS_DIR"
```

Outputs:

- `sequences/<sequence>/scene/gaze_object_hits.csv`
- `sequences/<sequence>/scene/gaze_object_hits_summary.json`
- `batch/batch_gaze_object_hits_summary.csv`
- `batch/batch_gaze_object_hits_report.json`

Useful columns:

- `object_hit`: whether the gaze ray hits an annotated cuboid.
- `hit_object_uid`, `hit_category`, `hit_motion_type`: first-hit object label.
- `hit_distance_m`, `hit_x_m/y_m/z_m`: first-hit point along the ray.
- `gaze_point_inside_any_box`: whether ADT's depth-defined gaze point is inside
  any candidate cuboid.
- `gaze_point_to_hit_distance_m`: difference between the ray-box hit point and
  ADT's depth-defined gaze point.

## 8. Extract Skeleton Samples

Skeleton extraction should run in the `adt` conda environment because it uses
the official ADT skeleton provider.

Single sequence:

```bash
conda run -n adt python scripts/extract_skeleton_samples.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --input-gaze-csv "$REPORTS_DIR/sequences/Apartment_release_decoration_skeleton_seq131_M1292/gaze/gaze_samples.csv" \
  --output-dir "$REPORTS_DIR"
```

Batch:

```bash
conda run -n adt python scripts/batch_extract_skeleton_samples.py \
  --reports-dir "$REPORTS_DIR"
```

Outputs:

- `sequences/<sequence>/skeleton/skeleton_samples.csv`
- `sequences/<sequence>/skeleton/skeleton_summary.json`
- `batch/batch_skeleton_samples_summary.csv`
- `batch/batch_skeleton_samples_report.json`

Meaning:

- rows are aligned to `gaze_samples.csv` timestamps.
- joints are in Scene frame.
- exported columns include root/head joints and all 51 ADT skeleton joints.

## 9. Analyze Head-Gaze Relationship

CPF-local / geometric head-gaze analysis:

```bash
python scripts/analyze_head_gaze_relationship.py --reports-dir "$REPORTS_DIR"
python scripts/report_head_gaze_relationship.py --reports-dir "$REPORTS_DIR"
```

Scene/world head-gaze analysis:

```bash
python scripts/analyze_scene_head_gaze_relationship.py --reports-dir "$REPORTS_DIR"
python scripts/report_scene_head_gaze_relationship.py --reports-dir "$REPORTS_DIR"
```

SparseGaze-oriented head utility analysis:

```bash
python scripts/analyze_sparsegaze_head_utility.py --reports-dir "$REPORTS_DIR"
python scripts/report_sparsegaze_head_utility.py --reports-dir "$REPORTS_DIR"
```

Main reports:

- `docs/head_gaze_relationship_report.md`
- `docs/scene_head_gaze_relationship_report.md`
- `docs/sparsegaze_head_utility_report.md`

These analyses are diagnostics. They do not create the base feature layers used
by the viewer.

## 10. Visualize Gaze Outputs

This opens ADT provider for a selected window and generates overlay/scanpath
figures and video.

```bash
python scripts/visualize_gaze_outputs.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --reports-dir "$REPORTS_DIR" \
  --start-row 0 \
  --end-row 60 \
  --stride 10 \
  --run-name row_0_60_stride_10
```

Outputs default to `outputs/figures/`.

Use this for:

- RGB overlay frames
- RGB overlay video
- 2D scanpath
- Scene-frame gaze ray figures

## 11. Visualize Scene Gaze Events

This does not reopen ADT provider. It only reads event CSVs.

```bash
python scripts/visualize_scene_gaze_events.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --reports-dir "$REPORTS_DIR" \
  --start-frame 0 \
  --end-frame 600
```

Outputs default to:

```text
outputs/figures/scene_gaze_events/
```

Use this to inspect:

- final `fixation` / `transition` labels
- Scene angular velocity
- Scene angular dispersion
- event segment boundaries

## 12. Interactive Notebooks

Current notebooks:

- `notebooks/02_gaze_head_scene_viewer.ipynb`
  - Matplotlib 3D view of Scene-frame gaze and head forward vectors.
  - Good for quick direction sanity checks.
- `notebooks/04_gaze_head_scene_viewer_interactive.ipynb`
  - Plotly 3D view of Scene-frame gaze/head vectors.
  - Supports interactive zoom and controlled azimuth rotation.
- `notebooks/05_scene_object_gaze_viewer.ipynb`
  - Plotly 3D scene viewer.
  - Shows object cuboids, skeleton, head/device trajectory, gaze rays, and
    depth-defined gaze points.
  - Requires `gaze_samples`, `head_samples`, `scene_object_boxes`, and
    `skeleton_samples`.
- `notebooks/06_scene_object_gaze_dynamic_viewer.ipynb`
  - Dynamic Plotly/IPyWidgets scene viewer with play/slider frame control.
  - Updates dynamic object boxes, skeleton pose, gaze ray, gaze point, ray-box
    hit point, and current ray-box hit outline over time.
  - Use larger `frame step`, larger `draw stride`, or fewer static boxes if
    playback is slow.
  - Key controls:
    - `start` / `end`: global frame range available to the play control.
    - `frame step`: frame increment for the play control and focus slider.
    - `context`: number of neighboring frames drawn before/after the current
      focus frame for trajectories and gaze rays.
    - `draw stride`: subsampling inside the rendered context window.
    - `max static`: maximum number of static object boxes to draw; `0` means no
      cap.
    - `categories`: optional comma-separated object categories to include.
      Empty means all categories.
    - `exclude`: comma-separated object categories to hide. Default `shelter`
      removes the room-envelope box.
    - `ray scale`: `fixed` draws fixed-length gaze rays for direction
      comparison; `depth` scales rays by ADT gaze depth.
    - `ray len`: fixed ray length in meters when `ray scale=fixed`.
    - `hit point`: shows the current ray-box intersection point.
    - `hit object outline`: adds an outline to the object cuboid intersected by
      the current gaze ray. This is a ray-box hit cue, not confirmed attention.
    - `auto render`: when enabled, slider/play changes rebuild the 3D figure.
      Disable it if the notebook UI becomes slow, then use `Render current`.
- `notebooks/07_multiview_gaze_dashboard.ipynb`
  - Interactive figure finder for one sequence/window.
  - Shows local gaze, motion magnitude, image-space gaze, object-hit context,
    and 3D Scene context in one coordinated figure.
  - Includes a lightweight prediction-track hook for future model-output CSVs.
  - Design notes: `docs/multiview_gaze_dashboard_design.md`.

To use the notebooks, first make sure the feature layers exist in:

```text
/mnt/d/SparseGaze/ADT-Gaze-structured
```

Then open the notebook in VS Code and select a Python/Jupyter kernel with
`pandas`, `plotly`, and `ipywidgets` installed. If the `adt` kernel is not
visible, register it once:

```bash
conda run -n adt python -m ipykernel install --user --name adt --display-name "Python (adt)"
```

## Suggested Full Pipeline

For a clean rerun from raw ADT data to current visualization-ready features:

```bash
export REPORTS_DIR=/mnt/d/SparseGaze/ADT-Gaze-structured

python scripts/batch_extract_gaze_samples.py --output-dir "$REPORTS_DIR" --stride 1
python scripts/check_gaze_quality.py --reports-dir "$REPORTS_DIR"

python scripts/batch_extract_head_proxy.py --reports-dir "$REPORTS_DIR"
python scripts/compute_gaze_dynamics_features.py --reports-dir "$REPORTS_DIR"
python scripts/detect_scene_gaze_events.py --reports-dir "$REPORTS_DIR"

python scripts/batch_extract_scene_object_boxes.py --output-dir "$REPORTS_DIR"
conda run -n adt python scripts/batch_extract_skeleton_samples.py --reports-dir "$REPORTS_DIR"
```

Then inspect:

```bash
python scripts/visualize_scene_gaze_events.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --reports-dir "$REPORTS_DIR" \
  --start-frame 0 \
  --end-frame 600
```

and open:

```text
notebooks/05_scene_object_gaze_viewer.ipynb
```

## Validation Commands

Static checks for the newly added scene/object viewer layer:

```bash
python -m py_compile \
  src/adt_sandbox/scene_features.py \
  src/adt_sandbox/scene_object_viewer.py \
  scripts/inspect_scene_assets.py \
  scripts/extract_scene_object_boxes.py \
  scripts/batch_extract_scene_object_boxes.py
```

Skeleton checks:

```bash
conda run -n adt python -m py_compile \
  src/adt_sandbox/skeleton_features.py \
  scripts/extract_skeleton_samples.py \
  scripts/batch_extract_skeleton_samples.py
```

Notebook JSON check:

```bash
python -m json.tool \
  notebooks/05_scene_object_gaze_viewer.ipynb \
  /tmp/scene_viewer_notebook_check.json
```
