# ADT Scene Feature Extraction Plan

## Goal

This track is paused from SparseGaze model analysis and focuses on extracting
Scene-level context that will later support visualization and evaluation:

- room / object layout
- object 3D boxes
- body / skeleton trajectory
- gaze rays and gaze-object interaction

The important motivation is that SparseGaze should eventually be evaluated not
only by angular MAE, but also by whether predicted gaze lands on the same object
or target region as GT gaze.

## Priority

Current priority:

1. `object boxes`
2. `gaze-object hit test`
3. interactive 3D scene viewer
4. skeleton/body trajectory
5. object-level fixation / hit-rate evaluation

Object boxes come first because they directly support:

- drawing a rough 3D room/object layout
- computing ray-box intersections
- comparing predicted-vs-GT object hit agreement

Skeleton/body trajectory is useful for explanation and visualization, but it is
less directly connected to SparseGaze evaluation than object hit metrics.

## Current Asset Availability

For `Apartment_release_decoration_skeleton_seq131_M1292`, file-level inspection
shows the relevant assets exist:

- `instances.json`
- `scene_objects.csv`
- `3d_bounding_box.csv`
- `2d_bounding_box.csv`
- `Skeleton_T.json`
- `skeleton_aria_association.json`

Single-sequence inspection result:

- instances: `354`
- object instances: `353`
- object pose rows: `139,513`
- object pose timestamps: `2,842`
- static object rows: `304`
- 3D object boxes: `353`
- invalid 3D box size rows: `0`
- 2D object box rows: `403,660`
- skeleton file exists: yes

Environment note: use the `adt` conda environment for official Project Aria ADT
APIs:

```bash
conda run -n adt python ...
```

The base conda environment has `projectaria-tools 1.1.0` and can produce
misleading provider errors. The `adt` environment has `projectaria-tools 2.1.2`
and can initialize this sequence. For object boxes, the current extractor still
uses files on disk directly because that is simpler and avoids opening large VRS
files. For skeleton, use the direct `AriaDigitalTwinSkeletonProvider` on
`Skeleton_T.json`.

## Implemented First Step

### Asset inspection

```bash
python scripts/inspect_scene_assets.py Apartment_release_decoration_skeleton_seq131_M1292
```

Optional skeleton JSON counting:

```bash
python scripts/inspect_scene_assets.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --include-skeleton-json
```

This reports:

- instance count and category / motion-type distributions
- object pose row count from `scene_objects.csv`
- 3D box row count from `3d_bounding_box.csv`
- 2D box row count and stream ids
- skeleton file presence and optional frame/joint/marker counts

### Scene object box extraction

```bash
python scripts/extract_scene_object_boxes.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --output-dir /mnt/d/SparseGaze/ADT-Gaze-structured
```

Inputs:

- `instances.json`
- `scene_objects.csv`
- `3d_bounding_box.csv`

Output:

- `sequences/<sequence>/scene/scene_object_boxes.csv`
- `sequences/<sequence>/scene/scene_object_boxes_summary.json`

Batch command:

```bash
python scripts/batch_extract_scene_object_boxes.py \
  --output-dir /mnt/d/SparseGaze/ADT-Gaze-structured
```

Batch output:

- `batch/batch_scene_object_boxes_summary.csv`
- `batch/batch_scene_object_boxes_report.json`

Each row joins:

- object id / category / motion type
- `T_scene_object` translation and quaternion from `scene_objects.csv`
- local object AABB from `3d_bounding_box.csv`
- eight Scene-frame 3D box corners

Validation on `Apartment_release_decoration_skeleton_seq131_M1292`:

- output rows: `139,513`
- unique objects: `353`
- unique timestamps: `2,842`
- object-level motion types: `304` static objects, `49` dynamic objects
- row-level motion types: `304` static rows, `139,209` dynamic rows

Coordinate-frame definitions:

- object pose is `T_scene_object`, derived from `t_wo` / `q_wo`
- local AABB is in object-local frame
- output corners are in ADT Scene/world frame, meters

## Implemented Skeleton Step

### Skeleton samples

```bash
conda run -n adt python scripts/extract_skeleton_samples.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --input-gaze-csv /mnt/d/SparseGaze/ADT-Gaze-structured/sequences/Apartment_release_decoration_skeleton_seq131_M1292/gaze/gaze_samples.csv \
  --output-dir /mnt/d/SparseGaze/ADT-Gaze-structured
```

Batch command:

```bash
conda run -n adt python scripts/batch_extract_skeleton_samples.py \
  --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured
```

Output:

- `sequences/<sequence>/skeleton/skeleton_samples.csv`
- `sequences/<sequence>/skeleton/skeleton_summary.json`
- `batch/batch_skeleton_samples_summary.csv`
- `batch/batch_skeleton_samples_report.json`

Each row contains:

- query timestamp
- nearest skeleton timestamp offset
- root joint Scene position
- head joint Scene position
- all 51 ADT skeleton joints as Scene-frame xyz columns

Validation on `Apartment_release_decoration_skeleton_seq131_M1292`:

- gaze-aligned rows: `2,840`
- valid skeleton rows: `2,840`
- valid ratio: `1.000`
- joints: `51`
- markers: `57`
- median abs skeleton timestamp offset: about `2.25 ms`
- max abs skeleton timestamp offset: about `10.66 ms`

### Interactive 3D scene viewer

Notebook:

```text
Experiments/visualization & Analysis/ADT/notebooks/05_scene_object_gaze_viewer.ipynb
```

Inputs under the reports directory:

- `sequences/<sequence>/gaze/gaze_samples.csv`
- `sequences/<sequence>/head/head_samples.csv`
- `sequences/<sequence>/scene/scene_object_boxes.csv`
- `sequences/<sequence>/skeleton/skeleton_samples.csv`

The viewer can render:

- static Scene-frame object boxes
- dynamic object boxes at the selected focus frame
- optional object centers
- skeleton joints and bone connections at the focus frame
- skeleton root trajectory
- head/device origin trajectory
- Scene-frame gaze rays
- depth-defined gaze points

Validation on `Apartment_release_decoration_skeleton_seq131_M1292`:

- loaded frames: `2,840`
- loaded object-box pose rows: `139,513`
- generated Plotly traces in a short smoke-test window: `8`

Interpretation notes:

- head trajectory here is the ADT device/CPF origin trajectory, not a separate
  body root estimate.
- gaze points are the ADT eye-gaze depth points; ray-box first-hit points are
  computed separately by `compute_gaze_object_hits.py`.
- object boxes are rough cuboids, enough for layout and hit testing, but not
  textured meshes.

## Implemented Gaze-Object Hit Step

```bash
python scripts/compute_gaze_object_hits.py \
  Apartment_release_decoration_skeleton_seq131_M1292 \
  --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured
```

Batch command:

```bash
python scripts/batch_compute_gaze_object_hits.py \
  --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured
```

Inputs:

- `sequences/<sequence>/gaze/gaze_samples.csv`
- `sequences/<sequence>/scene/scene_object_boxes.csv`

Output:

- `sequences/<sequence>/scene/gaze_object_hits.csv`
- `sequences/<sequence>/scene/gaze_object_hits_summary.json`
- `batch/batch_gaze_object_hits_summary.csv`
- `batch/batch_gaze_object_hits_report.json`

Method:

- ray origin: `gaze_origin_scene_xyz`
- ray direction: `gaze_dir_scene_unit_xyz`
- target geometry: oriented Scene-frame object cuboids
- static boxes are active for every gaze sample
- dynamic boxes use the nearest object timestamp within `20 ms`
- `shelter` is excluded by default because it is the room envelope and otherwise
  makes point-inside-box statistics almost always true

The output keeps two related but different interpretations:

- `object_hit`: first ray-box intersection against annotated cuboids.
- `gaze_point_inside_any_box`: whether ADT's depth-defined
  `gaze_point_scene_xyz` falls inside any candidate cuboid.

Batch validation on `/mnt/d/SparseGaze/ADT-Gaze-structured`:

- sequences: `34`
- failures: `0`
- mean `object_hit_ratio`: `0.917`
- min / max `object_hit_ratio`: `0.772` / `0.989`
- mean `gaze_point_inside_any_box_ratio`: `0.561`
- min / max `gaze_point_inside_any_box_ratio`: `0.397` / `0.691`

Interpretation:

- Ray-box hit rate is high because the gaze ray often intersects some annotated
  cuboid within `20 m`.
- The depth-defined gaze point falls inside annotated cuboids less often. This
  supports keeping `gaze_point_scene_xyz` and ray-box `object_hit` as separate
  measurements instead of treating them as equivalent.

## Remaining Next Steps

1. Extend `Experiments/visualization & Analysis/ADT/notebooks/05_scene_object_gaze_viewer.ipynb` later with predicted
   gaze rays, once SparseGaze predictions are exported in the same Scene-frame
   format.
2. Add prediction evaluation:
   - GT object hit vs predicted object hit
   - object-hit agreement
   - hit/no-hit confusion
   - hit-point distance on same object or same AOI
