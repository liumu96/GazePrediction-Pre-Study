# ADT Dataset Sandbox

This repository is a focused workspace for exploring the Aria Digital Twin
(ADT) dataset and preparing ADT-derived features, diagnostics, and
visualizations for SparseGaze-style gaze prediction research.

Large ADT data and generated feature tables stay outside the repository,
currently on the Windows D drive. This repo keeps only lightweight code,
notebooks, documentation, manifests, and small reports.

## Current Status

The repository has moved from basic dataset inspection to a structured
feature-and-visualization pipeline. The current pipeline can extract and inspect:

- gaze samples in CPF, RGB image, and Scene frames;
- device/CPF head proxy features aligned to gaze timestamps;
- CPF-local gaze dynamics features;
- Scene-direction fixation/transition event labels;
- Scene object cuboids;
- Scene ray-box gaze-object hit features;
- ADT skeleton joints aligned to gaze timestamps;
- head-gaze relationship diagnostics;
- interactive 3D and multi-view notebooks for sequence/window inspection.

The current standard output root is:

```bash
REPORTS_DIR=/mnt/d/SparseGaze/ADT-Gaze-structured
```

The old flat output directory `/mnt/d/SparseGaze/ADT-Gaze` is no longer the
expected layout for current code.

## Repository Layout

```text
.
├── configs/          # local path templates and environment examples
├── analysis/         # reusable model-vs-GT prediction analysis code
├── docs/             # pipeline notes, analysis plans, reports, command guide
├── external/         # optional local source checkouts, ignored
├── manifests/        # lightweight sequence lists or dataset notes
├── notebooks/        # interactive inspection notebooks
├── outputs/
│   ├── figures/      # generated plots, ignored by git
│   └── reports/      # small local reports, ignored by git
├── scripts/          # command-line feature extraction / analysis helpers
├── src/adt_sandbox/  # reusable Python utilities
└── visualization/    # reusable visualization modules and CLI viewers
```

`external/` is optional. It can hold a local `projectaria_tools` source checkout
for reading official examples or source code, but the normal workflow uses the
installed `projectaria-tools` Python package.

## Environment

Use the dedicated `adt` conda environment. Do not rely on `base`.

```bash
conda create -n adt python=3.10
conda activate adt
python -m pip install -e ".[dev]"
```

Quick check:

```bash
which python
python -c "import projectaria_tools; print(projectaria_tools.__file__)"
python -m py_compile src/adt_sandbox/adt_files.py scripts/inspect_adt_sequence.py
```

`which python` should point to the `adt` environment, for example:

```text
/home/liumu/miniconda3/envs/adt/bin/python
```

Some skeleton extraction commands use the official ADT skeleton provider and
should be run with `conda run -n adt ...` if the current terminal is not already
inside the `adt` environment.

## Data and Output Locations

Do not store ADT data inside the repository. Point the sandbox to the raw ADT
root with `ADT_DATA_ROOT`.

Example:

```bash
export ADT_DATA_ROOT=/mnt/d/Pose2Gaze-ADT
```

For repeated use, put the real path in a local untracked `.env` file. The
template is [configs/paths.example.env](configs/paths.example.env). Scripts
load `.env` from the repository root.

Raw ADT sequences are expected under:

```text
$ADT_DATA_ROOT/
└── <sequence_id>/
    ├── metadata.json
    ├── instances.json
    ├── scene_objects.csv
    ├── aria_trajectory.csv
    ├── 2d_bounding_box.csv
    ├── 3d_bounding_box.csv
    ├── video.vrs
    ├── depth_images.vrs
    ├── segmentations.vrs
    └── mps/
```

Generated feature outputs use a sequence-first layout:

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

Open one sequence by going to:

```text
$REPORTS_DIR/sequences/<sequence_id>/
```

## Feature Layers

| Layer | Main script | Main output | Meaning |
| --- | --- | --- | --- |
| Gaze samples | `batch_extract_gaze_samples.py` | `gaze/gaze_samples.csv` | CPF, RGB camera, and Scene gaze features |
| Gaze quality | `check_gaze_quality.py` | `batch/gaze_quality_report.csv/json` | sequence-level validity summary |
| Head proxy | `batch_extract_head_proxy.py` | `head/head_samples.csv` | device/CPF pose in Scene plus relative motion |
| CPF gaze dynamics | `compute_gaze_dynamics_features.py` | `dynamics/gaze_dynamics.csv` | eye-in-head velocity / dispersion |
| Scene gaze events | `detect_scene_gaze_events.py` | `events/scene_gaze_frame_labels.csv` | Scene-direction fixation / transition labels |
| Scene object boxes | `batch_extract_scene_object_boxes.py` | `scene/scene_object_boxes.csv` | object cuboids in Scene frame |
| Gaze-object hits | `batch_compute_gaze_object_hits.py` | `scene/gaze_object_hits.csv` | Scene ray-box first hit and depth-point box check |
| Skeleton samples | `batch_extract_skeleton_samples.py` | `skeleton/skeleton_samples.csv` | skeleton joints in Scene frame |
| Head-gaze diagnostics | `analyze_*`, `report_*` | docs + batch summaries | diagnostic analysis, not base features |

Important interpretation notes:

- CPF gaze dynamics are continuous local eye-in-head features, not final
  fixation labels.
- Current fixation/transition labels are Scene-direction event labels.
- Object geometry is cuboids, not meshes or textured surfaces.
- `gaze_point_scene_xyz` is ADT's depth-defined gaze point.
- `gaze_object_hits.csv` is a separate ray-box intersection approximation.
- A ray-box hit does not by itself mean confirmed visual attention to that
  object.
- The current head proxy is device/CPF pose, not skeleton head joint pose.

## Quick Start

From the repository root:

```bash
cd /home/liumu/Github_Projects/adt_dataset_sandbox
conda activate adt
export REPORTS_DIR=/mnt/d/SparseGaze/ADT-Gaze-structured
```

Inspect one sequence:

```bash
python scripts/inspect_adt_sequence.py Apartment_release_decoration_skeleton_seq131_M1292
python scripts/inspect_scene_assets.py Apartment_release_decoration_skeleton_seq131_M1292
```

Run the current feature pipeline:

```bash
python scripts/batch_extract_gaze_samples.py --output-dir "$REPORTS_DIR" --stride 1
python scripts/check_gaze_quality.py --reports-dir "$REPORTS_DIR"

python scripts/batch_extract_head_proxy.py --reports-dir "$REPORTS_DIR"
python scripts/compute_gaze_dynamics_features.py --reports-dir "$REPORTS_DIR"
python scripts/detect_scene_gaze_events.py --reports-dir "$REPORTS_DIR"

python scripts/batch_extract_scene_object_boxes.py --output-dir "$REPORTS_DIR"
python scripts/batch_compute_gaze_object_hits.py --reports-dir "$REPORTS_DIR"
conda run -n adt python scripts/batch_extract_skeleton_samples.py --reports-dir "$REPORTS_DIR"
```

Detailed commands, single-sequence variants, fields, and validation commands are
documented in:

- [docs/adt_feature_and_visualization_commands.md](docs/adt_feature_and_visualization_commands.md)

## Interactive Notebooks

Current notebooks:

- [notebooks/02_gaze_head_scene_viewer.ipynb](notebooks/02_gaze_head_scene_viewer.ipynb)
  - Matplotlib 3D sanity check for Scene-frame gaze and head forward vectors.
- [notebooks/04_gaze_head_scene_viewer_interactive.ipynb](notebooks/04_gaze_head_scene_viewer_interactive.ipynb)
  - Plotly 3D gaze/head direction viewer with controlled view rotation.
- [notebooks/05_scene_object_gaze_viewer.ipynb](notebooks/05_scene_object_gaze_viewer.ipynb)
  - Static Plotly 3D scene viewer for objects, skeleton, head/device
    trajectory, gaze rays, gaze points, and ray-box hit cues.
- [notebooks/06_scene_object_gaze_dynamic_viewer.ipynb](notebooks/06_scene_object_gaze_dynamic_viewer.ipynb)
  - Dynamic frame-by-frame scene viewer with play/slider control.
  - `hit object outline` means current ray-box intersection only, not confirmed
    attention.
- [notebooks/07_multiview_gaze_dashboard.ipynb](notebooks/07_multiview_gaze_dashboard.ipynb)
  - Multi-view figure finder for local gaze, motion magnitude, image-space gaze,
    object-hit context, and 3D Scene context.
  - Design notes: [docs/multiview_gaze_dashboard_design.md](docs/multiview_gaze_dashboard_design.md).
- [notebooks/08_prediction_gaze_evaluation_viewer.ipynb](notebooks/08_prediction_gaze_evaluation_viewer.ipynb)
  - SparseGaze prediction-result diagnostics: aggregate errors, per-frame
    error distributions, anchor/event breakdowns, and selected-window traces.
- [notebooks/09_npz_gaze_visualization_viewer.ipynb](notebooks/09_npz_gaze_visualization_viewer.ipynb)
  - Visualizes one per-sequence prediction `.npz` with the same Scene rays,
    scanpath overlays, overlay frames, and video outputs as CSV gaze
    visualization.
  - Uses `.npz` Scene-frame gaze directions plus extracted ADT CSV context.

If the `adt` Jupyter kernel is not visible in VS Code:

```bash
conda run -n adt python -m ipykernel install --user --name adt --display-name "Python (adt)"
```

## Analysis Reports and Plans

Project planning and analysis notes:

- [docs/adt_exploration_plan.md](docs/adt_exploration_plan.md)
- [docs/adt_feature_extraction_guide.md](docs/adt_feature_extraction_guide.md)
- [docs/tutorial_gaze_feature_extraction.md](docs/tutorial_gaze_feature_extraction.md)
- [docs/scene_feature_extraction_plan.md](docs/scene_feature_extraction_plan.md)
- [docs/gaze_event_analysis_notes.md](docs/gaze_event_analysis_notes.md)
- [docs/sparsegaze_modeling_notes.md](docs/sparsegaze_modeling_notes.md)

Quality and event reports:

- [docs/gaze_quality_report_notes.md](docs/gaze_quality_report_notes.md)
- [docs/gaze_event_detection_report.md](docs/gaze_event_detection_report.md)
- [docs/fixation_policy_comparison_notes.md](docs/fixation_policy_comparison_notes.md)

Head-gaze and SparseGaze-oriented diagnostics:

- [docs/head_gaze_relationship_analysis.md](docs/head_gaze_relationship_analysis.md)
- [docs/head_gaze_relationship_report.md](docs/head_gaze_relationship_report.md)
- [docs/scene_head_gaze_relationship_report.md](docs/scene_head_gaze_relationship_report.md)
- [docs/sparsegaze_head_utility_analysis_plan.md](docs/sparsegaze_head_utility_analysis_plan.md)
- [docs/sparsegaze_head_utility_report.md](docs/sparsegaze_head_utility_report.md)

Visualization design:

- [docs/multiview_gaze_dashboard_design.md](docs/multiview_gaze_dashboard_design.md)

## Current Research Direction

The dataset exploration layer is largely in place. The next useful work is to
turn extracted Scene context into SparseGaze evaluation and analysis tasks:

- object-level gaze event analysis by combining Scene events and ray-box hits;
- GT vs predicted gaze comparison in Scene/event/object space;
- model-output visual diagnostics in the existing scene viewers;
- device/CPF head proxy vs skeleton head sanity checks;
- qualitative case selection for paper figures.

## Working Conventions

- Activate the dedicated `adt` environment before running scripts or notebooks.
- Keep large raw data and generated feature tables outside the repo.
- Keep reusable code in `src/adt_sandbox`.
- Keep command-line entry points in `scripts`.
- Keep interactive exploration in `notebooks`.
- Keep analysis decisions in `docs`, not only in chat or notebook outputs.
- Keep README as the project entry point and use detailed docs for long command
  references.
