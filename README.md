# ADT Dataset Sandbox

This repository is a focused workspace for exploring the Aria Digital Twin
(ADT) dataset from WSL. Large data stays on the Windows D drive, this repo
keeps only lightweight code, notes, manifests, notebooks, and generated reports.

## Layout

```text
.
├── configs/          # local path templates and environment examples
├── docs/             # notes about ADT formats and workflows
├── external/         # optional local source checkouts, ignored
├── manifests/        # lightweight sequence lists or dataset notes
├── notebooks/        # exploratory notebooks
├── outputs/
│   ├── figures/      # generated plots, ignored by git
│   └── reports/      # generated summaries, ignored by git
├── scripts/          # command-line helpers
└── src/adt_sandbox/  # reusable Python utilities for this sandbox
```

`external/` is optional. It can hold a local `projectaria_tools` source checkout
for reading official examples or source code, but the default workflow uses the
installed `projectaria-tools` Python package.

## Setup

Use a dedicated environment for this sandbox. Do not rely on `base` or reuse
paper-specific environments, even if they already have `projectaria-tools`
installed.

Recommended conda setup:

```bash
conda create -n adt python=3.10
conda activate adt
python -m pip install -e ".[dev]"
```

This installs the sandbox package plus the dependencies declared in
`pyproject.toml`, including `projectaria-tools`, `numpy`, `pandas`,
`matplotlib`, and `tqdm`.

Quick check:

```bash
python -c "import projectaria_tools; print(projectaria_tools.__file__)"
python -m py_compile src/adt_sandbox/adt_files.py scripts/inspect_adt_sequence.py
```

If you need to inspect the upstream Project Aria source, keep that checkout
outside this repo or under ignored `external/projectaria_tools/`. It is not
required for normal use.

## Data Location

Do not store ADT data inside the WSL repository. Keep large datasets on the
Windows D drive and point the sandbox to them with `ADT_DATA_ROOT`.

Example:

```bash
export ADT_DATA_ROOT=/mnt/d/Pose2Gaze-ADT
```

For repeated use, put the real value in your shell config or a local untracked
`.env` file. The template is [configs/paths.example.env](configs/paths.example.env).

Inside that root, a typical sequence layout is:

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

The repository ignores local `data/` directories as a guardrail, but the
preferred path is to keep ADT files under `/mnt/d/...`.

## First Checks

Inspect one downloaded sequence without loading VRS data, using either an
absolute path:

```bash
python scripts/inspect_adt_sequence.py /mnt/d/path/to/adt/<sequence_id>
```

or a sequence id resolved under `ADT_DATA_ROOT`:

```bash
export ADT_DATA_ROOT=/mnt/d/Pose2Gaze-ADT
python scripts/inspect_adt_sequence.py <sequence_id>
```

Write the same summary as JSON:

```bash
python scripts/inspect_adt_sequence.py <sequence_id> --json \
  > outputs/reports/<sequence_id>_summary.json
```

## Working Conventions

- Activate the dedicated `adt` environment before running scripts or notebooks.
- Keep reusable code in `src/adt_sandbox`.
- Keep one-off exploration in `notebooks`.
- Keep downloaded ADT data on D drive, not in the WSL repo.
- Keep upstream tool source checkouts out of git; use `external/` only as an
  ignored local reference if needed.
