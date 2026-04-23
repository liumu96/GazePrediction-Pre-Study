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
which python
python -c "import projectaria_tools; print(projectaria_tools.__file__)"
python -m py_compile src/adt_sandbox/adt_files.py scripts/inspect_adt_sequence.py
```

`which python` should point to `/home/liumu/miniconda3/envs/adt/bin/python`.
If it points to `/home/liumu/miniconda3/bin/python`, the terminal is still using
the base environment.

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
Scripts in this repository automatically load `.env` from the repository root.
VS Code is also configured to inject this `.env` into Python terminals.

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

or a sequence id resolved under `ADT_DATA_ROOT` from the local `.env`:

```bash
python scripts/inspect_adt_sequence.py <sequence_id>
```

Write the same summary as JSON:

```bash
python scripts/inspect_adt_sequence.py <sequence_id> --json \
  > outputs/reports/<sequence_id>_summary.json
```

## Exploration Roadmap / 探索路线

ADT 探索路线记录在：

- [docs/adt_exploration_plan.md](docs/adt_exploration_plan.md)
- [docs/adt_feature_extraction_guide.md](docs/adt_feature_extraction_guide.md)
- [docs/tutorial_gaze_feature_extraction.md](docs/tutorial_gaze_feature_extraction.md)

这些文档以中文说明为主，同时保留官方 API、字段名和坐标系英文术语。
当 scripts、notebooks、APIs 或假设发生变化时，及时同步更新。

已实现的 gaze-first tutorial：

```bash
python scripts/extract_gaze_samples.py <sequence_id> --start-index 900 --end-index 905 --stride 1
```

它会导出完整 gaze CSV、RGB frames、RGB overlay frames 和 `manifest.json`。
调试局部片段时可以用 `--start-index/--end-index` 选帧区间，也可以用
`--start-offset-s/--end-offset-s` 按 sequence 内相对秒数选时间区间。
如果已有 CSV 和 manifest，只想调整可视化参数，使用
`scripts/visualize_gaze_outputs.py --stride <N>`，避免重新逐帧提取 gaze 或打开
ADT provider。
下一步是把这个 workflow 扩展成批量 gaze quality report，然后再做 pose、
skeleton 和 object feature extraction。

## Working Conventions

- Activate the dedicated `adt` environment before running scripts or notebooks.
- Keep exploration decisions in `docs/`, not only in chat or notebook outputs.
- Keep reusable code in `src/adt_sandbox`.
- Keep one-off exploration in `notebooks`.
- Keep downloaded ADT data on D drive, not in the WSL repo.
- Keep upstream tool source checkouts out of git; use `external/` only as an
  ignored local reference if needed.
