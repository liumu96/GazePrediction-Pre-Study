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
- [docs/gaze_quality_report_notes.md](docs/gaze_quality_report_notes.md)
- [docs/sparsegaze_modeling_notes.md](docs/sparsegaze_modeling_notes.md)
- [docs/gaze_event_analysis_notes.md](docs/gaze_event_analysis_notes.md)
- [docs/head_gaze_relationship_analysis.md](docs/head_gaze_relationship_analysis.md)
- [docs/head_gaze_relationship_report.md](docs/head_gaze_relationship_report.md)

这些文档以中文说明为主，同时保留官方 API、字段名和坐标系英文术语。
当 scripts、notebooks、APIs 或假设发生变化时，及时同步更新。

已实现的 gaze-first tutorial：

```bash
python scripts/extract_gaze_samples.py <sequence_id> --start-index 900 --end-index 905 --stride 1
```

它会导出 `gaze_samples.csv` 和一个轻量 `gaze_summary.json`，先把核心数据和质量
统计保存下来。调试局部片段时可以用 `--start-index/--end-index` 选帧区间，也可以
用 `--start-offset-s/--end-offset-s` 按 sequence 内相对秒数选时间区间。
如果后面想围绕某个 event/window 生成 scene rays、scanpath、overlay frames
和 overlay video，再运行 `scripts/visualize_gaze_outputs.py`。这样图片和视频
就作为后处理，而不是默认主流程。

如果下一步想直接批量处理所有 sequence 的 gaze 数据，不做可视化：

```bash
python scripts/batch_extract_gaze_samples.py
```

它会为每个 sequence 生成一份 `gaze_samples.csv` 和 `gaze_summary.json`，
同时在 `outputs/reports/` 下写一份批量总表，方便后续再做 quality report、
pose、skeleton 和 object feature extraction。

如果批量提取已经完成，只想汇总这些 summary 的 sequence-level 质量：

```bash
python scripts/check_gaze_quality.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

它只读取已有的 `*_gaze_summary.json`，输出：

- `gaze_quality_report.csv`
- `gaze_quality_report.json`

不会重新打开 ADT provider，也不会生成可视化。

如果下一步开始准备 event analysis，或者想先把可复用的 head 特征层落盘，
先提取和 gaze 行对齐的 `head_samples.csv`：

```bash
python scripts/extract_head_proxy.py <sequence_id>
```

它会读取已有的 `gaze_samples.csv`，按相同时间戳查询 pose，导出：

- `head_samples.csv`
- `head_summary.json`

当前 `head.py` 的定位已经不是只给 event 用的一张 context 摘要表，而是一层独立
的 head feature layer。第一版 head 仍然不从 skeleton 提取，而是使用
`device pose + CPF` 作为 tracker-mounted head proxy；输出同时包含：

- Scene frame 下的绝对 pose
- 相邻帧的相对平移 / 相对旋转
- 可直接用于 head-gaze 关系分析和后续模型设计的基础特征

如果下一步要系统分析 head 和 gaze 的关系，而不是立刻改模型，直接运行：

先确认 `head_samples.csv` 是用当前 `head.py` 重构后的 schema 重新导出的。
如果 D 盘上还是旧版 head CSV，先重跑：

```bash
python scripts/batch_extract_head_proxy.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

然后再运行：

```bash
python scripts/analyze_head_gaze_relationship.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

它会基于现有：

- `gaze_samples.csv`
- `head_samples.csv`

生成：

- 每条 sequence 的逐帧 joined table：`*_head_gaze_analysis_rows.csv`
- 每条 sequence 的统计摘要：`*_head_gaze_analysis_summary.json`
- 一份批量总表：`batch_head_gaze_analysis_summary.csv`
- 一份批量报告：`batch_head_gaze_analysis_report.json`

这一步的目标不是新的 detector，而是回答：

- Scene 里 head 和 gaze 的几何关系是什么
- local gaze dynamics 和 head motion 是否同步
- current head motion 对下一步 gaze change 有没有统计上的解释力

完整说明见：

- [docs/head_gaze_relationship_analysis.md](docs/head_gaze_relationship_analysis.md)

如果要把 whole-sequence CPF-local gaze dynamics feature 也跑出来：

```bash
python scripts/compute_gaze_dynamics_features.py --reports-dir /mnt/d/SparseGaze/ADT-Gaze
```

它只保存连续特征：

- `*_gaze_dynamics.csv`
- `*_gaze_dynamics_summary.json`
- `batch_gaze_dynamics_summary.csv`

注意：这一步不再生成 CPF-based fixation labels。CPF velocity / dispersion
保留为辅助 dynamics features；真正的 scene/object-level fixation 需要后续单独定义。

## Working Conventions

- Activate the dedicated `adt` environment before running scripts or notebooks.
- Keep exploration decisions in `docs/`, not only in chat or notebook outputs.
- Keep reusable code in `src/adt_sandbox`.
- Keep one-off exploration in `notebooks`.
- Keep downloaded ADT data on D drive, not in the WSL repo.
- Keep upstream tool source checkouts out of git; use `external/` only as an
  ignored local reference if needed.
