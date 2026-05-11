#!/usr/bin/env python
"""Generate a Scene-level head-gaze relationship report."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("/mnt/d/SparseGaze/ADT-Gaze"),
        help="Directory containing batch_scene_head_gaze_analysis_summary.csv.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "scene_head_gaze_relationship",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "docs" / "scene_head_gaze_relationship_report.md",
        help="Markdown report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.reports_dir / "batch_scene_head_gaze_analysis_summary.csv")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figures = generate_figures(rows, args.figure_dir)
    markdown = build_markdown(rows, args.output_md, figures)
    args.output_md.write_text(markdown, encoding="utf-8")

    print(f"figures_dir: {args.figure_dir}")
    for name, path in figures.items():
        print(f"{name}: {path}")
    print(f"report_md: {args.output_md}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generate_figures(rows: list[dict[str, str]], figure_dir: Path) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.boxplot(
        [
            numeric_column(rows, "corr_scene_velocity_vs_head_rotation_speed"),
            numeric_column(rows, "corr_cpf_velocity_vs_head_rotation_speed"),
            numeric_column(rows, "corr_scene_velocity_vs_head_translation_speed"),
            numeric_column(rows, "corr_cpf_velocity_vs_scene_velocity"),
        ],
        tick_labels=[
            "scene vel\nvs head rot",
            "CPF vel\nvs head rot",
            "scene vel\nvs head trans",
            "CPF vel\nvs scene vel",
        ],
        showmeans=True,
    )
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Scene/CPF gaze dynamics correlations")
    ax.grid(axis="y", alpha=0.3)
    path = figure_dir / "scene_cpf_correlation_distributions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["scene_cpf_correlation_distributions"] = path

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(2)
    width = 0.26
    scene_means = [
        distribution(rows, "mean_scene_velocity_fixation_deg_s")["mean"],
        distribution(rows, "mean_scene_velocity_transition_deg_s")["mean"],
    ]
    cpf_means = [
        distribution(rows, "mean_cpf_velocity_fixation_deg_s")["mean"],
        distribution(rows, "mean_cpf_velocity_transition_deg_s")["mean"],
    ]
    head_means = [
        distribution(rows, "mean_head_rotation_fixation_deg_s")["mean"],
        distribution(rows, "mean_head_rotation_transition_deg_s")["mean"],
    ]
    ax.bar(x - width, scene_means, width, label="Scene gaze vel")
    ax.bar(x, cpf_means, width, label="CPF gaze vel")
    ax.bar(x + width, head_means, width, label="Head rot speed")
    ax.set_xticks(x)
    ax.set_xticklabels(["fixation", "transition"])
    ax.set_ylabel("Mean speed [deg/s]")
    ax.set_title("Event-conditioned dynamics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = figure_dir / "event_conditioned_dynamics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["event_conditioned_dynamics"] = path

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    labels = ["low", "mid", "high"]
    scene = [
        distribution(rows, "low_head_mean_scene_velocity_deg_s")["mean"],
        distribution(rows, "mid_head_mean_scene_velocity_deg_s")["mean"],
        distribution(rows, "high_head_mean_scene_velocity_deg_s")["mean"],
    ]
    cpf = [
        distribution(rows, "low_head_mean_cpf_velocity_deg_s")["mean"],
        distribution(rows, "mid_head_mean_cpf_velocity_deg_s")["mean"],
        distribution(rows, "high_head_mean_cpf_velocity_deg_s")["mean"],
    ]
    x = np.arange(3)
    ax.plot(x, scene, marker="o", label="Scene gaze velocity")
    ax.plot(x, cpf, marker="o", label="CPF local gaze velocity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean gaze velocity [deg/s]")
    ax.set_xlabel("Within-sequence head rotation group")
    ax.set_title("Gaze dynamics by head rotation group")
    ax.legend()
    ax.grid(alpha=0.3)
    path = figure_dir / "gaze_velocity_by_head_rotation_group.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["gaze_velocity_by_head_rotation_group"] = path

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    fixation_fractions = [
        distribution(rows, "low_head_fixation_fraction")["mean"],
        distribution(rows, "mid_head_fixation_fraction")["mean"],
        distribution(rows, "high_head_fixation_fraction")["mean"],
    ]
    ax.bar(labels, fixation_fractions, color="#2ca25f")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean scene fixation fraction")
    ax.set_xlabel("Within-sequence head rotation group")
    ax.set_title("Scene fixation fraction by head rotation group")
    ax.grid(axis="y", alpha=0.3)
    path = figure_dir / "fixation_fraction_by_head_rotation_group.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["fixation_fraction_by_head_rotation_group"] = path

    return figure_paths


def build_markdown(
    rows: list[dict[str, str]],
    output_md: Path,
    figures: dict[str, Path],
) -> str:
    def rel(path: Path) -> str:
        return os.path.relpath(path.resolve(), start=output_md.parent.resolve())

    corr_scene_head = distribution(rows, "corr_scene_velocity_vs_head_rotation_speed")
    corr_cpf_head = distribution(rows, "corr_cpf_velocity_vs_head_rotation_speed")
    corr_scene_trans = distribution(rows, "corr_scene_velocity_vs_head_translation_speed")
    corr_cpf_scene = distribution(rows, "corr_cpf_velocity_vs_scene_velocity")
    fixation_fraction = distribution(rows, "scene_fixation_fraction")
    scene_fix = distribution(rows, "mean_scene_velocity_fixation_deg_s")
    scene_transition = distribution(rows, "mean_scene_velocity_transition_deg_s")
    cpf_fix = distribution(rows, "mean_cpf_velocity_fixation_deg_s")
    cpf_transition = distribution(rows, "mean_cpf_velocity_transition_deg_s")
    head_fix = distribution(rows, "mean_head_rotation_fixation_deg_s")
    head_transition = distribution(rows, "mean_head_rotation_transition_deg_s")
    low_fixation = distribution(rows, "low_head_fixation_fraction")
    mid_fixation = distribution(rows, "mid_head_fixation_fraction")
    high_fixation = distribution(rows, "high_head_fixation_fraction")
    low_scene = distribution(rows, "low_head_mean_scene_velocity_deg_s")
    mid_scene = distribution(rows, "mid_head_mean_scene_velocity_deg_s")
    high_scene = distribution(rows, "high_head_mean_scene_velocity_deg_s")
    low_cpf = distribution(rows, "low_head_mean_cpf_velocity_deg_s")
    mid_cpf = distribution(rows, "mid_head_mean_cpf_velocity_deg_s")
    high_cpf = distribution(rows, "high_head_mean_cpf_velocity_deg_s")

    lines: list[str] = []
    lines.append("# Scene-Level Head-Gaze Relationship Report")
    lines.append("")
    lines.append(
        f"Generated on {date.today().isoformat()} from "
        "`batch_scene_head_gaze_analysis_summary.csv`."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report analyzes final Scene/world gaze dynamics, not only CPF-local eye-in-head dynamics. "
        "It joins scene gaze velocity, CPF-local gaze velocity, head motion, and scene-direction event "
        "labels to test whether head motion changes world gaze, compensates world gaze, or both."
    )
    lines.append("")
    lines.append("## Research Questions")
    lines.append("")
    lines.append("1. Is Scene gaze velocity related to head rotation?")
    lines.append("2. Does this relation differ from CPF-local gaze velocity?")
    lines.append("3. During scene fixation versus transition, how do head and CPF dynamics differ?")
    lines.append("4. When head rotation is high, does Scene gaze remain stable or transition?")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Scene velocity is computed from `gaze_dir_scene_unit_xyz` and the scene event labels produced by "
        "`detect_scene_gaze_events.py`. CPF-local velocity is recomputed from `gaze_dir_cpf_unit_xyz`. "
        "Head rotation speed comes from the refactored head proxy layer. Correlations are computed per "
        "sequence and then summarized across 34 sequences."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Sequence count: `{len(rows)}`")
    lines.append(f"- Mean valid analysis ratio: `{fmt(distribution(rows, 'analysis_valid_ratio')['mean'])}`")
    lines.append(f"- Mean scene fixation frame fraction: `{fmt(fixation_fraction['mean'])}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### Table 1. Scene vs CPF Correlation Summary")
    lines.append("")
    lines.append(
        "Table 1 compares head-motion correlations in Scene and CPF spaces. This directly tests whether "
        "head rotation is more related to final world-gaze motion or to eye-in-head local motion."
    )
    lines.append("")
    lines.append("| Metric | Mean | Median | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, stats in [
        ("corr(scene gaze velocity, head rotation speed)", corr_scene_head),
        ("corr(CPF local velocity, head rotation speed)", corr_cpf_head),
        ("corr(scene gaze velocity, head translation speed)", corr_scene_trans),
        ("corr(CPF local velocity, scene gaze velocity)", corr_cpf_scene),
    ]:
        lines.append(
            f"| {label} | {fmt(stats['mean'])} | {fmt(stats['p50'])} | {fmt(stats['min'])} | {fmt(stats['max'])} |"
        )
    lines.append("")
    lines.append(
        "Scene gaze velocity has a slightly stronger relation to head rotation than CPF-local velocity in "
        "this batch. The very high CPF/Scene velocity correlation indicates that many rapid eye-in-head "
        "changes also appear as rapid world-gaze changes, so compensation is not the dominant aggregate "
        "pattern under the current scene-event settings."
    )
    lines.append("")
    lines.append(f"![Scene/CPF correlations]({rel(figures['scene_cpf_correlation_distributions'])})")
    lines.append("")
    lines.append("### Table 2. Event-Conditioned Dynamics")
    lines.append("")
    lines.append(
        "Table 2 compares scene fixation and transition frames. This is the first use of the scene-direction "
        "event labels in the head-gaze analysis."
    )
    lines.append("")
    lines.append("| Metric | Fixation Mean | Transition Mean |")
    lines.append("|---|---:|---:|")
    for label, first, second in [
        ("Scene gaze velocity (deg/s)", scene_fix, scene_transition),
        ("CPF local gaze velocity (deg/s)", cpf_fix, cpf_transition),
        ("Head rotation speed (deg/s)", head_fix, head_transition),
    ]:
        lines.append(f"| {label} | {fmt(first['mean'])} | {fmt(second['mean'])} |")
    lines.append("")
    lines.append(
        "Transition frames have much larger Scene and CPF gaze velocities than fixation frames. Head rotation "
        "also increases during transitions, but the gap is smaller than the gaze-velocity gap."
    )
    lines.append("")
    lines.append(f"![Event-conditioned dynamics]({rel(figures['event_conditioned_dynamics'])})")
    lines.append("")
    lines.append("### Table 3. Head-Rotation Group Summary")
    lines.append("")
    lines.append(
        "Table 3 groups frames by sequence-specific head rotation speed percentiles. It tests what happens "
        "to Scene and CPF gaze dynamics as head motion becomes relatively larger within each sequence."
    )
    lines.append("")
    lines.append("| Metric | Low | Mid | High |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Scene gaze velocity (deg/s) | {fmt(low_scene['mean'])} | {fmt(mid_scene['mean'])} | {fmt(high_scene['mean'])} |"
    )
    lines.append(
        f"| CPF local gaze velocity (deg/s) | {fmt(low_cpf['mean'])} | {fmt(mid_cpf['mean'])} | {fmt(high_cpf['mean'])} |"
    )
    lines.append(
        f"| Scene fixation fraction | {fmt(low_fixation['mean'])} | {fmt(mid_fixation['mean'])} | {fmt(high_fixation['mean'])} |"
    )
    lines.append("")
    lines.append(
        "As head rotation increases, both Scene and CPF gaze velocities increase and the fixation fraction "
        "drops. This supports a head-contributed transition pattern more strongly than a pure compensation "
        "pattern at the aggregate level."
    )
    lines.append("")
    lines.append(f"![Gaze velocity by head group]({rel(figures['gaze_velocity_by_head_rotation_group'])})")
    lines.append("")
    lines.append(
        f"![Fixation fraction by head group]({rel(figures['fixation_fraction_by_head_rotation_group'])})"
    )
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append(
        "1. In Scene space, head rotation is positively related to final world-gaze velocity and is still much "
        "more informative than translation."
    )
    lines.append(
        "2. Scene and CPF gaze velocities are highly correlated, so rapid local eye motion often coincides "
        "with rapid world-gaze motion in this dataset."
    )
    lines.append(
        "3. Scene fixation frames have low Scene velocity by definition, but they also have lower CPF gaze "
        "velocity and lower head rotation than transition frames."
    )
    lines.append(
        "4. High relative head-rotation frames are less likely to be scene fixations, which suggests that "
        "head rotation often participates in world-gaze transitions rather than being fully compensated."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Scene event labels are direction-level labels, not object-level fixation labels.")
    lines.append("- Head rotation groups are sequence-relative percentile groups, not absolute physical thresholds.")
    lines.append("- This is correlation and stratification analysis, not a prediction experiment.")
    lines.append("")
    return "\n".join(lines) + "\n"


def distribution(rows: list[dict[str, str]], column: str) -> dict[str, float]:
    return describe_values(numeric_column(rows, column))


def numeric_column(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "")
        if raw in ("", None):
            continue
        value = float(raw)
        if np.isfinite(value):
            values.append(value)
    return values


def describe_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
