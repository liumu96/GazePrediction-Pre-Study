#!/usr/bin/env python
"""Generate a report from head-gaze relationship analysis outputs.

Inputs:
- `batch_head_gaze_analysis_summary.csv`
- `batch_head_gaze_analysis_report.json`

Outputs:
- figures under `outputs/figures/head_gaze_relationship/`
- a markdown report, by default `docs/head_gaze_relationship_report.md`

This report intentionally does not use CPF-derived fixation labels. It describes
continuous head-gaze dynamics only, including motion magnitude and motion
direction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.results import batch_dir, find_sequence_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("/mnt/d/SparseGaze/ADT-Gaze-structured"),
        help="Organized reports root containing batch/head-gaze outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "head_gaze_relationship",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "docs" / "head_gaze_relationship_report.md",
        help="Markdown report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = existing_batch_file(args.reports_dir, "batch_head_gaze_analysis_summary.csv")
    report_json = existing_batch_file(args.reports_dir, "batch_head_gaze_analysis_report.json")
    rows = read_rows(summary_csv)
    report = json.loads(report_json.read_text(encoding="utf-8"))
    sequence_summaries = load_sequence_summaries(args.reports_dir, rows)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figures = generate_figures(rows, sequence_summaries, args.figure_dir)
    markdown = build_markdown_report(
        rows=rows,
        report=report,
        sequence_summaries=sequence_summaries,
        output_md=args.output_md,
        figures=figures,
    )
    args.output_md.write_text(markdown, encoding="utf-8")

    print(f"figures_dir: {args.figure_dir}")
    for name, path in figures.items():
        print(f"{name}: {path}")
    print(f"report_md: {args.output_md}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_sequence_summaries(
    reports_dir: Path,
    rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        sequence_name = row["sequence_name"]
        path = find_sequence_file(
            reports_dir,
            sequence_name,
            "analysis",
            "head_gaze_analysis_summary.json",
        )
        summaries[sequence_name] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def existing_batch_file(reports_dir: Path, filename: str) -> Path:
    path = batch_dir(reports_dir) / filename
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing batch file: {path}")


def generate_figures(
    rows: list[dict[str, str]],
    sequence_summaries: dict[str, dict[str, Any]],
    figure_dir: Path,
) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}

    current_rot = numeric_column(rows, "corr_current_local_velocity_vs_head_rotation_speed")
    current_trans = numeric_column(rows, "corr_current_local_velocity_vs_head_translation_speed")
    next_rot = numeric_column(rows, "corr_next_local_velocity_vs_current_head_rotation_speed")
    signed_yaw = numeric_column(rows, "corr_signed_delta_yaw_vs_head_rotvec_y")
    signed_pitch = numeric_column(rows, "corr_signed_delta_pitch_vs_head_rotvec_x")
    abs_yaw = numeric_column(rows, "corr_abs_delta_yaw_vs_abs_head_rotvec_y")
    abs_pitch = numeric_column(rows, "corr_abs_delta_pitch_vs_abs_head_rotvec_x")
    aligned_fraction = numeric_column(rows, "gaze_head_motion_aligned_fraction")
    opposed_fraction = numeric_column(rows, "gaze_head_motion_opposed_fraction")
    weak_fraction = numeric_column(rows, "gaze_head_motion_weak_or_orthogonal_fraction")

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.boxplot(
        [current_rot, current_trans],
        tick_labels=[
            "curr local gaze vel\nvs head rot",
            "curr local gaze vel\nvs head trans",
        ],
        showmeans=True,
    )
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Per-sequence current-step head-gaze correlations")
    ax.grid(axis="y", alpha=0.3)
    path = figure_dir / "correlation_distributions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["correlation_distributions"] = path

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.boxplot(
        [signed_yaw, signed_pitch, abs_yaw, abs_pitch],
        tick_labels=[
            "signed yaw\nvs head Y",
            "signed pitch\nvs head X",
            "|yaw|\nvs |head Y|",
            "|pitch|\nvs |head X|",
        ],
        showmeans=True,
    )
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Directional head-gaze component correlations")
    ax.grid(axis="y", alpha=0.3)
    path = figure_dir / "directional_component_correlations.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["directional_component_correlations"] = path

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    means = [
        float(np.mean(aligned_fraction)),
        float(np.mean(opposed_fraction)),
        float(np.mean(weak_fraction)),
    ]
    ax.bar(
        ["aligned", "opposed", "weak/orthogonal"],
        means,
        color=["#2ca25f", "#de2d26", "#9ecae1"],
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean fraction of valid frames")
    ax.set_title("2D gaze/head motion alignment categories")
    ax.grid(axis="y", alpha=0.3)
    path = figure_dir / "motion_alignment_fractions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["motion_alignment_fractions"] = path

    low_velocity = [
        stratum_metric_mean(summary, "low", "local_velocity_deg_s")
        for summary in sequence_summaries.values()
    ]
    mid_velocity = [
        stratum_metric_mean(summary, "mid", "local_velocity_deg_s")
        for summary in sequence_summaries.values()
    ]
    high_velocity = [
        stratum_metric_mean(summary, "high", "local_velocity_deg_s")
        for summary in sequence_summaries.values()
    ]
    mean_velocity = [
        float(summary["dynamics"]["local_velocity_deg_s"]["mean"])
        for summary in sequence_summaries.values()
    ]
    low_velocity_norm = safe_divide_list(low_velocity, mean_velocity)
    mid_velocity_norm = safe_divide_list(mid_velocity, mean_velocity)
    high_velocity_norm = safe_divide_list(high_velocity, mean_velocity)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.boxplot(
        [low_velocity, mid_velocity, high_velocity],
        tick_labels=["low", "mid", "high"],
        showmeans=True,
    )
    ax.set_ylabel("Mean local gaze velocity (deg/s)")
    ax.set_xlabel("Head rotation speed stratum")
    ax.set_title("Local gaze velocity by head-rotation stratum")
    ax.grid(axis="y", alpha=0.3)
    path = figure_dir / "local_gaze_velocity_by_head_rotation_stratum.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["local_gaze_velocity_by_head_rotation_stratum"] = path

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.boxplot(
        [low_velocity_norm, mid_velocity_norm, high_velocity_norm],
        tick_labels=["low", "mid", "high"],
        showmeans=True,
    )
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.0)
    ax.set_ylabel("Group mean / sequence mean")
    ax.set_xlabel("Within-sequence head rotation speed group")
    ax.set_title("Normalized local gaze velocity by head-rotation group")
    ax.grid(axis="y", alpha=0.3)
    path = figure_dir / "normalized_local_gaze_velocity_by_head_rotation_group.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["normalized_local_gaze_velocity_by_head_rotation_group"] = path

    x_values = current_rot
    y_values = next_rot
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    scatter = ax.scatter(
        x_values,
        y_values,
        c=numeric_column(rows, "median_gaze_head_angle_deg"),
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    min_axis = min(min(x_values), min(y_values))
    max_axis = max(max(x_values), max(y_values))
    ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlabel("corr(current local gaze velocity, current head rotation)")
    ax.set_ylabel("corr(next local gaze velocity, current head rotation)")
    ax.set_title("Current-vs-next head rotation relation")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Median gaze-head angle (deg)")
    ax.grid(alpha=0.3)
    path = figure_dir / "current_vs_next_rotation_correlation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["current_vs_next_rotation_correlation"] = path

    return figure_paths


def build_markdown_report(
    rows: list[dict[str, str]],
    report: dict[str, Any],
    sequence_summaries: dict[str, dict[str, Any]],
    output_md: Path,
    figures: dict[str, Path],
) -> str:
    def rel(path: Path) -> str:
        return relative_link(output_md.parent, path)

    valid_ratio_key = "dynamics_input_valid_ratio"
    if valid_ratio_key not in rows[0]:
        valid_ratio_key = "event_input_valid_ratio"

    angle_stats = distribution(rows, "median_gaze_head_angle_deg")
    curr_rot_stats = distribution(rows, "corr_current_local_velocity_vs_head_rotation_speed")
    curr_trans_stats = distribution(rows, "corr_current_local_velocity_vs_head_translation_speed")
    signed_yaw_stats = distribution(rows, "corr_signed_delta_yaw_vs_head_rotvec_y")
    signed_pitch_stats = distribution(rows, "corr_signed_delta_pitch_vs_head_rotvec_x")
    abs_yaw_stats = distribution(rows, "corr_abs_delta_yaw_vs_abs_head_rotvec_y")
    abs_pitch_stats = distribution(rows, "corr_abs_delta_pitch_vs_abs_head_rotvec_x")
    alignment_stats = distribution(rows, "mean_gaze_head_motion_alignment_2d")
    aligned_fraction_stats = distribution(rows, "gaze_head_motion_aligned_fraction")
    opposed_fraction_stats = distribution(rows, "gaze_head_motion_opposed_fraction")
    weak_fraction_stats = distribution(rows, "gaze_head_motion_weak_or_orthogonal_fraction")
    next_rot_stats = distribution(rows, "corr_next_local_velocity_vs_current_head_rotation_speed")
    next_trans_stats = distribution(rows, "corr_next_local_velocity_vs_current_head_translation_speed")
    curr_rot_values = numeric_column(rows, "corr_current_local_velocity_vs_head_rotation_speed")
    curr_trans_values = numeric_column(rows, "corr_current_local_velocity_vs_head_translation_speed")
    curr_rot_positive_count = sum(1 for value in curr_rot_values if value > 0)
    rotation_gt_translation_count = sum(
        1 for rotation, translation in zip(curr_rot_values, curr_trans_values, strict=True)
        if rotation > translation
    )

    low_stratum_velocity_values = [
        stratum_metric_mean(summary, "low", "local_velocity_deg_s")
        for summary in sequence_summaries.values()
    ]
    mid_stratum_velocity_values = [
        stratum_metric_mean(summary, "mid", "local_velocity_deg_s")
        for summary in sequence_summaries.values()
    ]
    high_stratum_velocity_values = [
        stratum_metric_mean(summary, "high", "local_velocity_deg_s")
        for summary in sequence_summaries.values()
    ]
    sequence_mean_velocity_values = [
        float(summary["dynamics"]["local_velocity_deg_s"]["mean"])
        for summary in sequence_summaries.values()
    ]
    low_stratum_velocity_norm_values = safe_divide_list(
        low_stratum_velocity_values,
        sequence_mean_velocity_values,
    )
    mid_stratum_velocity_norm_values = safe_divide_list(
        mid_stratum_velocity_values,
        sequence_mean_velocity_values,
    )
    high_stratum_velocity_norm_values = safe_divide_list(
        high_stratum_velocity_values,
        sequence_mean_velocity_values,
    )
    high_minus_low_velocity_values = [
        high - low
        for low, high in zip(
            low_stratum_velocity_values,
            high_stratum_velocity_values,
            strict=True,
        )
    ]
    high_minus_low_velocity_norm_values = safe_divide_list(
        high_minus_low_velocity_values,
        sequence_mean_velocity_values,
    )
    high_low_velocity_ratio_values = safe_divide_list(
        high_stratum_velocity_values,
        low_stratum_velocity_values,
    )
    low_stratum_velocity_stats = describe_values(low_stratum_velocity_values)
    mid_stratum_velocity_stats = describe_values(mid_stratum_velocity_values)
    high_stratum_velocity_stats = describe_values(high_stratum_velocity_values)
    low_stratum_velocity_norm_stats = describe_values(low_stratum_velocity_norm_values)
    mid_stratum_velocity_norm_stats = describe_values(mid_stratum_velocity_norm_values)
    high_stratum_velocity_norm_stats = describe_values(high_stratum_velocity_norm_values)
    high_minus_low_velocity_stats = describe_values(high_minus_low_velocity_values)
    high_minus_low_velocity_norm_stats = describe_values(
        high_minus_low_velocity_norm_values
    )
    high_low_velocity_ratio_stats = describe_values(high_low_velocity_ratio_values)
    high_stratum_velocity_gt_low_count = sum(
        1 for low_value, high_value in zip(low_stratum_velocity_values, high_stratum_velocity_values, strict=True)
        if high_value > low_value
    )
    low_stratum_angle_stats = describe_values(
        [stratum_metric_mean(summary, "low", "gaze_head_angle_deg") for summary in sequence_summaries.values()]
    )
    mid_stratum_angle_stats = describe_values(
        [stratum_metric_mean(summary, "mid", "gaze_head_angle_deg") for summary in sequence_summaries.values()]
    )
    high_stratum_angle_stats = describe_values(
        [stratum_metric_mean(summary, "high", "gaze_head_angle_deg") for summary in sequence_summaries.values()]
    )
    top_next = sorted(
        rows,
        key=lambda row: abs(float(row["corr_next_local_velocity_vs_current_head_rotation_speed"])),
        reverse=True,
    )[:5]

    lines: list[str] = []
    lines.append("# Head-Gaze Relationship Report")
    lines.append("")
    lines.append(f"Generated on {date.today().isoformat()} from `batch_head_gaze_analysis_summary.csv`.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report summarizes GT head-gaze relationship analysis over the extracted ADT sequences. It no "
        "longer uses CPF-derived fixation labels. The goal is to describe continuous relationships among "
        "scene gaze/head geometry, CPF-local gaze dynamics, and head motion before adding event-conditioned "
        "analysis."
    )
    lines.append("")
    lines.append("## Research Questions")
    lines.append("")
    lines.append("1. Does head rotation co-vary with CPF-local gaze dynamics within the same 30 Hz timestep?")
    lines.append("2. Is head rotation more informative than head translation for local gaze dynamics?")
    lines.append("3. Does head motion explain only gaze-change magnitude, or also gaze-change direction?")
    lines.append("4. Do larger head-rotation regimes contain larger local gaze changes?")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append("- `*_gaze_samples.csv`")
    lines.append("- `*_head_samples.csv`")
    lines.append("- `batch_head_gaze_analysis_summary.csv`")
    lines.append("- `batch_head_gaze_analysis_report.json`")
    lines.append("")
    lines.append("## Computation Method")
    lines.append("")
    lines.append(
        "CPF-local gaze dynamics are computed from `gaze_dir_cpf_unit_xyz`, while scene geometry uses "
        "`gaze_dir_scene_unit_xyz` and `head_forward_scene_unit`. The CPF dynamics are not converted into "
        "fixation labels in this report."
    )
    lines.append("")
    lines.append(
        "Coordinate-frame audit: this report does not compare CPF gaze direction with CPF head forward "
        "direction. CPF head forward would be constant by construction. CPF sections compare local "
        "eye-in-head gaze dynamics with inter-frame head motion (`R_{t-1}^{-1} R_t`, rotation speed, "
        "and related signed rotation-vector components). Scene geometry is the only place where "
        "`head_forward_scene_unit` is used as a direction."
    )
    lines.append("")
    lines.append(
        "`head_translation_speed_m_s` is a scalar Scene-origin displacement speed. It is not a CPF "
        "translation direction; the local directional translation feature is `translation_prev_head_*`."
    )
    lines.append("")
    lines.append("Core quantities:")
    lines.append("")
    lines.append("- `local_angle_step_deg = angle(g_t-1^cpf, g_t^cpf)`")
    lines.append("- `local_velocity_deg_s = local_angle_step_deg / dt`")
    lines.append("- `window_dispersion_deg = max pairwise CPF angular separation in a centered window`")
    lines.append("- `head_rotation_speed_deg_s = angle(R_{t-1}^{-1} R_t) / dt`")
    lines.append(
        "- `head_rotvec_prev_head_*`: signed axis-angle vector of "
        "`R_{t-1}^{-1} R_t`, expressed in the previous head/CPF frame"
    )
    lines.append(
        "- `gaze_head_motion_alignment_2d = cosine([delta_yaw, delta_pitch], "
        "[head_rotvec_y, head_rotvec_x])`"
    )
    lines.append("- `head_translation_speed_m_s = ||p_t - p_{t-1}|| / dt`")
    lines.append("- `gaze_head_angle_deg = angle(gaze_dir_scene_unit, head_forward_scene_unit)`")
    lines.append("")
    lines.append(
        "All correlations are Pearson correlations computed per sequence and then summarized across "
        "sequences. They measure linear association only, not causality or model performance."
    )
    lines.append(
        "The directional metrics use signed CPF angular deltas and signed relative head rotation vector "
        "components. Their signs are coordinate-frame conventions; the robust question is whether direction "
        "components carry additional structure beyond speed magnitude."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    sequence_count = report.get("sequence_count", len(rows))
    valid_stats = distribution(rows, valid_ratio_key)
    lines.append(f"- Sequence count: `{sequence_count}`")
    lines.append(f"- Mean dynamics-input valid ratio: `{fmt(valid_stats['mean'])}`")
    lines.append(f"- Positive current gaze/head-rotation correlations: `{curr_rot_positive_count}/{len(rows)}`")
    lines.append(
        f"- Current gaze/head-rotation correlation larger than translation correlation: "
        f"`{rotation_gt_translation_count}/{len(rows)}`"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### Table 1. Batch-Level Summary")
    lines.append("")
    lines.append(
        "Table 1 tests whether head rotation or head translation has the stronger relationship with CPF-local "
        "gaze velocity. This is a dynamics diagnostic, not a fixation analysis."
    )
    lines.append("")
    lines.append("| Metric | Mean | Median | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, stats in [
        ("Median gaze-head angle (deg)", angle_stats),
        ("corr(current local gaze vel, current head rot)", curr_rot_stats),
        ("corr(current local gaze vel, current head trans)", curr_trans_stats),
    ]:
        lines.append(
            f"| {label} | {fmt(stats['mean'])} | {fmt(stats['p50'])} | {fmt(stats['min'])} | {fmt(stats['max'])} |"
        )
    lines.append("")
    lines.append(
        f"Head rotation is consistently more related to local gaze velocity than translation: mean correlation "
        f"`{fmt(curr_rot_stats['mean'])}` vs `{fmt(curr_trans_stats['mean'])}`. The magnitude is still "
        "weak-to-moderate, so this should be read as evidence of useful dynamics context, not as evidence "
        "that head motion alone explains gaze."
    )
    lines.append("")
    lines.append("### Figure 1. Current-Step Correlation Distributions")
    lines.append("")
    lines.append(f"![Correlation distributions]({rel(figures['correlation_distributions'])})")
    lines.append("")
    lines.append(
        "The per-sequence distribution confirms that the rotation result is stable rather than driven by a "
        "small number of sequences. Translation remains close to zero on average."
    )
    lines.append("")
    lines.append("### Table 2. Directional Component Summary")
    lines.append("")
    lines.append(
        "Table 2 addresses the missing direction question. It compares signed gaze angular deltas with "
        "signed head rotation-vector components in the previous head frame, and also compares absolute "
        "component magnitudes. Signed correlations test whether motion directions co-vary. Absolute "
        "component correlations test whether larger component-wise head turns come with larger component-wise "
        "eye-in-head changes."
    )
    lines.append("")
    lines.append("| Metric | Mean | Median | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, stats in [
        ("corr(signed delta yaw, head rotvec Y)", signed_yaw_stats),
        ("corr(signed delta pitch, head rotvec X)", signed_pitch_stats),
        ("corr(|delta yaw|, |head rotvec Y|)", abs_yaw_stats),
        ("corr(|delta pitch|, |head rotvec X|)", abs_pitch_stats),
        ("mean 2D gaze/head motion alignment", alignment_stats),
        ("aligned frame fraction", aligned_fraction_stats),
        ("opposed frame fraction", opposed_fraction_stats),
        ("weak/orthogonal frame fraction", weak_fraction_stats),
    ]:
        lines.append(
            f"| {label} | {fmt(stats['mean'])} | {fmt(stats['p50'])} | {fmt(stats['min'])} | {fmt(stats['max'])} |"
        )
    lines.append("")
    lines.append(
        "The direction-level results should be read separately from the speed results. A strong speed "
        "correlation with weak signed component correlation means head motion is informative about how much "
        "local gaze changes, but less deterministic about which way the eyes move. Opposed alignment can "
        "indicate compensatory eye-in-head motion during head turns, while aligned motion can indicate "
        "co-moving eye/head behavior under the chosen CPF sign convention."
    )
    lines.append("")
    lines.append("### Figure 2. Directional Component Correlations")
    lines.append("")
    lines.append(
        f"![Directional component correlations]({rel(figures['directional_component_correlations'])})"
    )
    lines.append("")
    lines.append("### Figure 3. 2D Motion Alignment Fractions")
    lines.append("")
    lines.append(
        f"![Motion alignment fractions]({rel(figures['motion_alignment_fractions'])})"
    )
    lines.append("")
    lines.append("### Table 3. Head-Rotation Speed Group Summary")
    lines.append("")
    lines.append(
        "Table 3 groups frames by head rotation speed within each sequence using the 33.3% and 66.7% "
        "sequence-specific percentiles. The table then summarizes the 34 per-sequence group means, so the "
        "statistical unit is a sequence-level group mean rather than a pooled frame."
    )
    lines.append("")
    lines.append("| Metric | Mean | Median | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, stats in [
        ("Low-rotation-group mean local gaze velocity (deg/s)", low_stratum_velocity_stats),
        ("Mid-rotation-group mean local gaze velocity (deg/s)", mid_stratum_velocity_stats),
        ("High-rotation-group mean local gaze velocity (deg/s)", high_stratum_velocity_stats),
        ("Low-rotation-group mean gaze-head angle (deg)", low_stratum_angle_stats),
        ("Mid-rotation-group mean gaze-head angle (deg)", mid_stratum_angle_stats),
        ("High-rotation-group mean gaze-head angle (deg)", high_stratum_angle_stats),
    ]:
        lines.append(
            f"| {label} | {fmt(stats['mean'])} | {fmt(stats['p50'])} | {fmt(stats['min'])} | {fmt(stats['max'])} |"
        )
    lines.append("")
    lines.append(
        f"Mean local gaze velocity rises from `{fmt(low_stratum_velocity_stats['mean'])}` to "
        f"`{fmt(mid_stratum_velocity_stats['mean'])}` and then to "
        f"`{fmt(high_stratum_velocity_stats['mean'])}` deg/s across low/mid/high head-rotation groups. "
        "Because the groups are relative within each sequence, this result should be read as a within-sequence "
        "trend rather than an absolute physical threshold."
    )
    lines.append("")
    lines.append("### Figure 4. Local Gaze Velocity by Head-Rotation Speed Group")
    lines.append("")
    lines.append(
        f"![Local gaze velocity by head-rotation stratum]({rel(figures['local_gaze_velocity_by_head_rotation_stratum'])})"
    )
    lines.append("")
    lines.append(
        f"High-head-rotation frames have larger local gaze velocity than low-head-rotation frames in "
        f"`{high_stratum_velocity_gt_low_count}/{len(rows)}` sequences. This is the strongest result in the "
        "report, but it remains a statistical relationship rather than a deterministic predictor."
    )
    lines.append("")
    lines.append("### Table 4. Within-Sequence Effect Size Summary")
    lines.append("")
    lines.append(
        "Table 4 normalizes each sequence by its own mean local gaze velocity. This reduces the influence of "
        "sequences that are globally faster or slower and directly tests whether relatively higher head "
        "rotation corresponds to a relative increase in local gaze velocity."
    )
    lines.append("")
    lines.append("| Metric | Mean | Median | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, stats in [
        ("Low group / sequence mean", low_stratum_velocity_norm_stats),
        ("Mid group / sequence mean", mid_stratum_velocity_norm_stats),
        ("High group / sequence mean", high_stratum_velocity_norm_stats),
        ("High-low delta (deg/s)", high_minus_low_velocity_stats),
        ("High-low delta / sequence mean", high_minus_low_velocity_norm_stats),
        ("High / low ratio", high_low_velocity_ratio_stats),
    ]:
        lines.append(
            f"| {label} | {fmt(stats['mean'])} | {fmt(stats['p50'])} | {fmt(stats['min'])} | {fmt(stats['max'])} |"
        )
    lines.append("")
    lines.append(
        "The normalized view supports the same conclusion without relying on absolute velocity scale: within "
        "a typical sequence, high-rotation frames have substantially larger local gaze velocity than "
        "low-rotation frames."
    )
    lines.append("")
    lines.append("### Figure 5. Normalized Local Gaze Velocity by Head-Rotation Group")
    lines.append("")
    lines.append(
        f"![Normalized local gaze velocity by head-rotation group]({rel(figures['normalized_local_gaze_velocity_by_head_rotation_group'])})"
    )
    lines.append("")
    lines.append("### Figure 6. Current-vs-Next Rotation Correlation")
    lines.append("")
    lines.append(
        "The one-step lagged diagnostic asks whether current head rotation remains associated with next-frame "
        "local gaze velocity. It is included only as a temporal diagnostic, not as a SparseGaze model result."
    )
    lines.append("")
    lines.append(
        f"![Current vs next rotation correlation]({rel(figures['current_vs_next_rotation_correlation'])})"
    )
    lines.append("")
    lines.append("### Table 5. Auxiliary Next-Step Summary")
    lines.append("")
    lines.append("| Metric | Mean | Median | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, stats in [
        ("corr(next local gaze vel, current head rot)", next_rot_stats),
        ("corr(next local gaze vel, current head trans)", next_trans_stats),
    ]:
        lines.append(
            f"| {label} | {fmt(stats['mean'])} | {fmt(stats['p50'])} | {fmt(stats['min'])} | {fmt(stats['max'])} |"
        )
    lines.append("")
    lines.append(
        "The lagged rotation relation remains positive but is weaker than the same-step relation. This can "
        "motivate later temporal modeling, but it is not itself a prediction experiment."
    )
    lines.append("")
    lines.append("### Table 6. Sequences With Strongest Next-Step Head-Rotation Signal")
    lines.append("")
    lines.append("| Sequence | corr(next local gaze vel, current head rot) |")
    lines.append("|---|---:|")
    for row in top_next:
        lines.append(
            f"| {row['sequence_name']} | {fmt(float(row['corr_next_local_velocity_vs_current_head_rotation_speed']))} |"
        )
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append(
        "1. Head rotation is the useful head-motion family in this analysis; translation is weak."
    )
    lines.append(
        "2. The strongest relationship is in motion magnitude: larger head-rotation regimes contain larger "
        "CPF-local gaze velocities."
    )
    lines.append(
        "3. Directional metrics are necessary because speed-only analysis hides whether gaze moves with, "
        "against, or orthogonal to head rotation."
    )
    lines.append(
        "4. CPF-local velocity and dispersion are useful continuous diagnostics, but CPF-thresholded fixation "
        "labels are intentionally excluded because they do not define scene/object fixation."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- The report is based on correlation and stratification, not causal inference.")
    lines.append(
        "- Scene-direction event detection is implemented separately; object-level fixation "
        "detection is not implemented here."
    )
    lines.append("- CPF dynamics should be used as auxiliary features, not as final event labels.")
    lines.append("")
    return "\n".join(lines) + "\n"


def stratum_metric_mean(summary: dict[str, Any], label: str, metric: str) -> float:
    return float(summary["head_rotation_speed_strata"]["groups"][label][metric]["mean"])


def distribution(rows: list[dict[str, str]], column: str) -> dict[str, float]:
    return describe_values(numeric_column(rows, column))


def numeric_column(rows: list[dict[str, str]], column: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(column, "")
        if raw in ("", None):
            continue
        values.append(float(raw))
    return values


def describe_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def safe_divide_list(numerators: list[float], denominators: list[float]) -> list[float]:
    ratios: list[float] = []
    for numerator, denominator in zip(numerators, denominators, strict=True):
        if denominator == 0 or not np.isfinite(denominator):
            continue
        ratio = numerator / denominator
        if np.isfinite(ratio):
            ratios.append(float(ratio))
    return ratios


def relative_link(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=from_dir.resolve())


def fmt(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
