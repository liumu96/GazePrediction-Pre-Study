#!/usr/bin/env python
"""Generate a markdown report for SparseGaze head-utility diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory containing batch_sparsegaze_head_utility_*.csv outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "sparsegaze_head_utility",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "docs" / "sparsegaze_head_utility_report.md",
        help="Markdown report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows = read_csv(
        args.reports_dir / "batch_sparsegaze_head_utility_aggregate.csv"
    )
    lead_lag_rows = read_csv(
        args.reports_dir / "batch_sparsegaze_head_utility_lead_lag_aggregate.csv"
    )
    if not summary_rows:
        raise ValueError("SparseGaze head-utility aggregate summary is empty")
    if not lead_lag_rows:
        raise ValueError("SparseGaze head-utility lead-lag aggregate is empty")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "residual_vs_interval": plot_residual_vs_interval(
            summary_rows,
            args.figures_dir / "residual_vs_anchor_interval.png",
        ),
        "scene_r2_vs_interval": plot_scene_r2_vs_interval(
            summary_rows,
            args.figures_dir / "scene_residual_r2_vs_anchor_interval.png",
        ),
        "event_conditioned_r2": plot_event_conditioned_r2(
            summary_rows,
            args.figures_dir / "event_conditioned_scene_r2.png",
        ),
        "lead_lag": plot_lead_lag(
            lead_lag_rows,
            args.figures_dir / "lead_lag_head_rotation.png",
        ),
    }
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        build_report(
            summary_rows=summary_rows,
            lead_lag_rows=lead_lag_rows,
            figure_paths=figure_paths,
            output_md=args.output_md,
            reports_dir=args.reports_dir,
        ),
        encoding="utf-8",
    )
    print(f"report: {args.output_md}")
    for figure_path in figure_paths.values():
        print(f"figure: {figure_path}")


def build_report(
    summary_rows: list[dict[str, Any]],
    lead_lag_rows: list[dict[str, Any]],
    figure_paths: dict[str, Path],
    output_md: Path,
    reports_dir: Path,
) -> str:
    sequence_count = max(
        int_float(row.get("sequence_count"), default=0) for row in summary_rows
    )
    n10_hold = find_summary_row(summary_rows, 10, "hold_last", "all")
    n10_linear = find_summary_row(summary_rows, 10, "linear_interp", "all")
    n30_hold = find_summary_row(summary_rows, 30, "hold_last", "all")
    fixation_n10 = find_summary_row(summary_rows, 10, "hold_last", "fixation")
    transition_n10 = find_summary_row(summary_rows, 10, "hold_last", "transition")
    lead_peak_scene = strongest_lead_lag(
        lead_lag_rows,
        target_name="scene_velocity_deg_s",
        head_feature_name="head_rotation_speed_deg_s",
    )
    lead_peak_cpf = strongest_lead_lag(
        lead_lag_rows,
        target_name="cpf_local_velocity_deg_s",
        head_feature_name="head_rotation_speed_deg_s",
    )

    lines: list[str] = []
    lines.append("# SparseGaze Head Utility Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report evaluates whether head motion contains useful information for "
        "recovering high-frequency gaze between sparse gaze anchors. It is not a "
        "full model evaluation; it is a diagnostic step before changing the SparseGaze model."
    )
    lines.append("")
    lines.append(f"- Input directory: `{reports_dir}`")
    lines.append(f"- Sequence count: {sequence_count}")
    lines.append(
        "- Required inputs: `*_gaze_samples.csv`, `*_head_samples.csv`, "
        "`*_scene_gaze_event_features.csv`, `*_scene_gaze_frame_labels.csv`"
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "For each sequence, gaze is sparsified by keeping one anchor every N frames "
        "(N = 2, 3, 5, 10, 15, 30). Non-anchor frames are reconstructed with two "
        "baselines: hold-last and linear interpolation. The residual is the angular "
        "distance between the baseline prediction and ground-truth gaze, computed in "
        "both CPF-local gaze direction and Scene gaze direction."
    )
    lines.append("")
    lines.append(
        "Head utility is measured in two ways. First, residual magnitude is correlated "
        "with current and cumulative head rotation. Second, a blocked cross-validated "
        "ridge diagnostic predicts residual magnitude from three feature sets: gap-only "
        "(position inside the anchor interval), current-head, and head-history. A useful "
        "head signal should improve R2 over gap-only, especially as the anchor interval grows."
    )
    lines.append("")
    lines.append(
        "Lead-lag curves compute corr(head motion at frame t, gaze dynamics at frame "
        "t+k). Positive lag means the current head signal is compared with future gaze "
        "dynamics; negative lag means gaze dynamics precedes head motion."
    )
    lines.append("")
    lines.append("## Coordinate-Frame Audit")
    lines.append("")
    lines.append(
        "The CPF-related analyses do not compare CPF gaze direction with CPF head "
        "forward direction. In CPF coordinates, head forward is constant by "
        "construction because CPF/head is the local device frame. Therefore CPF "
        "results should be read as `CPF-local gaze dynamics vs inter-frame head "
        "motion`, not as `CPF gaze vs CPF head direction`."
    )
    lines.append("")
    lines.append(
        "Head rotation features are inter-frame motion features: "
        "`head_rotation_speed_deg_s`, `head_rotation_angle_step_deg`, and relative "
        "rotation vectors derived from `R_{t-1}^{-1} R_t` in the previous head/CPF "
        "frame. These are dynamic even though the current-frame CPF forward axis is fixed."
    )
    lines.append("")
    lines.append(
        "`head_translation_speed_m_s` is a scalar origin-speed feature computed from "
        "Scene/world displacement magnitude. It is useful as motion intensity, but it "
        "is not a CPF translation direction. Directional local translation would require "
        "`translation_prev_head_dxyz_m`, which this SparseGaze utility diagnostic does "
        "not currently use."
    )
    lines.append("")
    lines.append("## Main Results")
    lines.append("")
    lines.append(
        "At a 10-frame anchor interval, hold-last Scene residual is "
        f"{fmt_metric(n10_hold, 'mean_scene_residual_deg_mean')} deg on average, "
        "while linear interpolation gives "
        f"{fmt_metric(n10_linear, 'mean_scene_residual_deg_mean')} deg. "
        "At a 30-frame interval, hold-last Scene residual rises to "
        f"{fmt_metric(n30_hold, 'mean_scene_residual_deg_mean')} deg. This tells us "
        "how quickly missing-gaze error grows as gaze becomes sparse."
    )
    lines.append("")
    lines.append(
        "For N=10 hold-last Scene residuals, gap-only R2 is "
        f"{fmt_metric(n10_hold, 'ridge_scene_r2_gap_only_mean')}, current-head R2 is "
        f"{fmt_metric(n10_hold, 'ridge_scene_r2_current_head_mean')}, and head-history R2 is "
        f"{fmt_metric(n10_hold, 'ridge_scene_r2_head_history_mean')}. The important "
        "quantity is not the absolute R2 alone, but whether current/head-history improves "
        "over gap-only. A small gain means head motion is only weakly explanatory for "
        "baseline residual magnitude under this diagnostic."
    )
    lines.append("")
    lines.append(
        "Event conditioning checks whether head is more useful outside stable viewing. "
        "For N=10 hold-last, fixation Scene residual is "
        f"{fmt_metric(fixation_n10, 'mean_scene_residual_deg_mean')} deg and transition "
        "Scene residual is "
        f"{fmt_metric(transition_n10, 'mean_scene_residual_deg_mean')} deg. Head-history "
        "R2 is "
        f"{fmt_metric(fixation_n10, 'ridge_scene_r2_head_history_mean')} for fixation "
        "and "
        f"{fmt_metric(transition_n10, 'ridge_scene_r2_head_history_mean')} for transition."
    )
    lines.append("")
    lines.append(
        "Lead-lag results show where head rotation aligns most strongly with gaze dynamics. "
        "For Scene velocity, the strongest mean correlation with head rotation appears at "
        f"lag {fmt_lag(lead_peak_scene)}. For CPF-local velocity, it appears at "
        f"lag {fmt_lag(lead_peak_cpf)}. If the peak is near zero, head and gaze dynamics "
        "are mostly synchronous at 30 Hz; if it is positive, current head motion has more "
        "predictive relation to future gaze dynamics; if negative, gaze tends to lead head."
    )
    lines.append("")
    lines.append("## Tables")
    lines.append("")
    lines.append("### Table 1. Sparse-anchor residuals")
    lines.append("")
    lines.append(
        "This table measures how much error remains when gaze is removed between anchors. "
        "Scene residual is the more task-facing quantity; CPF residual isolates eye-in-head "
        "local gaze change."
    )
    lines.append("")
    lines.extend(
        markdown_table(
            residual_table_rows(summary_rows),
            [
                "N",
                "baseline",
                "event",
                "CPF residual deg",
                "Scene residual deg",
                "samples",
            ],
        )
    )
    lines.append("")
    lines.append("### Table 2. Head feature utility for residual magnitude")
    lines.append("")
    lines.append(
        "This table compares blocked-CV R2 for predicting residual magnitude. "
        "`gap` uses only position inside the sparse interval; `current` adds current "
        "head motion; `history` adds recent and cumulative head motion. The meaningful "
        "signal is the R2 gain from `gap` to `current` or `history`."
    )
    lines.append("")
    lines.extend(
        markdown_table(
            r2_table_rows(summary_rows),
            [
                "N",
                "baseline",
                "event",
                "gap R2",
                "current-head R2",
                "head-history R2",
            ],
        )
    )
    lines.append("")
    lines.append("### Table 3. Strongest lead-lag correlations")
    lines.append("")
    lines.append(
        "This table asks whether head motion is synchronized with, leads, or lags gaze "
        "dynamics. Positive lag means head at frame t is compared with gaze at a future frame."
    )
    lines.append("")
    lines.extend(
        markdown_table(
            strongest_lead_lag_table_rows(lead_lag_rows),
            ["target", "head feature", "best lag", "mean corr", "median corr"],
        )
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for title, key in [
        ("Figure 1. Residual vs sparse anchor interval", "residual_vs_interval"),
        ("Figure 2. Scene residual R2 vs sparse anchor interval", "scene_r2_vs_interval"),
        ("Figure 3. Event-conditioned Scene residual R2", "event_conditioned_r2"),
        ("Figure 4. Lead-lag relation for head rotation", "lead_lag"),
    ]:
        rel_path = relative_markdown_path(figure_paths[key], output_md.parent)
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({rel_path})")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This analysis is useful if it produces one of three outcomes. First, if head-history "
        "R2 consistently improves over gap-only, SparseGaze should explicitly model recent "
        "head trajectory rather than only current head. Second, if gain appears mainly in "
        "transition frames or long anchor intervals, head features should be event/gap-aware "
        "instead of uniformly weighted. Third, if lead-lag peaks are not near zero, the model "
        "should align head history with gaze targets using the observed temporal offset."
    )
    lines.append("")
    lines.append(
        "If these gains are small, that is also actionable: head motion may still help as a "
        "regularizer or context signal, but a large architecture change based only on head "
        "features would be weakly supported. The next step would then be model-output "
        "diagnostics: compare where SparseGaze errors grow relative to anchor gap, Scene event "
        "label, and head-motion regime."
    )
    lines.append("")
    return "\n".join(lines)


def plot_residual_vs_interval(rows: list[dict[str, Any]], output_path: Path) -> Path:
    plt.figure(figsize=(7, 4))
    for baseline in ("hold_last", "linear_interp"):
        selected = [
            row
            for row in rows
            if row.get("baseline") == baseline and row.get("event_group") == "all"
        ]
        selected = sorted(selected, key=lambda row: int_float(row["anchor_interval_frames"]))
        x = [int_float(row["anchor_interval_frames"]) for row in selected]
        y = [float_or_nan(row.get("mean_scene_residual_deg_mean")) for row in selected]
        plt.plot(x, y, marker="o", label=f"{baseline} Scene")
    plt.xlabel("Anchor interval [frames]")
    plt.ylabel("Mean Scene residual [deg]")
    plt.title("Sparse-anchor residual grows with missing-gaze interval")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_scene_r2_vs_interval(rows: list[dict[str, Any]], output_path: Path) -> Path:
    selected = [
        row
        for row in rows
        if row.get("baseline") == "hold_last" and row.get("event_group") == "all"
    ]
    selected = sorted(selected, key=lambda row: int_float(row["anchor_interval_frames"]))
    x = [int_float(row["anchor_interval_frames"]) for row in selected]
    plt.figure(figsize=(7, 4))
    for metric, label in [
        ("ridge_scene_r2_gap_only_mean", "gap only"),
        ("ridge_scene_r2_current_head_mean", "current head"),
        ("ridge_scene_r2_head_history_mean", "head history"),
    ]:
        y = [float_or_nan(row.get(metric)) for row in selected]
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Anchor interval [frames]")
    plt.ylabel("Blocked-CV R2 for Scene residual")
    plt.title("Does head motion explain missing-gaze residuals?")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_event_conditioned_r2(rows: list[dict[str, Any]], output_path: Path) -> Path:
    event_groups = ["fixation", "transition"]
    metrics = [
        ("ridge_scene_r2_gap_only_mean", "gap"),
        ("ridge_scene_r2_current_head_mean", "current"),
        ("ridge_scene_r2_head_history_mean", "history"),
    ]
    x = range(len(event_groups))
    width = 0.24
    plt.figure(figsize=(6.5, 4))
    for metric_index, (metric, label) in enumerate(metrics):
        values = [
            float_or_nan(
                find_summary_row(rows, 10, "hold_last", event_group).get(metric)
            )
            for event_group in event_groups
        ]
        offset = (metric_index - 1) * width
        plt.bar([value + offset for value in x], values, width=width, label=label)
    plt.xticks(list(x), event_groups)
    plt.ylabel("Blocked-CV R2 for Scene residual")
    plt.title("Head utility by Scene event label (N=10, hold-last)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_lead_lag(rows: list[dict[str, Any]], output_path: Path) -> Path:
    plt.figure(figsize=(7, 4))
    for target_name, label in [
        ("scene_velocity_deg_s", "Scene velocity"),
        ("cpf_local_velocity_deg_s", "CPF local velocity"),
    ]:
        selected = [
            row
            for row in rows
            if row.get("target_name") == target_name
            and row.get("head_feature_name") == "head_rotation_speed_deg_s"
        ]
        selected = sorted(selected, key=lambda row: int_float(row["lag_frames"]))
        x = [int_float(row["lag_frames"]) for row in selected]
        y = [float_or_nan(row.get("pearson_corr_mean")) for row in selected]
        plt.plot(x, y, marker="o", markersize=3, label=label)
    plt.axvline(0, color="black", linewidth=1, alpha=0.5)
    plt.xlabel("Lag [frames], positive = head compared with future gaze")
    plt.ylabel("Mean Pearson correlation")
    plt.title("Lead-lag relation between head rotation and gaze dynamics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def residual_table_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    out: list[list[str]] = []
    for interval in [5, 10, 15, 30]:
        for baseline in ["hold_last", "linear_interp"]:
            row = find_summary_row(rows, interval, baseline, "all")
            out.append(
                [
                    str(interval),
                    baseline,
                    "all",
                    fmt(row.get("mean_cpf_residual_deg_mean")),
                    fmt(row.get("mean_scene_residual_deg_mean")),
                    fmt(row.get("sample_count_mean"), digits=0),
                ]
            )
    return out


def r2_table_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    out: list[list[str]] = []
    for interval in [5, 10, 15, 30]:
        for event_group in ["all", "fixation", "transition"]:
            row = find_summary_row(rows, interval, "hold_last", event_group)
            out.append(
                [
                    str(interval),
                    "hold_last",
                    event_group,
                    fmt(row.get("ridge_scene_r2_gap_only_mean")),
                    fmt(row.get("ridge_scene_r2_current_head_mean")),
                    fmt(row.get("ridge_scene_r2_head_history_mean")),
                ]
            )
    return out


def strongest_lead_lag_table_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    out: list[list[str]] = []
    for target_name in [
        "scene_velocity_deg_s",
        "cpf_local_velocity_deg_s",
        "scene_transition_indicator",
    ]:
        for head_feature_name in [
            "head_rotation_speed_deg_s",
            "head_translation_speed_m_s",
        ]:
            peak = strongest_lead_lag(rows, target_name, head_feature_name)
            out.append(
                [
                    target_name,
                    head_feature_name,
                    fmt_lag(peak),
                    fmt(peak.get("pearson_corr_mean")),
                    fmt(peak.get("pearson_corr_median")),
                ]
            )
    return out


def strongest_lead_lag(
    rows: list[dict[str, Any]],
    target_name: str,
    head_feature_name: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("target_name") == target_name
        and row.get("head_feature_name") == head_feature_name
        and is_finite(row.get("pearson_corr_mean"))
    ]
    if not selected:
        return {}
    return max(selected, key=lambda row: abs(float(row["pearson_corr_mean"])))


def find_summary_row(
    rows: list[dict[str, Any]],
    interval: int,
    baseline: str,
    event_group: str,
) -> dict[str, Any]:
    for row in rows:
        if (
            int_float(row.get("anchor_interval_frames")) == interval
            and row.get("baseline") == baseline
            and row.get("event_group") == event_group
        ):
            return row
    return {}


def markdown_table(rows: list[list[str]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def relative_markdown_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(Path("..") / path.resolve().relative_to(REPO_ROOT.resolve()))


def fmt_metric(row: dict[str, Any], key: str) -> str:
    return fmt(row.get(key))


def fmt_lag(row: dict[str, Any]) -> str:
    if not row:
        return "n/a"
    lag = fmt(row.get("lag_frames"), digits=0)
    corr = fmt(row.get("pearson_corr_mean"))
    return f"{lag} frames (mean r={corr})"


def fmt(value: Any, digits: int = 3) -> str:
    if not is_finite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def float_or_nan(value: Any) -> float:
    if not is_finite(value):
        return math.nan
    return float(value)


def int_float(value: Any, default: int = 0) -> int:
    if not is_finite(value):
        return default
    return int(float(value))


def is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
