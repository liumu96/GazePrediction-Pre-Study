"""Figure and report generation for prediction-result analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COLOR = "#4C78A8"
SECONDARY_COLOR = "#F58518"
ACCENT_COLOR = "#54A24B"
ERROR_COLOR = "#C44E52"
GRID_ALPHA = 0.22


def write_analysis_figures(
    *,
    sequence_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    frame_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write high-level figures for model-vs-GT analysis."""

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_missing_error": figure_dir / "model_missing_error.png",
        "frequency_curve": figure_dir / "frequency_curve.png",
        "sequence_missing_error": figure_dir / "sequence_missing_error.png",
        "event_error": figure_dir / "event_error.png",
        "error_distribution": figure_dir / "error_distribution.png",
        "yaw_pitch_residual": figure_dir / "yaw_pitch_residual.png",
    }
    plot_model_missing_error(model_summary, paths["model_missing_error"])
    plot_frequency_curve(model_summary, paths["frequency_curve"])
    plot_sequence_missing_error(sequence_summary, paths["sequence_missing_error"])
    plot_event_error(event_summary, paths["event_error"])
    plot_error_distribution(frame_summary, paths["error_distribution"])
    plot_yaw_pitch_residual(frame_summary, paths["yaw_pitch_residual"])
    return paths


def write_anchor_gap_figures(
    *,
    gap_summary: pd.DataFrame,
    gap_event_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write anchor-gap position figures."""

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "anchor_gap_position": figure_dir / "anchor_gap_position.png",
        "anchor_gap_event_position": figure_dir / "anchor_gap_event_position.png",
    }
    plot_anchor_gap_position(gap_summary, paths["anchor_gap_position"])
    plot_anchor_gap_event_position(gap_event_summary, paths["anchor_gap_event_position"])
    return paths


def write_anchor_gap_report(
    *,
    gap_summary: pd.DataFrame,
    gap_event_summary: pd.DataFrame,
    figure_paths: dict[str, Path],
    output_path: Path,
) -> None:
    """Write a compact Markdown report for anchor-gap position analysis."""

    lines = [
        "# Anchor-Gap Position Analysis",
        "",
        "This report evaluates missing-frame angular error as a function of position between sparse gaze anchors.",
        "Anchor frames are excluded; only evaluated missing frames bracketed by a previous and next anchor are used.",
        "",
        "## Figures",
        "",
    ]
    for key, path in figure_paths.items():
        if path.exists():
            lines.extend([f"### {key}", "", f"![{key}]({path.relative_to(output_path.parent).as_posix()})", ""])

    lines.extend(["## Gap Position Summary", ""])
    if gap_summary.empty:
        lines.append("No anchor-gap summary available.")
    else:
        cols = [
            "model",
            "eval_kind",
            "target_hz",
            "phase",
            "mean_normalized_gap_position",
            "sequence_n",
            "frame_n",
            "sequence_macro_mae_deg",
            "frame_weighted_mae_deg",
            "p90_deg",
        ]
        lines.extend(dataframe_to_markdown(gap_summary[existing(cols, gap_summary)].head(80)))

    lines.extend(["", "## Event-Conditioned Summary", ""])
    if gap_event_summary.empty:
        lines.append("No event-conditioned anchor-gap summary available.")
    else:
        cols = [
            "model",
            "eval_kind",
            "target_hz",
            "phase",
            "scene_event_label",
            "mean_normalized_gap_position",
            "sequence_n",
            "frame_n",
            "sequence_macro_mae_deg",
        ]
        focus = gap_event_summary[
            gap_event_summary["scene_event_label"].isin(["fixation", "transition"])
        ]
        lines.extend(dataframe_to_markdown(focus[existing(cols, focus)].head(120)))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_report(
    *,
    sequence_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    figure_paths: dict[str, Path],
    output_path: Path,
) -> None:
    """Write a compact Markdown report next to CSV and figure outputs."""

    lines = [
        "# Prediction Analysis Report",
        "",
        "This report compares SparseGaze-style predicted gaze directions against GT gaze directions.",
        "",
        "Primary metric: 3D angular error in degrees. Missing-frame metrics are the main SparseGaze metrics.",
        "",
        "## Model Summary",
        "",
    ]
    if model_summary.empty:
        lines.append("No model summary available.")
    else:
        cols = [
            "model",
            "eval_kind",
            "target_hz",
            "phase",
            "missing_mean_angular_error_deg",
            "missing_median_angular_error_deg",
            "missing_mean_abs_yaw_error_deg",
            "missing_mean_abs_pitch_error_deg",
        ]
        lines.extend(dataframe_to_markdown(model_summary[existing(cols, model_summary)]))

    lines.extend(["", "## Figures", ""])
    figure_descriptions = {
        "model_missing_error": "Compares missing-frame mean/median angular error across models and eval modes.",
        "frequency_curve": "Shows missing-frame error as gaze anchor frequency changes.",
        "sequence_missing_error": "Ranks sequences by missing-frame mean angular error, useful for finding hard sequences.",
        "event_error": "Compares error under scene-direction fixation/transition labels.",
        "error_distribution": "Shows the full evaluated-frame angular error distribution.",
        "yaw_pitch_residual": "Shows yaw/pitch residual direction and bias, not just error magnitude.",
    }
    for key, path in figure_paths.items():
        if path.exists():
            lines.extend(
                [
                    f"### {key}",
                    "",
                    figure_descriptions.get(key, ""),
                    "",
                    f"![{key}]({path.relative_to(output_path.parent).as_posix()})",
                    "",
                ]
            )

    lines.extend(["## Event Summary", ""])
    if event_summary.empty:
        lines.append("No scene-event labels were attached.")
    else:
        cols = [
            "model",
            "eval_kind",
            "target_hz",
            "phase",
            "scene_event_label",
            "n_event",
            "mean_angular_error_deg",
            "median_angular_error_deg",
        ]
        grouped = (
            event_summary.groupby(["model", "eval_kind", "target_hz", "phase", "scene_event_label"], as_index=False)
            [["n_event", "mean_angular_error_deg", "median_angular_error_deg"]]
            .mean(numeric_only=True)
        )
        lines.extend(dataframe_to_markdown(grouped[existing(cols, grouped)]))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_model_missing_error(model_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot missing-frame mean/median angular error by model."""

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(model_summary)), 4.5))
    if model_summary.empty:
        draw_empty(ax, "No model summary")
    else:
        data = model_summary.copy()
        data["label"] = data.apply(format_model_label, axis=1)
        x = np.arange(len(data))
        ax.bar(x - 0.18, data["missing_mean_angular_error_deg"], width=0.36, color=DEFAULT_COLOR, label="mean")
        ax.bar(x + 0.18, data["missing_median_angular_error_deg"], width=0.36, color=SECONDARY_COLOR, label="median")
        ax.set_xticks(x)
        ax.set_xticklabels(data["label"], rotation=35, ha="right")
        ax.set_ylabel("Angular error [deg]")
        ax.set_title("Missing-frame prediction error")
        ax.grid(axis="y", alpha=GRID_ALPHA)
        ax.legend()
    save_figure(fig, output_path)


def plot_frequency_curve(model_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot missing-frame angular error across target frequencies."""

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if model_summary.empty or model_summary["target_hz"].nunique() <= 1:
        draw_empty(ax, "Need multiple target_hz values for frequency curve")
    else:
        for (model, eval_kind), group in model_summary.groupby(["model", "eval_kind"]):
            group = group.sort_values("target_hz")
            ax.plot(
                group["target_hz"],
                group["missing_mean_angular_error_deg"],
                marker="o",
                linewidth=1.8,
                label=f"{model}:{eval_kind}",
            )
        ax.set_xlabel("Target gaze rate [Hz]")
        ax.set_ylabel("Missing-frame mean angular error [deg]")
        ax.set_title("Frequency sensitivity")
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(fontsize=8)
    save_figure(fig, output_path)


def plot_sequence_missing_error(sequence_summary: pd.DataFrame, output_path: Path, max_sequences: int = 30) -> None:
    """Rank sequences by missing-frame mean angular error."""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if sequence_summary.empty:
        draw_empty(ax, "No sequence summary")
    else:
        data = sequence_summary.sort_values("missing_mean_angular_error_deg", ascending=False).head(max_sequences)
        labels = data["sequence"].str.replace("_skeleton_", "_", regex=False)
        ax.barh(labels, data["missing_mean_angular_error_deg"], color=DEFAULT_COLOR)
        ax.invert_yaxis()
        ax.set_xlabel("Missing-frame mean angular error [deg]")
        ax.set_title("Hardest sequences")
        ax.grid(axis="x", alpha=GRID_ALPHA)
    save_figure(fig, output_path)


def plot_event_error(event_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot fixation/transition missing-frame error when event labels exist."""

    fig, ax = plt.subplots(figsize=(9, 4.8))
    if event_summary.empty:
        draw_empty(ax, "No scene-event labels")
    else:
        data = (
            event_summary.groupby(["model", "eval_kind", "target_hz", "scene_event_label"], as_index=False)
            ["mean_angular_error_deg"]
            .mean()
        )
        labels = [format_model_label(row) for _, row in data.drop_duplicates(["model", "eval_kind", "target_hz"]).iterrows()]
        events = [event for event in ["fixation", "transition", "invalid", "unknown"] if event in set(data["scene_event_label"])]
        x = np.arange(len(labels))
        width = 0.78 / max(len(events), 1)
        colors = {
            "fixation": ACCENT_COLOR,
            "transition": SECONDARY_COLOR,
            "invalid": ERROR_COLOR,
            "unknown": "#999999",
        }
        for idx, event in enumerate(events):
            values = []
            for _, row in data.drop_duplicates(["model", "eval_kind", "target_hz"]).iterrows():
                match = data[
                    (data["model"] == row["model"])
                    & (data["eval_kind"] == row["eval_kind"])
                    & (data["target_hz"] == row["target_hz"])
                    & (data["scene_event_label"] == event)
                ]
                values.append(np.nan if match.empty else float(match["mean_angular_error_deg"].iloc[0]))
            ax.bar(x + (idx - (len(events) - 1) / 2) * width, values, width=width, color=colors.get(event), label=event)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Mean angular error [deg]")
        ax.set_title("Error by scene event")
        ax.grid(axis="y", alpha=GRID_ALPHA)
        ax.legend()
    save_figure(fig, output_path)


def plot_error_distribution(frame_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot evaluated-frame angular error distribution."""

    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = evaluated_frames(frame_summary)
    if data.empty:
        draw_empty(ax, "No evaluated frames")
    else:
        values = data["angular_error_deg"].dropna()
        ax.hist(values, bins=70, color=DEFAULT_COLOR, alpha=0.82)
        ax.axvline(values.mean(), color=ERROR_COLOR, linewidth=1.7, label=f"mean {values.mean():.2f}")
        ax.axvline(values.median(), color="#222222", linestyle="--", linewidth=1.5, label=f"median {values.median():.2f}")
        ax.set_xlabel("Angular error [deg]")
        ax.set_ylabel("Frame count")
        ax.set_title("Evaluated-frame error distribution")
        ax.grid(alpha=GRID_ALPHA)
        ax.legend()
    save_figure(fig, output_path)


def plot_yaw_pitch_residual(frame_summary: pd.DataFrame, output_path: Path, max_points: int = 50000) -> None:
    """Plot yaw/pitch residuals to reveal error direction and bias."""

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    data = evaluated_frames(frame_summary)
    if data.empty:
        draw_empty(ax, "No evaluated frames")
    else:
        if len(data) > max_points:
            data = data.sample(max_points, random_state=7)
        colors = data["scene_event_label"].map(
            {"fixation": ACCENT_COLOR, "transition": SECONDARY_COLOR, "invalid": ERROR_COLOR}
        ).fillna(DEFAULT_COLOR)
        ax.scatter(data["yaw_error_deg"], data["pitch_error_deg"], s=5, c=colors, alpha=0.24, linewidths=0)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_xlabel("Yaw residual [deg]")
        ax.set_ylabel("Pitch residual [deg]")
        ax.set_title("Yaw/pitch residual cloud")
        ax.grid(alpha=GRID_ALPHA)
        ax.set_aspect("equal", adjustable="box")
    save_figure(fig, output_path)


def plot_anchor_gap_position(gap_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot sequence-macro error as a function of normalized anchor-gap position."""

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if gap_summary.empty:
        draw_empty(ax, "No anchor-gap summary")
    else:
        for label, group in gap_summary.groupby(gap_method_label_columns(gap_summary), sort=False):
            x_col = gap_x_column(group)
            group = group.sort_values(x_col)
            ax.plot(
                group[x_col],
                group["sequence_macro_mae_deg"],
                marker="o",
                linewidth=1.8,
                label=format_gap_label(label),
            )
        ax.set_xlabel("Normalized position between gaze anchors")
        ax.set_ylabel("Sequence-macro MAE [deg]")
        ax.set_title("Missing-frame error across anchor gap")
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(fontsize=8)
    save_figure(fig, output_path)


def plot_anchor_gap_event_position(gap_event_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot anchor-gap curves split by fixation and transition labels."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    focus_events = ["fixation", "transition"]
    if gap_event_summary.empty:
        for ax in axes:
            draw_empty(ax, "No event-conditioned gap summary")
    else:
        for ax, event in zip(axes, focus_events):
            data = gap_event_summary[gap_event_summary["scene_event_label"] == event]
            if data.empty:
                draw_empty(ax, f"No {event} rows")
                continue
            for label, group in data.groupby(gap_method_label_columns(data), sort=False):
                x_col = gap_x_column(group)
                group = group.sort_values(x_col)
                ax.plot(
                    group[x_col],
                    group["sequence_macro_mae_deg"],
                    marker="o",
                    linewidth=1.8,
                    label=format_gap_label(label),
                )
            ax.set_title(event)
            ax.set_xlabel("Normalized position between gaze anchors")
            ax.set_xlim(0.0, 1.0)
            ax.grid(alpha=GRID_ALPHA)
        axes[0].set_ylabel("Sequence-macro MAE [deg]")
        axes[-1].legend(fontsize=8)
    save_figure(fig, output_path)


def gap_method_label_columns(frame: pd.DataFrame) -> list[str]:
    """Return grouping columns that identify one plotted method curve."""

    cols = ["model", "eval_kind", "target_hz", "phase"]
    return [col for col in cols if col in frame.columns]


def gap_x_column(frame: pd.DataFrame) -> str:
    """Prefer the observed mean position over the coarse bin center."""

    if "mean_normalized_gap_position" in frame.columns:
        return "mean_normalized_gap_position"
    return "gap_bin_center"


def format_gap_label(label: Any) -> str:
    """Format a groupby label tuple for gap-position plots."""

    if not isinstance(label, tuple):
        label = (label,)
    if len(label) == 4:
        model, eval_kind, target_hz, phase = label
        if str(model) == "HAGI++":
            return f"HAGI++ {int(target_hz)}Hz"
        names = {
            "rollout": "SparseGaze rollout",
            "rollout_linear": "Linear",
            "rollout_pchip": "PCHIP",
            "rollout_gt": "GT-repair",
            "oracle": "Oracle",
        }
        method = names.get(str(eval_kind), str(eval_kind))
        suffix = f" {int(target_hz)}Hz"
        if int(phase) != 0:
            suffix += f" p{int(phase)}"
        return method + suffix
    return ":".join(str(part) for part in label)


def evaluated_frames(frame_summary: pd.DataFrame) -> pd.DataFrame:
    """Return evaluated frames; fallback to all frames if no eval mask exists."""

    if frame_summary.empty:
        return frame_summary
    if "eval_mask" in frame_summary:
        return frame_summary[frame_summary["eval_mask"]].copy()
    return frame_summary.copy()


def dataframe_to_markdown(frame: pd.DataFrame) -> list[str]:
    """Small Markdown table writer that avoids optional tabulate dependency."""

    if frame.empty:
        return ["No rows."]
    display_frame = frame.copy()
    for col in display_frame.select_dtypes(include=[float]).columns:
        display_frame[col] = display_frame[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    rows = [list(display_frame.columns)] + display_frame.astype(str).values.tolist()
    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(rows[0]))]
    header = "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(rows[0])) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    body = [
        "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows[1:]
    ]
    return [header, divider, *body]


def existing(columns: list[str], frame: pd.DataFrame) -> list[str]:
    """Keep columns that exist in a frame."""

    return [col for col in columns if col in frame.columns]


def format_model_label(row: Any) -> str:
    """Compact label for model/eval/frequency plots."""

    return f"{row['model']}\n{row['eval_kind']} {int(row['target_hz'])}Hz"


def draw_empty(ax: Any, message: str) -> None:
    """Draw a placeholder for unavailable plots."""

    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()


def save_figure(fig: Any, output_path: Path) -> None:
    """Save a Matplotlib figure with consistent layout."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
