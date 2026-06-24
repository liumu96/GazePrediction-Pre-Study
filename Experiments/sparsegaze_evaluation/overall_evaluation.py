#!/usr/bin/env python
"""Dataset-level SparseGaze evaluation helpers.

This module aggregates the single-sequence event evaluation across all
available ADT sequences. The primary comparison uses common frames shared by
all selected methods, so method differences are not mixed with coverage
differences.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import event_evaluation as event_eval  # noqa: E402
from visualization.adt_hagi_sparsegaze_compare import METHOD_COLORS  # noqa: E402

EVENT_ORDER = ["fixation", "transition", "invalid", "unknown"]


def available_sequences(
    *,
    fps: int,
    adt_data: dict | None = None,
    sparsegaze_dir: Path = event_eval.DEFAULT_SPARSEGAZE_DIR,
    hagi_dir: Path = event_eval.DEFAULT_HAGI_DIR,
) -> list[str]:
    """Return sequences with ADT, SparseGaze rollout, and HAGI++ availability."""

    if adt_data is None:
        adt_data = event_eval.load_adt_data(event_eval.DEFAULT_ADT_DATA)
    return event_eval.sequence_names(
        adt_data=adt_data,
        fps=fps,
        sparsegaze_dir=sparsegaze_dir,
        hagi_dir=hagi_dir,
    )


def load_dataset_event_errors(
    *,
    fps: int,
    modes: Iterable[str],
    sequences: Iterable[str] | None = None,
    include_hagi: bool = True,
    common_frames: bool = True,
    reports_dir: Path = event_eval.DEFAULT_REPORTS_DIR,
    sparsegaze_dir: Path = event_eval.DEFAULT_SPARSEGAZE_DIR,
    hagi_dir: Path = event_eval.DEFAULT_HAGI_DIR,
    adt_data_path: Path = event_eval.DEFAULT_ADT_DATA,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> pd.DataFrame:
    """Load event-labeled angular errors for all selected sequences.

    When ``common_frames`` is true, only frames present for every selected
    method inside a sequence are retained. This is the default because HAGI++
    and SparseGaze can have different valid frame sets.
    """

    adt_data = event_eval.load_adt_data(adt_data_path)
    selected_sequences = list(sequences or available_sequences(
        fps=fps,
        adt_data=adt_data,
        sparsegaze_dir=sparsegaze_dir,
        hagi_dir=hagi_dir,
    ))
    frames: list[pd.DataFrame] = []
    for sequence in selected_sequences:
        labels = event_eval.load_event_labels(reports_dir, sequence)
        seq_start, seq_end = event_eval.resolve_window(
            labels,
            start_frame=start_frame,
            end_frame=end_frame,
            max_frames=0,
        )
        pred = event_eval.load_method_event_errors(
            sequence=sequence,
            fps=fps,
            modes=modes,
            adt_data=adt_data,
            labels=labels,
            sparsegaze_dir=sparsegaze_dir,
            start_frame=seq_start,
            end_frame=seq_end,
            include_hagi=include_hagi,
            hagi_dir=hagi_dir,
        )
        if pred.empty:
            continue
        pred = pred.copy()
        pred.insert(0, "sequence", sequence)
        frames.append(pred)
    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    if common_frames:
        data = keep_common_frames(data)
    return data.sort_values(["sequence", "method", "frame_index"]).reset_index(drop=True)


def keep_common_frames(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep only per-sequence frames observed by every selected method."""

    if predictions.empty:
        return predictions.copy()
    kept = []
    for _, seq_group in predictions.groupby("sequence", sort=False):
        methods = list(dict.fromkeys(seq_group["method"]))
        if len(methods) <= 1:
            kept.append(seq_group)
            continue
        frame_sets = [
            set(seq_group.loc[seq_group["method"] == method, "frame_index"].astype(int))
            for method in methods
        ]
        common = set.intersection(*frame_sets) if frame_sets else set()
        if common:
            kept.append(seq_group[seq_group["frame_index"].isin(common)])
    return pd.concat(kept, ignore_index=True) if kept else predictions.iloc[0:0].copy()


def summarize_overall(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize frame-weighted and sequence-macro error by method."""

    columns = [
        "method",
        "mode",
        "sequence_n",
        "frame_n",
        "frame_weighted_mae_deg",
        "sequence_macro_mae_deg",
        "median_deg",
        "p90_deg",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    seq_mae = summarize_by_sequence(predictions)
    rows = []
    for (method, mode), group in predictions.groupby(["method", "mode"], sort=False):
        seq_group = seq_mae[(seq_mae["method"] == method) & (seq_mae["mode"] == mode)]
        rows.append(
            {
                "method": method,
                "mode": mode,
                "sequence_n": int(seq_group["sequence"].nunique()),
                "frame_n": int(len(group)),
                "frame_weighted_mae_deg": float(group["angular_error_deg"].mean()),
                "sequence_macro_mae_deg": float(seq_group["mae_deg"].mean()),
                "median_deg": float(group["angular_error_deg"].median()),
                "p90_deg": float(np.percentile(group["angular_error_deg"], 90)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("sequence_macro_mae_deg")


def summarize_frame_coverage(
    all_predictions: pd.DataFrame,
    retained_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Report how many method frames remain after common-frame filtering."""

    columns = [
        "method",
        "mode",
        "sequence_n",
        "all_available_frame_n",
        "retained_frame_n",
        "retained_ratio",
    ]
    if all_predictions.empty:
        return pd.DataFrame(columns=columns)

    all_counts = (
        all_predictions.groupby(["method", "mode"], as_index=False, sort=False)
        .agg(
            sequence_n=("sequence", "nunique"),
            all_available_frame_n=("angular_error_deg", "size"),
        )
        .set_index(["method", "mode"])
    )
    retained_counts = (
        retained_predictions.groupby(["method", "mode"], as_index=False, sort=False)
        .agg(retained_frame_n=("angular_error_deg", "size"))
        .set_index(["method", "mode"])
        if not retained_predictions.empty
        else pd.DataFrame(columns=["retained_frame_n"])
    )
    rows = []
    for (method, mode), row in all_counts.iterrows():
        retained = (
            int(retained_counts.loc[(method, mode), "retained_frame_n"])
            if (method, mode) in retained_counts.index
            else 0
        )
        total = int(row["all_available_frame_n"])
        rows.append(
            {
                "method": method,
                "mode": mode,
                "sequence_n": int(row["sequence_n"]),
                "all_available_frame_n": total,
                "retained_frame_n": retained,
                "retained_ratio": retained / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_by_sequence(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return per-sequence MAE for every method."""

    if predictions.empty:
        return pd.DataFrame(
            columns=["sequence", "method", "mode", "n", "mae_deg", "median_deg", "p90_deg"]
        )
    return (
        predictions.groupby(["sequence", "method", "mode"], as_index=False, sort=False)
        .agg(
            n=("angular_error_deg", "size"),
            mae_deg=("angular_error_deg", "mean"),
            median_deg=("angular_error_deg", "median"),
            p90_deg=("angular_error_deg", lambda values: float(np.percentile(values, 90))),
        )
        .sort_values(["sequence", "mae_deg"])
    )


def summarize_by_event(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize errors by method and GT event with sequence-macro MAE."""

    columns = [
        "method",
        "mode",
        "scene_event_label",
        "sequence_n",
        "frame_n",
        "frame_weighted_mae_deg",
        "sequence_macro_mae_deg",
        "median_deg",
        "p90_deg",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    seq_event = (
        predictions.groupby(
            ["sequence", "method", "mode", "scene_event_label"],
            as_index=False,
            sort=False,
        )
        .agg(mae_deg=("angular_error_deg", "mean"))
    )
    rows = []
    for (method, mode, event), group in predictions.groupby(
        ["method", "mode", "scene_event_label"],
        sort=False,
    ):
        seq_group = seq_event[
            (seq_event["method"] == method)
            & (seq_event["mode"] == mode)
            & (seq_event["scene_event_label"] == event)
        ]
        rows.append(
            {
                "method": method,
                "mode": mode,
                "scene_event_label": event,
                "sequence_n": int(seq_group["sequence"].nunique()),
                "frame_n": int(len(group)),
                "frame_weighted_mae_deg": float(group["angular_error_deg"].mean()),
                "sequence_macro_mae_deg": float(seq_group["mae_deg"].mean()),
                "median_deg": float(group["angular_error_deg"].median()),
                "p90_deg": float(np.percentile(group["angular_error_deg"], 90)),
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    out["event_order"] = out["scene_event_label"].map(
        {label: index for index, label in enumerate(EVENT_ORDER)}
    ).fillna(len(EVENT_ORDER))
    return out.sort_values(["event_order", "sequence_macro_mae_deg"]).drop(columns="event_order")


def summarize_event_contrast(event_summary: pd.DataFrame) -> pd.DataFrame:
    """Return fixation/transition contrast for each method."""

    columns = [
        "method",
        "mode",
        "fixation_sequence_macro_mae_deg",
        "transition_sequence_macro_mae_deg",
        "transition_minus_fixation_mae_deg",
    ]
    if event_summary.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (method, mode), group in event_summary.groupby(["method", "mode"], sort=False):
        by_event = group.set_index("scene_event_label")
        fixation = by_event["sequence_macro_mae_deg"].get("fixation", np.nan)
        transition = by_event["sequence_macro_mae_deg"].get("transition", np.nan)
        rows.append(
            {
                "method": method,
                "mode": mode,
                "fixation_sequence_macro_mae_deg": fixation,
                "transition_sequence_macro_mae_deg": transition,
                "transition_minus_fixation_mae_deg": transition - fixation,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_baseline_delta(
    sequence_summary: pd.DataFrame,
    *,
    baseline: str = "HAGI++",
) -> pd.DataFrame:
    """Return per-sequence MAE delta against a baseline method."""

    if sequence_summary.empty or baseline not in set(sequence_summary["method"]):
        return pd.DataFrame()
    pivot = sequence_summary.pivot(index="sequence", columns="method", values="mae_deg")
    if baseline not in pivot:
        return pd.DataFrame()
    rows = []
    for method in pivot.columns:
        if method == baseline:
            continue
        delta = pivot[method] - pivot[baseline]
        for sequence, value in delta.dropna().items():
            rows.append(
                {
                    "sequence": sequence,
                    "method": method,
                    "baseline": baseline,
                    "delta_mae_deg": float(value),
                }
            )
    return pd.DataFrame(rows)


def summarize_win_counts(sequence_summary: pd.DataFrame) -> pd.DataFrame:
    """Count per-sequence wins and average rank by method."""

    columns = ["method", "win_n", "sequence_n", "mean_rank"]
    if sequence_summary.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for method, group in sequence_summary.groupby("method", sort=False):
        win_n = 0
        ranks = []
        for _, seq_group in sequence_summary.groupby("sequence", sort=False):
            ranked = seq_group.sort_values("mae_deg").reset_index(drop=True)
            winners = set(ranked.loc[ranked["mae_deg"] == ranked["mae_deg"].min(), "method"])
            if method in winners:
                win_n += 1
            method_rank = ranked.index[ranked["method"] == method]
            if len(method_rank):
                ranks.append(float(method_rank[0] + 1))
        rows.append(
            {
                "method": method,
                "win_n": int(win_n),
                "sequence_n": int(group["sequence"].nunique()),
                "mean_rank": float(np.mean(ranks)) if ranks else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["mean_rank", "method"])


def make_method_color_map(methods: Iterable[str]) -> dict[str, str]:
    """Use project method colors where available and fall back to matplotlib."""

    names = list(dict.fromkeys(methods))
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = {}
    fallback_index = 0
    for method in names:
        if method in METHOD_COLORS:
            colors[method] = METHOD_COLORS[method]
        else:
            colors[method] = cycle[fallback_index % len(cycle)] if cycle else None
            fallback_index += 1
    return colors


def make_overall_dashboard(
    predictions: pd.DataFrame,
    *,
    title: str = "SparseGaze dataset-level missing-frame evaluation",
) -> plt.Figure:
    """Create the main aggregate evaluation dashboard."""

    overall = summarize_overall(predictions)
    by_event = summarize_by_event(predictions)
    by_sequence = summarize_by_sequence(predictions)
    colors = make_method_color_map(overall["method"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_overall, ax_event, ax_box, ax_delta = axes.ravel()
    _plot_overall_bars(ax_overall, overall, colors)
    _plot_event_bars(ax_event, by_event, colors)
    _plot_sequence_box(ax_box, by_sequence, colors)
    _plot_baseline_delta(ax_delta, by_sequence, colors)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def make_sequence_heatmap(
    sequence_summary: pd.DataFrame,
    *,
    title: str = "Per-sequence MAE heatmap",
) -> plt.Figure:
    """Create a method-by-sequence MAE heatmap."""

    fig, ax = plt.subplots(figsize=(12, max(5, 0.34 * sequence_summary["sequence"].nunique())))
    if sequence_summary.empty:
        ax.set_axis_off()
        return fig
    pivot = sequence_summary.pivot(index="sequence", columns="method", values="mae_deg")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    pivot = pivot[pivot.mean(axis=0).sort_values().index]
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([short_sequence_name(seq) for seq in pivot.index])
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel("Sequence")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("MAE [deg]")
    fig.tight_layout()
    return fig


def _plot_overall_bars(
    axis: plt.Axes,
    overall: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    if overall.empty:
        axis.set_axis_off()
        return
    ordered = overall.sort_values("sequence_macro_mae_deg")
    x = np.arange(len(ordered))
    axis.bar(
        x,
        ordered["sequence_macro_mae_deg"],
        color=[colors.get(method) for method in ordered["method"]],
    )
    axis.scatter(
        x,
        ordered["frame_weighted_mae_deg"],
        color="black",
        marker="_",
        s=110,
        label="frame-weighted",
        zorder=3,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(ordered["method"], rotation=25, ha="right")
    axis.set_ylabel("MAE [deg]")
    axis.set_title("Overall MAE across sequences")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def _plot_event_bars(
    axis: plt.Axes,
    by_event: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    if by_event.empty:
        axis.set_axis_off()
        return
    events = [event for event in EVENT_ORDER if event in set(by_event["scene_event_label"])]
    methods = list(dict.fromkeys(by_event["method"]))
    x = np.arange(len(events), dtype=float)
    width = 0.82 / max(len(methods), 1)
    for index, method in enumerate(methods):
        values = []
        for event in events:
            row = by_event[
                (by_event["method"] == method)
                & (by_event["scene_event_label"] == event)
            ]
            values.append(
                float(row["sequence_macro_mae_deg"].iloc[0]) if not row.empty else np.nan
            )
        offset = (index - (len(methods) - 1) / 2) * width
        axis.bar(x + offset, values, width=width, label=method, color=colors.get(method))
    axis.set_xticks(x)
    axis.set_xticklabels(events)
    axis.set_ylabel("Sequence-macro MAE [deg]")
    axis.set_title("Error by GT scene-gaze event")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def _plot_sequence_box(
    axis: plt.Axes,
    by_sequence: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    if by_sequence.empty:
        axis.set_axis_off()
        return
    methods = list(dict.fromkeys(by_sequence["method"]))
    values = [
        by_sequence.loc[by_sequence["method"] == method, "mae_deg"].to_numpy(dtype=float)
        for method in methods
    ]
    box = axis.boxplot(values, tick_labels=methods, patch_artist=True, showfliers=True)
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(colors.get(method, "#cccccc"))
        patch.set_alpha(0.55)
    axis.set_xticklabels(methods, rotation=25, ha="right")
    axis.set_ylabel("Per-sequence MAE [deg]")
    axis.set_title("Sequence-to-sequence variability")
    axis.grid(axis="y", alpha=0.25)


def _plot_baseline_delta(
    axis: plt.Axes,
    by_sequence: pd.DataFrame,
    colors: dict[str, str],
    baseline: str = "HAGI++",
) -> None:
    delta = summarize_baseline_delta(by_sequence, baseline=baseline)
    if delta.empty:
        axis.set_axis_off()
        return
    methods = list(dict.fromkeys(delta["method"]))
    positions = np.arange(len(methods), dtype=float)
    for index, method in enumerate(methods):
        vals = delta.loc[delta["method"] == method, "delta_mae_deg"].to_numpy(dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        axis.scatter(
            positions[index] + jitter,
            vals,
            color=colors.get(method),
            alpha=0.75,
            s=22,
            label=method,
        )
        axis.hlines(
            np.nanmean(vals),
            positions[index] - 0.22,
            positions[index] + 0.22,
            color="black",
            linewidth=1.5,
        )
    axis.axhline(0, color="black", linewidth=0.9, alpha=0.7)
    axis.set_xticks(positions)
    axis.set_xticklabels(methods, rotation=25, ha="right")
    axis.set_ylabel(f"MAE delta vs {baseline} [deg]")
    axis.set_title("Per-sequence improvement over baseline")
    axis.grid(axis="y", alpha=0.25)


def short_sequence_name(sequence: str) -> str:
    """Make long ADT sequence names readable on plots."""

    prefix = "Apartment_release_"
    if sequence.startswith(prefix):
        sequence = sequence[len(prefix):]
    return sequence.replace("_skeleton_", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        choices=sorted(event_eval.SPARSEGAZE_MODES),
        help="SparseGaze mode to include. Repeat to include multiple modes.",
    )
    parser.add_argument("--no-hagi", action="store_true")
    parser.add_argument("--all-available", action="store_true")
    parser.add_argument("--reports-dir", type=Path, default=event_eval.DEFAULT_REPORTS_DIR)
    parser.add_argument("--sparsegaze-dir", type=Path, default=event_eval.DEFAULT_SPARSEGAZE_DIR)
    parser.add_argument("--hagi-dir", type=Path, default=event_eval.DEFAULT_HAGI_DIR)
    parser.add_argument("--adt-data", type=Path, default=event_eval.DEFAULT_ADT_DATA)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs" / "overall")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = tuple(args.modes or event_eval.SPARSEGAZE_MODES.keys())
    all_predictions = load_dataset_event_errors(
        fps=args.fps,
        modes=modes,
        include_hagi=not args.no_hagi,
        common_frames=False,
        reports_dir=args.reports_dir,
        sparsegaze_dir=args.sparsegaze_dir,
        hagi_dir=args.hagi_dir,
        adt_data_path=args.adt_data,
    )
    predictions = all_predictions if args.all_available else keep_common_frames(all_predictions)
    overall = summarize_overall(predictions)
    by_sequence = summarize_by_sequence(predictions)
    by_event = summarize_by_event(predictions)
    event_contrast = summarize_event_contrast(by_event)
    win_counts = summarize_win_counts(by_sequence)
    coverage = summarize_frame_coverage(all_predictions, predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / f"hz{args.fps}_frame_errors.csv", index=False)
    overall.to_csv(args.output_dir / f"hz{args.fps}_overall_summary.csv", index=False)
    by_sequence.to_csv(args.output_dir / f"hz{args.fps}_sequence_summary.csv", index=False)
    by_event.to_csv(args.output_dir / f"hz{args.fps}_event_summary.csv", index=False)
    event_contrast.to_csv(args.output_dir / f"hz{args.fps}_event_contrast.csv", index=False)
    win_counts.to_csv(args.output_dir / f"hz{args.fps}_win_counts.csv", index=False)
    coverage.to_csv(args.output_dir / f"hz{args.fps}_coverage.csv", index=False)

    fig = make_overall_dashboard(predictions, title=f"SparseGaze overall evaluation | {args.fps} Hz")
    fig.savefig(args.output_dir / f"hz{args.fps}_overall_dashboard.png", dpi=180)
    plt.close(fig)
    heatmap = make_sequence_heatmap(by_sequence, title=f"Per-sequence MAE | {args.fps} Hz")
    heatmap.savefig(args.output_dir / f"hz{args.fps}_sequence_heatmap.png", dpi=180)
    plt.close(heatmap)

    print(overall.round(4).to_string(index=False))
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
