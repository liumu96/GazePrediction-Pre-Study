"""CLI entrypoint for SparseGaze anchor-gap position analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.prediction_plots import write_anchor_gap_figures, write_anchor_gap_report
from analysis.prediction_results import (
    discover_prediction_files,
    load_prediction_frame,
    load_many_prediction_frames,
    summarize_anchor_gap_position,
)
from visualization.adt_hagi_sparsegaze_compare import DEFAULT_HAGI_DIR, load_hagi_primary


DEFAULT_EVAL_ROOT = Path("/home/liumu/Github_Projects/SparseGaze/outputs/eval/adt")
DEFAULT_REPORTS_DIR = Path("/mnt/d/SparseGaze/ADT-Gaze-structured")
DEFAULT_MODEL = "sparsegaze_cpf_forward_head_motion_residual_ss"
DEFAULT_EVAL_KINDS = ["rollout", "rollout_linear", "rollout_pchip", "rollout_gt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze missing-frame error by position between sparse gaze anchors."
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--hagi-dir", type=Path, default=DEFAULT_HAGI_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/anchor_gap_position"))
    parser.add_argument("--target-hz", type=int, default=6, help="Target gaze rate. Use --all-hz to disable.")
    parser.add_argument("--all-hz", action="store_true", help="Analyze all available target rates.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=f"Model directory name. Default: {DEFAULT_MODEL}. Repeatable.",
    )
    parser.add_argument(
        "--eval-kind",
        action="append",
        dest="eval_kinds",
        help=f"Eval kind. Default: {', '.join(DEFAULT_EVAL_KINDS)}. Repeatable.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--no-hagi", action="store_true", help="Do not include HAGI++ rows.")
    parser.add_argument(
        "--all-available",
        action="store_true",
        help="Use all available rows instead of common frames shared by selected methods.",
    )
    parser.add_argument("--no-events", action="store_true", help="Skip scene-event labels.")
    parser.add_argument("--no-figures", action="store_true", help="Write CSV/JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_hz = None if args.all_hz else args.target_hz
    models = args.models if args.models else [DEFAULT_MODEL]
    eval_kinds = args.eval_kinds if args.eval_kinds else DEFAULT_EVAL_KINDS
    reports_dir = None if args.no_events else args.reports_dir

    prediction_files = discover_prediction_files(
        args.eval_root,
        target_hz=target_hz,
        models=models,
        eval_kinds=eval_kinds,
        split=args.split,
    )
    if prediction_files.empty:
        raise FileNotFoundError(
            "No prediction NPZ files found. Check --eval-root, --model, --eval-kind, and --target-hz."
        )

    frame_summary = load_many_prediction_frames(prediction_files, reports_dir=reports_dir)
    if not args.no_hagi:
        hagi_frames = load_hagi_gap_frames(
            prediction_files,
            reports_dir=reports_dir,
            hagi_dir=args.hagi_dir,
            split=args.split,
        )
        if not hagi_frames.empty:
            frame_summary = pd.concat([frame_summary, hagi_frames], ignore_index=True)
    if not args.all_available:
        frame_summary = keep_common_anchor_gap_frames(frame_summary)

    gap_summary = summarize_anchor_gap_position(frame_summary, bins=args.bins)
    gap_event_summary = summarize_anchor_gap_position(
        frame_summary,
        bins=args.bins,
        event_conditioned=reports_dir is not None,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_files.to_csv(args.output_dir / "prediction_files.csv", index=False)
    gap_summary.to_csv(args.output_dir / "gap_position_summary.csv", index=False)
    if not gap_event_summary.empty:
        gap_event_summary.to_csv(args.output_dir / "gap_event_summary.csv", index=False)

    figure_paths = {}
    if not args.no_figures:
        figure_paths = write_anchor_gap_figures(
            gap_summary=gap_summary,
            gap_event_summary=gap_event_summary,
            output_dir=args.output_dir,
        )
        write_anchor_gap_report(
            gap_summary=gap_summary,
            gap_event_summary=gap_event_summary,
            figure_paths=figure_paths,
            output_path=args.output_dir / "summary.md",
        )

    config = {
        "eval_root": str(args.eval_root),
        "reports_dir": None if reports_dir is None else str(reports_dir),
        "output_dir": str(args.output_dir),
        "target_hz": target_hz,
        "models": models,
        "eval_kinds": eval_kinds,
        "split": args.split,
        "bins": args.bins,
        "hagi": not args.no_hagi,
        "common_frames": not args.all_available,
        "n_prediction_files": len(prediction_files),
        "n_sequences": int(prediction_files["sequence"].nunique()),
        "figures": not args.no_figures,
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"prediction_files: {len(prediction_files)}")
    print(f"sequences: {config['n_sequences']}")
    print(f"gap_position_summary: {args.output_dir / 'gap_position_summary.csv'}")
    if not gap_event_summary.empty:
        print(f"gap_event_summary: {args.output_dir / 'gap_event_summary.csv'}")
    if figure_paths:
        print(f"figures: {args.output_dir / 'figures'}")
        print(f"report: {args.output_dir / 'summary.md'}")


def load_hagi_gap_frames(
    prediction_files: pd.DataFrame,
    *,
    reports_dir: Path | None,
    hagi_dir: Path,
    split: str,
) -> pd.DataFrame:
    """Load HAGI++ rows and attach SparseGaze anchor-gap positions by frame."""

    if prediction_files.empty:
        return pd.DataFrame()
    references = prediction_files.drop_duplicates(["sequence", "target_hz", "phase"])
    rows = []
    hagi_cache: dict[int, dict] = {}
    for _, item in references.iterrows():
        fps = int(item["target_hz"])
        sequence = str(item["sequence"])
        phase = int(item["phase"])
        if fps not in hagi_cache:
            hagi_cache[fps] = load_hagi_primary(fps, hagi_dir=hagi_dir)
        hagi = hagi_cache[fps]
        sequence_names = hagi["sequence_name"].astype(str)
        mask = sequence_names == sequence
        if not np.any(mask):
            continue

        reference = load_prediction_frame(Path(item["path"]), reports_dir=reports_dir)
        gap_cols = [
            "frame_index",
            "anchor_mask",
            "prev_anchor_frame",
            "next_anchor_frame",
            "anchor_interval_frames",
            "missing_interval_frames",
            "frames_since_prev_anchor",
            "frames_until_next_anchor",
            "nearest_anchor_distance_frames",
            "normalized_gap_position",
            "anchor_gap_valid",
        ]
        if "scene_event_label" in reference:
            gap_cols.append("scene_event_label")
        reference = reference[gap_cols]

        hagi_frame = pd.DataFrame(
            {
                "frame_index": hagi["frame_index"][mask].astype(int),
                "angular_error_deg": hagi["angular_error_deg"][mask].astype(float),
            }
        )
        hagi_frame = hagi_frame.merge(reference, on="frame_index", how="inner")
        if hagi_frame.empty:
            continue
        hagi_frame["eval_mask"] = True
        hagi_frame["model"] = "HAGI++"
        hagi_frame["split"] = split
        hagi_frame["eval_kind"] = "hagi++"
        hagi_frame["target_hz"] = fps
        hagi_frame["phase"] = phase
        hagi_frame["sequence"] = sequence
        rows.append(hagi_frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def keep_common_anchor_gap_frames(frame_summary: pd.DataFrame) -> pd.DataFrame:
    """Keep evaluated missing gap frames shared by every selected method."""

    if frame_summary.empty:
        return frame_summary
    required = {
        "sequence",
        "target_hz",
        "phase",
        "model",
        "eval_kind",
        "frame_index",
        "eval_mask",
        "anchor_mask",
        "anchor_gap_valid",
    }
    if not required.issubset(frame_summary.columns):
        return frame_summary

    eligible = frame_summary[
        frame_summary["eval_mask"].astype(bool)
        & ~frame_summary["anchor_mask"].astype(bool)
        & frame_summary["anchor_gap_valid"].astype(bool)
    ].copy()
    if eligible.empty:
        return eligible

    kept = []
    group_cols = ["sequence", "target_hz", "phase"]
    method_cols = ["model", "eval_kind"]
    for _, group in eligible.groupby(group_cols, sort=False):
        methods = list(group[method_cols].drop_duplicates().itertuples(index=False, name=None))
        frame_sets = [
            set(
                group.loc[
                    (group["model"] == model) & (group["eval_kind"] == eval_kind),
                    "frame_index",
                ].astype(int)
            )
            for model, eval_kind in methods
        ]
        common = set.intersection(*frame_sets) if frame_sets else set()
        if common:
            kept.append(group[group["frame_index"].isin(common)])
    return pd.concat(kept, ignore_index=True) if kept else eligible.iloc[0:0].copy()


if __name__ == "__main__":
    main()
