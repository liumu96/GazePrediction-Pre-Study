"""CLI entrypoint for reusable SparseGaze prediction-result analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.prediction_plots import write_analysis_figures, write_markdown_report
from analysis.prediction_results import (
    discover_prediction_files,
    load_many_prediction_frames,
    summarize_many_predictions,
)


DEFAULT_EVAL_ROOT = Path("/home/liumu/Github_Projects/SparseGaze/outputs/eval/adt")
DEFAULT_REPORTS_DIR = Path("/mnt/d/SparseGaze/ADT-Gaze-structured")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SparseGaze-style per-sequence NPZ prediction outputs."
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/prediction_results"))
    parser.add_argument("--target-hz", type=int, default=6, help="Target gaze rate. Use --all-hz to disable.")
    parser.add_argument("--all-hz", action="store_true", help="Analyze all available target rates.")
    parser.add_argument("--model", action="append", dest="models", help="Model directory name. Repeatable.")
    parser.add_argument("--eval-kind", action="append", dest="eval_kinds", help="rollout, rollout_gt, etc.")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Skip attaching scene-event labels from reports-dir.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Only write CSV/JSON outputs; skip PNG figures and Markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_hz = None if args.all_hz else args.target_hz
    eval_kinds = args.eval_kinds if args.eval_kinds else ["rollout"]
    reports_dir = None if args.no_events else args.reports_dir

    prediction_files = discover_prediction_files(
        args.eval_root,
        target_hz=target_hz,
        models=args.models,
        eval_kinds=eval_kinds,
        split=args.split,
    )
    if prediction_files.empty:
        raise FileNotFoundError(
            "No prediction NPZ files found. Check --eval-root, --model, --eval-kind, and --target-hz."
        )

    sequence_summary, model_summary, event_summary = summarize_many_predictions(
        prediction_files,
        reports_dir=reports_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_files.to_csv(args.output_dir / "prediction_files.csv", index=False)
    sequence_summary.to_csv(args.output_dir / "sequence_summary.csv", index=False)
    model_summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    if not event_summary.empty:
        event_summary.to_csv(args.output_dir / "event_summary.csv", index=False)

    if not args.no_figures:
        frame_summary = load_many_prediction_frames(prediction_files, reports_dir=reports_dir)
        frame_summary.to_csv(args.output_dir / "frame_summary.csv", index=False)
        figure_paths = write_analysis_figures(
            sequence_summary=sequence_summary,
            model_summary=model_summary,
            event_summary=event_summary,
            frame_summary=frame_summary,
            output_dir=args.output_dir,
        )
        write_markdown_report(
            sequence_summary=sequence_summary,
            model_summary=model_summary,
            event_summary=event_summary,
            figure_paths=figure_paths,
            output_path=args.output_dir / "summary.md",
        )

    config = {
        "eval_root": str(args.eval_root),
        "reports_dir": None if reports_dir is None else str(reports_dir),
        "output_dir": str(args.output_dir),
        "target_hz": target_hz,
        "models": args.models,
        "eval_kinds": eval_kinds,
        "split": args.split,
        "n_prediction_files": len(prediction_files),
        "n_sequences": int(sequence_summary["sequence"].nunique()),
        "figures": not args.no_figures,
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"prediction_files: {len(prediction_files)}")
    print(f"sequences: {config['n_sequences']}")
    print(f"sequence_summary: {args.output_dir / 'sequence_summary.csv'}")
    print(f"model_summary: {args.output_dir / 'model_summary.csv'}")
    if not event_summary.empty:
        print(f"event_summary: {args.output_dir / 'event_summary.csv'}")
    if not args.no_figures:
        print(f"frame_summary: {args.output_dir / 'frame_summary.csv'}")
        print(f"figures: {args.output_dir / 'figures'}")
        print(f"report: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
