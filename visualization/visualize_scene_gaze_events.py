#!/usr/bin/env python
"""CLI entry point for scene-direction gaze event timeline figures.

Run ``detect_scene_gaze_events.py`` first.  The reusable plotting code lives in
``visualization.scene_gaze_events``; this file only resolves input/output paths.

Example:
    python visualization/visualize_scene_gaze_events.py \
      Apartment_release_decoration_skeleton_seq131_M1292 \
      --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured \
      --start-frame 0 \
      --end-frame 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.results import find_sequence_file  # noqa: E402
from adt_sandbox.scene_gaze_events import (  # noqa: E402
    read_scene_gaze_event_features_csv,
    read_scene_gaze_event_segments_csv,
    read_scene_gaze_frame_labels_csv,
)
from visualization.scene_gaze_events import (  # noqa: E402
    plot_scene_gaze_event_timeline,
    resolve_frame_window,
    select_feature_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", help="Sequence name used in scene event output filenames.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "reports",
        help="Directory containing scene event CSV files.",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=None,
        help="Optional explicit *_scene_gaze_event_features.csv path.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional explicit *_scene_gaze_frame_labels.csv path.",
    )
    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=None,
        help="Optional explicit *_scene_gaze_event_segments.csv path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "scene_gaze_events",
        help="Directory for the generated timeline figure.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive starting frame index.")
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive ending frame index. Default uses --max-frames from start.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Window length when --end-frame is omitted. Use 0 to plot to sequence end.",
    )
    parser.add_argument(
        "--velocity-threshold-deg-s",
        type=float,
        default=40.0,
        help="Draw the velocity threshold used for fixation labeling.",
    )
    parser.add_argument(
        "--dispersion-threshold-deg",
        type=float,
        default=2.5,
        help="Draw the dispersion threshold used for fixation labeling.",
    )
    parser.add_argument("--velocity-ymax", type=float, default=None, help="Optional y-axis max for velocity.")
    parser.add_argument("--dispersion-ymax", type=float, default=None, help="Optional y-axis max for dispersion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = Path(args.sequence).name
    feature_csv = args.features_csv or resolve_scene_event_file(
        args.reports_dir,
        sequence,
        "scene_gaze_event_features.csv",
    )
    label_csv = args.labels_csv or resolve_scene_event_file(
        args.reports_dir,
        sequence,
        "scene_gaze_frame_labels.csv",
    )
    segment_csv = args.segments_csv or resolve_scene_event_file(
        args.reports_dir,
        sequence,
        "scene_gaze_event_segments.csv",
    )

    require_file(feature_csv, "scene event features")
    require_file(label_csv, "scene event frame labels")
    require_file(segment_csv, "scene event segments")

    features = read_scene_gaze_event_features_csv(feature_csv)
    labels = read_scene_gaze_frame_labels_csv(label_csv)
    segments = read_scene_gaze_event_segments_csv(segment_csv)
    start_frame, end_frame = resolve_frame_window(
        features,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
    )
    selected_features = select_feature_window(features, start_frame, end_frame)
    if not selected_features:
        raise ValueError(
            f"No scene event feature rows selected for frames {start_frame}..{end_frame}"
        )

    output_path = args.output_dir / (
        f"{sequence}_scene_gaze_events_{start_frame}_{end_frame}.png"
    )
    plot_scene_gaze_event_timeline(
        output_path,
        sequence_name=sequence,
        features=selected_features,
        labels=labels,
        segments=segments,
        start_frame=start_frame,
        end_frame=end_frame,
        velocity_threshold_deg_s=args.velocity_threshold_deg_s,
        dispersion_threshold_deg=args.dispersion_threshold_deg,
        velocity_ymax=args.velocity_ymax,
        dispersion_ymax=args.dispersion_ymax,
    )

    print(f"sequence: {sequence}")
    print(f"features_csv: {feature_csv}")
    print(f"labels_csv: {label_csv}")
    print(f"segments_csv: {segment_csv}")
    print(f"frames: {start_frame}..{end_frame}")
    print(f"figure: {output_path}")


def resolve_scene_event_file(
    reports_dir: Path,
    sequence: str,
    filename: str,
) -> Path:
    """Resolve one organized scene-event CSV path."""

    return find_sequence_file(reports_dir, sequence, "events", filename)


def require_file(path: Path, description: str) -> None:
    """Fail early with the command that creates the missing event CSVs."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Run scripts/detect_scene_gaze_events.py first."
        )


if __name__ == "__main__":
    main()
