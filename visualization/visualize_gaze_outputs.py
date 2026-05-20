#!/usr/bin/env python
"""CLI entry point for gaze window visualizations.

The reusable plotting and export logic lives in ``visualization.gaze_outputs``.
This script only resolves command-line inputs, loads the extracted CSV, and
opens the ADT provider for the selected sequence.

Example:
    python visualization/visualize_gaze_outputs.py \
      Apartment_release_decoration_skeleton_seq131_M1292 \
      --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured \
      --start-row 0 \
      --end-row 60 \
      --stride 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import read_samples_csv  # noqa: E402
from adt_sandbox.providers import create_adt_providers  # noqa: E402
from adt_sandbox.results import find_sequence_file, reports_root  # noqa: E402
from visualization.gaze_outputs import generate_gaze_output_visualizations  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        help="ADT sequence id resolved under ADT_DATA_ROOT, or an absolute sequence path.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Explicit input gaze_samples.csv path. Overrides --reports-dir.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help=(
            "Organized reports root. Defaults to REPORTS_DIR env var when set, "
            "then /mnt/d/SparseGaze/ADT-Gaze-structured when it exists, otherwise "
            "outputs/reports."
        ),
    )
    parser.add_argument("--start-row", type=int, default=0, help="Starting CSV row index.")
    parser.add_argument("--end-row", type=int, default=None, help="Exclusive ending CSV row index.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth selected row for all generated visualizations.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output subdirectory name under outputs/figures/gaze/<sequence>/visualizations/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = resolve_reports_dir(args.reports_dir)
    sequence_name = Path(args.sequence).name
    csv_path = args.input_csv or find_gaze_samples_csv(reports_dir, sequence_name)
    samples = read_samples_csv(csv_path)

    # Skeleton loading is intentionally disabled here.  This visualization only
    # needs RGB frames and gaze projection, while some ADT metadata variants do
    # not expose skeleton subtours cleanly.
    providers = create_adt_providers(args.sequence, skeleton_flag=False)
    result = generate_gaze_output_visualizations(
        gt_provider=providers.gt_provider,
        csv_path=csv_path,
        samples=samples,
        sequence_name=providers.sequence_path.name,
        output_root=REPO_ROOT / "outputs" / "figures" / "gaze",
        start_row=args.start_row,
        end_row=args.end_row,
        stride=args.stride,
        run_name=args.run_name,
    )

    print(f"sequence: {providers.sequence_path.name}")
    print(f"sequence_path: {providers.sequence_path}")
    print(f"csv: {csv_path}")
    if args.input_csv is None:
        print(f"reports_dir: {reports_dir}")
    print(f"rows: {args.start_row}..{args.end_row if args.end_row is not None else len(samples)}")
    print(f"window_samples: {result['window_samples']} viz_samples={result['viz_samples']}")
    print(f"image_orientation: {'upright' if result['make_upright'] else 'raw'}")
    print(f"figures: {result['output_dir']}")


def resolve_reports_dir(explicit_reports_dir: Path | None) -> Path:
    """Resolve the organized report root used for implicit CSV lookup."""

    if explicit_reports_dir is not None:
        return reports_root(explicit_reports_dir)
    env_reports_dir = os.environ.get("REPORTS_DIR")
    if env_reports_dir:
        return reports_root(env_reports_dir)
    structured_reports_dir = Path("/mnt/d/SparseGaze/ADT-Gaze-structured")
    if structured_reports_dir.exists():
        return structured_reports_dir
    return reports_root(REPO_ROOT / "outputs" / "reports")


def find_gaze_samples_csv(reports_dir: Path, sequence_name: str) -> Path:
    """Find the extracted gaze CSV and raise an actionable error if missing."""

    try:
        return find_sequence_file(
            reports_dir,
            sequence_name,
            "gaze",
            "gaze_samples.csv",
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not find gaze_samples.csv. Pass either:\n"
            "  --input-csv /path/to/gaze_samples.csv\n"
            "or:\n"
            "  --reports-dir /mnt/d/SparseGaze/ADT-Gaze-structured\n"
            f"Current reports_dir: {reports_dir}\n"
            f"Original error: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
