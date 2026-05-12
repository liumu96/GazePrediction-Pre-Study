#!/usr/bin/env python
"""Inspect scene/object/skeleton assets for one ADT sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.providers import resolve_sequence_path  # noqa: E402
from adt_sandbox.scene_features import (  # noqa: E402
    format_scene_asset_report,
    inspect_scene_assets,
    write_json,
)

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        type=Path,
        help="Sequence id resolved under ADT_DATA_ROOT, or an explicit sequence directory.",
    )
    parser.add_argument(
        "--include-skeleton-json",
        action="store_true",
        help=(
            "Load Skeleton_T.json to count frames/joints/markers. This can take "
            "noticeable time and memory because the file is large."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = resolve_sequence_path(args.sequence)
    report = inspect_scene_assets(
        sequence_dir,
        include_skeleton_json=args.include_skeleton_json,
    )
    if args.output_json is not None:
        write_json(args.output_json, report)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_scene_asset_report(report))


if __name__ == "__main__":
    main()
