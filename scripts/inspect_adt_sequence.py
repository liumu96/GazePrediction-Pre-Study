#!/usr/bin/env python
"""Inspect the files present in one ADT sequence directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv, resolve_data_root  # noqa: E402
from adt_sandbox.adt_files import format_report, inspect_sequence  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        type=Path,
        help=(
            "Path to an ADT sequence directory, or a sequence id resolved under "
            "the ADT_DATA_ROOT environment variable."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = resolve_sequence_path(args.sequence)
    report = inspect_sequence(sequence_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))


def resolve_sequence_path(sequence: Path) -> Path:
    expanded = sequence.expanduser()
    if expanded.exists() or expanded.is_absolute():
        return expanded

    data_root = resolve_data_root()
    if data_root:
        return data_root / expanded

    return expanded


if __name__ == "__main__":
    main()
