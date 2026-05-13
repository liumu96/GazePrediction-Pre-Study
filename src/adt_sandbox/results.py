"""Helpers for organized ADT sandbox result paths.

Outputs use a sequence-first layout:

    <root>/sequences/<sequence>/gaze/gaze_samples.csv
    <root>/sequences/<sequence>/head/head_samples.csv
    <root>/sequences/<sequence>/events/scene_gaze_frame_labels.csv
    <root>/sequences/<sequence>/scene/scene_object_boxes.csv
    <root>/sequences/<sequence>/skeleton/skeleton_samples.csv
    <root>/batch/batch_gaze_extract_summary.csv

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def default_reports_root() -> Path:
    """Return the repository-local default reports root."""

    return Path(__file__).resolve().parents[2] / "outputs" / "reports"


def reports_root(root: str | Path | None = None) -> Path:
    """Return a normalized reports root."""

    return Path(root).expanduser() if root is not None else default_reports_root()


def sequence_dir(root: str | Path | None, sequence_name: str) -> Path:
    """Return the organized directory for one sequence."""

    return reports_root(root) / "sequences" / sequence_name


def feature_dir(root: str | Path | None, sequence_name: str, layer: str) -> Path:
    """Return a sequence-local feature layer directory."""

    return sequence_dir(root, sequence_name) / layer


def sequence_file_path(
    root: str | Path | None,
    sequence_name: str,
    layer: str,
    filename: str,
) -> Path:
    """Return an organized sequence feature file path."""

    return feature_dir(root, sequence_name, layer) / filename


def batch_dir(root: str | Path | None = None) -> Path:
    """Return the organized batch-summary directory."""

    return reports_root(root) / "batch"


def find_sequence_file(
    root: str | Path,
    sequence_name: str,
    layer: str,
    filename: str,
) -> Path:
    """Find a sequence file in the organized layout."""

    path = sequence_file_path(root, sequence_name, layer, filename)
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not find {filename} for {sequence_name}: {path}")


@dataclass(frozen=True)
class SequenceFile:
    """One discovered sequence file."""

    sequence_name: str
    path: Path


def discover_sequence_files(
    root: str | Path,
    layer: str,
    filename: str,
) -> list[SequenceFile]:
    """Discover sequence files in the organized layout."""

    base = reports_root(root)
    found: list[SequenceFile] = []
    organized_pattern = f"sequences/*/{layer}/{filename}"
    for path in sorted(base.glob(organized_pattern)):
        sequence_name = path.parents[1].name
        found.append(SequenceFile(sequence_name, path))
    return found


def discover_sequence_names(
    root: str | Path,
    layer: str,
    filename: str,
) -> list[str]:
    """Return discovered sequence names for one feature file type."""

    return [
        item.sequence_name
        for item in discover_sequence_files(root, layer, filename)
    ]
