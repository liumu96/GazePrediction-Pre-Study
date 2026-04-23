"""Helpers for creating official Project Aria ADT data providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import resolve_data_root


@dataclass(frozen=True)
class AdtProviders:
    """Container for a resolved ADT sequence and its Project Aria providers."""

    sequence_path: Path
    paths_provider: Any
    data_paths: Any
    gt_provider: Any
    provider_mode: str


def resolve_sequence_path(sequence: str | Path) -> Path:
    """Resolve a sequence id/path against ADT_DATA_ROOT when needed."""

    expanded = Path(sequence).expanduser()
    if expanded.exists() or expanded.is_absolute():
        return expanded

    data_root = resolve_data_root()
    if data_root:
        return data_root / expanded

    return expanded


def create_adt_providers(
    sequence: str | Path,
    skeleton_flag: bool = True,
) -> AdtProviders:
    """Create official ADT providers for one sequence.

    zh-CN:
    为一个 ADT sequence 创建官方 Project Aria provider。这个仓库默认使用
    `adt` conda 环境里的 `projectaria-tools 2.x`，因此不再保留 base 环境或
    旧版 API 的兼容读取路径。
    """

    sequence_path = resolve_sequence_path(sequence).resolve()
    if not sequence_path.exists():
        raise FileNotFoundError(f"ADT sequence does not exist: {sequence_path}")
    if not sequence_path.is_dir():
        raise NotADirectoryError(f"Expected ADT sequence directory: {sequence_path}")

    try:
        from projectaria_tools.projects.adt import (
            AriaDigitalTwinDataPathsProvider,
            AriaDigitalTwinDataProvider,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import projectaria_tools. Activate the `adt` environment and run "
            '`python -m pip install -e ".[dev]"` from the repository root.'
        ) from exc

    paths_provider = AriaDigitalTwinDataPathsProvider(str(sequence_path))
    data_paths = paths_provider.get_datapaths(skeleton_flag)
    gt_provider = AriaDigitalTwinDataProvider(data_paths)
    return AdtProviders(
        sequence_path=sequence_path,
        paths_provider=paths_provider,
        data_paths=data_paths,
        gt_provider=gt_provider,
        provider_mode="official_adt",
    )
