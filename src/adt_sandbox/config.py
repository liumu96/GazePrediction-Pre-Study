"""Project-local configuration helpers.

zh-CN:
这里集中处理本项目的轻量配置：从仓库根目录 `.env` 读取环境变量，并解析
`ADT_DATA_ROOT`。脚本会先调用 `load_dotenv()`，这样通常不需要每次手动
`export ADT_DATA_ROOT=...`。
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(dotenv_path: str | Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file without overwriting env vars."""

    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_data_root() -> Path | None:
    """Return ADT_DATA_ROOT if configured."""

    data_root = os.environ.get("ADT_DATA_ROOT")
    if not data_root:
        return None
    return Path(data_root).expanduser()
