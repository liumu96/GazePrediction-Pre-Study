"""Visualization helpers for ADT sandbox notebooks and scripts.

The visualization package lives at the repository root, while core data
extraction modules live under ``src/``.  Add ``src`` when importing the
root-level package directly from notebooks or small scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
