"""Root conftest: make repo modules importable for the whole test session.

The data/ modules (parse_mesh, parse_face_to_nodes) are written to be imported as
top-level names (no data/ package). The fem/ package imports them too (assembly,
direct_solver, postprocess), so the data/ directory must be on sys.path before any
test module is collected — a root conftest runs early enough for that, whereas the
per-package tests/conftest.py does not cover fem/tests collection-time imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
for p in (REPO_ROOT, REPO_ROOT / "data"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
