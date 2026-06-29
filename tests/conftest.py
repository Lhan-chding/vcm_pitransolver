"""Shared pytest fixtures and path setup for the Stage-0 test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# make `data/` modules importable as top-level (parse_mesh, parse_face_to_nodes)
sys.path.insert(0, str(REPO_ROOT / "data"))

# Real-data integration tests need an extracted variant. Not committed (multi-GB),
# so tests using this fixture skip cleanly in CI.
DEV_VARIANT = REPO_ROOT / "_devdata" / "VCM_COMPLEX_0001"


@pytest.fixture
def dev_variant() -> Path:
    if not (DEV_VARIANT / "nodes.csv").exists():
        pytest.skip(f"dev data not present at {DEV_VARIANT} (extract a variant to run)")
    return DEV_VARIANT
