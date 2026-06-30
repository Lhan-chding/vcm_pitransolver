"""models — neural backbone wrappers and hard-BC enforcement for the predictor.

Modules:
  bc_enforce : hard Dirichlet via mask blending (no in-place, gradient-safe).
"""

from __future__ import annotations
