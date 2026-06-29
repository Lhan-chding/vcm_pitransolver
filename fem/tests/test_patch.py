"""Constant-strain patch test for every quadratic element family.

This is the gold-standard correctness proof: a linear displacement field must
reproduce a constant strain/stress and the exact strain energy. If mid-edge node
ordering, shape functions, the B-matrix or the Voigt convention were inconsistent,
B @ u_e would not be constant and these tests would fail.
"""

from __future__ import annotations

import pytest

from fem.tests.patch_common import run_patch_test
from fem.tests.reference_elements import REFERENCES


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_constant_strain_patch(name):
    module, builder = REFERENCES[name]
    node_xyz = builder()
    run_patch_test(module, node_xyz)
