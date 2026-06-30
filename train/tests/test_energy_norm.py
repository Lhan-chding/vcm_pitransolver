"""Per-variant energy normalization for multi-geometry training.

On the analytic single-hex (unit cube, nu=0, u_x = delta*x), the exact strain
energy is U = 1/2 * E * delta^2 (V=1, L=1). The three U_ref methods are checked
against this known answer: directfem makes L==1 exactly, stiffness with K_ref=E
matches, and the dimensional "scale" estimate lands in the right ballpark.
"""

from __future__ import annotations

import numpy as np
import pytest

from parse_mesh import Mesh
from fem.tests.reference_elements import hex20_reference
from train.energy_norm import (
    characteristic_length,
    energy_scale,
    mesh_volume,
    normalized_loss,
)

E, DELTA = 127000.0, 0.005
U_TRUE = 0.5 * E * DELTA * DELTA          # unit cube uniaxial: V=1, L=1


def _unit_cube_mesh() -> Mesh:
    coords = hex20_reference()             # [0,1]^3
    row = np.arange(20, dtype=np.int64)
    return Mesh(
        node_ids=np.arange(20, dtype=np.int64), coords=coords,
        elem_ids=np.array([0], dtype=np.int64), connectivity=[row],
        elem_nnode=np.array([20], dtype=np.int8), id_to_row={i: i for i in range(20)},
    )


def test_volume_and_length_of_unit_cube():
    mesh = _unit_cube_mesh()
    assert mesh_volume(mesh) == pytest.approx(1.0, rel=1e-9)
    assert characteristic_length(mesh) == pytest.approx(1.0, rel=1e-9)


def test_directfem_method_makes_loss_unity_at_true_solution():
    mesh = _unit_cube_mesh()
    sc = energy_scale(mesh, E_MPa=E, delta_mm=DELTA, method="directfem",
                      U_directfem=U_TRUE)
    assert sc.U_ref == pytest.approx(U_TRUE)
    assert normalized_loss(U_TRUE, sc) == pytest.approx(1.0)


def test_stiffness_method_matches_when_Kref_is_E():
    """For the unit cube K = E, so 1/2 K delta^2 == U_true."""
    mesh = _unit_cube_mesh()
    sc = energy_scale(mesh, E_MPa=E, delta_mm=DELTA, method="stiffness", K_ref=E)
    assert sc.U_ref == pytest.approx(U_TRUE)


def test_scale_method_right_order_of_magnitude():
    """Dimensional estimate: 1/2 E V (delta/L)^2 == U_true for the unit cube."""
    mesh = _unit_cube_mesh()
    sc = energy_scale(mesh, E_MPa=E, delta_mm=DELTA, method="scale")
    # V=1, L=1 here so it is exact; in general it is an order-of-magnitude proxy.
    assert sc.U_ref == pytest.approx(U_TRUE, rel=1e-9)
    assert sc.volume_mm3 == pytest.approx(1.0)
    assert sc.characteristic_length_mm == pytest.approx(1.0)


def test_scale_normalizes_disparate_variants_to_order_one():
    """Two cubes of very different size must both normalize to ~O(1) loss.

    A 10x larger cube stores ~10^3 more energy at the same strain; the scale
    reference must absorb that so neither dominates a multi-geometry batch.
    """
    small = _unit_cube_mesh()
    big_coords = hex20_reference() * 10.0          # 10 mm cube
    row = np.arange(20, dtype=np.int64)
    big = Mesh(np.arange(20, dtype=np.int64), big_coords, np.array([0]), [row],
               np.array([20], dtype=np.int8), {i: i for i in range(20)})

    # true energies scale very differently (V*(delta/L)^2): small ~ 1, big ~ V/L^2
    sc_s = energy_scale(small, E_MPa=E, delta_mm=DELTA, method="scale")
    sc_b = energy_scale(big, E_MPa=E, delta_mm=DELTA, method="scale")
    # the raw references differ a lot...
    assert sc_b.U_ref != pytest.approx(sc_s.U_ref, rel=0.1)
    # ...but a field at each one's own true energy normalizes to ~1 for both.
    Ls = normalized_loss(sc_s.U_ref, sc_s)
    Lb = normalized_loss(sc_b.U_ref, sc_b)
    assert Ls == pytest.approx(1.0) and Lb == pytest.approx(1.0)


def test_missing_required_args_raise():
    mesh = _unit_cube_mesh()
    with pytest.raises(ValueError, match="K_ref"):
        energy_scale(mesh, E_MPa=E, delta_mm=DELTA, method="stiffness")
    with pytest.raises(ValueError, match="U_directfem"):
        energy_scale(mesh, E_MPa=E, delta_mm=DELTA, method="directfem")
    with pytest.raises(ValueError, match="unknown method"):
        energy_scale(mesh, E_MPa=E, delta_mm=DELTA, method="bogus")
