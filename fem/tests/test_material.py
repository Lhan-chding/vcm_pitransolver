"""Material loading + constitutive-matrix correctness.

Locks in three things the rest of the FEM engine depends on:
  1. config/material.yaml loads and the default (C1990) carries the right E/nu.
  2. The loaded E/nu are self-consistent with the file's *documented* derived
     moduli G and K_bulk (so a future edit to E/nu that desyncs the comment is
     caught), and the constitutive matrix C is symmetric positive-definite.
  3. The Material dataclass and the bare elastic_C() agree, and the error paths
     (unknown name, non-isotropic) fail loudly rather than silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem.material import Material, elastic_C, load_material

# C1990 spring-blade reference values, straight from config/material.yaml.
C1990_E_MPa = 127000.0
C1990_NU = 0.33
# Derived moduli the YAML documents (and claims match the ANSYS file to the
# quoted precision). G = E / (2(1+nu)); K_bulk = E / (3(1-2nu)).
C1990_G_MPa = 47744.36
C1990_K_BULK_MPa = 124509.80


def test_default_material_is_c1990():
    """The file's default must resolve to the copper-alloy spring blade C1990."""
    mat = load_material()  # no name -> use yaml `default`
    assert mat.name == "C1990"
    assert mat.E_MPa == pytest.approx(C1990_E_MPa)
    assert mat.nu == pytest.approx(C1990_NU)
    assert mat.density_kg_m3 == pytest.approx(8900.0)
    assert mat.tensile_yield_MPa == pytest.approx(1390.0)


def test_c1990_derived_moduli_match_documented_values():
    """E/nu must reproduce the G and K_bulk written in the YAML (self-consistency).

    This guards against an edit to E/nu that leaves the documented derived
    moduli stale — the failure the memory file warns about for material data.
    """
    mat = load_material("C1990")
    G = mat.E_MPa / (2.0 * (1.0 + mat.nu))
    K_bulk = mat.E_MPa / (3.0 * (1.0 - 2.0 * mat.nu))
    # YAML quotes G and K to 2 decimals; require agreement to that precision.
    assert G == pytest.approx(C1990_G_MPa, abs=5e-3)
    assert K_bulk == pytest.approx(C1990_K_BULK_MPa, abs=5e-3)


def test_c1990_constitutive_matrix_is_spd_and_symmetric():
    """C must be symmetric positive-definite for a physically valid material."""
    C = load_material("C1990").C()
    assert C.shape == (6, 6)
    assert np.allclose(C, C.T), "constitutive matrix must be symmetric"
    eigvals = np.linalg.eigvalsh(C)
    assert (eigvals > 0.0).all(), f"C not positive-definite; eigenvalues={eigvals}"


def test_dataclass_C_matches_bare_elastic_C():
    """Material.C() must be exactly elastic_C(E, nu) — no silent divergence."""
    mat = load_material("C1990")
    assert np.array_equal(mat.C(), elastic_C(mat.E_MPa, mat.nu))


def test_shear_diagonal_equals_shear_modulus():
    """Engineering-shear Voigt: shear diagonal of C must equal mu (== G), not 2*mu.

    Cross-checks the convention documented in material.py against the loaded
    material, so the B-matrix (which emits gamma) and C stay consistent.
    """
    mat = load_material("C1990")
    C = mat.C()
    mu = mat.E_MPa / (2.0 * (1.0 + mat.nu))
    for i in (3, 4, 5):
        assert C[i, i] == pytest.approx(mu)


def test_unknown_material_name_raises():
    with pytest.raises(KeyError):
        load_material("does-not-exist")


def test_non_isotropic_material_rejected(tmp_path):
    """A non-isotropic entry must be rejected — elastic_C is isotropic-only."""
    yaml_text = (
        "default: aniso\n"
        "materials:\n"
        "  aniso:\n"
        "    behavior: orthotropic\n"
        "    E_MPa: 100000.0\n"
        "    nu: 0.3\n"
    )
    bad = tmp_path / "material.yaml"
    bad.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match="isotropic"):
        load_material(yaml_path=bad)


def test_reference_material_without_yield_loads():
    """Materials lacking optional fields (e.g. Cu has no yield) load with None."""
    mat = load_material("Cu")
    assert mat.E_MPa == pytest.approx(75000.0)
    assert mat.tensile_yield_MPa is None
