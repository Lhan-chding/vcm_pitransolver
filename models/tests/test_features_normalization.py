"""test_features_normalization.py — pin the per-axis coords_net normalization.

The first energy-trained run overshot stiffness ~2.2e4x because build_node_inputs
normalized all coordinates by a SINGLE scalar (max bbox extent). On a VCM thin
plate (thickness:width ~1:190) that crushes the thin move axis to ~5e-3 in
coords_net, blinding the network to thickness-direction position. These tests
lock in the fix: each axis is normalized by its OWN extent so coords_net spans
~O(1) on every axis, while the isotropic distance features stay numerically
unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from parse_mesh import Mesh
from parse_face_to_nodes import BoundarySets
from models.features import build_node_inputs

E, NU, DELTA = 127000.0, 0.3, 0.005


def _thin_plate_coords(nx: int = 3, ny: int = 9, nz: int = 9,
                       tx: float = 0.06, wy: float = 11.42, wz: float = 11.42):
    """A thin-plate point cloud mimicking variant 0001: x thin, y/z wide."""
    xs = np.linspace(0.0, tx, nx)
    ys = np.linspace(0.0, wy, ny)
    zs = np.linspace(0.0, wz, nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float64)


def _dummy_mesh(coords: np.ndarray) -> Mesh:
    n = coords.shape[0]
    return Mesh(
        node_ids=np.arange(n, dtype=np.int64), coords=coords,
        elem_ids=np.array([0], dtype=np.int64), connectivity=[np.arange(min(n, 20))],
        elem_nnode=np.array([min(n, 20)], dtype=np.int8),
        id_to_row={i: i for i in range(n)},
    )


def _boundary(coords: np.ndarray) -> BoundarySets:
    """x=0 face fixed, x=max face move (x is the thin move axis, as in 0001)."""
    x = coords[:, 0]
    fixed = np.nonzero(np.isclose(x, x.min()))[0].astype(np.int64)
    move = np.nonzero(np.isclose(x, x.max()))[0].astype(np.int64)
    free = np.setdiff1d(np.arange(coords.shape[0], dtype=np.int64),
                        np.union1d(fixed, move))
    return BoundarySets(move_nodes=move, fixed_nodes=fixed, free_nodes=free,
                        move_face_descriptor={})


def test_coords_net_spans_order_one_on_every_axis_including_thin():
    """The core fix: the thin axis must NOT be crushed in coords_net.

    With per-axis normalization every axis spans ~[-0.5, 0.5]; the old single
    scalar left the thin x axis at ~5e-3, which is what this guards against.
    """
    coords = _thin_plate_coords()
    ni = build_node_inputs(coords, _boundary(coords), E_MPa=E, nu=NU, delta_mm=DELTA)
    cnet = ni.coords_net.cpu().numpy()
    span = cnet.max(0) - cnet.min(0)               # (3,) span per axis
    # every axis, including the thin move axis, spans ~1.0 (full [-0.5, 0.5])
    assert np.allclose(span, 1.0, atol=1e-9), f"per-axis span not ~1: {span}"
    # explicit anti-regression: thin axis is NOT ~5e-3 like the old scalar norm
    assert span[0] > 0.9, f"thin axis crushed in coords_net: span_x={span[0]}"


def test_scale_is_per_axis_extent_and_scale_iso_is_max():
    coords = _thin_plate_coords()
    ni = build_node_inputs(coords, _boundary(coords), E_MPa=E, nu=NU, delta_mm=DELTA)
    extent = coords.max(0) - coords.min(0)
    assert isinstance(ni.scale, np.ndarray) and ni.scale.shape == (3,)
    assert np.allclose(ni.scale, extent)
    # scale_iso == old scalar scale (max extent): distance features unchanged
    assert ni.scale_iso == pytest.approx(float(extent.max()))


def test_zero_extent_axis_is_guarded():
    """A degenerate axis (all nodes share one coordinate) must not divide by 0."""
    coords = _thin_plate_coords()
    coords[:, 2] = 3.0                              # collapse z to a single plane
    ni = build_node_inputs(coords, _boundary(coords), E_MPa=E, nu=NU, delta_mm=DELTA)
    cnet = ni.coords_net.cpu().numpy()
    assert np.all(np.isfinite(cnet)), "zero-extent axis produced non-finite coords_net"
    # collapsed axis maps to 0 (centered), guarded scale=1 -> no inf/nan
    assert np.allclose(cnet[:, 2], 0.0)
    assert ni.scale[2] == 1.0


def test_coords_phys_is_untouched_physical_mm():
    """coords_phys must remain the raw physical coordinates (energy depends on it)."""
    coords = _thin_plate_coords()
    ni = build_node_inputs(coords, _boundary(coords), E_MPa=E, nu=NU, delta_mm=DELTA)
    assert np.allclose(ni.coords_phys.cpu().numpy(), coords)


def test_distance_features_stay_order_one():
    """Isotropic distance features remain ~O(1) (divided by scale_iso, not vector)."""
    coords = _thin_plate_coords()
    ni = build_node_inputs(coords, _boundary(coords), E_MPa=E, nu=NU, delta_mm=DELTA)
    fx = ni.fx.cpu().numpy()
    d_fixed, d_move = fx[:, 3], fx[:, 4]            # columns 3,4 per the fx layout
    assert np.all(np.isfinite(d_fixed)) and np.all(np.isfinite(d_move))
    # normalized by max extent, so distances land within [0, ~sqrt(3)]
    assert d_fixed.max() <= 2.0 and d_move.max() <= 2.0
    assert d_fixed.min() >= 0.0 and d_move.min() >= 0.0
