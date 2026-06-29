"""
parse_mesh.py — Robust reader for VCM spring variant meshes.

Data contract (verified against real dataset, 2026-06-29):
  - nodes.csv     : header `node_id,x,y,z`, units = mm. node_id is NOT contiguous
                    and NOT zero-based -> we build an explicit id->row index map.
  - elements.csv  : header `element_id,node_ids`, node_ids is a quoted python-style
                    list of ints. NODE COUNT VARIES PER ELEMENT:
                        10 -> tet10   (quadratic tetrahedron)
                        15 -> wedge15 (quadratic prism / pentahedron)
                        20 -> hex20   (quadratic hexahedron)
                    The mix is variant-dependent (e.g. COMPLEX ~55% hex20 / ~44%
                    tet10; STRICT_0034 ~90% tet10 / ~10% hex20). There is NO
                    pure-tet10 shortcut: the FEM engine must support all three.
  - mesh.inp      : `*ELEMENT,TYPE=UNKNOWN`, NO *NSET/*ELSET. Useless for type or
                    boundary sets. We deliberately ignore it.

This module returns a single immutable `Mesh` object. It performs NO physics; it
only parses, validates, and indexes. Every assumption is asserted, not hoped for.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Supported quadratic element families, keyed by node count.
ELEM_TYPES = {10: "tet10", 15: "wedge15", 20: "hex20"}
SUPPORTED_NODE_COUNTS = frozenset(ELEM_TYPES)


@dataclass(frozen=True)
class Mesh:
    """Immutable parsed mesh.

    Elements are heterogeneous (tet10/wedge15/hex20), so connectivity cannot be a
    single rectangular array. We store, per element, a 0-based ROW-index tuple into
    `coords`, plus a parallel array of node counts so the FEM engine can dispatch by
    element family without re-parsing.
    """

    node_ids: np.ndarray            # (N,) int64  original mesh node ids (not contiguous)
    coords: np.ndarray              # (N,3) float64  coordinates in mm
    elem_ids: np.ndarray            # (E,) int64  original element ids
    connectivity: list             # length E; each item is np.ndarray ROW indices
    elem_nnode: np.ndarray          # (E,) int8  node count per element (10/15/20)
    id_to_row: dict                 # node_id (int) -> row index (int)

    @property
    def num_nodes(self) -> int:
        return self.coords.shape[0]

    @property
    def num_elements(self) -> int:
        return len(self.connectivity)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.coords.min(axis=0), self.coords.max(axis=0)

    def type_histogram(self) -> dict:
        """element family -> count."""
        out = {name: 0 for name in ELEM_TYPES.values()}
        vals, cnts = np.unique(self.elem_nnode, return_counts=True)
        for v, c in zip(vals, cnts):
            out[ELEM_TYPES[int(v)]] = int(c)
        return out

    def elements_of(self, nnode: int) -> np.ndarray:
        """Return a (M, nnode) int64 array of row-index connectivity for one family.

        This is the form the FEM engine consumes: one rectangular batch per family.
        """
        idx = np.nonzero(self.elem_nnode == nnode)[0]
        if idx.size == 0:
            return np.empty((0, nnode), dtype=np.int64)
        return np.stack([self.connectivity[i] for i in idx]).astype(np.int64)


def _read_nodes(nodes_csv: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    ids: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    with open(nodes_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header[:4] == ["node_id", "x", "y", "z"], (
            f"unexpected nodes.csv header: {header!r}"
        )
        for row in reader:
            if not row:
                continue
            ids.append(int(row[0]))
            xyz.append((float(row[1]), float(row[2]), float(row[3])))

    node_ids = np.asarray(ids, dtype=np.int64)
    coords = np.asarray(xyz, dtype=np.float64)

    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("duplicate node_id found in nodes.csv")

    id_to_row = {int(nid): i for i, nid in enumerate(node_ids)}
    return node_ids, coords, id_to_row


def _read_elements(
    elements_csv: Path, id_to_row: dict
) -> tuple[np.ndarray, list, np.ndarray]:
    elem_ids: list[int] = []
    connectivity: list = []
    nnodes: list[int] = []
    unsupported: dict[int, int] = {}

    with open(elements_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header[:2] == ["element_id", "node_ids"], (
            f"unexpected elements.csv header: {header!r}"
        )
        for row in reader:
            if not row:
                continue
            node_list = ast.literal_eval(row[1])
            k = len(node_list)
            if k not in SUPPORTED_NODE_COUNTS:
                unsupported[k] = unsupported.get(k, 0) + 1
                continue
            elem_ids.append(int(row[0]))
            connectivity.append(np.asarray([id_to_row[int(n)] for n in node_list], dtype=np.int64))
            nnodes.append(k)

    if unsupported:
        raise ValueError(
            f"unsupported element node-counts encountered: {unsupported}. "
            f"Supported: {sorted(SUPPORTED_NODE_COUNTS)} (tet10/wedge15/hex20)."
        )

    return np.asarray(elem_ids, dtype=np.int64), connectivity, np.asarray(nnodes, dtype=np.int8)


# Corner-swap that flips orientation, paired with the mid-edge index permutation that
# keeps each mid-edge node attached to the same physical edge after the corner swap.
# These restore a positive Jacobian when a family's winding is reversed vs. our
# reference convention. Verified against the real dataset (tet10 is reversed there:
# swapping corners 0<->1 takes all real tet10 from detJ<0 to detJ>0 AND passes the
# constant-strain patch test on real elements).
_ORIENT_FIX = {
    # tet10 C3D10: corners 0,1,2,3 ; edges 4(0-1)5(1-2)6(2-0)7(0-3)8(1-3)9(2-3)
    # swap corners 0<->1  => edge5(1-2)<->edge6(2-0), edge7(0-3)<->edge8(1-3)
    10: [1, 0, 2, 3, 4, 6, 5, 8, 7, 9],
    # wedge15 C3D15: swap the two triangle vertices 1<->2 on both faces.
    # corners b:0,1,2 t:3,4,5 ; edges 6(0-1)7(1-2)8(2-0) 9(3-4)10(4-5)11(5-3) vert 12,13,14
    15: [0, 2, 1, 3, 5, 4, 8, 7, 6, 11, 10, 9, 12, 14, 13],
    # hex20 C3D20: reflect bottom<->top to flip handedness.
    # corners b:0,1,2,3 t:4,5,6,7 ; edges 8-11(bottom)12-15(top)16-19(vertical)
    20: [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 16, 17, 18, 19],
}


def _element_detj_sign(coords: np.ndarray, row: np.ndarray, nnode: int) -> float:
    """detJ sign of an element, computed with its OWN FEM shape functions.

    This is the ground-truth orientation test: it uses the exact mapping the FEM
    engine will use, so there is no sign-convention to reconcile by hand. A
    negative value means the element is reversed relative to the engine's
    convention and must be repaired. Imports the element modules lazily so the
    parser has no hard dependency on fem/ when normalization is disabled.
    """
    # ensure the repo root (parent of data/) is importable even when this file is
    # run directly as a script (python data/parse_mesh.py), where only data/ is on
    # sys.path. pytest already puts the repo root on the path via conftest.
    import sys as _sys
    from pathlib import Path as _Path

    _root = str(_Path(__file__).resolve().parents[1])
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from fem.elements import tet10, wedge15, hex20  # lazy

    mod = {10: tet10, 15: wedge15, 20: hex20}[nnode]
    nat = mod.GAUSS[0][0]                      # evaluate at one Gauss point
    xyz = coords[row]
    J = xyz.T @ mod.shape_grads(nat)
    return float(np.linalg.det(J))


def _normalize_orientation(
    coords: np.ndarray, connectivity: list, elem_nnode: np.ndarray
) -> tuple[list, dict]:
    """Repair reversed-winding elements so every element has positive orientation.

    Uses each element's own FEM detJ as the orientation oracle. Robust to meshes
    that mix orientations. Returns (new_connectivity, per-family fix counts).
    """
    fixed = {10: 0, 15: 0, 20: 0}
    out = []
    for row, nn in zip(connectivity, elem_nnode):
        nn = int(nn)
        if _element_detj_sign(coords, row, nn) < 0.0:
            row = row[_ORIENT_FIX[nn]]
            fixed[nn] += 1
        out.append(row)
    return out, fixed


def load_mesh(variant_dir: str | Path, normalize_orientation: bool = True) -> Mesh:
    """Parse one variant directory into a validated, indexed Mesh.

    If normalize_orientation is True (default), reversed-winding elements are
    repaired so the FEM engine always receives positively-oriented elements
    (detJ > 0). The dataset's tet10 elements use the opposite winding from the
    standard C3D10 reference, so this is required for correct physics.
    """
    variant_dir = Path(variant_dir)
    nodes_csv = variant_dir / "nodes.csv"
    elements_csv = variant_dir / "elements.csv"
    if not nodes_csv.exists() or not elements_csv.exists():
        raise FileNotFoundError(f"missing nodes/elements in {variant_dir}")

    node_ids, coords, id_to_row = _read_nodes(nodes_csv)
    elem_ids, connectivity, elem_nnode = _read_elements(elements_csv, id_to_row)

    n = coords.shape[0]
    n_degenerate = 0
    for row in connectivity:
        if row.min() < 0 or row.max() >= n:
            raise ValueError("connectivity references a node row outside coords")
        if len(set(row.tolist())) != len(row):
            n_degenerate += 1
    if n_degenerate:
        raise ValueError(f"{n_degenerate} elements have repeated node ids (degenerate element)")

    if normalize_orientation:
        connectivity, _ = _normalize_orientation(coords, connectivity, elem_nnode)

    return Mesh(
        node_ids=node_ids,
        coords=coords,
        elem_ids=elem_ids,
        connectivity=connectivity,
        elem_nnode=elem_nnode,
        id_to_row=id_to_row,
    )


if __name__ == "__main__":
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "_devdata/VCM_COMPLEX_0001"
    m = load_mesh(d)
    lo, hi = m.bbox()
    print(f"variant dir         : {d}")
    print(f"num_nodes           : {m.num_nodes}")
    print(f"num_elements        : {m.num_elements}")
    hist = m.type_histogram()
    tot = m.num_elements
    print("element families    :")
    for name, c in hist.items():
        print(f"    {name:8s}: {c:7d}  ({100*c/tot:5.2f}%)")
    print(f"node_id range       : [{m.node_ids.min()}, {m.node_ids.max()}] "
          f"(contiguous={len(m.node_ids) == m.node_ids.max() - m.node_ids.min() + 1})")
    print(f"bbox x [mm]         : [{lo[0]:+.4f}, {hi[0]:+.4f}]  span={hi[0]-lo[0]:.4f}")
    print(f"bbox y [mm]         : [{lo[1]:+.4f}, {hi[1]:+.4f}]  span={hi[1]-lo[1]:.4f}")
    print(f"bbox z [mm]         : [{lo[2]:+.4f}, {hi[2]:+.4f}]  span={hi[2]-lo[2]:.4f}")
    thin_axis = int(np.argmin(hi - lo))
    print(f"thinnest axis       : {'XYZ'[thin_axis]}  (span={(hi-lo)[thin_axis]:.4f} mm)")
