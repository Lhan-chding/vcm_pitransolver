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


def load_mesh(variant_dir: str | Path) -> Mesh:
    """Parse one variant directory into a validated, indexed Mesh."""
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
