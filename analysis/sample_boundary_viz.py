"""sample_boundary_viz.py — render fixed/move/free node sets for representative variants.

Picks a small, design-diverse sample (baseline + parameter extremes: thickness,
width, effective length, R-angle, serpentine count, plus a few complex-diversity
variants) straight from the dataset zip, reconstructs each one's boundary node
sets, and renders the labeled point cloud (free/fixed/move) so the boundary
mapping can be eyeballed across the design space — not just on variant 0001.

Reuses analysis.plot_bc_check.plot for the actual rendering (one figure style).
Selection is driven by params.json design deltas, so the extremes are real
geometric extremes, not arbitrary indices.

Usage:
    python analysis/sample_boundary_viz.py [zip_path] [--out reports/boundary_visualization]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

from parse_face_to_nodes import _load_named_selections, reconstruct_boundary  # noqa: E402
from parse_mesh import load_mesh  # noqa: E402
from analysis.plot_bc_check import plot  # noqa: E402

_NEEDED = ("nodes.csv", "elements.csv", "named_selections.json", "params.json")


def _variant_ids(zf: zipfile.ZipFile) -> list[str]:
    return sorted(set(
        re.match(r"variants/([^/]+)/", n).group(1)
        for n in zf.namelist() if n.startswith("variants/")
    ))


def _params(zf: zipfile.ZipFile, vid: str) -> dict:
    with zf.open(f"variants/{vid}/params.json") as fh:
        return json.load(io.TextIOWrapper(fh, encoding="utf-8"))


def select_representative(zf: zipfile.ZipFile, vids: list[str]) -> dict[str, str]:
    """Return {label: variant_id} covering baseline + parameter extremes.

    Extremes are taken from params.json design deltas; a label maps to the
    variant achieving that extreme. Missing params just skip that label.
    """
    rows = []
    for vid in vids:
        p = _params(zf, vid)
        rows.append((vid, p))

    def extreme(key: str, want_max: bool) -> str | None:
        vals = [(p.get(key), vid) for vid, p in rows if isinstance(p.get(key), (int, float))]
        if not vals:
            return None
        vals.sort()
        return vals[-1][1] if want_max else vals[0][1]

    def near_zero(key: str) -> str | None:
        """A 'baseline-ish' variant: smallest |delta| on the given key."""
        vals = [(abs(p.get(key, 1e9)), vid) for vid, p in rows
                if isinstance(p.get(key), (int, float))]
        if not vals:
            return None
        vals.sort()
        return vals[0][1]

    picks: dict[str, str] = {}
    # baseline: smallest effective-length delta (closest to nominal geometry)
    b = near_zero("effective_length_delta_mm")
    if b:
        picks["baseline"] = b
    for label, key, want_max in [
        ("thickness_min", "beam_line_width_delta_mm", False),   # line width ~ thickness proxy
        ("thickness_max", "beam_line_width_delta_mm", True),
        ("eff_length_min", "effective_length_delta_mm", False),
        ("eff_length_max", "effective_length_delta_mm", True),
        ("r_angle_min", "r_angle_delta_mm", False),
        ("r_angle_max", "r_angle_delta_mm", True),
        ("serpentine_min", "serpentine_path_length_delta_mm", False),
        ("serpentine_max", "serpentine_path_length_delta_mm", True),
    ]:
        v = extreme(key, want_max)
        if v:
            picks[label] = v

    # a couple of complex-diversity variants for raw geometric variety
    complex_vids = [vid for vid, p in rows
                    if "COMPLEX" in vid and vid not in picks.values()]
    for i, vid in enumerate(complex_vids[:3]):
        picks[f"complex_{i+1}"] = vid

    return picks


def _extract(zf: zipfile.ZipFile, vid: str, dest: Path) -> None:
    for fname in _NEEDED:
        try:
            with zf.open(f"variants/{vid}/{fname}") as src, open(dest / fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
        except KeyError:
            pass


def render_variant(zf: zipfile.ZipFile, label: str, vid: str, out_dir: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="vcm_viz_") as td:
        tdp = Path(td)
        _extract(zf, vid, tdp)
        mesh = load_mesh(tdp)
        ns = _load_named_selections(tdp)
        bs = reconstruct_boundary(mesh, ns)
        labels = np.zeros(mesh.num_nodes, dtype=int)
        labels[bs.fixed_nodes] = 1
        labels[bs.move_nodes] = 2
        out_png = out_dir / f"{label}__{vid}.png"
        plot(mesh.coords, labels, out_png, f"{label}: {vid}")
    return {"label": label, "variant_id": vid,
            "n_fixed": int(bs.fixed_nodes.size), "n_move": int(bs.move_nodes.size),
            "png": str(out_png)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip_path", nargs="?",
                    default=str(_REPO.parent / "vcm_spring_valid_300_variants.zip"))
    ap.add_argument("--out", default=str(_REPO / "reports" / "boundary_visualization"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path) as zf:
        vids = _variant_ids(zf)
        picks = select_representative(zf, vids)
        print(f"rendering {len(picks)} representative variants:")
        results = []
        for label, vid in picks.items():
            r = render_variant(zf, label, vid, out_dir)
            results.append(r)
            print(f"  {label:16s} {vid:50s} fixed={r['n_fixed']} move={r['n_move']}")

    # write an index so the sample set is documented
    with open(out_dir / "index.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nfigures + index.json -> {out_dir}")


if __name__ == "__main__":
    main()
