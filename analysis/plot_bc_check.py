"""
plot_bc_check.py — One-shot visual verification of reconstructed boundary node sets.

Renders the labeled point cloud (0=free, 1=fixed, 2=move) so a human can confirm:
  - MOVE nodes sit on the central ring inner wall,
  - FIXED nodes sit on the four corner top pads,
before any physics is computed on top of these sets.

Two render paths:
  - input CSV  : reports/bc_check_<variant>.csv  (x,y,z,label) from parse_face_to_nodes
  - OR a variant dir: runs the reconstruction live, then plots.

14k-143k nodes choke a 3D scatter, so free nodes are downsampled while fixed/move
(the sets we actually need to verify) are always drawn in full. Produces:
  - a 3-panel projection (Y-Z / X-Y / X-Z) — best for a thin plate,
  - an optional 3D view.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt  # noqa: E402

LABEL_COLORS = {0: ("#cfcfcf", "free"), 1: ("#1f77b4", "fixed"), 2: ("#d62728", "move")}


def _load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, :3], data[:, 3].astype(int)


def _load_from_variant(variant_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    # import the Stage-0 modules from data/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from parse_mesh import load_mesh
    from parse_face_to_nodes import _load_named_selections, reconstruct_boundary

    mesh = load_mesh(variant_dir)
    ns = _load_named_selections(variant_dir)
    bs = reconstruct_boundary(mesh, ns)
    label = np.zeros(mesh.num_nodes, dtype=int)
    label[bs.fixed_nodes] = 1
    label[bs.move_nodes] = 2
    return mesh.coords, label


def _downsample_free(coords, label, max_free):
    """Keep all fixed/move; randomly thin free nodes to max_free for legible plots."""
    free_idx = np.nonzero(label == 0)[0]
    keep_free = free_idx
    if free_idx.size > max_free:
        # deterministic stride sampling (no RNG -> reproducible figures)
        stride = int(np.ceil(free_idx.size / max_free))
        keep_free = free_idx[::stride]
    bc_idx = np.nonzero(label != 0)[0]
    keep = np.concatenate([keep_free, bc_idx])
    return coords[keep], label[keep]


def _scatter(ax, coords, label, ax1, ax2, names):
    for lab, (color, name) in LABEL_COLORS.items():
        m = label == lab
        if not np.any(m):
            continue
        # draw free first (background), bc on top: handled by call order below
        ax.scatter(coords[m, ax1], coords[m, ax2], s=(2 if lab == 0 else 6),
                   c=color, label=name, alpha=(0.25 if lab == 0 else 0.9),
                   edgecolors="none")
    ax.set_xlabel(names[ax1]); ax.set_ylabel(names[ax2])
    ax.set_aspect("equal", adjustable="datalim")


def plot(coords, label, out_png: Path, title: str, max_free: int = 8000,
         with_3d: bool = False) -> None:
    coords, label = _downsample_free(coords, label, max_free)
    names = ["x (mm)", "y (mm)", "z (mm)"]

    n_panels = 3 + (1 if with_3d else 0)
    fig = plt.figure(figsize=(5 * n_panels, 5))

    # Y-Z is the plate face (most informative for a thin-in-X plate)
    ax_yz = fig.add_subplot(1, n_panels, 1)
    _scatter(ax_yz, coords, label, 1, 2, names)
    ax_yz.set_title("Y-Z (plate face)")
    ax_yz.legend(loc="upper right", markerscale=2, framealpha=0.9)

    ax_xy = fig.add_subplot(1, n_panels, 2)
    _scatter(ax_xy, coords, label, 0, 1, names)
    ax_xy.set_title("X-Y (thickness vs y)")

    ax_xz = fig.add_subplot(1, n_panels, 3)
    _scatter(ax_xz, coords, label, 0, 2, names)
    ax_xz.set_title("X-Z (thickness vs z)")

    if with_3d:
        ax3d = fig.add_subplot(1, n_panels, 4, projection="3d")
        for lab, (color, name) in LABEL_COLORS.items():
            m = label == lab
            if not np.any(m):
                continue
            ax3d.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                         s=(1 if lab == 0 else 6), c=color,
                         alpha=(0.15 if lab == 0 else 0.9), edgecolors="none")
        ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
        ax3d.set_title("3D")

    n_fixed = int(np.sum(label == 1))
    n_move = int(np.sum(label == 2))
    fig.suptitle(f"{title}   |   fixed={n_fixed}  move={n_move}  "
                 f"(free downsampled for display)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("usage: python analysis/plot_bc_check.py <bc_check.csv | variant_dir> "
              "[out.png] [--3d]")
        raise SystemExit(2)

    src = Path(argv[1])
    with_3d = "--3d" in argv
    out = None
    for a in argv[2:]:
        if not a.startswith("--"):
            out = Path(a)

    if src.is_dir():
        coords, label = _load_from_variant(src)
        title = src.name
        out = out or Path("reports") / f"bc_check_{src.name}.png"
    else:
        coords, label = _load_csv(src)
        title = src.stem
        out = out or src.with_suffix(".png")

    plot(coords, label, out, title, with_3d=with_3d)
    print(f"wrote {out}")
    print(f"  fixed={int(np.sum(label==1))}  move={int(np.sum(label==2))}  "
          f"free={int(np.sum(label==0))}")
    print("  CHECK: move=ring inner wall (a vertical strip), fixed=4 corner top pads")


if __name__ == "__main__":
    main(sys.argv)
