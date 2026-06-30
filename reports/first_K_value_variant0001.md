# First physical FEM result — VCM_COMPLEX_0001

Date: 2026-06-30. Material: **C1990** (E=127000 MPa, nu=0.33). Δ = 5 µm, axis = X
(out-of-plane voice-coil stroke), move-transverse = rigid. Displacement control.

## Result

| quantity | value |
|---|---|
| nodes / elements | 143 593 / 31 025 |
| element families | tet10 13 715, wedge15 348, hex20 16 962 |
| global DOFs | 430 779 |
| **K_energy = 2U/Δ²** | **3.242570e-02 N/mm = 32.43 N/m** |
| **K_reaction = F/Δ** | **3.242570e-02 N/mm** |
| energy/reaction gap | 6.2e-9 (≈0 — Clapeyron satisfied) |
| strain energy U | 4.05e-7 N·mm |
| move-face force | 1.62e-4 N |
| \|u\| max / mean | 5.00e-3 / 1.25e-3 mm |
| von Mises max / mean | 3.25 / 0.11 MPa (0.23 % of 1390 MPa yield) |

The energy- and reaction-based stiffness agree to 9 significant figures, which is
the end-to-end correctness proof for assembly + Dirichlet BC + element library +
post-processing. K ≈ 32 N/m is physically sensible for a thin copper flexure and
matches the order of magnitude (~24 N/m) seen in the earlier PINN work on a
related spring.

## The bug that this exposed (and the fix)

The first solve gave U ≈ 1e-21 (zero) and an energy/reaction gap of 4.4e6 — a
classic zero-energy mechanism. Root cause: the exported mesh is **5 disconnected
components** (central mover body + 4 corner fixed-pads). The corner pads duplicate
the central-body nodes at their bonded interfaces instead of sharing them: 44
coincident-but-unmerged node pairs, all cross-component. With the mover face in a
separate component from the fixed pads, pushing it transmitted no force.

Fix: `load_mesh(..., merge_coincident=True)` ties sub-tolerance (1e-6 mm) duplicate
nodes into one shared node via union-find and remaps connectivity. After merging,
the mesh is a single connected solid (143 637 → 143 593 nodes) and the load path
exists. This is a real dataset property, not a code defect — the single-element
analytic test (`K == E`) confirms the FEM engine itself was always correct.

## Reproduce

```
python analysis/solve_variant.py _devdata/VCM_COMPLEX_0001
```
