# Full-dataset health audit findings (300 variants)

Date: 2026-06-30. Source: `data/audit_health_300.py` streaming the dataset zip.
Audits use the SAME quadrature / merge logic the training & energy code uses.

## Headline

| audit | pass | of |
|---|---|---|
| element type (only tet10/wedge15/hex20, no unknown) | **300** | 300 |
| detJ (all elements, all Gauss points, >0 & not near-zero) | **300** | 300 |
| connectivity (single component after node-merge) | **293** | 300 |
| boundary (fixed/move/free reconstructed, 4 corners, no overlap) | **300** | 300 |
| **fully trainable (pass all four)** | **293** | 300 |
| audit errors | 0 | — |

Element family totals across all variants: tet10 4,847,272 (52.0%), hex20
4,361,731 (46.8%), wedge15 106,420 (1.1%). detJ checked on **9.3M elements** ×
their full Gauss rules — **zero** negative or near-zero Jacobians.

## The 7 non-trainable variants — a real dataset defect, not a tolerance bug

Seven STRICT variants stay multi-component after node merging and are correctly
flagged `connectivity_ok=False` → excluded from training:

| variant | comps after merge | merged nodes | failure |
|---|---|---|---|
| VCM_STRICT_0076_RPAIR_IN0p1655_OUT0p2545_DN010 | 5 | 0 | all 4 corner pads detached |
| VCM_STRICT_0122_RPAIR_IN0p2805_OUT0p3695_DP025 | 5 | 0 | all 4 corner pads detached |
| VCM_STRICT_0123_RPAIR_IN0p2805_OUT0p3695_DP030 | 5 | 0 | all 4 corner pads detached |
| VCM_STRICT_0174_WIDTH_DELTA_N040UM | 2 | 76 | 1 of 4 pads detached |
| VCM_STRICT_0175_WIDTH_DELTA_N030UM | 2 | 76 | 1 of 4 pads detached |
| VCM_STRICT_0213_SERP_COUNT_P1_PROFILE_REBUILD | 3 | 22 | 2 pads detached |
| VCM_STRICT_0214_SERP_COUNT_N1_PROFILE_REBUILD | 3 | 22 | 2 pads detached |

**Root cause (measured):** the detached corner pads sit a real **0.2055 mm** away
from the main body — three orders of magnitude past the 1e-6 mm merge tolerance.
This is a genuine geometric gap from the parametric rebuild, not unmerged
coincident nodes. A flexure whose fixed supports float 0.2 mm off the body has no
load path; solving it would reproduce the zero-energy-mechanism failure. The
correct action is exclusion (done), NOT loosening the tolerance — bonding a 0.2 mm
gap would fabricate a support that isn't in the geometry.

The other 293 variants merge to a single connected solid (the export's standard
~44 coincident interface nodes per variant) and are safe to train on.

## Outputs
- `element_type_audit_300.csv`, `detj_audit_300.csv`,
  `node_merge_audit_300.csv`, `boundary_node_audit_300.csv` (per-variant rows)
- `boundary_visualization/` — fixed/move/free figures for a design-diverse sample
- Reproduce: `python data/audit_health_300.py`
