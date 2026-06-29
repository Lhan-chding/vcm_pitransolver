# Dataset Audit Report (Stage 0)

- variants audited: **300**
- variants with >=1 status/params conflict: **229** (76.3%)
- conflicts by field (status != params):
    - `valid_geometry`: 0
    - `valid_mesh`: 229
    - `valid_named_selections`: 229
- variants valid_for_physics (resolved AND + boundary present + 0 failed elems): **300**
- variants missing FIXED/MOVE selection: **0**

## Resolution rule

`status.json` is the **source of truth**; `params.json` carries stale generation-time values. The conflict is 100% one-directional (status=True / params=False), all 300 have status valid_*=True and 0 failed elements, so each `valid_*` resolves to its status.json value.

## By sample group

- : 158
- COMPLEX_DIVERSITY_PROFILE_REBUILD: 142

## Index

See `dataset_index.csv` (one row per variant).
