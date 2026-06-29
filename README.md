# VCM Spring — Physics-Informed Transolver

Physics-informed structural-response prediction for VCM spring upper geometries.
Predicts **displacement / stress / equivalent stiffness K** for a *new* spring
geometry **without large supervised (ANSYS) label sets**, using
PhysicsNeMo Transolver / Transolver++ as the backbone and a custom finite-element
elastic-energy loss as the physics signal.

> Design rationale and full plan: `docs/design_plan_v1.md`
> (architecture, physics, module breakdown, stage gates, risk register).

## Architecture in one picture

```
ANSYS (5–20 anchors, calibrate)
   → direct-FEM (scipy Ku=f, pseudo-labels + verification oracle)
      → Transolver / Transolver++ (ms-latency predictor; trained by FEM energy loss)
         → deploy: new geometry → u / stress / K (ms), optional test-time energy refine (s)
```

- **Transolver = the predictor** you deploy.
- **FEM = the physics engine**, used twice:
  1. its tet10/wedge15/hex20 `B` matrices + elastic energy `U` are the *training loss*
     (this is what "physics-informed" means here — no labels needed);
  2. the same `B/C` assemble `Ku=f` for a **direct-FEM solver** that produces
     pseudo-labels and a verification oracle (solves the "no labels = no way to
     validate" problem).

## Verified dataset facts (drive every design choice)

| fact | value |
|---|---|
| variants | 300 valid (158 `VCM_STRICT_*` + 142 `VCM_COMPLEX_*`) |
| units | mm |
| geometry | thin plate, **X-thickness ≈ 0.06 mm**, lies in Y–Z plane |
| **mesh** | **mixed quadratic**: hex20 + tet10 + wedge15 (mix is variant-dependent) |
| boundary defs | STEP **face IDs** in `named_selections.json` — **NOT node sets**; mesh has no `*NSET`/`.cdb` → boundary node sets must be reconstructed geometrically |
| labels | none yet (we generate direct-FEM pseudo-labels; optional 5–20 ANSYS anchors) |

> ⚠️ The "all tet10" assumption from early sampling was **wrong**; the mesh is mixed
> quadratic. The FEM engine must support all three families.

## Repository layout

```
data/    parse_mesh.py            mixed quadratic mesh reader (tet10/wedge15/hex20)
         parse_face_to_nodes.py   reconstruct fixed/move node sets from face geometry
         audit_dataset.py         (Stage 0) status-vs-params conflict audit + index
fem/     tet10.py wedge15.py hex20.py   quadratic shape fns, B matrices, Gauss rules
         material.py assembly.py direct_solver.py energy.py postprocess.py
         tests/    patch test, cantilever (locking), torch==numpy energy consistency
models/  transolver.py bc_enforce.py features.py     PhysicsNeMo backbone + hard BC
train/   train_single.py train_operator.py refine.py losses.py
analysis/ slice_maps.py sensitivity.py rank_candidates.py
deploy/  predict.py geometry_to_features.py
config/  material.yaml physics.yaml model_transolver.yaml stage_*.yaml
```

## Quickstart (Stage 0, CPU only)

```bash
pip install -r requirements.txt
# extract one variant's nodes.csv/elements.csv/named_selections.json into a dir, then:
python data/parse_mesh.py           <variant_dir>   # element-type histogram + bbox
python data/parse_face_to_nodes.py  <variant_dir>   # reconstruct boundary node sets
#   -> writes reports/bc_check_<variant>.csv  (x,y,z,label) for visual sign-off
```

## Training environment (A800 server)

Runs inside the user's **PhysicsNeMo Docker**; do not reinstall torch in-container:

```
PhysicsNeMo 1.2.0 · PhysicsNeMo-Sym 2.2.0
PyTorch 2.8.0a0+...nv25.06 · CUDA 12.9 · cuDNN 91002
GPU: NVIDIA A800 80GB PCIe
```

Stage 0 + direct-FEM run on plain CPU (numpy/scipy). Transolver training (Stage 1+)
runs in the container on the A800.

## Status

- [x] Stage 0: mixed-quadratic mesh parser (validated on real variants)
- [x] Stage 0: boundary face→node reconstruction (area self-check < 5% on 0001)
- [ ] Stage 0: dataset audit (status vs params conflict) + full index
- [ ] Stage 1: quadratic FEM engine + direct-FEM solver + patch/cantilever tests
- [ ] Stage 2–4: Transolver operator training
- [ ] Stage 5: sensitivity analysis + deployment
```
