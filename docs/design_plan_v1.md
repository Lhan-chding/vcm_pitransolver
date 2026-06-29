# VCM Spring — Physics-Informed Transolver 结构响应预测系统：高精度详细设计方案

**版本**：v1.0（设计基线）
**日期**：2026-06-29
**作者角色**：算法/结构力学/AI4Science 联合设计
**状态**：架构已锁定，待按 milestone 实现

---

## ⚠️ 勘误（Stage 0 实测，2026-06-29，优先级高于正文）

正文部分基于"全是 tet10"的早期判断（来自只抽样 elements.csv 头几行）。**Stage 0 用 `parse_mesh.py` 全量解析后证实该判断错误**，以下结论以此勘误为准：

1. **网格是混合二次网格，不是纯 tet10**：每个单元节点数为
   - `10` → **tet10**（二次四面体）
   - `15` → **wedge15**（二次三棱柱）
   - `20` → **hex20**（二次六面体）

   比例随 variant 变化：`VCM_COMPLEX_0001` ≈ hex20 54.7% / tet10 44.2% / wedge15 1.1%；
   `VCM_STRICT_0034` ≈ tet10 89.9% / hex20 9.7% / wedge15 0.4%。
   **后果**：FEM 引擎必须实现三种二次单元的形函数/B 矩阵/高斯积分，没有"只做 tet10"
   的捷径。正文中所有写 `tet10.py` / "确认全 tet10" 的地方，应理解为 **`fem/elements/`
   下的 tet10 + wedge15 + hex20 三件套**。每种单元的高斯积分阶：tet10→4点，wedge15→6点，
   hex20→27点（或 14 点缩减积分，需对照 patch test 选定）。

2. **边界确认为几何重建路径（无现成节点集）**：`mesh.inp` 仅含 `*NODE`/`*ELEMENT`，
   **无 `*NSET`/`*ELSET`**；数据集**无 `.cdb`**。因此"先读网格 component"在本数据集走不通，
   `parse_face_to_nodes.py` 用 `named_selections.debug.move_candidates_top5` 的几何特征
   （中心/法向/面积）重建节点集，并用**重建面积 vs 报告面积**自检（variant 0001 误差 4.8%）。

3. **move face 法向在面内（Y-Z），不在 X**：variant 0001 的 move face 法向 `(ny,nz)=(0,-1)`，
   是中心环内壁的薄条；这使"move 方向 = X（面外）"成为**必须用户确认**的物理假设（见 §11）。

正文其余（物理原理、两阶段架构、direct-FEM 伪标签、风险表）不受影响。

---

## 0. 本文档的定位

本文档是把"利用 300 个 VCM 弹簧几何 → 训练一个 physics-informed Transolver → 部署时对新几何毫秒级预测 u/应力/K → 分析刚度/应力敏感区"这一目标，落到**可实现、可验证、可部署**的工程级方案。

与上一版 review_plan 的关键区别（基于数据集实测修正）：

| 维度 | 旧 plan 假设 | 实测真相 | 本方案处理 |
|---|---|---|---|
| 网格单元 | 可能 tet4/tet10，先 linearize 成 tet4 | **确认全是 tet10（二次四面体，10 节点/单元）** | 直接实现 tet10，**不降阶** |
| 边界条件 | 有节点级 `fixed_mask/move_mask` | **named_selections 给的是 STEP face ID，不是节点；SPRING_BODY/FREE_SURFACES 为空** | Stage 0 必须做 **face→node 几何重建** |
| 几何形态 | 一般 3D 实体 | **X 向仅 0.06mm 厚的薄片，躺在 Y-Z 平面** | move 方向/Δ 量级成硬约束，需用户确认 |
| 数据完整性 | status 字段可信 | **status.json 与 params.json 的 valid_* 字段互相矛盾** | Stage 0 全量交叉审计 |
| 单位 | 未明确 | **明确 mm（measurement.json 用 `*_mm`）** | 全链路 mm-N-MPa |
| 标签 | 纯无标签 | — | **direct-FEM 伪标签 + 5–20 ANSYS 锚点**（已确认） |
| 部署形态 | 未定 | — | **两阶段：amortized 前向 + 可选 test-time energy refine**（已确认） |

---

## 1. 总体原理：三个角色的分工

整个系统由三个角色构成，**必须分清谁是谁**：

```
ANSYS (5–20个，最准，最贵)
        │ 校准 direct-FEM 的系统偏差（接触/非线性等线弹性算不准的效应）
        ▼
direct-FEM (本机 scipy.sparse spsolve，秒级，线弹性精确解)
        │ ① 提供"伪标准答案"→ 让无标签的 Transolver 终于可被验证
        │ ② warm-start / anchor → 加速并稳定 Transolver 训练
        ▼
Transolver / Transolver++ (神经网络，毫秒级前向)
        │ 这是你最终部署的"快速预测器"
        │ 训练信号 = FEM 弹性能量损失（physics-informed，无需大规模标签）
        ▼
部署：新几何 → 前向 u/应力/K（毫秒）→ 可选 test-time energy refine（几秒，精度更高）
```

**关键认知（务必内化）**：

- **Transolver = 预测器**（你要的模型本体）。
- **FEM = 物理引擎**，有两个用途：
  - 用途 A：FEM 的应变-位移矩阵 `B` + 弹性能公式 `U`，**就是 Transolver 的 physics-informed 损失函数**。没有 FEM，"physics-informed" 无从谈起。
  - 用途 B：同一套 `B/C` 组装 `Ku=f` 直接求解 → direct-FEM 伪标签。
- **同一套 tet10 的 `B` 矩阵，被 direct-FEM 和 Transolver 损失共用**。这就是为什么自写 NumPy/SciPy（而非黑箱 FEM 库）：保证两边用的是**同一套数学**，且 `B` 可被 PyTorch 自动微分。

---

## 2. 物理基础（小变形线弹性）

### 2.1 控制方程

未知量：位移场 `u(x) = [u_x, u_y, u_z]`。

```
应变-位移：  ε(u) = ½(∇u + ∇uᵀ)              （6 分量 Voigt 记法）
本构：       σ = C(E,ν) : ε                    （各向同性线弹性）
平衡：       ∇·σ = 0                           （无体力）
```

### 2.2 最小势能原理（无标签训练的理论根基）

总势能泛函：
```
Π(u) = U_internal(u) − W_external(u)
U_internal = ½ ∫_Ω ε:σ dΩ = ½ ∫_Ω εᵀ C ε dΩ
W_external = ∫_Γt t·u dΓ + ∫_Ω b·u dΩ
```

**真实位移场 = 在满足 Dirichlet 边界（fixed/move）的所有容许场中，使 Π 最小的那个场。**

⚠️ **决定成败的前提（必须焊死在代码里）**：
> 本问题是**纯位移控制**（move face 施加 Δ，无外加 traction、无体力）。此时 `W_external = 0`，于是 `Π = U_internal`，**最小化 U 才等价于平衡**。

```python
# 代码必须有此断言
assert loading_mode == "displacement_control", \
    "min(U) 仅在纯位移控制下等价于平衡；若改力加载，必须用 min(U - W)，否则 u≡0 坍缩"
```

这呼应已知教训：位移加载优于牵引力加载；力加载下能量会坍缩到零位移。

### 2.3 等效刚度 K 的两种定义

```
K_energy   = 2U / Δ²                            （全局能量比，对局部错误不敏感）
K_reaction = R_axis / Δ                          （反力/位移，更接近 ANSYS 报告值）
R_axis = Σ_{fixed nodes} f_internal,axis         （fixed face 上内力沿 move 轴求和）
```

**两者之差 `|K_energy − K_reaction| / K` 是最强的无标签收敛诊断量**——精确解下两者相等，不一致即说明位移场没收敛对。**Stage 1 验收必须同时算这两个并报告差异**（不要像旧 plan 那样把 reaction 推到第二阶段）。

---

## 3. 数据集事实（实测，作为所有模块的输入契约）

每个 variant 目录（共 300 个，`VCM_STRICT_*` 158 + `VCM_COMPLEX_*` 142）：

```
geometry.step               STEP 几何（face ID 的来源）
geometry_native_file.scdocx SpaceClaim 原生
mesh.inp                    Abaqus 格式网格，但 *ELEMENT,TYPE=UNKNOWN（类型不可信，须数节点）
nodes.csv                   node_id, x, y, z（单位 mm）
elements.csv                element_id, "[10个 node_id]"（tet10，二次四面体）
named_selections.json       face ID + debug 几何特征（不是节点集！）
params.json                 设计参数 + valid_* 字段（可能与 status 冲突）
measurement.json            bbox_mm、loop_count、各 body 的 face/edge count
mesh_quality.json           num_nodes, num_elements, failed_elements_count
status.json                 valid_geometry/valid_mesh/valid_named_selections（与 params 可能冲突）
preview.png / preview_metadata.json
```

实测关键数字（variant 0001）：
- `num_nodes = 143637`, `num_elements = 31025` → 比值 4.63，**确认 tet10**。
- bbox：`x∈[-0.022, 0.042]`(span 0.06mm，厚度方向)，`y,z∈[-5.71, 5.71]`(span 11.42mm)。
- `named_selections.MOVE_INNER_RING_FACE = [11566]`（单个 face ID），`debug.move_candidates_top5` 含该 face 的中心 `cy/cz`、法向 `ny/nz`、面积 `area`。
- `status.valid_mesh=true` vs `params.valid_mesh=false` → **冲突**。

---

## 4. 系统架构与代码组织

```
vcm_pitransolver/
├── config/
│   ├── material.yaml              # E, ν, 单位（须用户确认弹片材料）
│   ├── physics.yaml               # move_axis, Δ, loading_mode
│   ├── model_transolver.yaml      # backbone 超参
│   └── stage_{0..5}.yaml
├── data/
│   ├── parse_mesh.py              # nodes/elements.csv → tensor，识别 tet10
│   ├── parse_face_to_nodes.py     # ★ STEP face ID → 节点集合（几何重建）
│   ├── audit_dataset.py           # status vs params 冲突审计，生成 dataset_index
│   └── dataset.py                 # PyTorch Dataset，padded batching + node_mask
├── fem/                           # ★ 物理引擎（NumPy/SciPy + torch 双实现）
│   ├── tet10.py                   # tet10 形函数、Jacobian、B 矩阵、4点高斯
│   ├── material.py                # C(E,ν) 本构矩阵
│   ├── assembly.py                # 全局 K 组装（scipy.sparse）
│   ├── direct_solver.py           # ★ Ku=f 直接求解 → 伪标签
│   ├── energy.py                  # ★ torch 版弹性能 U（Transolver 损失用，可微）
│   ├── postprocess.py             # ε, σ, von Mises, K_energy, K_reaction, 反力
│   └── tests/
│       ├── test_patch.py          # ★ 单 tet10 常应变 patch test
│       ├── test_cantilever.py     # ★ 悬臂梁解析对照（量化 locking）
│       └── test_energy_consistency.py  # torch energy == numpy energy
├── models/
│   ├── transolver.py              # PhysicsNeMo Transolver/Transolver++ wrapper
│   ├── bc_enforce.py              # ★ 硬 Dirichlet（mask blending，非 in-place）
│   └── features.py                # 节点特征 + design token 构造
├── train/
│   ├── losses.py                  # L_energy + (可选) L_anchor + 诊断量
│   ├── train_single.py            # Stage 1：单几何
│   ├── train_operator.py          # Stage 2/3/4：多几何 amortized
│   └── refine.py                  # ★ test-time energy refine（部署可选第二阶段）
├── analysis/
│   ├── slice_maps.py              # Transolver slice 可视化
│   ├── sensitivity.py             # K/应力 对设计参数/区域的梯度敏感性
│   └── rank_candidates.py         # Pareto / 推荐 ANSYS 验证子集
├── deploy/
│   ├── predict.py                 # ★ 新几何 → u/应力/K（部署入口）
│   └── geometry_to_features.py    # 新几何 → mesh+BC+material features
└── reports/                       # 各 stage 自动生成的 md/csv/png
```

`★` = 高风险或高价值、需重点实现与评审的模块。

---

## 5. 模块详细设计

### 5.1 `fem/tet10.py` — tet10 单元（地基中的地基）

**二次四面体（10 节点）**：4 个角节点 + 6 个边中点。形函数用体积坐标 `(L1,L2,L3,L4)`，`L4 = 1−L1−L2−L3`：

```
角节点 i：     N_i = L_i (2L_i − 1)            i=1..4
边中点 (ij)：  N_{ij} = 4 L_i L_j
```

实现要点（**易错点，逐条核验**）：

1. **B 矩阵随位置变化**：tet10 是二次单元，应变在单元内**不是常数**，`B = B(ξ)` 依赖高斯点位置。**不能像 tet4 那样用单个常 B**。
2. **数值积分**：tet10 弹性能必须用 **4 点高斯积分**（精度阶 2），单点积分会欠积分、刚度奇异。每个高斯点：算 Jacobian `J`、`det(J)`、`B(ξ_g)`，累加 `wᵍ·det(J)·BᵀCB`。
3. **Jacobian**：`J = ∂x/∂ξ`，由 `∂N/∂ξ` 和节点坐标算。`det(J)>0` 必须断言（负 = 单元翻转/编号错）。
4. **单元刚度阵**：`K_e = Σ_g w_g · det(J_g) · B_gᵀ C B_g`，尺寸 30×30（10 节点 × 3 自由度）。

接口：
```python
def tet10_shape_grads(natural_coords) -> dN_dxi          # (10,3) @ 一个高斯点
def tet10_jacobian(node_xyz, dN_dxi) -> (J, detJ, dN_dx) # 物理梯度
def tet10_B_matrix(dN_dx) -> B                            # (6,30) Voigt
def tet10_element_stiffness(node_xyz, C) -> K_e           # (30,30)，4点高斯
GAUSS_4PT = [...]  # (4,4) 体积坐标 + 权重，硬编码常量
```

### 5.2 `fem/direct_solver.py` — 伪标签求解器

```
1. 组装全局 K（scipy.sparse.lil → csr），自由度 = 3·num_nodes（~43万）
2. 施加 Dirichlet：fixed 节点 u=0，move 节点沿 move_axis = Δ
   用罚函数法或行列消去法（推荐消去法，更准）
3. spsolve(K_reduced, f_reduced) → u
4. 后处理：每单元 ε,σ,von Mises；K_energy, K_reaction
```

性能：14 万节点 / 43 万 DOF，scipy `spsolve`（SuperLU）单核约数秒~十几秒；可换 `pypardiso` 或 `scikit-umfpack` 提速。**这是伪标签，不要求毫秒级。**

### 5.3 `fem/energy.py` — 可微弹性能（Transolver 损失）

**与 5.1 共用同一套 B/C，但用 torch 实现**，使 `U` 对网络输出 `u` 可自动微分：

```python
def elastic_energy_torch(u, node_xyz, elements, C, gauss_pts) -> U_scalar:
    # u: (N,3) requires_grad（来自 Transolver）
    # 向量化所有单元的高斯点：gather 节点位移 → B@u_e → ε → σ → U_e
    # 返回标量 U = Σ_e Σ_g w·detJ·½·εᵀCε
```

**`test_energy_consistency.py` 必须验证：给同一个 u，torch energy 与 numpy energy 数值相等**（这是两套实现一致性的护栏）。

### 5.4 `data/parse_face_to_nodes.py` — face ID → 节点集合（最高风险）

**问题**：named_selections 给 STEP face ID（如 move=11566），mesh 给节点坐标，**两者无现成映射**，且 mesh.inp 不含面信息。

**重建策略**（用 `debug.move_candidates_top5` 的几何特征）：
```
对 MOVE_INNER_RING_FACE：
  从 debug 取该 face 的中心 (cy,cz)、法向 (ny,nz)、面积 area
  在 mesh 节点中筛选：到该平面距离 < tol 且 在面投影范围内 的节点
  → move_node_ids

对 FIXED_TOP_FACES（4个角）：
  弹片是 X 向 0.06mm 薄片，FIXED 是"四个角固定区上表面"
  按 (y,z) 落在四角 + x 接近 x_max（上表面）筛选
  → fixed_node_ids
```

⚠️ **必须可视化人工核验**：把选中的 fixed/move 节点染色叠加到 preview，目视确认选对了。**选错 move face → K 整个错。** 这是 Stage 0 的头号交付物，需人工 sign-off。

护栏断言：
```python
assert len(move_node_ids) > 0 and len(fixed_node_ids) > 0
assert set(move_node_ids).isdisjoint(set(fixed_node_ids))  # 不能重叠
# free = all − fixed − move
```

### 5.5 `models/bc_enforce.py` — 硬 Dirichlet（mask blending）

⚠️ **禁止 in-place 赋值**（`u[fixed]=0` 会切断梯度、违反不可变）。**只用 mask blending**：

```python
def enforce_bc(u_raw, fixed_mask, move_mask, free_mask, u_prescribed):
    # u_raw: (N,3) 网络原始输出
    # 三个 mask 互斥且并集=全集（Stage 0 已断言）
    u = free_mask * u_raw + move_mask * u_prescribed   # fixed 的 prescribed=0
    return u    # 能量必须在这个 u 上算，绝不在 u_raw 上算
```

### 5.6 `models/transolver.py` — backbone

第一版配置（150k 节点级）：
```yaml
backbone: Transolver++         # plus=True
out_dim: 3                     # u_x,u_y,u_z（不直接出应力/K）
n_layers: 6                    # 稳定后→8
n_hidden: 128                  # 稳定后→256
n_head: 8
slice_num: 64                  # 150k 节点偏少，建议试 96/128
embedding_dim: 3               # 坐标
functional_dim: C_in           # 节点特征维度
use_te: false                  # 调试期关，环境稳定后开
```

节点特征 `fx`（5.7 features.py 构造）：
```
fixed_flag, move_flag, free_flag
distance_to_fixed, distance_to_move
material_E_norm, material_nu
prescribed_disp_{x,y,z}_norm
+ design token（厚度/宽度/有效长/R角/蛇形段 delta，broadcast 到节点）
```

⚠️ **坐标双轨**（旧 plan 对的一点，保留）：
```
coords_net  = (coords − bbox_center) / bbox_scale   # 喂网络
coords_phys = coords（原始 mm）                      # 算能量/应变能，绝不用归一化坐标
```

> **关于 Transolver 的清醒认知**：近期工作（LinearNO, arXiv:2511.06294）发现 Transolver 的 physics-attention 去掉 slice 间 attention 反而更好——有效性主要来自 slice/deslice。**因此 slice map 只能当探索性可视化，不能直接当"因果物理分区"进工业报告**（见 §8）。

---

## 6. 两阶段部署架构（已确认）

### 阶段 A：amortized operator（毫秒级前向）

一个 Transolver 在 300 个几何上用能量损失训练。部署时新几何**只跑一次前向** → u/应力/K。这是默认快速路径。

### 阶段 B：test-time energy refine（可选，几秒，精度更高）

```python
def predict(geometry, refine_steps=0):
    feats = geometry_to_features(geometry)      # mesh+BC+material
    u = transolver(feats)                        # 毫秒，amortized 初值
    u = enforce_bc(u, ...)
    if refine_steps > 0:                          # 可选第二阶段
        u = energy_refine(u, feats, steps=refine_steps)  # 对该几何微调降能量
    return postprocess(u)                         # 应力/K
```

`refine.py`：以 amortized `u` 为初值，对**该几何**做少量 L-BFGS/Adam 步直接最小化能量（不更新网络权重，只更新该几何的 u，或微调最后几层）。**每个新几何都有能量保障**，代价几秒。

> 设计含义：amortized 给"快"，refine 给"准且有物理保障"。部署接口用 `refine_steps` 一个参数切换，满足你"快速预测 + 高精度可选"的双需求。

---

## 7. 标签策略（已确认：direct-FEM 伪标签 + 5–20 ANSYS 锚点）

| 来源 | 数量 | 速度 | 角色 |
|---|---|---|---|
| ANSYS | 5–20 | 慢/贵 | 校准 direct-FEM 系统偏差；最终可信度背书 |
| direct-FEM | 全 300 可算 | 秒级 | 主伪标签：验证基准 + warm-start + test 集"真解" |
| 能量损失 | — | — | 主训练信号（physics-informed，无监督） |

损失：
```
L = λ_energy · L_energy                          # 主项（无标签）
  + λ_anchor · L_anchor                          # 可选：‖u_pred − u_directFEM‖² 在锚点几何上
  + 诊断量（不参与梯度）：|K_energy − K_reaction|, BC 残差
```

**ANSYS 锚点的用法**：先用 ANSYS 校准 direct-FEM（修正 E/ν 或边界模型，使 direct-FEM 在这 5–20 个上对齐 ANSYS），**之后 direct-FEM 即可作为全 300 的可信伪标签**——这样 5–20 个 ANSYS 撬动了全数据集的可验证性。

---

## 8. 分析能力：K/应力敏感区（你的终极目标之一）

### 8.1 设计参数敏感性（可信，优先）

对 K 关于设计参数（厚度/宽度/长度/R角）求梯度：`∂K/∂θ`。**这是因果可信的**（参数是真实可控变量）。输出 OAT 趋势曲线 + Pareto 候选。

### 8.2 区域敏感性（需谨慎，配验证）

```
应力区域：直接看 element-level von Mises 场（优先 element-level，避免节点平均抹平 R 角 hotspot）
K 敏感区：∂K/∂(节点位移或局部刚度扰动)，或 slice-token 梯度
```

⚠️ **slice map ≠ 因果分区**（§5.6 的 LinearNO 警告）。区域影响必须用以下至少一种交叉验证：
- 局部几何扰动（改 R 角 → 看 K 变化）；
- occlusion / 局部刚度敏感；
- direct-FEM 重算对照。

---

## 9. 实施阶段与验收标准

### Stage 0 — 数据地基（实测发现使其成为最重工作）

交付：
```
data/parse_mesh.py, parse_face_to_nodes.py, audit_dataset.py
reports/dataset_index.csv          # 300 样本路径+节点数+有效性
reports/element_type_report.md     # 确认全 tet10
reports/face_to_node_report.md     # ★ 含可视化核验图
reports/status_vs_params_conflict.md  # ★ 字段冲突全量审计
config/physics.yaml                # 用户确认的 move_axis, Δ
```
验收：
- [ ] 300 样本 nodes/elements 全部可读，单元类型确认 tet10；
- [ ] 每个样本的 fixed/move 节点集重建成功，**抽样可视化人工 sign-off**；
- [ ] fixed∩move=∅，mask 并集=全集；
- [ ] status vs params 冲突有明确裁决（哪个是 source of truth）；
- [ ] **用户确认 move_axis（X 面外？）和 Δ 量级（≪0.06mm）**。

### Stage 1 — FEM 引擎 + 单几何（不上 Transolver）

> **关键修正**：Stage 1 用最朴素 coordinate-MLP，先把物理引擎调通。Transolver 在单几何上无优势且更难调，推迟到 Stage 2。

交付：
```
fem/tet10.py, material.py, assembly.py, direct_solver.py, energy.py, postprocess.py
fem/tests/test_patch.py, test_cantilever.py, test_energy_consistency.py
reports/baseline_directFEM_0001/  # direct-FEM 解 + K + 应力场
reports/single_mlp_0001/          # MLP+能量 收敛对照 direct-FEM
```
验收：
- [ ] **patch test 通过**：单 tet10 给线性位移场，应变/应力解析正确；
- [ ] **悬臂梁测试**：K 随厚度/宽/长方向正确，**量化 tet4 若降阶的 locking 误差**（证明保留 tet10 的必要）；
- [ ] **torch energy == numpy energy**（一致性护栏）；
- [ ] direct-FEM 解：BC 满足、K>0、量级合理、无 NaN；
- [ ] MLP+能量训练收敛到接近 direct-FEM（L2 误差报告）；
- [ ] **|K_energy − K_reaction|/K < tol**。

### Stage 2 — 10 几何 Transolver（amortized 雏形）

选样：baseline + 厚度高低 + 宽高低 + 长高低 + R高低 + 1 个 complex。
验收：一个 Transolver 同时处理 10 个几何，每个对 direct-FEM 的 L2 误差稳定；padded batching + node_mask 工作。

### Stage 3 — 60 代表样本（分层抽样）

形成第一个可用 geometry-conditioned operator。验收：train/val/test 物理残差稳定；OAT 趋势符合工程逻辑；**held-out 几何对 direct-FEM 的误差可接受**（这是无标签泛化的真正考验）。

### Stage 4 — 300 全量

`hidden 128→256, layers 6→8, slice 64→96/128`。验收：extreme 样本不崩；K/应力排名稳定；test 集误差不显著恶化。

### Stage 5 — 分析与部署

交付：`deploy/predict.py`（含 refine 开关）、敏感性图、Pareto 候选、推荐 ANSYS 验证子集。

---

## 10. 风险登记（合并实测发现）

| 风险 | 后果 | 对策 | 责任阶段 |
|---|---|---|---|
| face→node 选错 | K 整个错 | 可视化人工核验 + 断言 | Stage 0 |
| move_axis 设错 | K 方向错 | 打印 bbox+法向给用户确认 | Stage 0 |
| Δ 超薄壁厚度 | 破坏小变形 | Δ≪0.06mm，写进 config | Stage 0 |
| tet10 当 tet4 | K 偏硬、hotspot 抹平 | 直接实现 tet10，4点高斯 | Stage 1 |
| 在 u_raw 上算能量 | BC 失效 | 能量必在 blended u 上 | Stage 1 |
| in-place BC 赋值 | 切断梯度 | 只用 mask blending | Stage 1 |
| 力加载 min(U)坍缩 | u≡0 | 断言纯位移控制 | Stage 1 |
| 归一化坐标算物理量 | 能量尺度错 | coords_phys 用真实 mm | Stage 1 |
| 节点平均应力 | hotspot 抹平 | element-level von Mises | Stage 1 |
| 单点积分 | 刚度奇异 | tet10 用 4 点高斯 | Stage 1 |
| 无标签 test 无法验证 | 不知模型对错 | direct-FEM 当"真解" | Stage 1+ |
| 纯物理多几何不泛化 | held-out 错 | anchor 损失 + refine + 稠密采样 | Stage 3+ |
| slice 当因果分区 | 误导工业决策 | 仅探索用，配扰动/occlusion验证 | Stage 5 |
| status/params 冲突 | 有效集判错 | 全量审计裁决 | Stage 0 |

---

## 11. 待用户最终确认项（开工前）

1. **材料**：VCM 弹片的 E、ν（不锈钢？铍铜？）—— 直接决定 K 量级。
2. **move_axis**：X（面外 0.06mm 厚度方向，voice-coil 行程）确认？move face 法向实测朝 Z，需厘清"施加位移方向"vs"面法向"。
3. **Δ 量级**：建议 ≪0.06mm（µm 级），具体值。
4. **move face 非加载方向约束**：`u_y=u_z=0`（刚性平移）还是 remote displacement（可转动）？后者更接近真实 VCM。
5. **PhysicsNeMo 版本 / 部署环境**：Transolver++ 是否可用 `plus=True`；是否上 A100。

---

## 12. 一句话总结

> **Transolver 是你部署的毫秒级预测器；FEM 是它的物理引擎——既当损失（无标签训练）又当伪标签（direct-FEM 验证）。** 地基是 tet10 的 B 矩阵和 face→node 的 BC 重建（这俩做对，全盘皆活）。两阶段部署（amortized 前向 + 可选 energy refine）兼顾快与准；5–20 个 ANSYS 锚点校准 direct-FEM，撬动全 300 的可验证性。**先用 direct-FEM + tet10 把物理验证基准建起来，再上 Transolver。**
</content>
</invoke>
