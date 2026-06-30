# Stage 2 实施级设计：`fem/energy.py` + Transolver 接入

**版本**：v1.1（规格 + 已实现状态）
**日期**：2026-06-30
**状态**：✅ **P3-P5 已实现并通过测试（本地 PINN env，107 tests）**。energy.py / bc_enforce.py / features.py / transolver_wrap.py(含 FakeBackbone) / train_single.py 全部落地。**端到端验证：fake backbone 用能量损失训练，K_energy 收敛到解析解 E=127000 N/mm 误差 0.01%（127012.7），|u|max→δ 精确**——证明"最小化可微弹性能 == 恢复真实物理"。剩 P6+(容器内 Transolver API 实测 + 真训练)待服务器。Stage 1 引擎已建成（65 tests）。
**上游**：[`docs/design_plan_v1.md`](design_plan_v1.md)（架构基线）。本文档不重复物理原理，只补"怎么和已有代码对接 + 真实 API + 验证方案"。

---

## 0. 本阶段目标与边界

把"无标签 physics-informed 训练"的最小可验证链路建起来，**先不上多几何、先不追泛化**：

```
Transolver(单几何前向) → enforce_bc(硬Dirichlet) → energy.py(torch可微U) → min U == 平衡
                                                          ▲
                                  与已建 numpy 引擎 total_strain_energy 数值对齐(护栏)
```

**交付物**（本阶段）：
1. `fem/energy.py` — torch 可微弹性应变能，与 numpy 版数值相等。
2. `models/bc_enforce.py` — 硬 Dirichlet（mask blending，无 in-place）。
3. `models/features.py` — 节点特征构造。
4. `models/transolver_wrap.py` — PhysicsNeMo Transolver 封装（含 B=1 路径）。
5. `train/losses.py` — 能量损失 + 诊断量（K_energy/K_reaction gap、BC 残差）。
6. `train/train_single.py` — 单几何训练 loop（裸 torch，不依赖 Sym）。
7. 测试：`fem/tests/test_energy_consistency.py`（torch==numpy）、`models/tests/test_bc_enforce.py`。

**明确不在本阶段**：多几何 padded batching、test-time refine、敏感性分析、部署 predict.py。

---

## 1. 验证环境（关键，先焊死）

| 用途 | 环境 | 说明 |
|---|---|---|
| **本地数值验证** | conda env **`PINN`** = torch **2.8.0+cpu** | 与服务器容器 torch 2.8.0a0 **同主版本**；numpy/scipy/pytest/yaml 齐全，能同时跑 torch 能量和现有 numpy 引擎做对照。 |
| 服务器训练 | PhysicsNeMo 1.2.0 Docker（A800 80GB, CUDA 12.9） | 真正训练 Transolver 用。容器内不重装 torch。 |

跑测试：`C:\Users\LHan1\anaconda3\envs\PINN\python.exe -m pytest fem/tests/test_energy_consistency.py -q`

⚠️ **PhysicsNeMo 只在服务器有**。所以 `models/transolver_wrap.py` 本地无法 import 真模型 → 设计成**惰性 import + 可注入 backbone**，本地用一个 shape 兼容的假 backbone 跑 `bc_enforce`/`energy`/`losses` 的单测（见 §5、§7）。

---

## 2. 已建引擎的对接契约（torch 版必须逐字复用）

来自 `fem/elements/common.py` / `material.py` / `parse_mesh.py` 实测：

- **Voigt 顺序**：`[xx, yy, zz, xy, yz, zx]`，B 矩阵、C 矩阵、ε/σ 处处一致。
- **工程剪切**：B 在剪切行直接发 `gamma = 2ε`，故能量是纯二次型 `½ εᵀ C ε`，**剪切无额外 ×2**；`elastic_C` 剪切对角 = `mu`（不是 2mu）。
- **单元 DOF 排序**：node-major `[ux0,uy0,uz0, ux1,...]`；第 i 节点占 B 列 `3i:3i+3`。`u_e.reshape(-1)` 即此展平。
- **能量逐元累加公式**（numpy 版 `element_strain_energy`，torch 要对齐）：
  ```
  U = Σ_family Σ_elem Σ_gauss  w_g · detJ_g · ½ · εᵀ C ε ,  ε = B(ξ_g) @ u_flat
  J = node_xyz.T @ dN_dnat ;  dN_dx = dN_dnat @ inv(J) ;  detJ = det(J)
  ```
- **各族高斯点数固定**：tet10=**4**，wedge15=**6**，hex20=**27**。`GAUSS[g] = ((c0,c1,c2), w)`，自然坐标恒 3 维。
- **`shape_grads(ξ)` 与 u/坐标无关，是常量** → 可对每族在所有高斯点预算 `dN_dnat`，缓存为 `(NGAUSS, NNODE, 3)` tensor，**无需让 shape_grads 可微**。可微性只经由 `u`。
- **Mesh 消费形式**：`mesh.elements_of(nnode) -> (M, nnode) int64` 行索引批；`coords[conn]→(M,nnode,3)`、`u[conn]→(M,nnode,3)` 一次 gather。
- **C 直接复用**：`elastic_C(E,nu)` 的 `(6,6)` numpy → `torch.as_tensor` 即可，保证数值一致。

---

## 3. `fem/energy.py` — torch 可微弹性应变能（本阶段地基）

### 3.1 设计：按族向量化，梯度只经 `u`

每族（tet10/hex20/wedge15）独立向量化，三族能量相加。坐标是常量、shape_grads 是常量，**唯一 requires_grad 的是位移 `u`**。

```python
# fem/energy.py
from __future__ import annotations
import numpy as np
import torch
from fem.elements import tet10, wedge15, hex20

_FAMILY = {10: tet10, 15: wedge15, 20: hex20}

class FamilyKernel:
    """一个单元族的预计算常量(与 u 无关)，搬一次到 device 后反复用。"""
    nnode: int
    dN_dnat: torch.Tensor     # (G, nnode, 3)  各高斯点自然坐标梯度(常量)
    weights: torch.Tensor     # (G,)           高斯权重
    conn: torch.Tensor        # (M, nnode)     该族单元的行索引(long)

def build_family_kernels(mesh, device, dtype=torch.float64) -> dict[int, FamilyKernel]:
    """对 mesh 每个非空族预计算 dN_dnat / weights / conn。dtype 默认 float64
    以便和 numpy 引擎做严格数值对照(训练时可切 float32)。"""
    ...

def elastic_energy_torch(
    u: torch.Tensor,            # (N, 3) requires_grad, device 上
    coords: torch.Tensor,       # (N, 3) 常量, 物理 mm 坐标(绝不用归一化坐标!)
    kernels: dict[int, FamilyKernel],
    C: torch.Tensor,            # (6, 6) = elastic_C(E,nu) 转 tensor
) -> torch.Tensor:              # 标量 U
    """U = Σ_family Σ_elem Σ_gauss w·detJ·½ εᵀCε。整段对 u 可微。"""
    total = u.new_zeros(())
    for nnode, k in kernels.items():
        xe = coords[k.conn]                 # (M, nnode, 3)
        ue = u[k.conn]                      # (M, nnode, 3)
        # 批量 Jacobian: J[m,g,i,j] = ∂x_i/∂ξ_j = Σ_n xe[m,n,i]·dN_dnat[g,n,j] -> (M,G,3,3)
        J = torch.einsum('mni,gnj->mgij', xe, k.dN_dnat)
        detJ = torch.linalg.det(J)                          # (M, G); 需要它当体积权重
        # assert (detJ > 0).all()  # debug 模式开: mesh 已 orientation 修复, detJ 应恒>0
        # dN/dx = dN/dξ · J^{-1}。不显式求逆: 解 Jᵀ·dN_dxᵀ = dN_dξᵀ 更稳更快(MEDIUM-2 评审)
        # dN_dx[...,n,k] from solving; broadcast dN_dnat (G,nnode,3)->(M,G,nnode,3)
        dN_dnat_b = k.dN_dnat.unsqueeze(0).expand(J.shape[0], -1, -1, -1)        # (M,G,nnode,3)
        dN_dx = torch.linalg.solve(J.transpose(-1, -2).unsqueeze(2),             # (M,G,1,3,3)
                                   dN_dnat_b.unsqueeze(-1)).squeeze(-1)          # (M,G,nnode,3)
        # 装配 B 并算 eps = B @ u_flat，等价于直接用 dN_dx 缩并 ue:
        eps = _strain_from_grad(dN_dx, ue)                  # (M, G, 6) 工程剪切 Voigt
        # 能量密度 ½ εᵀ C ε
        sig = torch.einsum('ij,mgj->mgi', C, eps)           # (M,G,6)
        dens = 0.5 * (eps * sig).sum(-1)                    # (M,G)
        total = total + (k.weights[None, :] * detJ * dens).sum()
    return total

def _strain_from_grad(dN_dx, ue):
    """dN_dx:(M,G,nnode,3), ue:(M,nnode,3) -> eps:(M,G,6) [xx,yy,zz,xy,yz,zx] 工程剪切。
    du_i/dx_j = Σ_n dN_dx[...,n,j] * ue[...,n,i]"""
    # grad[...,i,j] = ∂u_i/∂x_j
    grad = torch.einsum('mgnj,mni->mgij', dN_dx, ue)        # (M,G,3,3)
    exx, eyy, ezz = grad[...,0,0], grad[...,1,1], grad[...,2,2]
    gxy = grad[...,0,1] + grad[...,1,0]   # 工程剪切 = 2*eps_xy
    gyz = grad[...,1,2] + grad[...,2,1]
    gzx = grad[...,2,0] + grad[...,0,2]
    return torch.stack([exx,eyy,ezz,gxy,gyz,gzx], dim=-1)
```

> **要点**：`_strain_from_grad` 的 Voigt 行序与 `common.B_matrix` 必须逐行一致（xx,yy,zz, 然后 xy=du/dy+dv/dx, yz, zx）。这是和 numpy 引擎对齐的命门。**已被独立评审数值核对通过（相对误差 2.6e-15）**。

> **单几何性能优化（LOW-2，本阶段适用）**：`J`/`detJ`/`dN_dx` 只依赖 `coords`，**单几何训练中是常量**，只有 `u` 每步变。故可在 `FamilyKernel` 里把 `detJ`(M,G) 和 `dN_dx`(M,G,nnode,3) **预算一次缓存**，训练每步只重复 `grad = einsum('mgnj,mni->mgij', dN_dx, ue)` 这步。正确性不变，省掉每步的 det/solve。多几何阶段再视情况权衡。

### 3.2 数值验证（本阶段头号护栏）

`fem/tests/test_energy_consistency.py`：
- 用现有 `_single_hex_mesh()`（test_solver.py 已有）+ 一个**任意非平凡位移场**（如线性场 + 小随机扰动）。
- `U_torch = elastic_energy_torch(u, coords, kernels, C)`（float64）。
- `U_numpy = fem.postprocess.total_strain_energy(mesh, u_np, C_np)`。
- 断言 `abs(U_torch - U_numpy) / (abs(U_numpy)+eps) < 1e-10`。
- **三族各测一遍**（tet10 用单 tet 参考、wedge15 用单 prism 参考、hex20 用单 cube），复用 `fem/tests/reference_elements.py`。
- 再测**可微性**：`U_torch.backward()`，断言 `u.grad` 有限、形状 (N,3)、非全零。

> 通过这个测试 = torch 能量和 numpy 引擎是同一套数学。这是后面所有训练可信的前提（design_plan_v1 §5.3 明确要求）。

---

## 4. `models/bc_enforce.py` — 硬 Dirichlet（mask blending）

来自 design_plan_v1 §5.5。**禁止 `u[fixed]=0` in-place**（切断梯度）。

```python
# models/bc_enforce.py
import torch

def build_bc_tensors(bs, num_nodes, move_axis: int, delta_mm: float,
                     move_transverse_rigid: bool, device, dtype):
    """从 BoundarySets 造 (free_mask, u_prescribed):
      free_mask     (N,3) {0,1}: 1=该 DOF 自由(网络说了算), 0=被 Dirichlet 钉住
      u_prescribed  (N,3): 被钉 DOF 的目标值(fixed=0; move 的 move_axis=delta; rigid 时 move 其余=0)
    与 fem/direct_solver._build_dirichlet 的 BC 口径必须一致(同一物理)。"""
    ...

def enforce_bc(u_raw: torch.Tensor, free_mask: torch.Tensor,
               u_prescribed: torch.Tensor) -> torch.Tensor:
    """u = free_mask * u_raw + (1-free_mask) * u_prescribed。无 in-place，梯度只流经 free DOF。"""
    return free_mask * u_raw + (1.0 - free_mask) * u_prescribed
```

> **与 design_plan_v1 的分歧（MEDIUM-1 评审，已确认本式更对）**：v1 §5.5(line 291)用 `u = free_mask*u_raw + move_mask*u_prescribed`(两 mask)。本式用 `(1-free_mask)*u_prescribed`(单 free_mask)，**能正确钉住 move 节点的 transverse 分量**(rigid 时 u_y=u_z=0)——这正是两-mask 式漏掉的 DOF。前提：`u_prescribed` 必须编码**所有**被钉 DOF(fixed=0 / move_axis=δ / rigid 时 move-transverse=0)，`build_bc_tensors` 已如此。测试须专门断言 move-transverse-rigid 的 DOF 被钉为 0(两-mask 式在此处会错)。

测试 `models/tests/test_bc_enforce.py`：
- enforce 后 fixed 节点三分量 == 0，move 节点 move_axis == delta，transverse(rigid)==0。
- 梯度只在 free DOF 上非零（对 u_raw 求 grad，被钉 DOF 的 grad == 0）。
- **BC 口径和 direct_solver 一致**：同一 bs/delta 下，两边钉住的 DOF 集合相同。

---

## 5. `models/transolver_wrap.py` — PhysicsNeMo 封装

> ⚠️ **API 来源与待验证声明（务必先读）**：下面的签名来自对 PhysicsNeMo **v1.2.0 GitHub tag 源码**的网络调研，但**本地无 physicsnemo 无法 import 验证**（本地 PINN 环境只有 torch，没有 physicsnemo）。因此本节所有 kwargs 标记为**待容器内核实**，并设为 milestone-4 的硬门槛（§7 步4）：进服务器后**第一件事**用 `import inspect; inspect.signature(Transolver.__init__)` + 一次 dummy forward 核对，再开训练。
>
> **与 design_plan_v1 的矛盾裁决**：design_plan_v1 §5.6（line 299）写 `backbone: Transolver++ / plus=True`。调研发现 **`plus=` 是 PhysicsNeMo 2.0.0 才加的参数，1.2.0（容器版）没有**。**以本文档为准：1.2.0 用基础 Transolver，不传 plus**。若容器内 `inspect` 发现实际有 plus，再回头修正——这正是 milestone-4 门槛的意义。

调研所得 API（v1.2.0，**待容器核实**）：`from physicsnemo.models.transolver import Transolver`；`fx`/`embedding` 分开传（不预拼）；支持非结构化 ~14万节点（`structured_shape=None`）；裸 torch 自定义损失 OK（physicsnemo.Module 即 nn.Module，不需 Sym）。

```python
# models/transolver_wrap.py
import torch

def make_transolver(functional_dim: int, *, n_layers=8, n_hidden=256, n_head=8,
                    slice_num=64, use_te=False):
    """惰性 import 真模型(只服务器有)。返回的 nn.Module forward(fx, embedding=coords)->(B,N,3)。"""
    from physicsnemo.models.transolver import Transolver  # 惰性: 本地无 physicsnemo 不报错
    return Transolver(
        functional_dim=functional_dim,  # 每节点物理特征数(不含坐标)
        out_dim=3,                       # u_x,u_y,u_z
        embedding_dim=3,                 # 坐标维, unified_pos=False 时必填
        structured_shape=None,           # 非结构化点云
        unified_pos=False,
        n_layers=n_layers, n_hidden=n_hidden, n_head=n_head, slice_num=slice_num,
        use_te=use_te,                   # bring-up 期 False, 稳定后 True 提速
    )

def forward_single(model, fx, coords):
    """单几何 B=1 路径(避免变长 padding)。
      fx:(N,F) coords:(N,3) -> u_raw:(N,3)。"""
    u = model(fx.unsqueeze(0), embedding=coords.unsqueeze(0))  # (1,N,3)
    return u.squeeze(0)
```

⚠️ **变长点云无内置 mask** → 本阶段坚持 **B=1 每 variant**，不碰 padding（多几何 batching 留到 Stage 3，那时再自己写 pad+mask 并在损失里排除 padding 节点）。

**本地可测性**：`train_single` 接受**注入的 backbone**（依赖倒置）。本地用一个假 backbone（`nn.Linear(F+3, 3)` 包成同签名）跑通 enforce_bc→energy→loss→backward 的管线单测；真 Transolver 只在服务器替换进去。

---

## 6. `models/features.py` + `train/losses.py` + `train/train_single.py`

### 6.1 features.py（design_plan_v1 §5.6/§5.7）
节点特征 `fx (N, F)`：`[fixed_flag, move_flag, free_flag, dist_to_fixed, dist_to_move, E_norm, nu, prescribed_disp_xyz_norm]`，单几何阶段 design token 可暂省。
**坐标双轨**：`coords_net=(coords-center)/scale` 喂网络 embedding；`coords_phys=coords`(原始 mm) 喂 energy（绝不用归一化坐标算能量）。

### 6.2 losses.py
```
L = λ_energy · U(enforce_bc(u_raw))          # 主项, 纯位移控制下 min U == 平衡
诊断(不反传): rel_gap=|K_energy-K_reaction|/K, bc_residual, U, |u|max
```
- `K_energy = 2U/δ²`（U 来自 energy.py）。
- `K_reaction`：对 enforced u 调已建 `direct_solver`/`postprocess` 思路在 move 面求反力——本阶段可先只报 K_energy + BC 残差，gap 留到能跑通后加（避免一次引入太多）。

### 6.3 train_single.py（裸 torch，不依赖 Sym）
```python
u_raw = forward_single(model, fx, coords_net)
u = enforce_bc(u_raw, free_mask, u_prescribed)   # 能量必在 enforced u 上算
U = elastic_energy_torch(u, coords_phys, kernels, C)
loss = U
loss.backward(); opt.step()
# 每 K 步打印诊断: U, |u|max, (后续) rel_gap —— 呼应 PINN 教训:不能只看 loss
```
**收敛判据**：U 下降并稳定 + |u|max≈δ 量级 + （加上后）rel_gap < tol。对照 direct-FEM 的 u 报 L2 误差（direct-FEM 解当"真解"）。

---

## 7. 实施顺序与验收（本阶段 milestone）

| 步 | 交付 | 验收 | 环境 |
|---|---|---|---|
| 1 | `fem/energy.py` + test_energy_consistency | **torch U == numpy U**(三族, <1e-10) + u.grad 有限非零 | 本地 PINN |
| 2 | `models/bc_enforce.py` + test | fixed/move 值正确; 梯度只流 free; BC 口径==direct_solver | 本地 PINN |
| 3 | features.py + losses.py + train_single(假 backbone) | 管线跑通; U 随步数下降; 诊断量打印 | 本地 PINN |
| 4a | **Transolver API 核实**(门槛) | 容器内 `inspect.signature(Transolver.__init__)` 核对 kwargs(functional_dim/embedding_dim/out_dim/有无 plus) + dummy `(1,N,F)`/`(1,N,3)` forward 出 `(1,N,3)`。**API 不符先停, 改 wrap 再训** | 服务器 |
| 4b | transolver_wrap 真 backbone | 单 variant 训练 U 下降; 对 direct-FEM L2 报告 | 服务器 |

> **遵循 [[feedback_pinn_mixed_form_debug]]**：每步都监控物理诊断量(U、|u|max、rel_gap、BC 残差)，不能只看 loss；纯位移控制断言焊死(`assert loading_mode=="displacement_control"`)；smoke test 只验不报错，物理收敛要看趋势。

---

## 8. 风险与对策（本阶段相关项）

| 风险 | 对策 |
|---|---|
| torch 能量与 numpy 不一致 | test_energy_consistency 三族 <1e-10 是硬门；Voigt 行序逐行核对 B_matrix |
| 在 u_raw 上算能量(BC 失效) | 能量只在 enforce_bc 后的 u 上算，train loop 焊死顺序 |
| in-place BC 切断梯度 | 只用 mask blending，单测查 free-only 梯度 |
| 归一化坐标算能量(尺度错) | coords_phys 用真实 mm 喂 energy；coords_net 只喂网络 embedding |
| 力加载 min(U) 坍缩到 u=0 | 断言纯位移控制(design_plan_v1 §2.2) |
| 本地无 physicsnemo 卡住开发 | 惰性 import + 注入假 backbone，管线单测本地全过 |
| Transolver API 与调研不符(本地无法验) | milestone-4a 容器内 inspect 签名+dummy forward 当硬门槛, 不符先停再训; 与 v1 的 plus 矛盾以本文档裁决(1.2.0 无 plus) |
| float32 训练精度 | 验证用 float64，训练切 float32；energy.py dtype 可参数化 |

---

## 9. 一句话总结

> 本阶段把 design_plan_v1 的 energy/bc/transolver 蓝图，对接到**已实测建成的 numpy 引擎**(同一套 B/C/高斯)和**核实过的 PhysicsNeMo 1.2.0 真实 API**(无 plus、fx/embedding 分开、裸 torch 损失)，用**本地 PINN(torch2.8 CPU)环境**做"torch U==numpy U"数值护栏。**先 B=1 单几何把无标签能量训练跑通并验证物理诊断量,再上多几何。**
