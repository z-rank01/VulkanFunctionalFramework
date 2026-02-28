# ARAP & N-ICP —— 非刚性网格配准工具集

基于 Sorkine & Alexa (2007) ARAP 算法与 Amberg et al. (2007) Optimal Step N-ICP 算法的 Python 实现，并提供可在 Maya 2020+ 中直接运行的图形化工具窗口。

---

## 文件结构

```
.
├── ARAP.py        # ARAP 核心算法（纯 numpy/scipy，无 DCC 依赖）
├── MayaARAP.py    # ARAP 的 Maya 应用层（PySide2 UI + om2 API）
├── NICP.py        # Optimal Step N-ICP 核心算法（纯 numpy/scipy）
└── MayaNICP.py    # N-ICP 的 Maya 应用层（PySide2 UI + om2 API）
```

---

## 算法实现

### ARAP.py — As-Rigid-As-Possible 变形

**论文**：Sorkine & Alexa, *As-Rigid-As-Possible Surface Modeling*, SGP 2007

**问题描述**：给定固定拓扑的源网格与一组硬约束（已知对应点对），求变形后顶点坐标 $P'$，使 ARAP 能量最小：

$$E(P') = \sum_i \sum_{j \in \mathcal{N}(i)} w_{ij} \| (p'_i - p'_j) - R_i(p_i - p_j) \|^2$$

**算法要点**：

| 模块 | 说明 |
|---|---|
| `MeshData` | 与 DCC 无关的三角网格数据结构（顶点 + 面片索引） |
| `CholeskyFactor` | 稀疏 Cholesky 分解封装，支持 CHOLMOD（scikit-sparse）与 scipy 双后端 |
| `ArapSolver` | 局部-全局交替优化主类 |
| `create_solver` | 快速工厂函数 |

**迭代策略（局部-全局交替优化）**：

- **局部步**：固定 $P'$，对每个顶点用 SVD 求解最优旋转矩阵 $R_i \in SO(3)$
- **全局步**：固定 $R_i$，求解全局稀疏线性方程组（余切 Laplacian + 约束项），Cholesky 预分解加速
- 多约束：硬约束以大权重项注入矩阵，一次分解后每步全局求解极快

---

### NICP.py — Optimal Step Non-Rigid ICP

**论文**：Amberg, Romdhani & Vetter, *Optimal Step Nonrigid ICP Algorithms for Surface Registration*, CVPR 2007

**问题描述**：给定源网格 $\{v_i\}$ 与目标网格 $\mathcal{T}$，为每个顶点求解仿射变换矩阵 $X_i \in \mathbb{R}^{3 \times 4}$，使总能量最小：

$$E(X) = \alpha \cdot E_s + E_d + \gamma_l \cdot E_l$$

其中：
- $E_s$（刚性项）：惩罚相邻顶点变换差异，$G = \mathrm{diag}(1,1,1,\gamma)$ 控制平移分量权重
- $E_d$（数据项）：驱使变形顶点贴近目标网格表面（KD-Tree 最近点）
- $E_l$（Landmark 项）：精确满足手动指定对应约束

**正规方程（闭合解）**：

$$(\alpha M_s + M_d + \gamma_l M_l) \cdot X = \mathrm{rhs}_d + \gamma_l \cdot \mathrm{rhs}_l$$

**算法要点**：

| 模块 | 说明 |
|---|---|
| `_build_edge_list` | 由三角面片提取无向边（刚性项依赖边图 Laplacian） |
| `_compute_vertex_normals` | 面积加权顶点法线（用于法线角度过滤） |
| `_find_closest_points` | KD-Tree 最近点查询，支持距离阈值 + 法线角度双重离群点过滤 |
| `_build_stiffness_normal_matrix` | 刚性项正规方程矩阵 $M_s$（带 G 权重的图 Laplacian，全向量化） |
| `_build_data_normal_matrix` | 数据项正规方程矩阵 $M_d$ 与右端项（块对角，全向量化，无 Python 循环） |
| `_build_landmark_normal_matrix` | Landmark 约束正规方程矩阵 $M_l$ 与右端项 |
| `_apply_transformations` | 将仿射变换 $X$ 应用于源网格（`einsum` 向量化） |
| `NicpSolver` | 多尺度 N-ICP 主求解器 |
| `create_nicp_solver` | 快速工厂函数 |

**多尺度策略**：刚性权重 $\alpha$ 由 `np.logspace(α_start, α_end, n_steps)` 生成等比数列，从大到小衰减，实现粗→细配准。

**坐标自动归一化**：`execute()` 内部自动将源/目标网格归一化到 $[-1, 1]$ 包围盒，保证 $M_s$ 与 $M_d$ 量纲一致（防止大坐标下刚性项被压制），算法结束后自动反归一化输出。

---

## Maya 工具

### MayaARAP.py — ARAP 变形工具

**适用场景**：固定拓扑的同网格变形，Source 和 Target 必须是**相同或相似拓扑**的网格。  
典型用途：将 MetaHuman 模板头部网格朝已有的扫描/雕刻结果变形。

**功能列表**：

| 功能 | 说明 |
|---|---|
| 目标 / 对象设置 | 在 Maya 视口选中网格一键设置 Target / Source |
| 手动配对 Landmark | F9 顶点模式，分别选 Source + Target 各 1 个顶点建立精确对应 |
| 投影最近点 Landmark | 批量将选中 Source 顶点投影到 Target 表面，快速建立轮廓约束 |
| Landmark 表格管理 | 可查看、删除单条或清空全部约束 |
| 迭代次数控制 | 可调全局迭代步数（默认 10，建议 10~15） |
| 逐迭代预览 | 勾选后每步写回网格并刷新视口，可直观观察收敛过程 |
| 一键撤销 | 最终结果通过 `undoInfo chunk` 写入，支持 Ctrl+Z 整体撤销 |
| 可停靠窗口 | 基于 `MayaQWidgetDockableMixin`，可浮动或嵌入 Maya 面板 |

---

### MayaNICP.py — N-ICP 配准工具

**适用场景**：源网格与目标网格**拓扑可以不同**，将 Source 形状配准到 Target 表面。  
典型用途：将写实角色头部网格配准到 MetaHuman DNA 模板。

**功能列表**：

| 功能 | 说明 |
|---|---|
| 目标 / 源网格设置 | 在视口选中网格一键设置 |
| 手动配对 Landmark | F9 顶点模式，分别选 Source + Target 各 1 个顶点，支持精确解剖位置标记 |
| 投影最近点 Landmark | 批量投影选中 Source 顶点到 Target 表面，适合头部轮廓批量添加 |
| Landmark 表格管理 | 可查看（含世界坐标）、删除单条或清空全部 |
| 刚性权重 α 序列 | 起始值 / 终止值 / 步数三参数配置，内部自动生成对数等间距等比数列 |
| 内层迭代次数 | 每个 α 下 ICP 内层循环次数（默认 3，建议 3~5） |
| 平移刚性 γ | G 矩阵平移分量权重（默认 1.0） |
| Landmark 权重 γ_l | 约束强度控制（默认 10，建议 5~50） |
| 收敛阈值 | 最大顶点位移低于阈值时提前终止 |
| 正则化系数 λ | Tikhonov 正则化防止矩阵奇异（默认 1e-6） |
| 离群点距离阈值 | 留空自适应（中位数×2），或手动指定排除远离区域的错误对应 |
| 法线角度阈值 | 可勾选启用，过滤源/目标对应顶点法线夹角超过阈值的错误对应（建议 60°~90°） |
| 逐迭代预览 | 每次内层迭代后实时刷新视口，反归一化坐标后准确写回 |
| 一键撤销 | 最终结果通过 `undoInfo chunk` 写入，支持 Ctrl+Z 整体撤销 |
| 可停靠窗口 | 基于 `MayaQWidgetDockableMixin`，可浮动或嵌入 Maya 面板 |

---

## 快速开始

在 Maya Script Editor 中粘贴以下代码运行：

```python
# ARAP 变形工具
import importlib, sys
sys.path.insert(0, r"<脚本目录绝对路径>")
import ARAP, MayaARAP
importlib.reload(ARAP)
importlib.reload(MayaARAP)
MayaARAP.show()
```

```python
# N-ICP 配准工具
import importlib, sys
sys.path.insert(0, r"<脚本目录绝对路径>")
import ARAP, NICP, MayaNICP
importlib.reload(ARAP)
importlib.reload(NICP)
importlib.reload(MayaNICP)
MayaNICP.show()
```

---

## 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.7+ | Maya 2020+ 内置 |
| numpy | >= 1.20 | |
| scipy | >= 1.6 | 稀疏矩阵 + Cholesky 求解 |
| scikit-sparse | 可选 | CHOLMOD 后端，Cholesky 速度更快 |
| Maya | 2020+ | PySide2 内置，om2 API |

---

## 参数调优建议

### ARAP

- 约束点对建议 6~10 个，分布在鼻尖、眼角、嘴角、下巴、耳根等解剖位置
- 迭代次数 10~15 通常已收敛，复杂变形可增到 20

### N-ICP

1. **先验粗对齐**：若 Source 与 Target 偏移较大，先手动粗对齐再运行
2. **Landmark**：建议 6~20 个，眼角/鼻尖/嘴角/耳根均匀覆盖
3. **α 序列**：`alpha_start=500~1000`，`alpha_end=1.0`，`n_steps=10~16`（形状差异大时增步数）
4. **法线过滤**：眼眶/鼻翼等内凹区域推荐启用，阈值 60°；一般情况用 90°
5. **距离阈值**：留空自适应即可；仅当 Source 有大面积无对应区域时手动指定
