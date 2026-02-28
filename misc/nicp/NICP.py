"""
Optimal Step Non-Rigid Iterative Closest Point (N-ICP) 核心算法模块。

算法参考
--------
Amberg, B., Romdhani, S., & Vetter, T. (2007).
    Optimal Step Nonrigid ICP Algorithms for Surface Registration.
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1–8.

算法概述
--------
给定源网格顶点坐标 V = {v_1, ..., v_N} 与目标网格 T，
为每个顶点求解一个仿射变换矩阵 X_i ∈ ℝ^{3×4}，使总能量最小：

    E(X) = α·E_s(X) + E_d(X) + γ_l·E_l(X)

各项定义
--------
    E_s  (刚性项, Stiffness Term)：
        E_s = Σ_{(i,j)∈E} ‖(X_i − X_j)·G‖_F²

        其中：
            E        — 网格边集合（由三角面片提取的无向边）
            G        — diag(1, 1, 1, γ)，控制旋转与平移分量的相对刚性权重
            α        — 刚性权重，多尺度策略中从大到小衰减（粗到细配准）

    E_d  (数据项, Data Term)：
        E_d = Σ_i w_i ‖X_i·ṽ_i − u_i‖²

        其中：
            ṽ_i = [v_i; 1] ∈ ℝ^4  — 顶点 v_i 的齐次坐标
            u_i                     — v_i 在目标网格上的最近点坐标（KD-Tree 查询）
            w_i ∈ {0, 1}           — 有效权重；距离超过阈值时置 0（离群点抑制）

    E_l  (Landmark 约束项, Landmark Term)：
        E_l = Σ_{k∈L} ‖X_k·ṽ_k − l_k‖²

        其中：
            L    — 已知手动对应的源顶点索引集合
            l_k  — 第 k 个 landmark 的目标坐标

求解原理（闭合解）
------------------
令 ∂E/∂X = 0，得到正规方程（Normal Equations）：

    (α·M_s + M_d + γ_l·M_l) · X = rhs_d + γ_l·rhs_l

其中：
    X     ∈ ℝ^{4N×3}  — 全局未知量（X[4i:4i+4, :] = X_i^T）
    M_s   ∈ ℝ^{4N×4N} — 刚性项 Hessian（由边图邻接结构给出的 Laplacian 型矩阵）
    M_d   ∈ ℝ^{4N×4N} — 数据项 Hessian（块对角，每块 = ṽ_i ṽ_i^T）
    M_l   ∈ ℝ^{4N×4N} — Landmark 项 Hessian（稀疏块对角）
    rhs_d ∈ ℝ^{4N×3}  — 数据项梯度
    rhs_l ∈ ℝ^{4N×3}  — Landmark 项梯度

对应方程 (A^T A)^{-1} A^T B 的等价矩阵形式（参见论文 Eq.(10)~(12)）：

    A = [ sqrt(α) (G ⊗ A_stiffness) ]    B = [    0   ]
        [    W^{1/2} A_d             ]        [ W^{1/2} B_d ]
        [ sqrt(γ_l) A_l              ]        [ sqrt(γ_l) B_l ]

显式组装正规方程（M_s + M_d + γ_l M_l）避免构造冗余大矩阵，节省内存。

多尺度策略（Coarse-to-Fine）
----------------------------
给定衰减刚性权重列表 [α_1, α_2, ..., α_K]（α_1 > ... > α_K 通常跨越数量级），
每个 α 执行 inner_iterations 次内层迭代：
    1. 最近点更新   — KD-Tree 重新查询当前变形顶点对应的目标最近点
    2. 矩阵组装    — 重新组装 M_d、rhs_d（M_s 和 M_l 对当前 α 不变时可缓存）
    3. 线性求解    — Cholesky 分解，求解 4N×4N 稀疏正定方程组
    4. 顶点更新    — deformed[i] = X_i @ ṽ_i，即 ṽ_i^T @ X[4i:4i+4, :]
    5. 收敛检测    — max‖Δv‖ < convergence_tol 时提前终止

设计原则
--------
本模块为纯算法层，仅依赖 numpy / scipy，与任何 DCC 工具完全解耦。
MeshData / CholeskyFactor 直接复用 ARAP.py 中的定义，无需重复实现。

依赖
----
    numpy  >= 1.20
    scipy  >= 1.6
    ARAP.py（同目录，提供 MeshData / CholeskyFactor）
    scikit-sparse（可选，CHOLMOD 后端，速度更快）
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# 将本文件所在目录加入 sys.path，使 import ARAP 不依赖调用环境的工作目录
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ARAP import CholeskyFactor, MeshData  # noqa: E402


# =============================================================================
# 辅助函数：边图构建
# =============================================================================

def _build_edge_list(mesh: MeshData) -> List[Tuple[int, int]]:
    """
    由三角面片索引构建无向边列表（去重有序）。

    遍历所有三角面片，对每个三角形的三条边取 (min(i,j), max(i,j)) 形式，
    通过 set 去重后返回升序排列的边列表。

    刚性项能量 E_s 定义在**边**上（而非 ARAP 的一环邻域），故需此辅助。

    Parameters
    ----------
    mesh : MeshData
        输入三角网格。

    Returns
    -------
    edges : List[Tuple[int, int]]
        去重后的无向边列表，每条边 (i, j) 满足 i < j，按 (i, j) 升序排列。
    """
    edge_set: set = set()
    for tri in mesh.faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        edge_set.add((min(i, j), max(i, j)))
        edge_set.add((min(j, k), max(j, k)))
        edge_set.add((min(i, k), max(i, k)))
    return sorted(edge_set)


# =============================================================================
# 辅助函数：顶点法线计算
# =============================================================================

def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    由三角面片计算面积加权顶点法线（单位化）。

    算法
    ----
    1. 对每个三角面片，计算叉积法线（其模长 = 2 × 面积，可作面积权重）
    2. 用 np.add.at 将面法线散射累加到三顶点
    3. 对每个顶点法线归一化（退化法线置为 [0, 0, 1]）

    面积加权保证大三角形对相邻顶点法线贡献更大，
    比均等权重在曲率变化大区域更准确。

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        顶点坐标。
    faces : np.ndarray, shape (F, 3), dtype int
        三角面片顶点索引。

    Returns
    -------
    normals : np.ndarray, shape (N, 3)
        单位化的逐顶点法线。退化顶点（全零法线）置为 (0, 0, 1)。
    """
    v0 = vertices[faces[:, 0]]   # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # 面法线（带面积权重：模 = 2 × 三角形面积）
    face_normals = np.cross(v1 - v0, v2 - v0)   # (F, 3)

    # 散射累加到顶点法线
    vertex_normals = np.zeros_like(vertices)     # (N, 3)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)

    # 归一化，退化法线保底置为 Z 轴方向
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)  # (N, 1)
    degenerate = (norms[:, 0] < 1e-10)
    norms = np.where(norms < 1e-10, 1.0, norms)
    vertex_normals /= norms
    vertex_normals[degenerate] = np.array([0.0, 0.0, 1.0])

    return vertex_normals


# =============================================================================
# 辅助函数：KD-Tree 最近点查询
# =============================================================================

def _find_closest_points(
    source_pts:             np.ndarray,
    target_pts:             np.ndarray,
    distance_threshold:     Optional[float] = None,
    source_normals:         Optional[np.ndarray] = None,
    target_normals:         Optional[np.ndarray] = None,
    normal_angle_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 KD-Tree 在目标点集中查找源点集的最近点，并生成有效对应掩码。

    离群点抑制策略
    -------------
    1. **距离过滤**：源点与最近目标点距离超过阈值时置 0（必选）。
       阈值为 None 时自动取所有对应距离中位数的 2 倍（自适应）。

    2. **法线角度过滤**（可选）：当源顶点法线与对应目标顶点法线之间的
       夹角超过 normal_angle_threshold（度数）时置 0。
       适合拒绝表面朝向不一致的错误对应，例如眼睑外侧对应到眼球内侧。
       需要同时传入 source_normals 和 target_normals 才生效。

    Parameters
    ----------
    source_pts : np.ndarray, shape (N, 3)
        当前迭代的源网格顶点坐标（已应用变换 X）。
    target_pts : np.ndarray, shape (M, 3)
        目标网格顶点坐标。
    distance_threshold : float, optional
        离群点距离阈值。若为 None，则自动取所有对应距离中位数的 2 倍。
    source_normals : np.ndarray, shape (N, 3), optional
        当前迭代变形后源网格顶点的单位法线。
    target_normals : np.ndarray, shape (M, 3), optional
        目标网格顶点的单位法线。
    normal_angle_threshold : float, optional
        法线夹角阈值（度数，0~180）。超过此角度的对应被视为无效。
        建议值：60~90°。典型情形：
            60° — 严格约束，眼部、鼻翼等曲率大区域效果好；
            90° — 拒绝背面对应；
            None — 不做法线过滤（同旧行为）。

    Returns
    -------
    closest_pts : np.ndarray, shape (N, 3)
        每个源顶点对应的目标网格最近顶点坐标。
    valid_mask : np.ndarray, shape (N,), dtype bool
        有效对应掩码：True 表示该点纳入数据项优化，False 为离群点（权重=0）。
    """
    tree = cKDTree(target_pts)
    # workers=-1 利用所有 CPU 核心并行查询
    dists, indices = tree.query(source_pts, k=1, workers=-1)

    # ── 距离过滤 ────────────────────────────────────────────────────────────
    # 自适应阈值：若未指定，取距离中位数的 2 倍
    if distance_threshold is None:
        median_dist = float(np.median(dists))
        threshold   = 2.0 * median_dist if median_dist > 1e-10 else np.inf
    else:
        threshold = float(distance_threshold)

    closest_pts = target_pts[indices]   # (N, 3)
    valid_mask  = dists <= threshold    # (N,) bool

    # ── 法线角度过滤（可选） ─────────────────────────────────────────────────
    if (
        normal_angle_threshold is not None
        and source_normals is not None
        and target_normals is not None
    ):
        cos_thresh = float(np.cos(np.deg2rad(float(normal_angle_threshold))))
        # 对应目标顶点的法线，shape (N, 3)
        matched_tgt_normals = target_normals[indices]         # (N, 3)
        # 逐顶点点积（已单位化，结果即 cos θ），shape (N,)
        cos_angles = np.einsum("ni,ni->n", source_normals, matched_tgt_normals)
        valid_mask &= (cos_angles >= cos_thresh)

    return closest_pts, valid_mask


# =============================================================================
# 步骤 1：刚性项正规方程矩阵 M_s = α·A_s^T·(G² ⊗ I)·A_s
# =============================================================================

def _build_stiffness_normal_matrix(
    n_verts: int,
    edges:   List[Tuple[int, int]],
    alpha:   float,
    gamma:   float,
) -> sp.csr_matrix:
    """
    组装刚性项的正规方程矩阵 M_s（对称正半定稀疏矩阵）。

    刚性能量定义
    -----------
    E_s = α Σ_{(i,j)∈E} ‖(X_i − X_j)·G‖_F²

    其中 G = diag(1, 1, 1, γ) 对仿射矩阵的 4 个输入维度施加差异化权重：
        - 前 3 维（旋转/缩放分量）权重为 1
        - 第 4 维（平移分量）权重为 γ

    正规化后块结构推导
    -----------------
    对边 (i, j) 的每个输入维度 c = 0..3（G 对角元 g_c）：

        M_s[4i+c, 4i+c] += α·g_c²
        M_s[4i+c, 4j+c] -= α·g_c²        (对称)
        M_s[4j+c, 4j+c] += α·g_c²

    本质上是带 G 权重的图 Laplacian，作用于 4N 维变换空间。

    Parameters
    ----------
    n_verts : int
        源网格顶点总数 N，矩阵尺寸为 (4N, 4N)。
    edges : List[Tuple[int, int]]
        无向边列表，每条 (i, j) 满足 i < j。
    alpha : float
        当前刚性权重 α（从大到小逐步衰减）。
    gamma : float
        G 矩阵第 4 个对角元（平移分量刚性）。

    Returns
    -------
    M_s : sp.csr_matrix, shape (4N, 4N)
        刚性项正规方程矩阵，对称正半定。
    """
    dim    = 4 * n_verts
    # G 对角元 [1, 1, 1, γ] 的平方，出现于正规方程
    g_sq   = np.array([1.0, 1.0, 1.0, gamma ** 2], dtype=np.float64)

    n_edges = len(edges)
    # 每条边贡献 4 个维度，每维 4 个非零项（对角2 + 交叉2），共 4*4*n_edges 个非零项
    capacity = 16 * n_edges
    rows = np.empty(capacity, dtype=np.int32)
    cols = np.empty(capacity, dtype=np.int32)
    vals = np.empty(capacity, dtype=np.float64)

    ptr = 0
    for (vi, vj) in edges:
        for c in range(4):
            ri  = 4 * vi + c
            rj  = 4 * vj + c
            w   = alpha * g_sq[c]

            # 对角块 (vi, vi) 和 (vj, vj)
            rows[ptr] = ri;  cols[ptr] = ri;  vals[ptr] =  w;  ptr += 1
            rows[ptr] = rj;  cols[ptr] = rj;  vals[ptr] =  w;  ptr += 1
            # 交叉块 (vi, vj) 和 (vj, vi)（对称保证矩阵对称性）
            rows[ptr] = ri;  cols[ptr] = rj;  vals[ptr] = -w;  ptr += 1
            rows[ptr] = rj;  cols[ptr] = ri;  vals[ptr] = -w;  ptr += 1

    M_s = sp.csr_matrix(
        (vals[:ptr], (rows[:ptr], cols[:ptr])),
        shape=(dim, dim),
        dtype=np.float64,
    )
    return M_s


# =============================================================================
# 步骤 2：数据项正规方程矩阵 M_d = A_d^T·W·A_d，右端项 rhs_d = A_d^T·W·B_d
# =============================================================================

def _build_data_normal_matrix(
    source_pts:  np.ndarray,
    closest_pts: np.ndarray,
    valid_mask:  np.ndarray,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    组装数据项的正规方程矩阵 M_d 与右端项 rhs_d（全向量化实现）。

    数据能量定义
    -----------
    E_d = Σ_i w_i ‖X_i·ṽ_i − u_i‖²

    其中 ṽ_i = [v_i; 1] ∈ ℝ^4（齐次坐标），w_i ∈ {0, 1}。

    正规化后矩阵结构
    ----------------
    M_d 为**块对角矩阵**，第 i 个有效顶点对应的 4×4 块为：
        M_d[4i:4i+4, 4i:4i+4] = ṽ_i ṽ_i^T

    rhs_d 为形如 (4N, 3) 的矩阵，其中：
        rhs_d[4i:4i+4, :] = ṽ_i ⊗ u_i^T  （外积，4×3）

    实现说明（全向量化，无 Python 循环）
    -------------------------------------
    1. 提取有效顶点的齐次坐标 vh ∈ ℝ^{K×4} 和最近点 uc ∈ ℝ^{K×3}
    2. 利用广播计算 K 个 4×4 外积块 → flat COO 三元组
    3. 利用 np.add.at 稀疏散射到 rhs_d

    Parameters
    ----------
    source_pts : np.ndarray, shape (N, 3)
        当前迭代的变形后源网格顶点坐标（用于构造 ṽ_i）。
    closest_pts : np.ndarray, shape (N, 3)
        对应的目标最近点坐标 u_i。
    valid_mask : np.ndarray, shape (N,), dtype bool
        有效对应掩码；False 的顶点对 M_d / rhs_d 贡献为零。

    Returns
    -------
    M_d : sp.csr_matrix, shape (4N, 4N)
        数据项正规方程矩阵，对称正半定（块对角）。
    rhs_d : np.ndarray, shape (4N, 3)
        数据项右端项，3 列分别对应 x、y、z 方向。
    """
    n   = source_pts.shape[0]
    dim = 4 * n

    # 齐次坐标 ṽ_i = [v_x, v_y, v_z, 1]，shape (N, 4)
    homogeneous = np.hstack(
        [source_pts, np.ones((n, 1), dtype=np.float64)]
    )

    valid_idx = np.where(valid_mask)[0]   # (K,)
    K = len(valid_idx)

    if K == 0:
        return (
            sp.csr_matrix((dim, dim), dtype=np.float64),
            np.zeros((dim, 3), dtype=np.float64),
        )

    vh    = homogeneous[valid_idx]        # (K, 4)：有效顶点齐次坐标
    uc    = closest_pts[valid_idx]        # (K, 3)：对应目标最近点
    bases = (4 * valid_idx).astype(np.int32)  # (K,)：每顶点在 4N 向量中的起始索引

    c_range = np.arange(4, dtype=np.int32)  # [0, 1, 2, 3]

    # ------------------------------------------------------------------
    # 组装 M_d（块对角 COO → CSR）
    # rows[k, c, cp] = bases[k] + c
    # cols[k, c, cp] = bases[k] + cp
    # vals[k, c, cp] = vh[k, c] * vh[k, cp]
    # ------------------------------------------------------------------
    # 生成行、列索引张量，均需为 (K, 4, 4) → ravel 后 K*16 个元素
    # 注意：(bases+c)[:,:,None] 形状 (K,4,1)，需用 np.tile 扩展至 (K,4,4)
    row_3d = np.tile(
        (bases[:, None] + c_range[None, :])[:, :, None], (1, 1, 4)
    ).ravel()   # (K*16,)
    col_3d = np.tile(
        (bases[:, None] + c_range[None, :])[:, None, :], (1, 4, 1)
    ).ravel()   # (K*16,)
    # 外积 vh[k,:] ⊗ vh[k,:]，shape (K, 4, 4)
    val_3d = (vh[:, :, None] * vh[:, None, :]).ravel()                 # (K*16,)

    M_d = sp.csr_matrix(
        (val_3d, (row_3d, col_3d)),
        shape=(dim, dim),
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # 组装 rhs_d（外积 vh[k,:] ⊗ uc[k,:] 散射到 rhs_d）
    # rhs_d[bases[k]+c, coord] += vh[k, c] * uc[k, coord]
    # ------------------------------------------------------------------
    rhs_d       = np.zeros((dim, 3), dtype=np.float64)
    # rhs_block[k, c, coord] = vh[k,c] * uc[k,coord]，shape (K, 4, 3)
    rhs_block   = vh[:, :, None] * uc[:, None, :]                   # (K, 4, 3)
    rhs_rows    = (bases[:, None] + c_range[None, :]).ravel()       # (K*4,)：行索引
    rhs_vals    = rhs_block.reshape(K * 4, 3)                       # (K*4, 3)

    np.add.at(rhs_d, rhs_rows, rhs_vals)

    return M_d, rhs_d


# =============================================================================
# 步骤 3：Landmark 项正规方程矩阵 M_l = A_l^T·A_l，右端项 rhs_l = A_l^T·B_l
# =============================================================================

def _build_landmark_normal_matrix(
    source_pts: np.ndarray,
    landmarks:  Dict[int, np.ndarray],
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    组装 Landmark 约束项的正规方程矩阵 M_l 与右端项 rhs_l（向量化实现）。

    Landmark 能量定义
    -----------------
    E_l = Σ_{k∈L} ‖X_k·ṽ_k − l_k‖²

    结构与数据项完全相同，区别在于：
        1. 只对 landmark 顶点（|L| 个）构造矩阵行
        2. 目标坐标直接使用用户指定的 l_k（精确 landmark 位置），
           而非 KD-Tree 查询的最近点

    权重 γ_l 由调用方（NicpSolver）在组装总矩阵时乘入，
    本函数返回"未加权"版本 M_l 和 rhs_l。

    Parameters
    ----------
    source_pts : np.ndarray, shape (N, 3)
        当前迭代的变形后源网格顶点坐标（用于更新齐次坐标 ṽ_k）。
    landmarks : Dict[int, np.ndarray]
        Landmark 约束字典 {源顶点索引 k: 目标坐标 l_k (shape (3,))}。

    Returns
    -------
    M_l : sp.csr_matrix, shape (4N, 4N)
        Landmark 项正规方程矩阵（未乘 γ_l），对称正半定（块对角）。
    rhs_l : np.ndarray, shape (4N, 3)
        Landmark 项右端项（未乘 γ_l）。
    """
    n   = source_pts.shape[0]
    dim = 4 * n

    if not landmarks:
        return (
            sp.csr_matrix((dim, dim), dtype=np.float64),
            np.zeros((dim, 3), dtype=np.float64),
        )

    # 齐次坐标
    homogeneous = np.hstack(
        [source_pts, np.ones((n, 1), dtype=np.float64)]
    )  # (N, 4)

    # 枚举 landmark 顶点（按索引升序处理，保持确定性）
    lm_keys  = np.array(sorted(landmarks.keys()), dtype=np.int32)     # (L,)
    lm_tgts  = np.array(
        [landmarks[int(k)] for k in lm_keys], dtype=np.float64
    )  # (L, 3)

    vh    = homogeneous[lm_keys]               # (L, 4)：landmark 齐次坐标
    bases = (4 * lm_keys).astype(np.int32)    # (L,)
    L     = len(lm_keys)
    c_range = np.arange(4, dtype=np.int32)

    # 外积块 M_l（与 _build_data_normal_matrix 结构完全相同）
    # row/col 需广播至 (L, 4, 4) → ravel 后 L*16 个元素，再与 val_3d 长度匹配
    row_3d = np.tile(
        (bases[:, None] + c_range[None, :])[:, :, None], (1, 1, 4)
    ).ravel()
    col_3d = np.tile(
        (bases[:, None] + c_range[None, :])[:, None, :], (1, 4, 1)
    ).ravel()
    val_3d = (vh[:, :, None] * vh[:, None, :]).ravel()

    M_l = sp.csr_matrix(
        (val_3d, (row_3d, col_3d)),
        shape=(dim, dim),
        dtype=np.float64,
    )

    # 右端项 rhs_l
    rhs_l     = np.zeros((dim, 3), dtype=np.float64)
    rhs_block = vh[:, :, None] * lm_tgts[:, None, :]    # (L, 4, 3)
    rhs_rows  = (bases[:, None] + c_range[None, :]).ravel()  # (L*4,)
    rhs_vals  = rhs_block.reshape(L * 4, 3)

    np.add.at(rhs_l, rhs_rows, rhs_vals)

    return M_l, rhs_l


# =============================================================================
# 辅助：由求解结果 X 更新变形顶点坐标
# =============================================================================

def _apply_transformations(
    source_pts: np.ndarray,
    X:          np.ndarray,
) -> np.ndarray:
    """
    将每顶点仿射变换矩阵 X 作用于源网格顶点，计算变形后坐标。

    变换公式
    --------
    deformed_i = X_i · ṽ_i

    其中：
        X_i   ∈ ℝ^{3×4} — 顶点 i 的仿射变换，存储为 X[4i:4i+4, :].T
        ṽ_i   ∈ ℝ^4    — 齐次坐标 [v_i; 1]

    等价向量化写法（无 Python 循环）：
        deformed[i] = ṽ_i @ X[4i:4i+4, :]
        → 将 X reshape 为 (N, 4, 3)，用 einsum 一次完成所有顶点

    Parameters
    ----------
    source_pts : np.ndarray, shape (N, 3)
        当前迭代起点的源网格顶点坐标（齐次化的基准位置，
        注意：每轮内层迭代 ṽ_i 始终以**原始源顶点**为基准，
        由外层调用者传入 original_source_pts，详见 NicpSolver.execute()）。
    X : np.ndarray, shape (4N, 3)
        全局仿射变换矩阵，X[4i:4i+4, :] = X_i^T ∈ ℝ^{4×3}。

    Returns
    -------
    deformed : np.ndarray, shape (N, 3)
        变形后的顶点坐标。
    """
    n = source_pts.shape[0]

    # 齐次坐标 ṽ_i = [v_x, v_y, v_z, 1]，shape (N, 4)
    homogeneous = np.hstack(
        [source_pts, np.ones((n, 1), dtype=np.float64)]
    )

    # X reshape 为 (N, 4, 3)，每个 [4i:4i+4, :] 对应顶点 i 的 X_i^T
    X_blocked = X.reshape(n, 4, 3)

    # deformed[i] = ṽ_i @ X_i^T，等价于 X_i @ ṽ_i
    # einsum 'ni,nij->nj': (N,4),(N,4,3)->(N,3)
    deformed = np.einsum("ni,nij->nj", homogeneous, X_blocked)

    return deformed


# =============================================================================
# NICP 求解器（核心类）
# =============================================================================

class NicpSolver:
    """
    基于 Optimal Step N-ICP 的非刚性网格配准求解器。

    算法输入
    --------
        source : MeshData  — 待配准的源网格（将被变形）
        target : MeshData  — 目标网格（形状参考）

    算法输出
    --------
        deformed : np.ndarray, shape (N, 3)
            源网格变形后的顶点坐标，使其形状尽量贴近目标网格。

    能量函数（三项之和）
    --------------------
        E(X) = α·E_s  +  E_d  +  γ_l·E_l

        E_s — 刚性项（Stiffness）：惩罚相邻顶点变换差异，防止过度非刚性形变
        E_d — 数据项（Data）    ：驱使变形后顶点贴近目标网格表面
        E_l — Landmark 项       ：精确满足已知手动对应的约束

    多尺度策略
    ----------
    stiffness_weights 列表中每个 α 对应一轮"外层迭代"，
    每轮执行 inner_iterations 次内层迭代（闭点更新 + 线性求解 + 顶点更新）。
    α 从大到小衰减，逐步放松刚性约束，实现粗到细的精细配准。

    Parameters
    ----------
    landmarks : Dict[int, np.ndarray]
        已知对应约束字典 {源顶点索引 k: 目标坐标 l_k (shape (3,))}。
        建议提供 6~20 个分布均匀的关键点（如人脸的鼻尖、眼角、嘴角等）。
        为空字典时仍可运行，但纯数据项驱动的配准收敛可能较慢。
    stiffness_weights : List[float], optional
        手动指定多尺度刚性权重列表（从大到小）。
        若提供此参数，则 alpha_start / alpha_end / n_steps 三个参数均被忽略。
        通常无需手动指定，使用对数等间距自动生成即可。
    alpha_start : float, optional
        自动生成刚性权重序列的起始值（最大刚性），默认 100.0。
        大 α → 网格近似刚体，宏观对齐阶段。
    alpha_end : float, optional
        自动生成刚性权重序列的终止值（最小刚性），默认 0.1。
        小 α → 精细非刚性形变，局部贴合阶段。
    n_steps : int, optional
        自动生成刚性权重序列的步数，默认 8。
        权重序列由 np.logspace(log10(alpha_start), log10(alpha_end), n_steps) 生成，
        保证每步相对变化量相同（等比数列），比手动指定的等差/稀疏列表更均匀。
        例如默认参数生成：[100, 46.4, 21.5, 10.0, 4.64, 2.15, 1.0, 0.1]。
    inner_iterations : int, optional
        每个刚性权重下的内层迭代次数（最近点更新 + 线性求解），默认 3。
    gamma : float, optional
        G 矩阵第 4 个对角元，控制平移分量的刚性惩罚权重，默认 1.0。
        增大 gamma 会使相邻顶点的**平移分量**也趋于一致（更保守）；
        减小 gamma 允许更自由的局部平移（更灵活）。
    gamma_landmark : float, optional
        Landmark 约束的能量权重 γ_l，默认 10.0。
        过大会导致非 landmark 区域配准质量下降；过小则 landmark 不精确。
    distance_threshold : float, optional
        数据项离群点距离阈值（None = 自适应）。
    normal_angle_threshold : float, optional
        法线夹角过滤阈值（度数，0~180）。
        源顶点法线与目标最近点法线夹角超过此值时，该对应被标记为无效（权重=0）。
        建议值：60~90°，None 表示不启用法线过滤（默认）。
        启用后可显著改善眼眶、鼻翼、嘴角等曲率变化大区域的配准质量，
        避免把曲面正面顶点错误对应到曲面背面（如眼睑内外翻转）。
    convergence_tol : float, optional
        内层收敛阈值：当相邻两次迭代变形顶点最大位移 < 此值时提前终止，默认 1e-4。
    regularization : float, optional
        Tikhonov 正则化系数 λ，加在总矩阵对角线上确保正定性，默认 1e-6。
        若遇到 "Factor is exactly singular" 错误可适当增大（如 1e-4）。
    verbose : bool, optional
        是否打印迭代进度日志，默认 False。
    step_callback : Callable[[int, float, np.ndarray], None], optional
        每次内层迭代结束后的回调函数：(iteration_index, alpha, deformed_pts)。
        可用于将中间结果实时更新到 DCC 视口（预览模式）。
    """

    def __init__(
        self,
        landmarks:              Dict[int, np.ndarray],
        stiffness_weights:      Optional[List[float]] = None,
        alpha_start:            float                 = 100.0,
        alpha_end:              float                 = 0.1,
        n_steps:                int                   = 8,
        inner_iterations:       int                   = 3,
        gamma:                  float                 = 1.0,
        gamma_landmark:         float                 = 10.0,
        distance_threshold:     Optional[float]       = None,
        normal_angle_threshold: Optional[float]       = None,
        convergence_tol:        float                 = 1e-4,
        regularization:         float                 = 1e-6,
        verbose:                bool                  = False,
        step_callback:          Optional[Callable[[int, float, np.ndarray], None]] = None,
    ) -> None:
        # 规范化 landmarks 字典
        self.landmarks: Dict[int, np.ndarray] = {
            int(k): np.asarray(v, dtype=np.float64)
            for k, v in landmarks.items()
        }

        # 多尺度刚性权重
        # 优先使用手动指定列表；否则用对数等间距自动生成（等比数列）。
        # 等比数列保证每步相对变化量相同，粗/精配准阶段均有均匀覆盖，
        # 比手动枚举的稀疏列表收敛更平滑。
        if stiffness_weights is not None:
            self.stiffness_weights: List[float] = list(stiffness_weights)
        else:
            self.stiffness_weights = list(
                np.logspace(
                    np.log10(float(alpha_start)),
                    np.log10(float(alpha_end)),
                    int(n_steps),
                )
            )

        self.alpha_start = float(alpha_start)
        self.alpha_end   = float(alpha_end)
        self.n_steps     = int(n_steps)

        self.inner_iterations:       int            = inner_iterations
        self.gamma:                   float          = gamma
        self.gamma_landmark:          float          = gamma_landmark
        self.distance_threshold:      Optional[float]= distance_threshold
        self.normal_angle_threshold:  Optional[float]= normal_angle_threshold
        self.convergence_tol:         float          = convergence_tol
        self.regularization:          float          = regularization
        self.verbose:                 bool           = verbose
        self.step_callback            = step_callback

    # =========================================================================
    # 私有：验证输入并初始化变换矩阵 X
    # =========================================================================

    @staticmethod
    def _initial_transformations(n_verts: int) -> np.ndarray:
        """
        初始化每顶点仿射变换矩阵 X 为恒等变换（无形变初始状态）。

        X ∈ ℝ^{4N×3}，其中 X[4i:4i+4, :] = X_i^T。

        恒等变换意味着 X_i = [I_3 | 0]（3×3 单位矩阵加零平移），
        即 X_i^T = [I_3; 0^T]，存储为 X[4i:4i+4, :]：

            X[4i:4i+3, :]  = I_3   (旋转/缩放部分)
            X[4i+3,    :]  = [0,0,0] (平移部分)

        使用恒等初始值时，第一轮最近点查询使用原始源顶点坐标，
        有助于在 α 较大的早期迭代中保持刚性约束的稳定性。

        Parameters
        ----------
        n_verts : int
            源网格顶点总数 N。

        Returns
        -------
        X : np.ndarray, shape (4N, 3), dtype float64
            初始化为恒等变换的全局仿射变换矩阵。
        """
        X = np.zeros((4 * n_verts, 3), dtype=np.float64)
        # 每个顶点的 3×3 旋转块置为单位矩阵
        for i in range(n_verts):
            X[4 * i:4 * i + 3, :] = np.eye(3, dtype=np.float64)
        return X

    # =========================================================================
    # 主入口：执行完整 N-ICP 配准
    # =========================================================================

    def execute(self, source: MeshData, target: MeshData) -> np.ndarray:
        """
        执行完整的 Optimal Step N-ICP 非刚性配准，返回变形后的源网格顶点坐标。

        完整流程
        --------
        [预计算阶段]
        1. 输入合法性校验（landmark 索引范围、网格维度）
        2. 从源网格提取无向边列表（用于刚性项）
        3. 预计算 Landmark 约束矩阵 M_l、rhs_l（不随迭代变化，仅在 X 更新时重新计算）
        4. 初始化变换矩阵 X 为恒等变换

        [外层循环：遍历 stiffness_weights 列表中每个 α]
        5. 组装刚性项矩阵 M_s（仅依赖 α 和边图，本轮内层迭代共享）

        [内层循环：每个 α 下迭代 inner_iterations 次]
        6. 计算当前变形顶点（apply_transformations）
        7. KD-Tree 最近点查询（数据项对应更新）
        8. 组装数据项矩阵 M_d、rhs_d
        9. 组装 Landmark 矩阵 M_l、rhs_l（用当前变形顶点更新齐次坐标）
        10. 组装总矩阵 M = M_s + M_d + γ_l·M_l，施加正则化
        11. Cholesky 分解并求解 M·X = rhs_d + γ_l·rhs_l
        12. 更新变形顶点，计算收敛指标，条件性提前退出
        13. step_callback 回调（用于 DCC 视口预览）

        Parameters
        ----------
        source : MeshData
            原始（未变形）源网格。顶点坐标 shape (N, 3)，面片 shape (F, 3)。
        target : MeshData
            目标网格（配准参考）。顶点坐标 shape (M, 3)，面片 shape (T, 3)。

        Returns
        -------
        deformed : np.ndarray, shape (N, 3)
            源网格配准变形后的顶点坐标。

        Raises
        ------
        ValueError
            若 landmark 索引超出 [0, N-1] 或源/目标网格顶点维度不为 3。
        RuntimeError
            若线性求解失败（通常由正则化不足导致，可增大 regularization）。
        """
        n_src: int = source.num_vertices
        n_tgt: int = target.num_vertices

        # ── 输入合法性校验 ──────────────────────────────────────────────────
        for lm_idx in self.landmarks:
            if not (0 <= lm_idx < n_src):
                raise ValueError(
                    f"Landmark 索引 {lm_idx} 超出源网格合法范围 [0, {n_src - 1}]。"
                )

        if self.verbose:
            print(
                f"[N-ICP] 源网格：{n_src} 顶点，{source.num_faces} 面片"
            )
            print(
                f"[N-ICP] 目标网格：{n_tgt} 顶点，{target.num_faces} 面片"
            )
            print(
                f"[N-ICP] Landmark 约束：{len(self.landmarks)} 个"
            )
            print(
                f"[N-ICP] 刚性权重序列：{self.stiffness_weights}"
            )

        # ── 坐标归一化（关键！）────────────────────────────────────────────
        # 算法中刚性项 M_s 对角元 ≈ α，数据项 M_d 对角元 ≈ ‖ṽ_i‖² = vx²+vy²+vz²+1。
        # 若网格坐标较大（如 Maya 世界空间 ~100 单位），M_d 远大于 M_s，
        # 刚性项完全被压制，导致无刚性约束效果。
        # 解决方案：将 source + target 整体归一化到 [-1, 1] 内（基于联合包围盒），
        # 算法全程在归一化坐标系内运行，最后将结果反归一化回原始坐标系。
        all_verts = np.vstack([source.vertices, target.vertices])
        bbox_min  = all_verts.min(axis=0)   # (3,)
        bbox_max  = all_verts.max(axis=0)   # (3,)
        norm_center: np.ndarray = (bbox_min + bbox_max) * 0.5
        norm_scale: float = float(np.max(bbox_max - bbox_min)) * 0.5
        if norm_scale < 1e-10:
            norm_scale = 1.0   # 防止退化

        src_verts_norm = (source.vertices - norm_center) / norm_scale  # [-1, 1]
        tgt_verts_norm = (target.vertices - norm_center) / norm_scale

        # 归一化后的 MeshData（仅替换顶点坐标，面片拓扑不变）
        source_norm = MeshData(src_verts_norm, source.faces)
        target_norm = MeshData(tgt_verts_norm, target.faces)

        # Landmark 目标坐标也须同步归一化
        landmarks_norm: Dict[int, np.ndarray] = {
            k: (v - norm_center) / norm_scale
            for k, v in self.landmarks.items()
        }

        if self.verbose:
            print(
                f"[N-ICP] 坐标归一化：center={norm_center}, scale={norm_scale:.4g}"
                f"  （原始坐标范围：{bbox_min} ~ {bbox_max}）"
            )

        # ── 以下所有计算均使用归一化坐标 ──────────────────────────────────
        source  = source_norm    # type: ignore[assignment]
        target  = target_norm    # type: ignore[assignment]

        # 距离阈值同步归一化（若用户手动指定）
        dist_threshold_norm: Optional[float] = (
            self.distance_threshold / norm_scale
            if self.distance_threshold is not None else None
        )

        # ── 预计算：边列表（刚性项提取一次即可） ────────────────────────────
        if self.verbose:
            print("[N-ICP] 构建无向边列表...")
        edges = _build_edge_list(source)
        if self.verbose:
            print(f"[N-ICP] 共 {len(edges)} 条无向边")

        # ── 预计算：目标网格顶点法线（固定不变，提取一次）──────────────────
        use_normals = self.normal_angle_threshold is not None
        target_normals: Optional[np.ndarray] = None
        if use_normals:
            target_normals = _compute_vertex_normals(target.vertices, target.faces)
            if self.verbose:
                print(
                    f"[N-ICP] 法线角度过滤已启用，阈值 = {self.normal_angle_threshold:.1f}°"
                )

        # ── 初始化变换矩阵 X（恒等变换）────────────────────────────────────
        X: np.ndarray = self._initial_transformations(n_src)

        # Cholesky 求解器实例（每次 α 改变重新分解）
        chol = CholeskyFactor()

        # ── 全局迭代计数器（用于 step_callback） ────────────────────────────
        global_iter = 0

        # ── 外层循环：遍历刚性权重 α ─────────────────────────────────────────
        for alpha in self.stiffness_weights:

            if self.verbose:
                print(f"\n[N-ICP] ── 刚性权重 α = {alpha:.4g} ──")

            # 步骤 5：本级 α 的刚性项矩阵（内层迭代共享，只组装一次）
            M_s = _build_stiffness_normal_matrix(
                n_verts = n_src,
                edges   = edges,
                alpha   = alpha,
                gamma   = self.gamma,
            )

            # ── 内层循环：最近点更新 + 线性求解 ──────────────────────────────
            for inner in range(self.inner_iterations):

                # 步骤 6：当前变形顶点坐标
                # 注意：ṽ_i 始终基于原始源顶点（论文中的 v_i），而不是前一步的
                # 变形顶点，这确保仿射变换 X_i 的几何意义保持一致。
                deformed: np.ndarray = _apply_transformations(source.vertices, X)

                # 步骤 6.5：若启用法线过滤，对当前变形后网格重新计算顶点法线
                # （每次内层迭代形状均有变化，须重新计算以保持法线精度）
                source_normals: Optional[np.ndarray] = None
                if use_normals:
                    source_normals = _compute_vertex_normals(deformed, source.faces)

                # 步骤 7：KD-Tree 最近点查询（用当前变形顶点查询目标表面）
                closest_pts, valid_mask = _find_closest_points(
                    source_pts             = deformed,
                    target_pts             = target.vertices,
                    distance_threshold     = dist_threshold_norm,
                    source_normals         = source_normals,
                    target_normals         = target_normals,
                    normal_angle_threshold = self.normal_angle_threshold,
                )
                n_valid = int(np.sum(valid_mask))

                # 步骤 8：数据项正规方程矩阵（用‌原始源顶点的齐次坐标构造 ṽ_i）
                # 理由：X_i 变换的输入是原始 ṽ_i，数据项须与 X 的语义一致
                M_d, rhs_d = _build_data_normal_matrix(
                    source_pts  = source.vertices,
                    closest_pts = closest_pts,
                    valid_mask  = valid_mask,
                )

                # 步骤 9：Landmark 项（同样基于原始源顶点的齐次坐标）
                # 使用归一化后的 landmarks_norm（与 source.vertices 坐标系一致）
                M_l, rhs_l = _build_landmark_normal_matrix(
                    source_pts = source.vertices,
                    landmarks  = landmarks_norm,
                )

                # 步骤 10：组装总矩阵 M 和右端项 rhs
                # M   = M_s + M_d + γ_l·M_l
                # rhs = rhs_d + γ_l·rhs_l
                M   = M_s + M_d + self.gamma_landmark * M_l
                rhs = rhs_d + self.gamma_landmark * rhs_l  # (4N, 3)

                # Tikhonov 正则化：M += λ·I，防止奇异
                if self.regularization > 0.0:
                    M = M + self.regularization * sp.eye(
                        M.shape[0], format="csc", dtype=np.float64
                    )

                # 步骤 11：Cholesky 分解并求解
                # 每次内层迭代 M_d 改变（最近点更新），需重新分解
                try:
                    chol.factorization(M)
                except Exception as exc:
                    raise RuntimeError(
                        f"Cholesky 分解失败（α={alpha:.4g}，内层迭代 {inner+1}）：{exc}\n"
                        "建议增大 regularization 参数（如 1e-4）。"
                    ) from exc

                # 求解 M·X_new = rhs（3 列分别求解，由 CholeskyFactor 内部处理）
                X_new: np.ndarray = chol.solve(rhs)  # (4N, 3)

                # 步骤 12：计算收敛指标（变形顶点位移最大值）
                deformed_new = _apply_transformations(source.vertices, X_new)
                max_disp     = float(np.max(np.linalg.norm(deformed_new - deformed, axis=1)))

                X = X_new  # 更新变换矩阵

                if self.verbose:
                    n_rejected = n_src - n_valid
                    print(
                        f"[N-ICP]   α={alpha:.4g}  内层迭代 {inner+1:2d}/{self.inner_iterations}"
                        f"  有效对应: {n_valid}/{n_src}"
                        + (f"  剔除: {n_rejected}（距离+法线）" if use_normals and n_rejected > 0
                           else (f"  剔除: {n_rejected}" if n_rejected > 0 else ""))
                        + f"  最大位移: {max_disp:.6e}"
                    )

                # 步骤 13：step_callback 回调（例如更新 DCC 视口）
                # 注意：回调接收的是原始坐标系下的顶点，需先反归一化
                if self.step_callback is not None:
                    deformed_world = deformed_new * norm_scale + norm_center
                    self.step_callback(global_iter, alpha, deformed_world)

                global_iter += 1

                # 提前收敛检测
                if max_disp < self.convergence_tol:
                    if self.verbose:
                        print(
                            f"[N-ICP]   已收敛（位移 {max_disp:.2e} "
                            f"< 阈值 {self.convergence_tol:.2e}），提前终止本级迭代。"
                        )
                    break

        # ── 返回最终变形顶点坐标（反归一化回原始坐标系）───────────────────
        final_deformed_norm = _apply_transformations(source.vertices, X)

        # 反归一化：将归一化坐标系下的结果还原为原始 Maya/世界坐标
        final_deformed = final_deformed_norm * norm_scale + norm_center

        if self.verbose:
            # 统计最终配准质量（在原始坐标系下计算，更直观）
            tree_final  = cKDTree(target.vertices * norm_scale + norm_center)
            final_dists, _ = tree_final.query(final_deformed, k=1, workers=-1)
            print(
                f"\n[N-ICP] 配准完成。"
                f"  平均距离: {float(np.mean(final_dists)):.6e}"
                f"  最大距离: {float(np.max(final_dists)):.6e}"
            )

        return final_deformed


# =============================================================================
# 便利工厂函数
# =============================================================================

def create_nicp_solver(
    landmarks:              Dict[int, np.ndarray],
    stiffness_weights:      Optional[List[float]] = None,
    alpha_start:            float                 = 100.0,
    alpha_end:              float                 = 0.1,
    n_steps:                int                   = 8,
    inner_iterations:       int                   = 3,
    gamma:                  float                 = 1.0,
    gamma_landmark:         float                 = 10.0,
    distance_threshold:     Optional[float]       = None,
    normal_angle_threshold: Optional[float]       = None,
    convergence_tol:        float                 = 1e-4,
    regularization:         float                 = 1e-6,
    verbose:                bool                  = False,
    step_callback:          Optional[Callable[[int, float, np.ndarray], None]] = None,
) -> NicpSolver:
    """
    快速创建 NicpSolver 实例的工厂函数。

    Parameters
    ----------
    landmarks : Dict[int, np.ndarray]
        已知对应约束字典 {源顶点索引: 目标坐标 (3,)}。
        建议至少提供 6 个分布在关键位置的 landmark。
    stiffness_weights : List[float], optional
        手动指定刚性权重列表（从大到小）。
        若提供，则 alpha_start / alpha_end / n_steps 被忽略。
    alpha_start : float, optional
        刚性权重序列起始值（最大刚性），默认 100.0。
    alpha_end : float, optional
        刚性权重序列终止值（最小刚性），默认 0.1。
    n_steps : int, optional
        刚性权重步数，默认 8。由 np.logspace 生成对数等间距等比数列，
        例如默认参数生成：[100, 46.4, 21.5, 10.0, 4.64, 2.15, 1.0, 0.1]。
    inner_iterations : int, optional
        每个刚性权重下的内层迭代次数，默认 3。
    gamma : float, optional
        G 矩阵平移分量权重，默认 1.0。
    gamma_landmark : float, optional
        Landmark 约束能量权重 γ_l，默认 10.0。
    distance_threshold : float, optional
        离群点距离阈值（None = 自适应中位数×2）。
    normal_angle_threshold : float, optional
        法线夹角过滤阈值（度数）。源与目标对应顶点法线夹角超过此值时，
        该对应被丢弃。建议值：60~90°，None 表示不启用（默认）。
    convergence_tol : float, optional
        内层迭代收敛阈值，默认 1e-4。
    regularization : float, optional
        正则化系数 λ，默认 1e-6。出现奇异错误时可增大至 1e-4。
    verbose : bool, optional
        是否打印迭代日志，默认 False。
    step_callback : Callable[[int, float, np.ndarray], None], optional
        每次内层迭代的回调：(全局迭代序号, 当前 α, 变形顶点) → None。

    Returns
    -------
    solver : NicpSolver
        配置好的 N-ICP 求解器实例，可直接调用 solver.execute(source, target)。

    Examples
    --------
    >>> import numpy as np
    >>> from NICP import MeshData, create_nicp_solver
    >>>
    >>> # source / target 网格由上层应用层（Maya / Blender 等）提供
    >>> src_verts = np.array(...)   # shape (N, 3)
    >>> src_faces = np.array(...)   # shape (F, 3)
    >>> tgt_verts = np.array(...)   # shape (M, 3)
    >>> tgt_faces = np.array(...)   # shape (T, 3)
    >>>
    >>> source = MeshData(src_verts, src_faces)
    >>> target = MeshData(tgt_verts, tgt_faces)
    >>>
    >>> # 指定 landmark 约束：源顶点 42 → 目标位置 [1, 0, 0]
    >>> landmarks = {42: [1.0, 0.0, 0.0], 100: [0.0, 1.0, 0.5]}
    >>>
    >>> solver   = create_nicp_solver(landmarks, verbose=True)
    >>> deformed = solver.execute(source, target)   # shape (N, 3)
    """
    return NicpSolver(
        landmarks              = landmarks,
        stiffness_weights      = stiffness_weights,
        alpha_start            = alpha_start,
        alpha_end              = alpha_end,
        n_steps                = n_steps,
        inner_iterations       = inner_iterations,
        gamma                  = gamma,
        gamma_landmark         = gamma_landmark,
        distance_threshold     = distance_threshold,
        normal_angle_threshold = normal_angle_threshold,
        convergence_tol        = convergence_tol,
        regularization         = regularization,
        verbose                = verbose,
        step_callback          = step_callback,
    )
