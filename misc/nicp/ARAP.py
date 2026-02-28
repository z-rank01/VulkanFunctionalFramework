"""
基于 ARAP (As-Rigid-As-Possible) 变形能量的非刚性网格配准核心算法模块。

算法参考
--------
Sorkine, O., & Alexa, M. (2007).
    As-rigid-as-possible surface modeling.
    Eurographics Symposium on Geometry Processing (SGP), pp. 109-116.

算法概述
--------
给定源网格顶点坐标 P = {p_1, ..., p_N}, 以及一组硬约束 (已知的对应点对) , 
求解变形后坐标 P' = {p'_1, ..., p'_N}, 使下式能量最小:

    E(P') = Σ_i Σ_{j∈N(i)} w_ij ‖(p'_i - p'_j) - R_i(p_i - p_j)‖²

    其中:
        N(i)        — 顶点 i 的一环邻域 (one-ring neighbors) 
        w_ij        — 余切权重 (cotangent weight) 
        R_i ∈ SO(3) — 顶点 i 处的局部最优旋转矩阵

迭代策略 (局部-全局交替优化) 
    局部步 (Local Step) :  固定 P', 对每个顶点用 SVD 求解最优 R_i
    全局步 (Global Step) :  固定 R_i, 求解全局稀疏线性方程组更新 P'

设计原则
--------
本模块为纯算法层, 仅依赖 numpy 与 scipy, 与任何 DCC 工具完全解耦。
可在 Maya、Blender、Houdini 等任意支持 Python 的环境中使用。

依赖
----
    numpy  >= 1.20
    scipy  >= 1.6
    scikit-sparse (可选, 提供 cholmod 后端, 速度更快) 
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =============================================================================
# 数据结构:与 DCC 解耦的三角网格
# =============================================================================

class MeshData:
    """
    与 DCC 工具无关的三角网格数据结构。

    本类仅存储算法所需的顶点坐标与面片索引, 不包含任何 DCC 特定对象。
    应用层 (Maya / Blender 等) 负责将 DCC 内的网格数据转换为本类实例。

    Attributes
    ----------
    vertices : np.ndarray, shape (N, 3), dtype float64
        顶点坐标数组, N 为顶点数。
    faces : np.ndarray, shape (F, 3), dtype int32
        三角面片索引数组, 每行为一个三角形的三顶点索引。
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """
        Parameters
        ----------
        vertices : array_like, shape (N, 3)
            顶点坐标, 将被转换为 float64 存储。
        faces : array_like, shape (F, 3)
            三角面片顶点索引, 将被转换为 int32 存储。

        Raises
        ------
        ValueError
            若 vertices 不为 (N, 3) 或 faces 不为 (F, 3)。
        """
        self.vertices: np.ndarray = np.asarray(vertices, dtype=np.float64)
        self.faces:    np.ndarray = np.asarray(faces,    dtype=np.int32)

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(
                f"vertices 应为 shape (N, 3), 实际 shape: {self.vertices.shape}"
            )
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(
                f"faces 应为 shape (F, 3), 实际 shape: {self.faces.shape}"
            )

    @property
    def num_vertices(self) -> int:
        """顶点总数 N。"""
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        """面片总数 F。"""
        return self.faces.shape[0]

    def __repr__(self) -> str:
        return f"MeshData(vertices={self.num_vertices}, faces={self.num_faces})"


# =============================================================================
# Cholesky 分解封装 (稀疏对称正定矩阵求解器) 
# =============================================================================

class CholeskyFactor:
    """
    稀疏对称正定矩阵的线性方程组求解器。

    优先尝试使用 scikit-sparse 提供的 CHOLMOD 超节点 Cholesky 分解; 
    若不可用, 退回到 scipy.sparse.linalg.factorized (基于 SuperLU 的直接 LU 分解) 。

    CHOLMOD 通常比 SuperLU 快 5~10 倍, 建议在处理大规模网格时安装:
        pip install scikit-sparse

    Attributes
    ----------
    _factor : object or None
        已完成分解的因子对象, factorization() 调用后有效。
    _use_cholmod : bool
        标识当前是否使用 CHOLMOD 后端。
    """

    def __init__(self) -> None:
        self._factor = None
        self._use_cholmod: bool = False

        try:
            from sksparse.cholmod import cholesky as _cholmod_fn  # type: ignore
            self._cholmod_fn = _cholmod_fn
            self._use_cholmod = True
        except ImportError:
            self._use_cholmod = False

    @property
    def backend(self) -> str:
        """返回当前实际使用的求解后端名称:'cholmod' 或 'superlu'。"""
        return "cholmod" if self._use_cholmod else "superlu"

    def factorization(self, A: sp.spmatrix) -> None:
        """
        对稀疏对称正定矩阵 A 进行分解, 并将因子缓存供后续 solve() 调用。

        Parameters
        ----------
        A : sp.spmatrix, shape (M, M)
            约束消元后的自由子系统矩阵, 须为对称正定稀疏矩阵。
            M = N - len(constraints), N 为网格顶点总数。

        Raises
        ------
        ValueError
            若 A 不是方阵。
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"矩阵 A 须为方阵, 实际 shape: {A.shape}")

        A_csc = A.tocsc()
        if self._use_cholmod:
            self._factor = self._cholmod_fn(A_csc)
        else:
            self._factor = spla.factorized(A_csc)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        利用已缓存的分解因子求解线性方程组 A x = b。

        Parameters
        ----------
        b : np.ndarray, shape (M,) or (M, K)
            右端向量 (K=1) 或矩阵 (K 列同时求解) 。
            M 须与 factorization() 时矩阵阶数一致。

        Returns
        -------
        x : np.ndarray, shape (M,) or (M, K)
            线性方程组的解。

        Raises
        ------
        RuntimeError
            若在 factorization() 之前调用本方法。
        """
        if self._factor is None:
            raise RuntimeError("请先调用 factorization() 完成矩阵分解, 再调用 solve()。")

        if self._use_cholmod:
            return self._factor.solve_A(b)
        else:
            # factorized() 返回可直接调用的求解函数
            if b.ndim == 1:
                return self._factor(b)
            # 多列:逐列求解后拼接
            return np.column_stack(
                [self._factor(b[:, k]) for k in range(b.shape[1])]
            )


# =============================================================================
# ARAP 求解器 (核心算法) 
# =============================================================================

class ArapSolver:
    """
    基于 ARAP 能量的非刚性网格变形 (ARAP) 求解器。

    能量函数
    --------
    E(P') = Σ_i Σ_{j∈N(i)} w_ij ‖(p'_i - p'_j) - R_i(p_i - p_j)‖²

    约束 (landmark 对应点) 通过行列消元法强制施加, 保证精确满足硬约束。

    求解流程
    --------
    预计算:
        1. 构建顶点一环邻接表
        2. 计算余切权重, 组装 Laplacian 矩阵 L ((N, N), 对称正半定) 
        3. 约束消元:提取自由子矩阵 L_free ((M, M), 对称正定) 
        4. 对 L_free 进行 Cholesky 分解

    迭代 (重复 iterations 次) :
        局部步 (Local Step) :   对每个顶点用 SVD 求最优旋转 R_i ∈ SO(3)
        全局步 (Global Step) :  构建右端项 b, 分离约束贡献, 求解线性方程组

    Parameters
    ----------
    constraints : Dict[int, array_like]
        硬约束字典, key 为顶点索引 (int) , value 为目标三维坐标 array_like (3,)。
        例如 {0: [1.0, 2.0, 3.0], 42: [0.0, 0.0, 1.5]}。
    iterations : int
        最大迭代次数, 通常 5~20 次, 建议默认值 10。
    cholesky_factor : CholeskyFactor
        线性求解器实例, 由外部注入便于替换或扩展。
    convergence_tol : float, optional
        收敛判据:若相邻两次迭代间所有顶点位移的最大 L2 范数小于此值则提前退出。
        默认 1e-6; 设为 0 可禁用提前终止。
    regularization : float, optional
        Tikhonov 正则化系数 λ。在 Cholesky 分解前将 L_free 替换为
        L_free + λ·I，使矩阵严格正定，避免「Factor is exactly singular」错误。
        余切 Laplacian 在约束不足或网格存在退化三角形时可能奇异，
        λ 取 1e-8 ~ 1e-4 通常对变形结果影响极小。默认 1e-8。
    verbose : bool, optional
        是否打印迭代进度日志, 默认 False。
    """

    def __init__(
        self,
        constraints:     Dict[int, np.ndarray],
        iterations:      int,
        cholesky_factor: CholeskyFactor,
        convergence_tol: float = 1e-6,
        regularization:  float = 1e-8,
        verbose:         bool  = False,
        step_callback:   Optional["Callable[[int, np.ndarray], None]"] = None,
    ) -> None:
        # 规范化约束字典, 确保值为 float64 ndarray
        self.constraints: Dict[int, np.ndarray] = {
            int(k): np.asarray(v, dtype=np.float64)
            for k, v in constraints.items()
        }
        self.iterations:      int            = iterations
        self.cholesky_factor: CholeskyFactor = cholesky_factor
        self.convergence_tol: float          = convergence_tol
        self.regularization:  float          = regularization
        self.verbose:         bool           = verbose
        #: 每次迭代结束后的回调 f(iteration_index: int, p_prime: np.ndarray)。
        #: 可用于将中间结果实时更新到 DCC 视口（预览模式）。
        self.step_callback = step_callback

        # ---- 预计算缓存 (execute() 调用后填充) --------------------------

        #: 顶点一环邻接表, adjacency[i] = [j1, j2, ...] (已排序) 
        self._adjacency: Optional[List[List[int]]] = None

        #: 余切权重表, key=(min(i,j), max(i,j)) → float w_ij
        self._cotan_weights: Optional[Dict[Tuple[int, int], float]] = None

        #: 自由顶点全局索引数组, shape (M,), 升序排列
        self._free_indices: Optional[np.ndarray] = None

        #: 约束顶点全局索引数组, shape (C,), 升序排列
        self._constrained_indices: Optional[np.ndarray] = None

        #: 完整 Laplacian ((N, N) csr_matrix) , 全局步时用于修正右端项
        self._L_full: Optional[sp.csr_matrix] = None

    # =========================================================================
    # 私有工具:邻接表
    # =========================================================================

    def _build_adjacency(self, mesh: MeshData) -> List[List[int]]:
        """
        由三角面片索引构建顶点一环邻接表。

        Parameters
        ----------
        mesh : MeshData
            输入网格。

        Returns
        -------
        adjacency : List[List[int]]
            adjacency[i] 为与顶点 i 共边的所有邻居顶点索引 (升序排列) 。
        """
        n: int = mesh.num_vertices
        adj: List[set] = [set() for _ in range(n)]
        for tri in mesh.faces:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            adj[i].update((j, k))
            adj[j].update((i, k))
            adj[k].update((i, j))
        return [sorted(s) for s in adj]

    # =========================================================================
    # 步骤 1:构建余切 Laplacian
    # =========================================================================

    def build_cotan_laplacian(self, mesh: MeshData) -> sp.csr_matrix:
        """
        构建网格余切权重 Laplacian 矩阵 L。

        余切权重定义
        -----------
        对三角形 (i, j, k), 以顶点 i 为角顶点, 其对角为边 (j, k):
            w_{jk} += cot(α_i) / 2
        其中 α_i 是顶点 i 处的内角。

        最终 Laplacian (对角占优, 对称正半定) :
            L[i, i] =  Σ_{j∈N(i)} w_ij
            L[i, j] = -w_ij         (i ≠ j) 

        注意:钝角三角形可能产生负余切值, 实现中将负权重截断为 0
        以保证矩阵正半定性。

        Parameters
        ----------
        mesh : MeshData
            输入三角网格。

        Returns
        -------
        L : sp.csr_matrix, shape (N, N)
            对称正半定余切 Laplacian 矩阵。
        """
        n:  int          = mesh.num_vertices
        vp: np.ndarray   = mesh.vertices           # (N, 3)

        cotan_weights: Dict[Tuple[int, int], float] = {}

        for tri in mesh.faces:
            idx = [int(tri[0]), int(tri[1]), int(tri[2])]

            for local_vtx in range(3):
                # 以 local_vtx 为角顶点, 计算余切值, 贡献给其对边
                vi = idx[local_vtx]
                vj = idx[(local_vtx + 1) % 3]
                vk = idx[(local_vtx + 2) % 3]

                e_ij = vp[vj] - vp[vi]   # 从 vi 到 vj 的边向量, (3,)
                e_ik = vp[vk] - vp[vi]   # 从 vi 到 vk 的边向量, (3,)

                cos_a = float(np.dot(e_ij, e_ik))
                sin_a = float(np.linalg.norm(np.cross(e_ij, e_ik)))

                # 退化三角形 (共线 / 面积为零) :跳过, 避免除零
                if sin_a < 1e-10:
                    continue

                cot_a = cos_a / sin_a

                # 角 vi 的余切贡献给对边 (vj, vk)
                edge_key = (min(vj, vk), max(vj, vk))
                cotan_weights[edge_key] = cotan_weights.get(edge_key, 0.0) + 0.5 * cot_a

        # 缓存余切权重, 供局部步 (SVD 旋转) 复用
        self._cotan_weights = cotan_weights

        # ---- 组装稀疏 Laplacian -------------------------------------------
        # 每条边 (i, j) 对应 4 个非零项:
        #   L[i,j] += -w,  L[j,i] += -w,  L[i,i] += w,  L[j,j] += w
        rows: List[int]   = []
        cols: List[int]   = []
        vals: List[float] = []

        for (vi, vj), w in cotan_weights.items():
            w = max(w, 0.0)     # 截断负权重 (钝角三角形产生) 
            rows += [vi, vj, vi, vj]
            cols += [vj, vi, vi, vj]
            vals += [-w, -w,  w,  w]

        L = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(n, n),
            dtype=np.float64,
        )
        return L

    # =========================================================================
    # 步骤 2:约束消元
    # =========================================================================

    def constraints_elimination(
        self,
        L:           sp.spmatrix,
        constraints: Dict[int, np.ndarray],
    ) -> sp.csr_matrix:
        """
        从完整 Laplacian 矩阵中消去约束顶点对应的行与列, 得到自由子系统矩阵。

        消元原理
        --------
        将顶点分为自由集 f 和约束集 c, 分块方程为:

            [ L_ff  L_fc ] [ p'_f ]   [ b_f ]
            [ L_cf  L_cc ] [ p'_c ] = [ b_c ]

        因 p'_c = q_c (约束目标值已知) , 由第一行得:
            L_ff p'_f = b_f - L_fc q_c

        本方法返回 L_ff, 右端项修正在 separate_constraints() 中完成。

        Parameters
        ----------
        L : sp.spmatrix, shape (N, N)
            完整 Laplacian 矩阵。
        constraints : Dict[int, np.ndarray]
            硬约束字典 {顶点索引: 目标坐标 (3,)}。

        Returns
        -------
        L_free : sp.csr_matrix, shape (M, M)
            自由子系统矩阵, M = N - len(constraints)。
            该矩阵对称正定, 可直接进行 Cholesky 分解。
        """
        n: int = L.shape[0]
        all_idx = np.arange(n, dtype=np.int32)

        constrained_arr = np.array(sorted(constraints.keys()), dtype=np.int32)
        free_mask = np.ones(n, dtype=bool)
        free_mask[constrained_arr] = False
        free_arr = all_idx[free_mask]

        # 缓存索引, 供后续步骤复用
        self._free_indices        = free_arr
        self._constrained_indices = constrained_arr

        L_csr  = L.tocsr()
        L_free = L_csr[free_arr, :][:, free_arr]
        return L_free.tocsr()

    # =========================================================================
    # 步骤 3:初始猜测
    # =========================================================================

    def initial_guess(self, mesh: MeshData) -> np.ndarray:
        """
        以原始顶点坐标作为迭代初始值 P'⁰, 并强制满足约束。

        对约束顶点直接赋予目标坐标, 使初始值更接近最终解, 有助于加速收敛。

        Parameters
        ----------
        mesh : MeshData
            原始 (未变形) 网格。

        Returns
        -------
        p_prime : np.ndarray, shape (N, 3), dtype float64
            迭代初始坐标, 约束顶点已赋为目标值。
        """
        p_prime = mesh.vertices.copy()
        for idx, target in self.constraints.items():
            p_prime[idx] = target
        return p_prime

    # =========================================================================
    # 步骤 4 (局部步) :逐顶点 SVD 求最优旋转
    # =========================================================================

    def compute_local_rotation(
        self,
        mesh:    MeshData,
        p_prime: np.ndarray,
    ) -> List[np.ndarray]:
        """
        局部步 (Local Step) :对每个顶点用 SVD 求最优旋转矩阵 R_i ∈ SO(3)。

        推导
        ----
        固定 P', 对顶点 i 的局部能量:
            E_i(R_i) = Σ_{j∈N(i)} w_ij ‖(p'_i - p'_j) - R_i(p_i - p_j)‖²

        构建加权协方差矩阵:
            S_i = Σ_{j∈N(i)} w_ij · (p_i - p_j) · (p'_i - p'_j)ᵀ   ∈ ℝ^{3x3}

        SVD 分解:S_i = U Σ Vᵀ

        最优旋转:R_i = V Uᵀ

        若 det(R_i) = -1 (反射矩阵) , 翻转奇异值最小对应的 V 列, 
        将 R_i 强制映射回 SO(3) (行列式为 +1) 。

        Parameters
        ----------
        mesh : MeshData
            原始 (未变形) 网格, 提供原始顶点坐标 p。
        p_prime : np.ndarray, shape (N, 3)
            当前迭代的变形顶点坐标。

        Returns
        -------
        rotations : List[np.ndarray]
            长度为 N 的列表, rotations[i] 为 shape (3, 3) 的旋转矩阵 R_i。
        """
        p:   np.ndarray              = mesh.vertices   # 原始坐标, (N, 3)
        n:   int                     = mesh.num_vertices
        adj: List[List[int]]         = self._adjacency
        cw:  Dict[Tuple[int,int], float] = self._cotan_weights

        rotations: List[np.ndarray] = []

        for i in range(n):
            neighbors = adj[i]
            if not neighbors:
                # 孤立顶点 (无邻居) :退回单位旋转
                rotations.append(np.eye(3, dtype=np.float64))
                continue

            # 构建协方差矩阵 S_i: Σ_j w_ij * outer(e_orig, e_deformed)
            S = np.zeros((3, 3), dtype=np.float64)
            for j in neighbors:
                edge_key = (min(i, j), max(i, j))
                w        = cw.get(edge_key, 0.0)

                e_orig     = p[i]       - p[j]          # 原始边向量, (3,)
                e_deformed = p_prime[i] - p_prime[j]    # 变形后边向量, (3,)

                S += w * np.outer(e_orig, e_deformed)   # (3, 3)

            # SVD 分解
            try:
                U, _sigma, Vt = np.linalg.svd(S)
            except np.linalg.LinAlgError:
                # SVD 不收敛:退回单位旋转
                rotations.append(np.eye(3, dtype=np.float64))
                continue

            R = Vt.T @ U.T

            # 保证 det(R) = +1, 避免反射
            if np.linalg.det(R) < 0.0:
                Vt[-1, :] *= -1.0    # 翻转最小奇异值对应的 V 列
                R = Vt.T @ U.T

            rotations.append(R)

        return rotations

    # =========================================================================
    # 步骤 5 (全局步) :构建线性方程组右端项
    # =========================================================================

    def build_rhs(
        self,
        vertices:  np.ndarray,
        rotations: List[np.ndarray],
    ) -> np.ndarray:
        """
        全局步 (Global Step) :构建线性方程组右端项 b。

        对固定的旋转矩阵 {R_i}, 对 P' 各分量求偏导并令其为零, 得:

            L p' = b

        其中右端项:
            b_i = Σ_{j∈N(i)} (w_ij / 2) · (R_i + R_j) · (p_i - p_j)

        Parameters
        ----------
        vertices : np.ndarray, shape (N, 3)
            原始顶点坐标 (p, 未变形) 。
        rotations : List[np.ndarray]
            当前迭代的旋转矩阵列表, 每个元素 shape (3, 3), 长度须为 N。

        Returns
        -------
        b : np.ndarray, shape (N, 3)
            完整右端项 (含约束顶点行) , 后续由 separate_constraints 修正。
        """
        n:   int                              = vertices.shape[0]
        adj: List[List[int]]                  = self._adjacency
        cw:  Dict[Tuple[int, int], float]     = self._cotan_weights

        b = np.zeros((n, 3), dtype=np.float64)

        for i in range(n):
            b_i = np.zeros(3, dtype=np.float64)
            for j in adj[i]:
                edge_key = (min(i, j), max(i, j))
                w        = cw.get(edge_key, 0.0)

                e_ij  = vertices[i] - vertices[j]    # 原始边向量, (3,)
                R_sum = rotations[i] + rotations[j]  # (3, 3)

                b_i += (w * 0.5) * (R_sum @ e_ij)

            b[i] = b_i

        return b

    # =========================================================================
    # 步骤 6:分离约束, 修正自由子系统右端项
    # =========================================================================

    def separate_constraints(
        self,
        b:           np.ndarray,
        L:           sp.spmatrix,
        constraints: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将约束顶点的已知坐标贡献从右端项中分离, 得到自由子系统右端项。

        根据消元原理 (见 constraints_elimination 的文档) :
            b_free = b[free] - L_fc · q_c

        其中 L_fc 为 Laplacian 中自由行 x 约束列的子矩阵, 
        q_c 为约束目标坐标矩阵 ((C, 3)) 。

        Parameters
        ----------
        b : np.ndarray, shape (N, 3)
            完整右端项 (由 build_rhs 计算得到) 。
        L : sp.spmatrix, shape (N, N)
            完整 Laplacian 矩阵。
        constraints : Dict[int, np.ndarray]
            硬约束字典 {顶点索引: 目标坐标 (3,)}。

        Returns
        -------
        constraint_positions : np.ndarray, shape (C, 3)
            约束顶点目标坐标矩阵, 按 constrained_indices 升序排列。
        b_free : np.ndarray, shape (M, 3)
            修正后的自由子系统右端项, M = N - C。
        """
        free:        np.ndarray = self._free_indices         # (M,)
        constrained: np.ndarray = self._constrained_indices  # (C,)
        L_csr = L.tocsr()

        # 按升序索引排列约束目标坐标, shape (C, 3)
        constraint_positions = np.array(
            [constraints[int(c)] for c in constrained],
            dtype=np.float64,
        )

        # L_fc = L[free行, constrained列], shape (M, C)
        L_fc   = L_csr[free, :][:, constrained]
        b_free = b[free] - L_fc @ constraint_positions   # (M, 3)

        return constraint_positions, b_free

    # =========================================================================
    # 步骤 7:合并自由顶点结果与约束顶点坐标
    # =========================================================================

    def merge_constraints(
        self,
        p_prime_free: np.ndarray,
        n_total:      int,
    ) -> np.ndarray:
        """
        将自由顶点求解结果与约束顶点目标坐标合并, 重建完整顶点坐标数组。

        Parameters
        ----------
        p_prime_free : np.ndarray, shape (M, 3)
            自由顶点求解得到的变形坐标, M = N - len(constraints)。
        n_total : int
            网格顶点总数 N。

        Returns
        -------
        p_prime : np.ndarray, shape (N, 3)
            完整变形坐标数组:自由顶点来源于求解结果, 
            约束顶点强制赋为 self.constraints 中指定的目标坐标。
        """
        p_prime = np.empty((n_total, 3), dtype=np.float64)

        # 自由顶点:来自线性求解结果
        p_prime[self._free_indices] = p_prime_free

        # 约束顶点:强制赋为目标值, 精确满足硬约束
        for c_global in self._constrained_indices:
            p_prime[c_global] = self.constraints[int(c_global)]

        return p_prime

    # =========================================================================
    # 收敛判定
    # =========================================================================

    @staticmethod
    def _compute_max_displacement(
        p_old: np.ndarray,
        p_new: np.ndarray,
    ) -> float:
        """
        计算两次迭代间顶点位移的最大 L2 范数, 用于收敛判定。

        Parameters
        ----------
        p_old : np.ndarray, shape (N, 3)
            上一次迭代的顶点坐标。
        p_new : np.ndarray, shape (N, 3)
            当前迭代的顶点坐标。

        Returns
        -------
        max_disp : float
            所有顶点位移向量 L2 范数的最大值 (即 max_i ‖p'_i^new - p'_i^old‖₂) 。
        """
        diff = p_new - p_old                           # (N, 3)
        return float(np.max(np.linalg.norm(diff, axis=1)))

    # =========================================================================
    # 主入口:执行完整 ARAP 求解
    # =========================================================================

    def execute(self, mesh: MeshData) -> np.ndarray:
        """
        执行完整的 ARAP 求解, 返回变形后的顶点坐标。

        完整流程
        --------
        [预计算阶段]
        1. 输入合法性检查
        2. 构建顶点一环邻接表
        3. 计算余切权重, 组装 Laplacian L ((N, N)) 
        4. 约束消元, 提取自由子矩阵 L_free ((M, M)) 
        5. 对 L_free 进行 Cholesky 分解 (仅执行一次) 

        [迭代求解:共最多 iterations 轮]
        6. 局部步:对每个顶点用 SVD 求最优旋转 R_i
        7. 全局步:构建右端项 b → 分离约束修正为 b_free → 求解 L_free p'_f = b_free
        8. 合并自由顶点与约束顶点, 更新 P'
        9. 计算收敛指标 max ‖Δp'‖, 若 < convergence_tol 则提前退出

        Parameters
        ----------
        mesh : MeshData
            原始 (未变形) 源网格。

        Returns
        -------
        p_prime : np.ndarray, shape (N, 3)
            经 ARAP 变形后的顶点坐标。

        Raises
        ------
        ValueError
            若 constraints 中含有超出合法范围 [0, N-1] 的顶点索引。
        RuntimeError
            若自由顶点数 M ≤ 0 (所有顶点均被约束, 方程组退化) 。
        """
        n: int = mesh.num_vertices

        # ---------- 输入合法性检查 ------------------------------------------
        for c_idx in self.constraints:
            if not (0 <= c_idx < n):
                raise ValueError(
                    f"约束顶点索引 {c_idx} 超出合法范围 [0, {n - 1}]。"
                )
        n_free = n - len(self.constraints)
        if n_free <= 0:
            raise RuntimeError(
                "所有顶点均被约束 (M = 0) , 自由子系统为空, 无需求解。"
            )

        # ---------- 预计算阶段 -----------------------------------------------
        if self.verbose:
            print(
                f"[ARAP] 网格:{n} 顶点, {mesh.num_faces} 面片, "
                f"{len(self.constraints)} 处约束, {n_free} 个自由顶点"
            )
            print("[ARAP] 构建邻接表...")

        self._adjacency = self._build_adjacency(mesh)

        if self.verbose:
            print("[ARAP] 构建余切 Laplacian 矩阵...")
        L = self.build_cotan_laplacian(mesh)
        self._L_full = L

        if self.verbose:
            print("[ARAP] 约束消元...")
        L_free = self.constraints_elimination(L, self.constraints)

        # Tikhonov 正则化：L_free += λ·I，使矩阵严格正定
        # 解决余切 Laplacian 在约束不足或含退化三角形时奇异的问题
        if self.regularization > 0.0:
            L_free = L_free + self.regularization * sp.eye(
                L_free.shape[0], format="csr", dtype=np.float64
            )

        if self.verbose:
            print(f"[ARAP] Cholesky 分解 (后端: {self.cholesky_factor.backend}, "
                  f"正则化 λ={self.regularization:.2e}) ...")
        self.cholesky_factor.factorization(L_free)

        # ---------- 初始化 ---------------------------------------------------
        p_prime: np.ndarray = self.initial_guess(mesh)

        # ---------- 迭代求解 -------------------------------------------------
        for iteration in range(self.iterations):
            p_prev = p_prime.copy()

            # 局部步:SVD 求每个顶点处的最优旋转
            rotations: List[np.ndarray] = self.compute_local_rotation(mesh, p_prime)

            # 全局步:构建右端项 b (完整, 含约束行) 
            b: np.ndarray = self.build_rhs(mesh.vertices, rotations)

            # 分离约束, 修正自由子系统右端项
            _, b_free = self.separate_constraints(b, L, self.constraints)

            # 求解自由顶点新坐标:L_free @ p'_f = b_free, 结果 (M, 3)
            p_prime_free: np.ndarray = self.cholesky_factor.solve(b_free)

            # 合并自由顶点结果与约束顶点目标坐标
            p_prime = self.merge_constraints(p_prime_free, n)

            # 每次迭代回调（可用于实时预览）
            if self.step_callback is not None:
                self.step_callback(iteration, p_prime)

            # 收敛检测
            if self.convergence_tol > 0.0:
                max_disp = self._compute_max_displacement(p_prev, p_prime)
                if self.verbose:
                    print(
                        f"[ARAP] 迭代 {iteration + 1:3d}/{self.iterations}"
                        f"  最大位移: {max_disp:.6e}"
                    )
                if max_disp < self.convergence_tol:
                    if self.verbose:
                        print(
                            f"[ARAP] 已收敛 (位移 {max_disp:.2e} "
                            f"< 阈值 {self.convergence_tol:.2e}) , 提前终止。"
                        )
                    break
            elif self.verbose:
                print(f"[ARAP] 迭代 {iteration + 1:3d}/{self.iterations}")

        if self.verbose:
            print("[ARAP] 求解完成。")

        return p_prime


# =============================================================================
# 便利工厂函数
# =============================================================================

def create_solver(
    constraints:     Dict[int, np.ndarray],
    iterations:      int   = 10,
    convergence_tol: float = 1e-6,
    regularization:  float = 1e-8,
    verbose:         bool  = False,
    step_callback:   Optional["Callable[[int, np.ndarray], None]"] = None,
) -> ArapSolver:
    """
    快速创建带默认 CholeskyFactor 的 ArapSolver 实例。

    Parameters
    ----------
    constraints : Dict[int, array_like]
        硬约束字典 {顶点索引: 目标坐标 (3,)}。
    iterations : int, optional
        最大迭代次数, 默认 10。
    convergence_tol : float, optional
        收敛阈值, 默认 1e-6。设为 0 可禁用提前终止。
    regularization : float, optional
        Tikhonov 正则化系数 λ，默认 1e-8。
        若遇到 "Factor is exactly singular" 错误可适当增大（如 1e-5）。
    verbose : bool, optional
        是否打印迭代日志, 默认 False。

    Returns
    -------
    solver : ArapSolver
        配置好的 ARAP 求解器实例, 可直接调用 solver.execute(mesh)。

    Examples
    --------
    >>> import numpy as np
    >>> from ARAP import MeshData, create_solver
    >>>
    >>> # vertices / faces 由上层应用层 (Maya / Blender 等) 提供
    >>> vertices = np.array(...)   # shape (N, 3)
    >>> faces    = np.array(...)   # shape (F, 3)
    >>> mesh     = MeshData(vertices, faces)
    >>>
    >>> # 指定 landmark 约束:顶点 0 → [1,0,0], 顶点 5 → [0,1,0]
    >>> constraints = {0: [1.0, 0.0, 0.0], 5: [0.0, 1.0, 0.0]}
    >>>
    >>> solver     = create_solver(constraints, iterations=15, verbose=True)
    >>> p_deformed = solver.execute(mesh)   # shape (N, 3)
    """
    chol = CholeskyFactor()
    return ArapSolver(
        constraints     = constraints,
        iterations      = iterations,
        cholesky_factor = chol,
        convergence_tol = convergence_tol,
        regularization  = regularization,
        verbose         = verbose,
        step_callback   = step_callback,
    )
