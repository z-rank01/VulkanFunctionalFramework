"""
MayaNICP — Maya 端 Optimal Step N-ICP 网格配准工具（应用层）

本脚本是 NICP.py 核心算法的 Maya 应用层封装，提供：
  - MayaMeshBridge  : 纯 Maya API (om2) 数据读写工具类，与 UI/算法解耦
  - NicpToolWindow  : 基于 PySide2 的可停靠图形化工具窗口
  - show()          : 模块入口，在 Maya 中调用此函数打开工具窗口

与 MayaARAP 的区别
------------------
    MayaARAP  — ARAP 局部-全局交替优化，固定拓扑变形（同一网格自变形）
    MayaNICP  — Optimal Step N-ICP，源网格配准到目标网格（可拓扑不同）

        ARAP  : source 在约束力作用下变形，无需 target 网格
        N-ICP : source 向 target 表面靠拢，刚性权重 α 多尺度衰减

使用流程
--------
1. 在 Maya 大纲视图选中"目标网格 (Target)"  → 点击"设置目标"
2. 选中"源网格 (Source)"                    → 点击"设置源网格"
3. 在顶点模式下选择对应点对（先 Source 顶点，Shift 再 Target 顶点）
   → 点击"添加配对 Landmark"
   或：单独选 Source 顶点 → "投影最近点 Landmark" 批量添加
   （建议 6~20 个分布在鼻尖、眼角、嘴角等关键位置的 Landmark）
4. 调整求解参数，点击"执行 N-ICP 配准"
5. 结果原地写回 Source 网格顶点坐标（支持 Ctrl+Z 撤销）

参数解释
--------
1. 先固定 γ = 1（或根据网格尺度归一化后设为1）
2. 调整 α 序列：
   - 如果配准结果"撕裂"（局部形变过大）→ 增大 α_min
   - 如果配准结果"僵硬"（无法贴合细节）→ 减小 α_min
   - 如果初始对齐就失败（模板飞走）→ 增大 α_max
3. 如果网格未归一化：
   - 先将模板和目标归一化到单位包围盒
   - 或者手动调整 γ ≈ 1 / avg_edge_len


Maya 版本要求
------------
    Maya 2020+（Python 3，PySide2 内置）

在 Script Editor 中运行
-----------------------
import importlib, sys
sys.path.insert(0, r"<此文件所在目录的绝对路径>")
import ARAP, NICP, MayaNICP
importlib.reload(ARAP)
importlib.reload(NICP)
importlib.reload(MayaNICP)
MayaNICP.show()
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 将本文件所在目录插入 sys.path
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ARAP import MeshData                          # noqa: E402
from NICP import NicpSolver, create_nicp_solver    # noqa: E402

# ---------------------------------------------------------------------------
# Maya 外部运行保护
# ---------------------------------------------------------------------------
try:
    import maya.api.OpenMaya as om2
    import maya.cmds as cmds
    import maya.OpenMayaUI as omui
    from maya.app.general.mayaMixin import MayaQWidgetDockableMixin

    from PySide2 import QtCore, QtGui, QtWidgets
    import shiboken2

    _IN_MAYA = True
except ImportError:
    _IN_MAYA = False


# =============================================================================
# Maya Bridge 层：所有 Maya API 调用集中在此
# =============================================================================

class MayaMeshBridge:
    """
    Maya API 数据读写工具类（静态方法集合）。

    封装所有与 Maya 交互的操作；上层代码（UI、算法调用）
    不直接调用任何 Maya API，保持层次清晰。
    所有坐标统一使用世界空间（kWorld）。
    """

    @staticmethod
    def _get_fn_mesh(node_name: str) -> "om2.MFnMesh":
        """通过 transform 或 shape 名称获取 MFnMesh 实例。"""
        sel = om2.MSelectionList()
        try:
            sel.add(node_name)
        except Exception:
            raise ValueError(f"节点 '{node_name}' 在场景中不存在。")
        dag_path = sel.getDagPath(0)
        try:
            dag_path.extendToShape()
        except Exception:
            pass
        if dag_path.apiType() != om2.MFn.kMesh:
            raise ValueError(f"节点 '{node_name}' 不是多边形网格。")
        return om2.MFnMesh(dag_path)

    @staticmethod
    def _get_mesh_dag_path(transform: str) -> "om2.MDagPath":
        """将 transform 名解析为 mesh shape 的 MDagPath。"""
        sel = om2.MSelectionList()
        try:
            sel.add(transform)
        except Exception:
            raise ValueError(f"节点 '{transform}' 在场景中不存在。")
        dag_path = sel.getDagPath(0)
        try:
            dag_path.extendToShape()
        except Exception:
            pass
        if dag_path.apiType() != om2.MFn.kMesh:
            raise ValueError(f"节点 '{transform}' 不是多边形网格。")
        return dag_path

    @staticmethod
    def resolve_shape(transform: str) -> str:
        """由 transform 名称解析出其 mesh shape 节点名。"""
        shapes = cmds.listRelatives(transform, shapes=True, type="mesh") or []
        if not shapes:
            raise ValueError(f"'{transform}' 下未找到 mesh shape 节点。")
        return shapes[0]

    @staticmethod
    def get_mesh_data(transform: str) -> MeshData:
        """
        从 Maya 网格节点读取世界空间顶点坐标和三角面片索引，
        构建 MeshData 对象供 NICP 算法使用。
        """
        fn_mesh = MayaMeshBridge._get_fn_mesh(transform)
        raw_pts: "om2.MPointArray" = fn_mesh.getPoints(om2.MSpace.kWorld)
        vertices = np.array(
            [[p.x, p.y, p.z] for p in raw_pts], dtype=np.float64
        )
        _tri_counts, tri_verts = fn_mesh.getTriangles()
        faces = np.array(tri_verts, dtype=np.int32).reshape(-1, 3)
        return MeshData(vertices, faces)

    @staticmethod
    def get_selected_mesh_transform() -> Optional[str]:
        """获取当前选中的第一个 mesh transform 名称。"""
        sel = cmds.ls(selection=True, long=False) or []
        for node in sel:
            shapes = cmds.listRelatives(node, shapes=True, type="mesh") or []
            if shapes:
                return node
            if cmds.nodeType(node) == "mesh":
                parent = cmds.listRelatives(node, parent=True, fullPath=False)
                return parent[0] if parent else node
        return None

    @staticmethod
    def get_selected_vertex_indices(transform: str) -> List[int]:
        """获取当前在 Maya 视口中选中的属于指定 transform 的顶点索引列表。"""
        sel = cmds.ls(selection=True, flatten=True) or []
        try:
            shape_name = MayaMeshBridge.resolve_shape(transform)
        except ValueError:
            shape_name = transform
        pattern = re.compile(r'^(.+)\.vtx\[(\d+)\]$')
        indices: List[int] = []
        for item in sel:
            m = pattern.match(item)
            if not m:
                continue
            node_part = m.group(1)
            idx       = int(m.group(2))
            if node_part in (transform, shape_name):
                indices.append(idx)
        return sorted(set(indices))

    @staticmethod
    def get_vertex_world_pos(transform: str, vtx_idx: int) -> np.ndarray:
        """获取指定顶点的世界空间坐标。"""
        fn_mesh = MayaMeshBridge._get_fn_mesh(transform)
        pts     = fn_mesh.getPoints(om2.MSpace.kWorld)
        p       = pts[vtx_idx]
        return np.array([p.x, p.y, p.z], dtype=np.float64)

    @staticmethod
    def get_closest_point_on_mesh(
        target:    str,
        query_pos: np.ndarray,
    ) -> np.ndarray:
        """在目标网格表面查找距给定世界空间点最近的表面点坐标。"""
        fn_mesh   = MayaMeshBridge._get_fn_mesh(target)
        query_mpt = om2.MPoint(
            float(query_pos[0]), float(query_pos[1]), float(query_pos[2])
        )
        closest_pt, _face_idx = fn_mesh.getClosestPoint(query_mpt, om2.MSpace.kWorld)
        return np.array([closest_pt.x, closest_pt.y, closest_pt.z], dtype=np.float64)

    @staticmethod
    def apply_vertices(transform: str, p_prime: np.ndarray) -> None:
        """
        将 N-ICP 计算得到的新顶点坐标（世界空间）写回 Maya 网格。
        使用 undoInfo chunk 包裹，令操作在撤销历史中表现为单步。
        """
        dag_path = MayaMeshBridge._get_mesh_dag_path(transform)
        new_mpa  = om2.MPointArray(
            [om2.MPoint(float(r[0]), float(r[1]), float(r[2])) for r in p_prime]
        )
        cmds.undoInfo(openChunk=True, chunkName="nicpDeform")
        try:
            fn_mesh = om2.MFnMesh(dag_path)
            fn_mesh.setPoints(new_mpa, om2.MSpace.kWorld)
            fn_mesh.updateSurface()
            cmds.refresh(force=True)
        finally:
            cmds.undoInfo(closeChunk=True)

    @staticmethod
    def set_vertices_direct(dag_path: "om2.MDagPath", p_prime: np.ndarray) -> None:
        """直接写入顶点坐标并刷新视口，用于逐迭代预览的中间步骤。"""
        mpa     = om2.MPointArray(
            [om2.MPoint(float(r[0]), float(r[1]), float(r[2])) for r in p_prime]
        )
        fn_mesh = om2.MFnMesh(dag_path)
        fn_mesh.setPoints(mpa, om2.MSpace.kWorld)
        fn_mesh.updateSurface()
        cmds.refresh(force=True)


# =============================================================================
# 表格列常量
# =============================================================================

_COL_VTX    = 0   # Source 顶点
_COL_TX     = 1   # Target X
_COL_TY     = 2   # Target Y
_COL_TZ     = 3   # Target Z
_TABLE_COLS = 4

_WINDOW_OBJECT_NAME = "NicpToolWindowDockable"


# =============================================================================
# 主工具窗口
# =============================================================================

class NicpToolWindow(MayaQWidgetDockableMixin, QtWidgets.QWidget):  # type: ignore[misc]
    """
    N-ICP 网格配准工具主窗口。

    状态字段
    --------
    _target_mesh  : Optional[str]            目标网格 transform 名（配准参考）
    _source_mesh  : Optional[str]            源网格 transform 名（将被变形）
    _landmarks    : Dict[int, np.ndarray]    {source 顶点索引 → target 世界坐标}
    """

    TOOL_NAME = _WINDOW_OBJECT_NAME

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super(NicpToolWindow, self).__init__(parent=parent)

        self._target_mesh: Optional[str]         = None
        self._source_mesh: Optional[str]         = None
        self._landmarks:   Dict[int, np.ndarray] = {}

        self.setWindowTitle("N-ICP 网格配准工具")
        self.setObjectName(self.TOOL_NAME)
        self.setMinimumWidth(460)
        self._build_ui()

    # =========================================================================
    # UI 构建
    # =========================================================================

    def _build_ui(self) -> None:
        """构建完整的工具窗口布局。"""
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # ── 组1：对象设置 ──────────────────────────────────────────────────
        grp_obj    = QtWidgets.QGroupBox("对象设置")
        lay_obj    = QtWidgets.QFormLayout(grp_obj)
        lay_obj.setLabelAlignment(QtCore.Qt.AlignRight)

        # 目标网格（Target）
        row_tgt = QtWidgets.QHBoxLayout()
        self._le_target = QtWidgets.QLineEdit()
        self._le_target.setReadOnly(True)
        self._le_target.setPlaceholderText("未设置（请先选中目标网格）")
        btn_set_tgt = QtWidgets.QPushButton("设置目标 ▶")
        btn_set_tgt.setToolTip(
            "在大纲视图或视口中选中「配准参考」网格，再点击此按钮。\n"
            "例如：MetaHuman 模板头部网格。"
        )
        btn_set_tgt.clicked.connect(self._on_set_target)
        row_tgt.addWidget(self._le_target)
        row_tgt.addWidget(btn_set_tgt)
        lay_obj.addRow("目标网格 (Target)：", row_tgt)

        # 源网格（Source）
        row_src = QtWidgets.QHBoxLayout()
        self._le_source = QtWidgets.QLineEdit()
        self._le_source.setReadOnly(True)
        self._le_source.setPlaceholderText("未设置（请先选中源网格）")
        btn_set_src = QtWidgets.QPushButton("设置源网格 ▶")
        btn_set_src.setToolTip(
            "在大纲视图或视口中选中「待配准」网格，再点击此按钮。\n"
            "例如：写实人物头部网格。\n"
            "配准完成后此网格顶点坐标将被原地修改。"
        )
        btn_set_src.clicked.connect(self._on_set_source)
        row_src.addWidget(self._le_source)
        row_src.addWidget(btn_set_src)
        lay_obj.addRow("源网格 (Source)：", row_src)

        root.addWidget(grp_obj)

        # ── 组2：Landmark 管理 ─────────────────────────────────────────────
        self._grp_lm = QtWidgets.QGroupBox("Landmark 约束管理")
        self._grp_lm.setEnabled(False)
        lay_lm = QtWidgets.QVBoxLayout(self._grp_lm)

        # 说明标签
        lbl_hint = QtWidgets.QLabel(
            "Landmark 为 Source↔Target 的手动对应点对，引导 N-ICP 全局对齐。\n"
            "建议在鼻尖、眼角、嘴角、下巴尖、耳根等关键位置添加 6~20 个。"
        )
        lbl_hint.setWordWrap(True)
        lbl_hint.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        lay_lm.addWidget(lbl_hint)

        # 约束表格
        self._table = QtWidgets.QTableWidget(0, _TABLE_COLS)
        self._table.setHorizontalHeaderLabels(
            ["Source 顶点", "Target X", "Target Y", "Target Z"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_VTX, QtWidgets.QHeaderView.ResizeToContents
        )
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(140)
        lay_lm.addWidget(self._table)

        # 按钮行1：添加
        row_add = QtWidgets.QHBoxLayout()
        btn_add_pair = QtWidgets.QPushButton("添加配对 Landmark")
        btn_add_pair.setToolTip(
            "进入顶点模式（F9）后，先选 Source 网格上 1 个顶点，\n"
            "再 Shift 选 Target 网格上 1 个对应顶点，\n"
            "点击此按钮建立精确手动对应关系。\n"
            "适合鼻尖、眼角等关键解剖位置。"
        )
        btn_add_pair.clicked.connect(self._on_add_pair)

        btn_add_nearest = QtWidgets.QPushButton("投影最近点 Landmark")
        btn_add_nearest.setToolTip(
            "在 Source 网格上选中一个或多个顶点，\n"
            "点击此按钮将每个顶点自动投影到 Target 网格最近表面点，\n"
            "快速批量添加约束。适合沿头部轮廓线批量添加。\n"
            "若同一顶点已有配对约束，投影会覆盖旧值。"
        )
        btn_add_nearest.clicked.connect(self._on_add_nearest)
        row_add.addWidget(btn_add_pair)
        row_add.addWidget(btn_add_nearest)
        lay_lm.addLayout(row_add)

        # 按钮行2：删除 / 清空
        row_del = QtWidgets.QHBoxLayout()
        btn_remove = QtWidgets.QPushButton("删除选中 Landmark")
        btn_remove.setToolTip("删除表格中当前选中行对应的 Landmark 约束。")
        btn_remove.clicked.connect(self._on_remove_landmark)

        btn_clear = QtWidgets.QPushButton("清空全部 Landmark")
        btn_clear.setToolTip("清除所有已添加的 Landmark，重新开始配置。")
        btn_clear.clicked.connect(self._on_clear_landmarks)
        row_del.addWidget(btn_remove)
        row_del.addWidget(btn_clear)
        lay_lm.addLayout(row_del)

        self._lbl_lm_count = QtWidgets.QLabel("当前 Landmark 数：0")
        self._lbl_lm_count.setAlignment(QtCore.Qt.AlignRight)
        lay_lm.addWidget(self._lbl_lm_count)

        root.addWidget(self._grp_lm)

        # ── 组3：求解参数 ──────────────────────────────────────────────────
        grp_params = QtWidgets.QGroupBox("求解参数")
        lay_params = QtWidgets.QFormLayout(grp_params)
        lay_params.setLabelAlignment(QtCore.Qt.AlignRight)

        # 刚性权重：起始 / 终止 / 步数
        row_alpha = QtWidgets.QHBoxLayout()
        self._dspin_alpha_start = QtWidgets.QDoubleSpinBox()
        self._dspin_alpha_start.setRange(0.01, 10000.0)
        self._dspin_alpha_start.setValue(100.0)
        self._dspin_alpha_start.setDecimals(2)
        self._dspin_alpha_start.setToolTip(
            "刚性权重起始值（最大刚性）。\n"
            "较大的 α → 网格近似刚体，用于宏观全局对齐。\n"
            "建议范围：50 ~ 500。"
        )

        lbl_alpha_to = QtWidgets.QLabel("→")
        lbl_alpha_to.setAlignment(QtCore.Qt.AlignCenter)

        self._dspin_alpha_end = QtWidgets.QDoubleSpinBox()
        self._dspin_alpha_end.setRange(0.001, 100.0)
        self._dspin_alpha_end.setValue(0.1)
        self._dspin_alpha_end.setDecimals(3)
        self._dspin_alpha_end.setToolTip(
            "刚性权重终止值（最小刚性）。\n"
            "较小的 α → 精细非刚性形变，用于局部表面贴合。\n"
            "建议范围：0.01 ~ 1.0。"
        )

        self._spin_n_steps = QtWidgets.QSpinBox()
        self._spin_n_steps.setRange(2, 50)
        self._spin_n_steps.setValue(8)
        self._spin_n_steps.setPrefix("步数 ")
        self._spin_n_steps.setToolTip(
            "对数等间距刚性权重的步数。\n"
            "权重由 logspace(alpha_start, alpha_end, n_steps) 自动生成，\n"
            "保证每步相对变化量相同（等比序列）。\n"
            "默认 8 步，通常足够；形状差异较大时可增加到 12~16。"
        )
        row_alpha.addWidget(self._dspin_alpha_start)
        row_alpha.addWidget(lbl_alpha_to)
        row_alpha.addWidget(self._dspin_alpha_end)
        row_alpha.addWidget(self._spin_n_steps)
        lay_params.addRow("刚性权重 α（起 → 止，步数）：", row_alpha)

        # 内层迭代次数
        self._spin_inner = QtWidgets.QSpinBox()
        self._spin_inner.setRange(1, 50)
        self._spin_inner.setValue(3)
        self._spin_inner.setToolTip(
            "每个刚性权重 α 下的内层迭代次数。\n"
            "每次内层迭代：重新查找最近点 + 重新求解线性方程组。\n"
            "建议值：3~5，过多收益递减且耗时增加。"
        )
        lay_params.addRow("内层迭代次数：", self._spin_inner)

        # G 矩阵平移权重 γ
        self._dspin_gamma = QtWidgets.QDoubleSpinBox()
        self._dspin_gamma.setRange(0.0, 100.0)
        self._dspin_gamma.setValue(1.0)
        self._dspin_gamma.setDecimals(3)
        self._dspin_gamma.setToolTip(
            "G 矩阵第 4 个对角元 γ，控制相邻顶点平移分量的刚性惩罚。\n"
            "γ=1（默认）：旋转与平移分量受到相同程度的刚性约束。\n"
            "γ→0：允许自由的局部平移（更灵活但可能产生撕裂）。\n"
            "γ→大：相邻顶点平移趋于一致（更保守，近似刚体平移）。"
        )
        lay_params.addRow("平移刚性 γ：", self._dspin_gamma)

        # Landmark 权重 γ_l
        self._dspin_gamma_lm = QtWidgets.QDoubleSpinBox()
        self._dspin_gamma_lm.setRange(0.1, 10000.0)
        self._dspin_gamma_lm.setValue(10.0)
        self._dspin_gamma_lm.setDecimals(2)
        self._dspin_gamma_lm.setToolTip(
            "Landmark 约束的能量权重 γ_l。\n"
            "建议范围：5 ~ 50。\n"
            "过大：Landmark 位置非常精确，但非 Landmark 区域配准质量可能下降。\n"
            "过小：Landmark 约束不够强，全局对齐可能偏差。"
        )
        lay_params.addRow("Landmark 权重 γ_l：", self._dspin_gamma_lm)

        # 收敛阈值
        self._le_tol = QtWidgets.QLineEdit("1e-4")
        self._le_tol.setToolTip(
            "内层收敛阈值：当相邻两次迭代所有顶点最大位移 < 此值时提前退出。\n"
            "设为 0 可禁用提前终止，强制完成全部迭代。"
        )
        lay_params.addRow("收敛阈值：", self._le_tol)

        # 正则化系数
        self._le_reg = QtWidgets.QLineEdit("1e-6")
        self._le_reg.setToolTip(
            "Tikhonov 正则化系数 λ：在 Cholesky 分解前对矩阵施加 M += λ·I，\n"
            "使矩阵严格正定，防止 'Factor is exactly singular' 错误。\n"
            "通常 1e-6 即可；若仍报奇异错误可逐步增大到 1e-4。"
        )
        lay_params.addRow("正则化系数 λ：", self._le_reg)

        # 距离阈值
        self._le_dist_thresh = QtWidgets.QLineEdit("")
        self._le_dist_thresh.setPlaceholderText("留空 = 自适应（中位数×2）")
        self._le_dist_thresh.setToolTip(
            "数据项离群点距离阈值。\n"
            "当 Source 顶点与 Target 最近点距离超过此值时，该对应被忽略（权重=0）。\n"
            "留空：自动取当前所有对应距离中位值的 2 倍（推荐）。\n"
            "手动指定：适合 Source 与 Target 有较大区域无法对应时（如半张脸配准全脸）。"
        )
        lay_params.addRow("离群点距离阈值：", self._le_dist_thresh)

        # 法线角度阈值
        row_normal = QtWidgets.QHBoxLayout()
        self._chk_normal = QtWidgets.QCheckBox("启用")
        self._chk_normal.setChecked(False)
        self._chk_normal.setToolTip(
            "勾选后启用法线角度过滤：当 Source 顶点法线与 Target 最近点法线\n"
            "夹角超过阈值时，该对应被忽略（权重＝0）。\n"
            "可有效避免映射到对面或背面的错误对应。"
        )
        self._dspin_normal_angle = QtWidgets.QDoubleSpinBox()
        self._dspin_normal_angle.setRange(1.0, 180.0)
        self._dspin_normal_angle.setValue(60.0)
        self._dspin_normal_angle.setDecimals(1)
        self._dspin_normal_angle.setSuffix("°")
        self._dspin_normal_angle.setEnabled(False)
        self._dspin_normal_angle.setToolTip(
            "法线夹角阈值（度）。\n"
            "  60°：严格，推荐用于眼周/鼻翼等曲率大的区域\n"
            "  90°：标准，过滤背面对应点（推荐默认值）\n"
            " 120°：宽松，仅排除严重反向对应"
        )
        self._chk_normal.toggled.connect(self._dspin_normal_angle.setEnabled)
        row_normal.addWidget(self._chk_normal)
        row_normal.addWidget(self._dspin_normal_angle)
        row_normal.addStretch()
        lay_params.addRow("法线角度阈值：", row_normal)

        # 预览模式
        self._chk_preview = QtWidgets.QCheckBox("逐迭代预览（每次内层迭代后刷新视口）")
        self._chk_preview.setChecked(False)
        self._chk_preview.setToolTip(
            "勾选后每次内层迭代完成后即将当前顶点写回网格并刷新视口，\n"
            "可直观观察配准过程，但速度变慢。\n"
            "不勾选则仅在全部迭代完成后一次性写入（更快）。"
        )
        lay_params.addRow("", self._chk_preview)

        root.addWidget(grp_params)

        # ── 分隔线 ──────────────────────────────────────────────────────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        root.addWidget(sep)

        # ── 执行按钮 ────────────────────────────────────────────────────────
        self._btn_execute = QtWidgets.QPushButton("  执行 N-ICP 配准  ")
        self._btn_execute.setMinimumHeight(36)
        self._btn_execute.setToolTip(
            "读取 Source 和 Target 网格数据，运行 Optimal Step N-ICP 求解，\n"
            "将配准结果原地写回 Source 网格顶点坐标。\n"
            "操作支持 Ctrl+Z 撤销。\n\n"
            "注意：无 Landmark 时也可运行（纯数据项驱动），\n"
            "但建议至少添加 6 个 Landmark 以确保全局对齐稳定。"
        )
        self._btn_execute.setStyleSheet(
            "QPushButton {"
            "  background-color: #1A6EB5;"
            "  color: white;"
            "  border-radius: 4px;"
            "  font-size: 13px;"
            "  font-weight: bold;"
            "}"
            "QPushButton:hover   { background-color: #2280D2; }"
            "QPushButton:pressed { background-color: #0F5591; }"
            "QPushButton:disabled{ background-color: #666666; color: #AAAAAA; }"
        )
        self._btn_execute.clicked.connect(self._on_execute)
        root.addWidget(self._btn_execute)

        # ── 状态栏 ──────────────────────────────────────────────────────────
        self._lbl_status = QtWidgets.QLabel("")
        self._lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_status.setStyleSheet("color: #888888; font-size: 11px;")
        root.addWidget(self._lbl_status)

        root.addStretch()

    # =========================================================================
    # 辅助
    # =========================================================================

    def _set_status(self, msg: str, color: str = "#888888") -> None:
        self._lbl_status.setStyleSheet(f"color: {color}; font-size: 11px;")
        self._lbl_status.setText(msg)

    def _refresh_landmark_table(self) -> None:
        """根据内存中的 _landmarks 字典重建表格显示。"""
        self._table.setRowCount(0)
        for vtx_idx in sorted(self._landmarks.keys()):
            pos   = self._landmarks[vtx_idx]
            row_n = self._table.rowCount()
            self._table.insertRow(row_n)
            items = [
                QtWidgets.QTableWidgetItem(f"vtx[{vtx_idx}]"),
                QtWidgets.QTableWidgetItem(f"{pos[0]:.4f}"),
                QtWidgets.QTableWidgetItem(f"{pos[1]:.4f}"),
                QtWidgets.QTableWidgetItem(f"{pos[2]:.4f}"),
            ]
            for col, item in enumerate(items):
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if col == _COL_VTX:
                    item.setData(QtCore.Qt.UserRole, vtx_idx)
                self._table.setItem(row_n, col, item)
        self._lbl_lm_count.setText(f"当前 Landmark 数：{len(self._landmarks)}")

    # =========================================================================
    # 按钮回调：对象设置
    # =========================================================================

    def _on_set_target(self) -> None:
        """将当前选中的第一个 mesh 设为目标网格（Target）。"""
        name = MayaMeshBridge.get_selected_mesh_transform()
        if name is None:
            QtWidgets.QMessageBox.warning(
                self, "未选中网格",
                "请先在场景中选中一个多边形网格，再点击「设置目标」。"
            )
            return
        if name != self._target_mesh:
            self._landmarks.clear()
            self._refresh_landmark_table()
        self._target_mesh = name
        self._le_target.setText(name)
        self._set_status(f"目标网格已设置：{name}", "#4CAF50")

    def _on_set_source(self) -> None:
        """将当前选中的第一个 mesh 设为源网格（Source），并解锁 Landmark 区域。"""
        name = MayaMeshBridge.get_selected_mesh_transform()
        if name is None:
            QtWidgets.QMessageBox.warning(
                self, "未选中网格",
                "请先在场景中选中一个多边形网格，再点击「设置源网格」。"
            )
            return
        if name != self._source_mesh:
            self._landmarks.clear()
            self._refresh_landmark_table()
        self._source_mesh = name
        self._le_source.setText(name)
        self._grp_lm.setEnabled(True)
        self._set_status(f"源网格已设置：{name}", "#4CAF50")

    # =========================================================================
    # 按钮回调：Landmark 管理
    # =========================================================================

    def _on_add_pair(self) -> None:
        """
        手动配对 Landmark：
        在顶点模式下同时选中 Source 上 1 个顶点 + Target 上 1 个顶点，
        建立精确对应关系。
        """
        if not self._source_mesh or not self._target_mesh:
            self._set_status("请先设置目标网格和源网格。", "#FF6B6B")
            return

        sel_flat  = cmds.ls(selection=True, flatten=True) or []
        vtx_items = [s for s in sel_flat if ".vtx[" in s]

        if len(vtx_items) != 2:
            QtWidgets.QMessageBox.warning(
                self, "选择错误",
                "请在顶点模式（F9）下恰好选中 2 个顶点：\n"
                "  • 第 1 个：Source 网格上的顶点\n"
                "  • 第 2 个：Target 网格上的对应顶点\n\n"
                f"当前选中的顶点数：{len(vtx_items)}"
            )
            return

        def _parse_vtx(item: str) -> Tuple[str, int]:
            m = re.match(r'^(.+)\.vtx\[(\d+)\]$', item)
            if not m:
                raise ValueError(f"无法解析顶点选择项：{item}")
            return m.group(1), int(m.group(2))

        try:
            node_a, idx_a = _parse_vtx(vtx_items[0])
            node_b, idx_b = _parse_vtx(vtx_items[1])
        except ValueError as exc:
            self._set_status(str(exc), "#FF6B6B")
            return

        try:
            src_shape = MayaMeshBridge.resolve_shape(self._source_mesh)
        except ValueError:
            src_shape = self._source_mesh
        try:
            tgt_shape = MayaMeshBridge.resolve_shape(self._target_mesh)
        except ValueError:
            tgt_shape = self._target_mesh

        src_names = (self._source_mesh, src_shape)
        tgt_names = (self._target_mesh, tgt_shape)

        if node_a in src_names and node_b in tgt_names:
            src_idx, tgt_idx = idx_a, idx_b
        elif node_b in src_names and node_a in tgt_names:
            src_idx, tgt_idx = idx_b, idx_a
        else:
            QtWidgets.QMessageBox.warning(
                self, "选择错误",
                "两个顶点必须分属当前设置的 Source 和 Target 网格。\n\n"
                f"选中的节点：{node_a}、{node_b}\n"
                f"Source：{self._source_mesh}，Target：{self._target_mesh}"
            )
            return

        target_pos = MayaMeshBridge.get_vertex_world_pos(self._target_mesh, tgt_idx)
        self._landmarks[src_idx] = target_pos
        self._refresh_landmark_table()
        self._set_status(
            f"已添加配对 Landmark：vtx[{src_idx}] → Target vtx[{tgt_idx}]",
            "#4CAF50",
        )

    def _on_add_nearest(self) -> None:
        """
        投影最近点 Landmark：
        在 Source 上选中一个或多个顶点，自动投影到 Target 表面最近点，
        批量添加 Landmark。
        """
        if not self._source_mesh or not self._target_mesh:
            self._set_status("请先设置目标网格和源网格。", "#FF6B6B")
            return

        src_indices = MayaMeshBridge.get_selected_vertex_indices(self._source_mesh)
        if not src_indices:
            QtWidgets.QMessageBox.warning(
                self, "未选中顶点",
                f"请在顶点模式下，在 Source 网格 '{self._source_mesh}' 上\n"
                "选中一个或多个顶点，再点击此按钮。"
            )
            return

        added = skipped = 0
        for vtx_idx in src_indices:
            try:
                src_pos    = MayaMeshBridge.get_vertex_world_pos(self._source_mesh, vtx_idx)
                target_pos = MayaMeshBridge.get_closest_point_on_mesh(
                    self._target_mesh, src_pos
                )
                self._landmarks[vtx_idx] = target_pos
                added += 1
            except Exception:
                skipped += 1

        self._refresh_landmark_table()
        msg = f"投影完成：新增/更新 {added} 个 Landmark"
        if skipped:
            msg += f"，{skipped} 个失败（可忽略）"
        self._set_status(msg, "#4CAF50")

    def _on_remove_landmark(self) -> None:
        """删除表格中当前选中行对应的 Landmark。"""
        selected_items = self._table.selectedItems()
        if not selected_items:
            self._set_status("请先在表格中选中要删除的 Landmark 行。", "#FFCC00")
            return
        rows_to_delete: List[int] = []
        seen: set = set()
        for item in selected_items:
            r = item.row()
            if r in seen:
                continue
            seen.add(r)
            vtx_item = self._table.item(r, _COL_VTX)
            if vtx_item:
                rows_to_delete.append(vtx_item.data(QtCore.Qt.UserRole))
        for vtx_idx in rows_to_delete:
            self._landmarks.pop(vtx_idx, None)
        self._refresh_landmark_table()
        self._set_status(f"已删除 {len(rows_to_delete)} 个 Landmark。", "#4CAF50")

    def _on_clear_landmarks(self) -> None:
        """清空全部 Landmark。"""
        if not self._landmarks:
            self._set_status("当前没有任何 Landmark。", "#FFCC00")
            return
        reply = QtWidgets.QMessageBox.question(
            self, "确认清空",
            f"确认要清除全部 {len(self._landmarks)} 个 Landmark 吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._landmarks.clear()
            self._refresh_landmark_table()
            self._set_status("已清空全部 Landmark。", "#888888")

    # =========================================================================
    # 执行回调
    # =========================================================================

    def _on_execute(self) -> None:
        """
        执行完整的 N-ICP 配准。

        执行流程
        --------
        1. 校验 source / target 是否已设置
        2. 读取并验证所有求解参数
        3. 从 Maya 读取 source / target 网格数据 → MeshData
        4. 创建并运行 NicpSolver
        5. 以 undoChunk 写回 source 网格，支持 Ctrl+Z 撤销
        """
        # ── 校验 ────────────────────────────────────────────────────────────
        if not self._source_mesh or not self._target_mesh:
            QtWidgets.QMessageBox.warning(
                self, "配置不完整",
                "请先设置「目标网格」和「源网格」。"
            )
            return

        if self._source_mesh == self._target_mesh:
            QtWidgets.QMessageBox.warning(
                self, "配置错误",
                "Source 和 Target 不能是同一个网格。\n"
                "N-ICP 用于将 Source 网格配准到 Target 网格，两者须不同。"
            )
            return

        # ── 解析参数 ─────────────────────────────────────────────────────────
        alpha_start = float(self._dspin_alpha_start.value())
        alpha_end   = float(self._dspin_alpha_end.value())
        n_steps     = int(self._spin_n_steps.value())
        inner_iter  = int(self._spin_inner.value())
        gamma       = float(self._dspin_gamma.value())
        gamma_lm    = float(self._dspin_gamma_lm.value())

        if alpha_end >= alpha_start:
            QtWidgets.QMessageBox.warning(
                self, "参数错误",
                f"刚性权重终止值（{alpha_end}）须小于起始值（{alpha_start}）。"
            )
            return

        try:
            tol = float(self._le_tol.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "参数错误",
                f"收敛阈值 '{self._le_tol.text()}' 不是有效的浮点数。"
            )
            return

        try:
            reg = float(self._le_reg.text().strip())
            if reg < 0.0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "参数错误",
                f"正则化系数 '{self._le_reg.text()}' 不是有效的非负浮点数。"
            )
            return

        dist_thresh_text = self._le_dist_thresh.text().strip()
        if dist_thresh_text:
            try:
                dist_thresh: Optional[float] = float(dist_thresh_text)
                if dist_thresh <= 0.0:
                    raise ValueError
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "参数错误",
                    f"离群点距离阈值 '{dist_thresh_text}' 须为正浮点数，或留空使用自适应阈值。"
                )
                return
        else:
            dist_thresh = None

        # 法线角度阈值
        normal_angle_thresh: Optional[float] = None
        if self._chk_normal.isChecked():
            normal_angle_thresh = float(self._dspin_normal_angle.value())

        # ── 禁用按钮，防止重复点击 ───────────────────────────────────────────
        self._btn_execute.setEnabled(False)
        self._set_status("正在执行 N-ICP 配准，请稍候…", "#FFCC00")
        QtWidgets.QApplication.processEvents()

        try:
            # ── 读取网格数据 ─────────────────────────────────────────────────
            self._set_status("正在读取网格数据…", "#FFCC00")
            QtWidgets.QApplication.processEvents()
            source_data: MeshData = MayaMeshBridge.get_mesh_data(self._source_mesh)
            target_data: MeshData = MayaMeshBridge.get_mesh_data(self._target_mesh)

            preview_mode = self._chk_preview.isChecked()

            if preview_mode:
                # ── 预览模式：每次内层迭代后刷新视口 ────────────────────────
                dag_path_src = MayaMeshBridge._get_mesh_dag_path(self._source_mesh)
                total_iters  = n_steps * inner_iter

                def _preview_cb(
                    iter_idx: int,
                    alpha:    float,
                    pts:      np.ndarray,
                ) -> None:
                    self._set_status(
                        f"α={alpha:.3g}  迭代 {iter_idx + 1}/{total_iters}…",
                        "#FFCC00",
                    )
                    QtWidgets.QApplication.processEvents()
                    MayaMeshBridge.set_vertices_direct(dag_path_src, pts)

                solver = create_nicp_solver(
                    landmarks              = self._landmarks,
                    alpha_start            = alpha_start,
                    alpha_end              = alpha_end,
                    n_steps                = n_steps,
                    inner_iterations       = inner_iter,
                    gamma                  = gamma,
                    gamma_landmark         = gamma_lm,
                    distance_threshold     = dist_thresh,
                    normal_angle_threshold = normal_angle_thresh,
                    convergence_tol        = tol,
                    regularization         = reg,
                    verbose                = True,
                    step_callback          = _preview_cb,
                )
                deformed: np.ndarray = solver.execute(source_data, target_data)
                # 最终结果以 undoChunk 方式写回（覆盖预览中间态，可整体撤销）
                MayaMeshBridge.apply_vertices(self._source_mesh, deformed)

            else:
                # ── 普通模式：仅在最终结果上执行一次可撤销写入 ──────────────
                solver = create_nicp_solver(
                    landmarks              = self._landmarks,
                    alpha_start            = alpha_start,
                    alpha_end              = alpha_end,
                    n_steps                = n_steps,
                    inner_iterations       = inner_iter,
                    gamma                  = gamma,
                    gamma_landmark         = gamma_lm,
                    distance_threshold     = dist_thresh,
                    normal_angle_threshold = normal_angle_thresh,
                    convergence_tol        = tol,
                    regularization         = reg,
                    verbose                = True,
                )
                self._set_status(
                    f"正在求解（α: {alpha_start}→{alpha_end}，{n_steps} 步"
                    f"  ×{inner_iter} 内层迭代，Landmark: {len(self._landmarks)} 个）…",
                    "#FFCC00",
                )
                QtWidgets.QApplication.processEvents()
                deformed = solver.execute(source_data, target_data)

                self._set_status("正在将结果写回网格…", "#FFCC00")
                QtWidgets.QApplication.processEvents()
                MayaMeshBridge.apply_vertices(self._source_mesh, deformed)

            # ── 成功 ─────────────────────────────────────────────────────────
            self._set_status(
                f"✓ 配准完成，共处理 {source_data.num_vertices} 个顶点。"
                f"（可 Ctrl+Z 撤销）",
                "#4CAF50",
            )
            try:
                cmds.inViewMessage(
                    amg=(
                        f"<hl>N-ICP</hl> 配准完成！"
                        f"顶点数：{source_data.num_vertices}"
                    ),
                    pos="botCenter",
                    fade=True,
                    fadeStayTime=2500,
                )
            except Exception:
                pass

        except Exception as exc:
            error_msg = str(exc)
            self._set_status(f"执行失败：{error_msg}", "#FF6B6B")
            QtWidgets.QMessageBox.critical(
                self, "N-ICP 执行失败",
                f"求解过程中发生错误：\n\n{error_msg}\n\n"
                "可能的原因：\n"
                "  • 网格数据存在退化三角形 → 尝试增大「正则化系数 λ」\n"
                "  • Source 与 Target 坐标差异过大 → 先手动粗对齐，再运行配准\n"
                "  • Landmark 顶点索引超出范围 → 清空 Landmark 后重新添加"
            )

        finally:
            self._btn_execute.setEnabled(True)


# =============================================================================
# 模块入口
# =============================================================================

_WINDOW_INSTANCE: Optional["NicpToolWindow"] = None


def show() -> "NicpToolWindow":
    """
    在 Maya 中打开 N-ICP 网格配准工具窗口。

    若窗口已存在则先关闭旧窗口再重建，保证每次打开的都是干净的实例。

    Returns
    -------
    window : NicpToolWindow
        已显示的工具窗口实例。

    在 Maya Script Editor 中调用
    ----------------------------
    .. code-block:: python

        import importlib, sys
        sys.path.insert(0, r"<此文件所在目录的绝对路径>")
        import ARAP, NICP, MayaNICP
        importlib.reload(ARAP)
        importlib.reload(NICP)
        importlib.reload(MayaNICP)
        MayaNICP.show()
    """
    if not _IN_MAYA:
        raise RuntimeError("MayaNICP.show() 只能在 Maya 内部运行。")

    # 重新加载核心模块，确保始终使用最新版本
    import importlib as _importlib
    for _mod_name in ("ARAP", "NICP"):
        _mod = sys.modules.get(_mod_name)
        if _mod is not None:
            _importlib.reload(_mod)
    global create_nicp_solver, MeshData, NicpSolver
    from ARAP import MeshData                        # noqa: F811, E402
    from NICP import NicpSolver, create_nicp_solver  # noqa: F811, E402

    global _WINDOW_INSTANCE

    # 关闭旧实例
    if _WINDOW_INSTANCE is not None:
        try:
            _WINDOW_INSTANCE.close()
            _WINDOW_INSTANCE.deleteLater()
        except RuntimeError:
            pass
        _WINDOW_INSTANCE = None

    # 删除 Maya 中残留的 WorkspaceControl 节点
    workspace_ctrl = _WINDOW_OBJECT_NAME + "WorkspaceControl"
    if cmds.workspaceControl(workspace_ctrl, query=True, exists=True):
        cmds.deleteUI(workspace_ctrl)

    _WINDOW_INSTANCE = NicpToolWindow()
    _WINDOW_INSTANCE.show(dockable=True, floating=True)
    return _WINDOW_INSTANCE
