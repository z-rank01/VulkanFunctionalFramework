"""
MayaARAP — Maya 端 ARAP 网格变形工具 (应用层) 

本脚本是 ARAP.py 核心算法的 Maya 应用层封装, 提供：
  - MayaMeshBridge  : 纯 Maya API (om2) 数据读写工具类, 与 UI/算法解耦
  - ArapToolWindow  : 基于 PySide2 的可停靠图形化工具窗口
  - show()          : 模块入口, 在 Maya 中调用此函数打开工具窗口

使用流程
--------
1. 在 Maya 大纲视图选中"变形参照 (Target) 网格" → 点击"设置目标"
2. 选中"待变形 (Source) 网格" → 点击"设置对象"
3. 在场景中选顶点添加 landmark, F9 进入顶点模式，先选 source 顶点再 Shift 选 target 对应顶点 → "添加配对约束"<br>重复添加鼻尖、下巴、眼角、嘴角、耳根等 6-10 对
4. 调整迭代参数 (一般是 10-15), 点击"执行 ARAP 变形"
5. 结果原地修改 Source 网格顶点坐标 (支持 Ctrl+Z 撤销) 

Maya 版本要求
------------
    Maya 2020+ (Python 3, PySide2 内置) 

在 Script Editor 中运行
-----------------------
import importlib, sys # 将脚本目录加入路径 (仅首次需要, 或放入 userSetup.py) 
sys.path.insert(0, r"<此文件所在目录的绝对路径>")
import ARAP, MayaARAP
importlib.reload(ARAP)      # 必须先 reload ARAP, 否则 MayaARAP 会持有旧版本函数引用
importlib.reload(MayaARAP)
MayaARAP.show()
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 将本文件所在目录插入 sys.path, 使 Maya 能 import 同目录的 ARAP.py
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ARAP import CholeskyFactor, MeshData, ArapSolver, create_solver  # noqa: E402

# ---------------------------------------------------------------------------
# Maya 外部运行保护 (如在普通 IDE 中打开文件做静态分析时不会报错) 
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
# Maya Bridge 层：所有 Maya API 调用集中在此, 与 UI / 算法完全隔离
# =============================================================================

class MayaMeshBridge:
    """
    Maya API 数据读写工具类 (静态方法集合) 。

    本类封装所有与 Maya 交互的操作, 上层代码 (UI、算法调用) 
    不直接调用任何 Maya API, 保持层次清晰。

    所有坐标统一使用 **世界空间 (kWorld) **, 避免两网格 transform 偏移
    引起坐标系不一致的问题。
    """

    # -------------------------------------------------------------------------
    # 私有工具：通过节点名获取 MFnMesh
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_fn_mesh(node_name: str) -> "om2.MFnMesh":
        """
        通过 transform 或 shape 名称获取 MFnMesh 实例。

        Parameters
        ----------
        node_name : str
            Maya 场景中的节点名称 (transform 或直接是 mesh shape 均可) 。

        Returns
        -------
        fn_mesh : om2.MFnMesh
            对应网格的 MFnMesh 对象。

        Raises
        ------
        ValueError
            若节点不存在或不是 mesh 类型。
        """
        sel = om2.MSelectionList()
        try:
            sel.add(node_name)
        except Exception:
            raise ValueError(f"节点 '{node_name}' 在场景中不存在。")

        dag_path = sel.getDagPath(0)
        # 若传入的是 transform, 需要 extend 到 mesh shape
        try:
            dag_path.extendToShape()
        except Exception:
            pass  # 已是 shape 节点, 忽略

        if dag_path.apiType() != om2.MFn.kMesh:
            raise ValueError(f"节点 '{node_name}' 不是多边形网格 (PolyMesh) 。")

        return om2.MFnMesh(dag_path)

    # -------------------------------------------------------------------------
    # 公共方法
    # -------------------------------------------------------------------------

    @staticmethod
    def resolve_shape(transform: str) -> str:
        """
        由 transform 名称解析出其 mesh shape 节点名。

        Parameters
        ----------
        transform : str
            场景中的 transform 节点名 (或直接是 shape 名) 。

        Returns
        -------
        shape_name : str
            对应的 mesh shape 节点名称。

        Raises
        ------
        ValueError
            若该 transform 下没有 mesh shape 子节点。
        """
        shapes = cmds.listRelatives(transform, shapes=True, type="mesh") or []
        if not shapes:
            raise ValueError(f"'{transform}' 下未找到 mesh shape 节点。")
        return shapes[0]

    @staticmethod
    def get_mesh_data(transform: str) -> MeshData:
        """
        从 Maya 网格节点读取世界空间顶点坐标和三角面片索引, 
        构建与 DCC 无关的 MeshData 对象供 ARAP 算法使用。

        调用 MFnMesh.getTriangles() 读取 Maya 内部三角化缓存, 
        **不会修改或重建原始网格拓扑**。

        Parameters
        ----------
        transform : str
            源/目标网格的 transform 节点名。

        Returns
        -------
        mesh_data : MeshData
            vertices shape (N, 3), faces shape (F, 3), 均为世界空间坐标。
        """
        fn_mesh = MayaMeshBridge._get_fn_mesh(transform)

        # 世界空间顶点坐标 → (N, 3) float64
        raw_pts: om2.MPointArray = fn_mesh.getPoints(om2.MSpace.kWorld)
        vertices = np.array(
            [[p.x, p.y, p.z] for p in raw_pts],
            dtype=np.float64,
        )

        # 三角面片索引 (不修改原始网格) 
        _tri_counts, tri_verts = fn_mesh.getTriangles()
        faces = np.array(tri_verts, dtype=np.int32).reshape(-1, 3)

        return MeshData(vertices, faces)

    @staticmethod
    def get_selected_mesh_transform() -> Optional[str]:
        """
        获取当前选中的第一个 mesh transform 名称。

        若选中多个对象, 取第一个且为 mesh 类型的节点。

        Returns
        -------
        transform_name : Optional[str]
            mesh transform 名称, 若选中内容不含 mesh 则返回 None。
        """
        sel = cmds.ls(selection=True, long=False) or []
        for node in sel:
            shapes = cmds.listRelatives(node, shapes=True, type="mesh") or []
            if shapes:
                return node
            # 选中的本身可能就是 shape
            if cmds.nodeType(node) == "mesh":
                parent = cmds.listRelatives(node, parent=True, fullPath=False)
                return parent[0] if parent else node
        return None

    @staticmethod
    def get_selected_vertex_indices(transform: str) -> List[int]:
        """
        获取当前在 Maya 视口中选中的属于指定 transform 的顶点索引列表。

        Parameters
        ----------
        transform : str
            必须属于哪个 mesh transform (过滤其他 mesh 上的选中组件) 。

        Returns
        -------
        indices : List[int]
            顶点索引列表 (升序) , 若无选中顶点则返回空列表。
        """
        sel = cmds.ls(selection=True, flatten=True) or []
        # 尝试获取其 shape 名 (解析时做宽松匹配) 
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
            node_part = m.group(1)    # e.g. "pSphere1" or "pSphereShape1"
            idx       = int(m.group(2))
            # 允许 transform 名或 shape 名匹配
            if node_part in (transform, shape_name):
                indices.append(idx)

        return sorted(set(indices))

    @staticmethod
    def get_vertex_world_pos(transform: str, vtx_idx: int) -> np.ndarray:
        """
        获取指定顶点的世界空间坐标。

        Parameters
        ----------
        transform : str
            mesh transform 节点名。
        vtx_idx : int
            顶点索引。

        Returns
        -------
        pos : np.ndarray, shape (3,), dtype float64
            顶点世界空间坐标 [x, y, z]。
        """
        fn_mesh = MayaMeshBridge._get_fn_mesh(transform)
        pts     = fn_mesh.getPoints(om2.MSpace.kWorld)
        p       = pts[vtx_idx]
        return np.array([p.x, p.y, p.z], dtype=np.float64)

    @staticmethod
    def get_closest_point_on_mesh(
        target:     str,
        query_pos:  np.ndarray,
    ) -> np.ndarray:
        """
        在目标网格表面查找距给定世界空间点最近的表面点坐标。

        Parameters
        ----------
        target : str
            目标网格 transform 名称。
        query_pos : np.ndarray, shape (3,)
            世界空间查询点坐标。

        Returns
        -------
        closest_pos : np.ndarray, shape (3,), dtype float64
            目标网格表面上距 query_pos 最近的点的世界空间坐标。
        """
        fn_mesh   = MayaMeshBridge._get_fn_mesh(target)
        query_mpt = om2.MPoint(
            float(query_pos[0]),
            float(query_pos[1]),
            float(query_pos[2]),
        )
        closest_pt, _face_idx = fn_mesh.getClosestPoint(query_mpt, om2.MSpace.kWorld)
        return np.array([closest_pt.x, closest_pt.y, closest_pt.z], dtype=np.float64)

    @staticmethod
    def _get_mesh_dag_path(transform: str) -> "om2.MDagPath":
        """
        将 transform 名解析为 mesh shape 的 MDagPath。

        Parameters
        ----------
        transform : str
            网格 transform 节点名 (或 shape 名) 。

        Returns
        -------
        dag_path : om2.MDagPath
            已 extendToShape 后的 mesh DAG 路径。
        """
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
    def apply_vertices(transform: str, p_prime: np.ndarray) -> None:
        """
        将 ARAP 计算得到的新顶点坐标 (世界空间) 写回 Maya 网格。

        使用 undoInfo chunk 包裹整个写入操作，令其在 Maya 历史记录中
        表现为单步（可通过 Ctrl+Z 触发 chunk 回滚，但注意 API 层的
        MFnMesh.setPoints 不经过命令引擎，实际撤销效果依赖 Maya 版本）。

        Parameters
        ----------
        transform : str
            source 网格 transform 节点名。
        p_prime : np.ndarray, shape (N, 3), dtype float64
            ARAP 变形后的顶点坐标 (世界空间) 。
        """
        dag_path = MayaMeshBridge._get_mesh_dag_path(transform)
        new_mpa  = om2.MPointArray(
            [om2.MPoint(float(r[0]), float(r[1]), float(r[2])) for r in p_prime]
        )
        cmds.undoInfo(openChunk=True, chunkName="arapDeform")
        try:
            fn_mesh = om2.MFnMesh(dag_path)
            fn_mesh.setPoints(new_mpa, om2.MSpace.kWorld)
            fn_mesh.updateSurface()
            cmds.refresh(force=True)
        finally:
            cmds.undoInfo(closeChunk=True)

    @staticmethod
    def set_vertices_direct(dag_path: "om2.MDagPath", p_prime: np.ndarray) -> None:
        """
        直接写入顶点坐标并刷新视口，用于逐迭代预览的中间步骤。

        预览模式下每次迭代均调用此方法，最终迭代完成后由
        apply_vertices() 以 undoChunk 方式写入最终结果。

        Parameters
        ----------
        dag_path : om2.MDagPath
            已 extendToShape 的 mesh DAG 路径。
        p_prime : np.ndarray, shape (N, 3), dtype float64
            当前迭代后的顶点坐标 (世界空间) 。
        """
        mpa     = om2.MPointArray(
            [om2.MPoint(float(r[0]), float(r[1]), float(r[2])) for r in p_prime]
        )
        fn_mesh = om2.MFnMesh(dag_path)
        fn_mesh.setPoints(mpa, om2.MSpace.kWorld)
        fn_mesh.updateSurface()
        cmds.refresh(force=True)


# =============================================================================
# 主工具窗口 (PySide2 + MayaQWidgetDockableMixin) 
# =============================================================================

# 列索引常量, 避免魔术数字
_COL_VTX    = 0
_COL_TX     = 1
_COL_TY     = 2
_COL_TZ     = 3
_TABLE_COLS = 4

_WINDOW_OBJECT_NAME = "ArapToolWindowDockable"


class ArapToolWindow(MayaQWidgetDockableMixin, QtWidgets.QWidget):  # type: ignore[misc]
    """
    ARAP 网格变形工具主窗口。

    状态字段
    --------
    _target_mesh  : Optional[str]            变形参照网格 transform 名
    _source_mesh  : Optional[str]            被变形网格 transform 名
    _constraints  : Dict[int, np.ndarray]    {source 顶点索引 → target 世界坐标}
    """

    TOOL_NAME = _WINDOW_OBJECT_NAME

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super(ArapToolWindow, self).__init__(parent=parent)

        # ---- 工具状态 -------------------------------------------------------
        self._target_mesh:  Optional[str]            = None
        self._source_mesh:  Optional[str]            = None
        self._constraints:  Dict[int, np.ndarray]    = {}

        # ---- 构建 UI --------------------------------------------------------
        self.setWindowTitle("ARAP 网格变形工具")
        self.setObjectName(self.TOOL_NAME)
        self.setMinimumWidth(420)
        self._build_ui()

    # =========================================================================
    # UI 构建
    # =========================================================================

    def _build_ui(self) -> None:
        """构建完整的工具窗口布局。"""
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setSpacing(8)
        root_layout.setContentsMargins(10, 10, 10, 10)

        # ---- 组 1：对象设置 --------------------------------------------------
        grp_objects = QtWidgets.QGroupBox("对象设置")
        layout_obj  = QtWidgets.QFormLayout(grp_objects)
        layout_obj.setLabelAlignment(QtCore.Qt.AlignRight)

        # 变形目标 (Target) 
        row_target = QtWidgets.QHBoxLayout()
        self._le_target = QtWidgets.QLineEdit()
        self._le_target.setReadOnly(True)
        self._le_target.setPlaceholderText("未设置 (请先在大纲视图中选中目标网格) ")
        btn_set_target = QtWidgets.QPushButton("设置目标 ▶")
        btn_set_target.setToolTip(
            "在大纲视图/视口中选中「变形参照」网格后点击此按钮。\n"
            " (例如 MetaHuman 模板头部网格) "
        )
        btn_set_target.clicked.connect(self._on_set_target)
        row_target.addWidget(self._le_target)
        row_target.addWidget(btn_set_target)
        layout_obj.addRow("变形目标 (Target)：", row_target)

        # 变形对象 (Source) 
        row_source = QtWidgets.QHBoxLayout()
        self._le_source = QtWidgets.QLineEdit()
        self._le_source.setReadOnly(True)
        self._le_source.setPlaceholderText("未设置 (请先在大纲视图中选中源网格) ")
        btn_set_source = QtWidgets.QPushButton("设置对象 ▶")
        btn_set_source.setToolTip(
            "在大纲视图/视口中选中「待变形」网格后点击此按钮。\n"
            " (例如写实人物头部网格) "
        )
        btn_set_source.clicked.connect(self._on_set_source)
        row_source.addWidget(self._le_source)
        row_source.addWidget(btn_set_source)
        layout_obj.addRow("变形对象 (Source)：", row_source)

        root_layout.addWidget(grp_objects)

        # ---- 组 2：约束点管理 ------------------------------------------------
        self._grp_constraints = QtWidgets.QGroupBox("约束点管理 (Landmarks)")
        self._grp_constraints.setEnabled(False)   # 未设置 source 前禁用
        layout_con = QtWidgets.QVBoxLayout(self._grp_constraints)

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
        self._table.setMinimumHeight(130)
        layout_con.addWidget(self._table)

        # 按钮行 1：添加
        row_add = QtWidgets.QHBoxLayout()
        btn_add_pair = QtWidgets.QPushButton("添加配对约束")
        btn_add_pair.setToolTip(
            "同时选中 Source 网格上的 1 个顶点 + Target 网格上的 1 个顶点, \n"
            "点击此按钮建立精确的手动对应关系。\n"
            "适合少量关键 landmark (鼻尖、眼角、嘴角等) 。"
        )
        btn_add_pair.clicked.connect(self._on_add_pair)

        btn_add_nearest = QtWidgets.QPushButton("投影最近点约束")
        btn_add_nearest.setToolTip(
            "在 Source 网格上选中一个或多个顶点, \n"
            "点击此按钮将每个顶点自动投影到 Target 网格上最近的表面点, \n"
            "快速批量添加约束。"
        )
        btn_add_nearest.clicked.connect(self._on_add_nearest)

        row_add.addWidget(btn_add_pair)
        row_add.addWidget(btn_add_nearest)
        layout_con.addLayout(row_add)

        # 按钮行 2：删除 / 清空
        row_del = QtWidgets.QHBoxLayout()
        btn_remove = QtWidgets.QPushButton("删除选中约束")
        btn_remove.setToolTip("删除表格中当前选中的约束行。")
        btn_remove.clicked.connect(self._on_remove_constraint)

        btn_clear = QtWidgets.QPushButton("清空全部约束")
        btn_clear.setToolTip("清除所有已添加的约束, 重新开始配置。")
        btn_clear.clicked.connect(self._on_clear_constraints)

        row_del.addWidget(btn_remove)
        row_del.addWidget(btn_clear)
        layout_con.addLayout(row_del)

        # 约束数量标签
        self._lbl_con_count = QtWidgets.QLabel("当前约束数：0")
        self._lbl_con_count.setAlignment(QtCore.Qt.AlignRight)
        layout_con.addWidget(self._lbl_con_count)

        root_layout.addWidget(self._grp_constraints)

        # ---- 组 3：求解参数 --------------------------------------------------
        grp_params = QtWidgets.QGroupBox("求解参数")
        layout_params = QtWidgets.QFormLayout(grp_params)
        layout_params.setLabelAlignment(QtCore.Qt.AlignRight)

        self._spin_iterations = QtWidgets.QSpinBox()
        self._spin_iterations.setRange(1, 200)
        self._spin_iterations.setValue(10)
        self._spin_iterations.setToolTip(
            "ARAP 局部-全局交替迭代次数。\n"
            "建议范围：5 (快速预览) ~ 30 (高精度) 。\n"
            "面数越多可适当提高, 一般 10 次已足够。"
        )
        layout_params.addRow("迭代次数：", self._spin_iterations)

        self._le_tol = QtWidgets.QLineEdit("1e-6")
        self._le_tol.setToolTip(
            "收敛阈值：当相邻两次迭代间所有顶点位移最大值 < 此阈值时提前终止。\n"
            "设为 0 可禁用提前终止, 强制跑完全部迭代轮次。"
        )
        layout_params.addRow("收敛阈值：", self._le_tol)

        self._le_reg = QtWidgets.QLineEdit("1e-8")
        self._le_reg.setToolTip(
            "Tikhonov 正则化系数 λ：在 Cholesky 分解前对矩阵施加 L += λ·I，\n"
            "使矩阵严格正定，防止出现 'Factor is exactly singular' 错误。\n"
            "通常 1e-8 即可；若仍报奇异错误可逐步增大到 1e-5。"
        )
        layout_params.addRow("正则化系数 λ：", self._le_reg)

        self._chk_preview = QtWidgets.QCheckBox("逐迭代预览（每次迭代后更新视口）")
        self._chk_preview.setChecked(False)
        self._chk_preview.setToolTip(
            "勾选后每完成一次迭代就将当前顶点坐标写入网格并刷新视口，\n"
            "可直观看到变形逐步收敛的过程，但速度会变慢。\n"
            "不勾选则仅在全部迭代完成后一次性写入（更快）。\n"
            "两种模式最终均支持 Ctrl+Z 撤销。"
        )
        layout_params.addRow("", self._chk_preview)

        root_layout.addWidget(grp_params)

        # ---- 分隔线 ----------------------------------------------------------
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        root_layout.addWidget(sep)

        # ---- 执行按钮 -------------------------------------------------------
        self._btn_execute = QtWidgets.QPushButton("  执行 ARAP 变形  ")
        self._btn_execute.setMinimumHeight(36)
        self._btn_execute.setToolTip(
            "读取 Source 网格数据, 运行 ARAP 求解, \n"
            "将变形结果原地写回 Source 网格顶点坐标。\n"
            "操作支持 Ctrl+Z 撤销。\n\n"
            "注意：至少需要 3 个约束点才能开始求解。"
        )
        # 蓝色高亮样式
        self._btn_execute.setStyleSheet(
            "QPushButton {"
            "  background-color: #1A6EB5;"
            "  color: white;"
            "  border-radius: 4px;"
            "  font-size: 13px;"
            "  font-weight: bold;"
            "}"
            "QPushButton:hover  { background-color: #2280D2; }"
            "QPushButton:pressed{ background-color: #0F5591; }"
            "QPushButton:disabled{ background-color: #666666; color: #AAAAAA; }"
        )
        self._btn_execute.clicked.connect(self._on_execute)
        root_layout.addWidget(self._btn_execute)

        # ---- 状态栏 ---------------------------------------------------------
        self._lbl_status = QtWidgets.QLabel("")
        self._lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_status.setStyleSheet("color: #888888; font-size: 11px;")
        root_layout.addWidget(self._lbl_status)

        root_layout.addStretch()

    # =========================================================================
    # 辅助：更新状态栏文字
    # =========================================================================

    def _set_status(self, msg: str, color: str = "#888888") -> None:
        """
        更新底部状态标签文字和颜色。

        Parameters
        ----------
        msg   : str    要显示的状态信息。
        color : str    CSS 颜色字符串, 默认灰色。
        """
        self._lbl_status.setStyleSheet(f"color: {color}; font-size: 11px;")
        self._lbl_status.setText(msg)

    # =========================================================================
    # 辅助：刷新约束表格
    # =========================================================================

    def _refresh_constraint_table(self) -> None:
        """
        根据内存中的 _constraints 字典重建表格显示。

        每行 4 列：Source 顶点索引、Target X、Target Y、Target Z。
        所有单元格只读, 行按顶点索引升序排列。
        """
        self._table.setRowCount(0)
        for vtx_idx in sorted(self._constraints.keys()):
            pos   = self._constraints[vtx_idx]
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
                # 在 item.data 中存储顶点索引, 方便删除时还原 key
                if col == _COL_VTX:
                    item.setData(QtCore.Qt.UserRole, vtx_idx)
                self._table.setItem(row_n, col, item)

        count = len(self._constraints)
        self._lbl_con_count.setText(f"当前约束数：{count}")

    # =========================================================================
    # 按钮回调：对象设置
    # =========================================================================

    def _on_set_target(self) -> None:
        """
        将当前在 Maya 中选中的第一个 mesh 设为"变形目标 (Target) "。

        操作步骤：
            1. 在 Maya 大纲视图或视口中选中 MetaHuman 模板头部网格
            2. 点击此按钮
        """
        name = MayaMeshBridge.get_selected_mesh_transform()
        if name is None:
            QtWidgets.QMessageBox.warning(
                self, "未选中网格",
                "请先在场景中选中一个多边形网格, 再点击\"设置目标\"。",
            )
            return

        # 若目标发生变化, 已有约束中的 target 位置全部失效, 清空
        if name != self._target_mesh:
            self._constraints.clear()
            self._refresh_constraint_table()

        self._target_mesh = name
        self._le_target.setText(name)
        self._set_status(f"变形目标已设置：{name}", "#4CAF50")

    def _on_set_source(self) -> None:
        """
        将当前在 Maya 中选中的第一个 mesh 设为"变形对象 (Source) "。

        操作步骤：
            1. 在 Maya 大纲视图或视口中选中写实人物头部网格
            2. 点击此按钮

        设置成功后解锁"约束点管理"区域。
        """
        name = MayaMeshBridge.get_selected_mesh_transform()
        if name is None:
            QtWidgets.QMessageBox.warning(
                self, "未选中网格",
                "请先在场景中选中一个多边形网格, 再点击\"设置对象\"。",
            )
            return

        if name != self._source_mesh:
            self._constraints.clear()
            self._refresh_constraint_table()

        self._source_mesh = name
        self._le_source.setText(name)
        self._grp_constraints.setEnabled(True)   # 解锁约束管理区
        self._set_status(f"变形对象已设置：{name}", "#4CAF50")

    # =========================================================================
    # 按钮回调：约束管理
    # =========================================================================

    def _on_add_pair(self) -> None:
        """
        手动配对约束：同时选中 Source 上 1 个顶点 + Target 上 1 个顶点, 
        建立精确的对应关系并添加到约束列表。

        选中方式 (在 Maya 中) ：
            1. 进入顶点选择模式 (F9) 
            2. 先点选 Source 上的一个顶点
            3. 再 Shift 点选 Target 上的对应顶点
            4. 回到工具窗口, 点击"添加配对约束"

        典型用途：鼻尖、眼角、嘴角、下巴尖等关键 landmark 精确对应。
        """
        if not self._source_mesh or not self._target_mesh:
            self._set_status("请先设置变形目标和变形对象。", "#FF6B6B")
            return

        sel_flat = cmds.ls(selection=True, flatten=True) or []
        vtx_items = [s for s in sel_flat if ".vtx[" in s]

        if len(vtx_items) != 2:
            QtWidgets.QMessageBox.warning(
                self, "选择错误",
                "请在进入顶点模式后, 恰好选中 2 个顶点：\n"
                "  • 第 1 个：Source 网格上的顶点\n"
                "  • 第 2 个：Target 网格上的对应顶点\n\n"
                f"当前选中的顶点数：{len(vtx_items)}",
            )
            return

        # 解析两个顶点分别属于哪个 mesh
        def parse_vtx(item: str) -> Tuple[str, int]:
            m = re.match(r'^(.+)\.vtx\[(\d+)\]$', item)
            if not m:
                raise ValueError(f"无法解析顶点选择项：{item}")
            return m.group(1), int(m.group(2))

        try:
            node_a, idx_a = parse_vtx(vtx_items[0])
            node_b, idx_b = parse_vtx(vtx_items[1])
        except ValueError as exc:
            self._set_status(str(exc), "#FF6B6B")
            return

        # 尝试获取 shape 名做宽松匹配
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

        # 根据 node 归属判断哪个是 source、哪个是 target
        if node_a in src_names and node_b in tgt_names:
            src_idx, tgt_idx = idx_a, idx_b
        elif node_b in src_names and node_a in tgt_names:
            src_idx, tgt_idx = idx_b, idx_a
        else:
            QtWidgets.QMessageBox.warning(
                self, "选择错误",
                "两个顶点必须分属当前设置的 Source 和 Target 网格。\n\n"
                f"选中的节点：{node_a}、{node_b}\n"
                f"Source：{self._source_mesh}, Target：{self._target_mesh}",
            )
            return

        # 获取 target 顶点的世界坐标, 作为约束目标位置
        target_pos = MayaMeshBridge.get_vertex_world_pos(self._target_mesh, tgt_idx)
        self._constraints[src_idx] = target_pos
        self._refresh_constraint_table()
        self._set_status(
            f"已添加配对约束：vtx[{src_idx}] → Target vtx[{tgt_idx}]",
            "#4CAF50",
        )

    def _on_add_nearest(self) -> None:
        """
        投影最近点约束：在 Source 网格上选中一个或多个顶点, 
        自动投影到 Target 网格表面最近点, 批量添加约束。

        选中方式 (在 Maya 中) ：
            1. 进入顶点选择模式 (F9) , 确保当前操作 Source 网格
            2. 框选或逐一选中需要约束的顶点 (如头部轮廓点) 
            3. 回到工具窗口, 点击"投影最近点约束"

        典型用途：沿头部轮廓线批量添加约束, 引导整体形状对齐。
        若同一顶点已有配对约束, 投影结果会覆盖旧值。
        """
        if not self._source_mesh or not self._target_mesh:
            self._set_status("请先设置变形目标和变形对象。", "#FF6B6B")
            return

        src_indices = MayaMeshBridge.get_selected_vertex_indices(self._source_mesh)
        if not src_indices:
            QtWidgets.QMessageBox.warning(
                self, "未选中顶点",
                f"请在进入顶点模式后, 在 Source 网格 '{self._source_mesh}' 上\n"
                "选中一个或多个顶点, 再点击此按钮。",
            )
            return

        added   = 0
        skipped = 0
        for vtx_idx in src_indices:
            try:
                src_pos    = MayaMeshBridge.get_vertex_world_pos(self._source_mesh, vtx_idx)
                target_pos = MayaMeshBridge.get_closest_point_on_mesh(
                    self._target_mesh, src_pos
                )
                self._constraints[vtx_idx] = target_pos
                added += 1
            except Exception:
                skipped += 1

        self._refresh_constraint_table()
        msg = f"投影完成：新增/更新 {added} 个约束"
        if skipped:
            msg += f", {skipped} 个失败 (可忽略) "
        self._set_status(msg, "#4CAF50")

    def _on_remove_constraint(self) -> None:
        """
        删除约束表格中当前选中的行对应的约束。

        在表格中单击选中一行后点击此按钮即可删除该约束。
        """
        selected_rows = self._table.selectedItems()
        if not selected_rows:
            self._set_status("请先在表格中选中要删除的约束行。", "#FFCC00")
            return

        # 收集选中行的顶点索引 (从 UserRole 取, 避免字符串解析) 
        rows_to_delete: List[int] = []
        seen: set = set()
        for item in selected_rows:
            r = item.row()
            if r in seen:
                continue
            seen.add(r)
            vtx_item = self._table.item(r, _COL_VTX)
            if vtx_item:
                vtx_idx = vtx_item.data(QtCore.Qt.UserRole)
                rows_to_delete.append(vtx_idx)

        for vtx_idx in rows_to_delete:
            self._constraints.pop(vtx_idx, None)

        self._refresh_constraint_table()
        self._set_status(f"已删除 {len(rows_to_delete)} 个约束。", "#4CAF50")

    def _on_clear_constraints(self) -> None:
        """清空全部约束, 重置约束表格。"""
        if not self._constraints:
            self._set_status("当前没有任何约束。", "#FFCC00")
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "确认清空",
            f"确认要清除全部 {len(self._constraints)} 个约束吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._constraints.clear()
            self._refresh_constraint_table()
            self._set_status("已清空全部约束。", "#888888")

    # =========================================================================
    # 执行回调
    # =========================================================================

    def _on_execute(self) -> None:
        """
        执行完整的 ARAP 变形, 将结果原地写回 Source 网格。

        执行流程
        --------
        1. 校验 source / target / constraints 是否就绪
        2. 读取求解参数 (iterations、convergence_tol) 
        3. 从 Maya 读取 source 网格数据 (世界空间) → MeshData
        4. 创建并运行 ArapSolver
        5. 用 undoChunk 包裹写回操作, 支持 Ctrl+Z 撤销

        错误处理
        --------
        任何步骤抛出异常均会：
          - 关闭 undoChunk (保证 Maya undo 栈完整) 
          - 弹出错误对话框
          - 恢复执行按钮可交互状态
        """
        # ---- 校验 -----------------------------------------------------------
        if not self._source_mesh or not self._target_mesh:
            QtWidgets.QMessageBox.warning(
                self, "配置不完整",
                "请先设置\"变形目标\"和\"变形对象\"。",
            )
            return

        if len(self._constraints) < 3:
            QtWidgets.QMessageBox.warning(
                self, "约束不足",
                f"当前约束数为 {len(self._constraints)} 个, 至少需要 3 个才能进行稳定求解。\n\n"
                "建议添加 6~10 个分布在头部关键位置的 landmark 约束\n"
                " (鼻尖、下巴尖、左右眼角、左右嘴角、耳根等) 。",
            )
            return

        # 解析收敛阈值
        try:
            tol = float(self._le_tol.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "参数错误",
                f"收敛阈值 '{self._le_tol.text()}' 不是有效的浮点数。",
            )
            return

        # 解析正则化系数
        try:
            reg = float(self._le_reg.text().strip())
            if reg < 0.0:
                raise ValueError("正则化系数不能为负数。")
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "参数错误",
                f"正则化系数 '{self._le_reg.text()}' 不是有效的非负浮点数。",
            )
            return

        iterations = self._spin_iterations.value()

        # ---- 禁用按钮, 防止重复点击 -----------------------------------------
        self._btn_execute.setEnabled(False)
        self._set_status("正在执行 ARAP 变形, 请稍候…", "#FFCC00")
        QtWidgets.QApplication.processEvents()   # 强制刷新 UI

        try:
            # ---- 读取网格数据 ------------------------------------------------
            self._set_status("正在读取网格数据…", "#FFCC00")
            QtWidgets.QApplication.processEvents()
            mesh_data: MeshData = MayaMeshBridge.get_mesh_data(self._source_mesh)

            # ---- 创建求解器并运行 -------------------------------------------
            self._set_status(
                f"正在求解 (约束 {len(self._constraints)} 个, "
                f"迭代 {iterations} 次) …",
                "#FFCC00",
            )
            QtWidgets.QApplication.processEvents()

            preview_mode = self._chk_preview.isChecked()

            if preview_mode:
                # ---- 预览模式：每次迭代直接写入视口，最终结果用 undoChunk 包裹 ----
                dag_path_src = MayaMeshBridge._get_mesh_dag_path(self._source_mesh)

                def _preview_callback(iter_idx: int, p_iter: np.ndarray) -> None:
                    self._set_status(
                        f"迭代 {iter_idx + 1}/{iterations}…", "#FFCC00"
                    )
                    QtWidgets.QApplication.processEvents()
                    MayaMeshBridge.set_vertices_direct(dag_path_src, p_iter)

                solver = create_solver(
                    constraints     = self._constraints,
                    iterations      = iterations,
                    convergence_tol = tol,
                    regularization  = reg,
                    verbose         = True,
                    step_callback   = _preview_callback,
                )
                p_prime: np.ndarray = solver.execute(mesh_data)

                # 最终结果以 undoChunk 方式写回（覆盖预览过程中的中间状态）
                MayaMeshBridge.apply_vertices(self._source_mesh, p_prime)

            else:
                # ---- 普通模式：仅在最终结果上执行一次可撤销写入 -------------
                solver = create_solver(
                    constraints     = self._constraints,
                    iterations      = iterations,
                    convergence_tol = tol,
                    regularization  = reg,
                    verbose         = True,
                )
                p_prime: np.ndarray = solver.execute(mesh_data)

                self._set_status("正在将结果写回网格…", "#FFCC00")
                QtWidgets.QApplication.processEvents()

                MayaMeshBridge.apply_vertices(self._source_mesh, p_prime)

            # 成功通知
            self._set_status(
                f"✓ 变形完成, 共处理 {mesh_data.num_vertices} 个顶点。"
                f" (可 Ctrl+Z 撤销) ",
                "#4CAF50",
            )
            # Maya 视口内弹出通知 (Maya 2015+ 支持 inViewMessage) 
            try:
                cmds.inViewMessage(
                    amg=f"<hl>ARAP</hl> 变形完成！顶点数：{mesh_data.num_vertices}",
                    pos="botCenter",
                    fade=True,
                    fadeStayTime=2000,
                )
            except Exception:
                pass   # inViewMessage 不可用时静默忽略

        except Exception as exc:
            error_msg = str(exc)
            self._set_status(f"执行失败：{error_msg}", "#FF6B6B")
            QtWidgets.QMessageBox.critical(
                self, "ARAP 执行失败",
                f"求解过程中发生错误：\n\n{error_msg}\n\n"
                "请检查网格数据是否有效, 约束点是否合理。",
            )

        finally:
            self._btn_execute.setEnabled(True)


# =============================================================================
# 模块入口
# =============================================================================

# 全局窗口实例, 防止重复创建
_WINDOW_INSTANCE: Optional["ArapToolWindow"] = None


def show() -> "ArapToolWindow":
    """
    在 Maya 中打开 ARAP 网格变形工具窗口。

    若窗口已存在则先关闭旧窗口再重建, 保证每次打开的都是干净的实例。

    Returns
    -------
    window : ArapToolWindow
        已显示的工具窗口实例。

    在 Maya Script Editor 中调用
    ----------------------------
    .. code-block:: python

        import importlib, sys
        sys.path.insert(0, r"D:/Repository/graphicsalgorithm/Tools/MetaHuman Rig Logic/Python File/maya/core/NICP")
        import ARAP, MayaARAP
        importlib.reload(ARAP)
        importlib.reload(MayaARAP)
        MayaARAP.show()
    """
    if not _IN_MAYA:
        raise RuntimeError("MayaARAP.show() 只能在 Maya 内部运行。")

    # 强制重新加载 ARAP 核心模块, 确保 create_solver / MeshData 等名称
    # 始终绑定到最新版本, 防止 sys.modules 缓存旧版本导致参数不匹配。
    import importlib as _importlib
    _arap_mod = sys.modules.get("ARAP")
    if _arap_mod is not None:
        _importlib.reload(_arap_mod)
        # 重新绑定本模块顶层 from-import 导入的名称
        global create_solver, MeshData, CholeskyFactor, ArapSolver
        from ARAP import CholeskyFactor, MeshData, ArapSolver, create_solver  # noqa: F811

    global _WINDOW_INSTANCE

    # 关闭旧实例
    if _WINDOW_INSTANCE is not None:
        try:
            _WINDOW_INSTANCE.close()
            _WINDOW_INSTANCE.deleteLater()
        except RuntimeError:
            pass   # C++ 对象已被 Maya 删除, 忽略
        _WINDOW_INSTANCE = None

    # 删除 Maya 中残留的 WorkspaceControl 节点
    # Maya dockable 窗口会在场景中创建名为 "{objectName}WorkspaceControl" 的节点,
    # 若上次窗口未被彻底清理 (如 Maya 崩溃或脚本重载), 再次 show() 时
    # Maya 会报 "对象名称不唯一" 错误, 必须手动删除后再创建。
    workspace_ctrl = _WINDOW_OBJECT_NAME + "WorkspaceControl"
    if cmds.workspaceControl(workspace_ctrl, query=True, exists=True):
        cmds.deleteUI(workspace_ctrl)

    _WINDOW_INSTANCE = ArapToolWindow()
    # dockable=True：允许停靠到 Maya 布局中；floating=True：默认以浮动窗口显示
    _WINDOW_INSTANCE.show(dockable=True, floating=True)
    return _WINDOW_INSTANCE
