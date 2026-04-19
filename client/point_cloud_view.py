from __future__ import annotations

from pathlib import Path
from typing import Callable

try:
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
except ImportError:  # pragma: no cover - client deps are optional in tests
    QLabel = object  # type: ignore[assignment]
    QVBoxLayout = object  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment]

try:  # pragma: no cover - client deps are optional in tests
    import numpy as np
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError:  # pragma: no cover
    np = None
    pv = None
    QtInteractor = None


class PointCloudView(QWidget):  # pragma: no cover - UI widgets are not unit-tested
    def __init__(self, on_part_selected: Callable[[int, str], None] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.on_part_selected = on_part_selected
        self.mesh = None
        self.highlight_actor = None
        self.length_map: dict[int, float | None] = {}
        self.part_names: dict[int, str] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 去除边距，使渲染区域更饱满
        
        if QtInteractor is None or pv is None:
            self.placeholder = QLabel("尚未安装 PyVista 或 PySide6，点云预览不可用。")
            self.placeholder.setStyleSheet("color: #888888; alignment: center;")
            layout.addWidget(self.placeholder)
            self.plotter = None
            return

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        
        # 将背景修改为深色，以匹配全局的工业风主题
        self.plotter.set_background("#1e1e1e")
        self.plotter.add_text("暂无点云数据", font_size=11, color="#888888")
        self.plotter.enable_point_picking(
            callback=self._handle_picked_point,
            show_message=False,
            use_mesh=True,
            left_clicking=True,
        )

    def set_lookup(self, part_names: dict[int, str], length_map: dict[int, float | None]) -> None:
        self.part_names = dict(part_names)
        self.length_map = dict(length_map)

    def load_segmentation_ply(self, path: Path) -> None:
        if self.plotter is None or pv is None:
            return
        self.plotter.clear()
        self.mesh = pv.read(str(path))
        scalars = "pred_label" if "pred_label" in self.mesh.point_data else None
        
        # 调整点云的基础渲染参数，增强立体感
        self.plotter.add_mesh(
            self.mesh,
            scalars=scalars,
            rgb=scalars is None,
            render_points_as_spheres=True,
            point_size=4,  # 稍微减小基础点大小，使高亮部分更突出
            ambient=0.2,
            name="main_cloud",
        )
        self.plotter.reset_camera()

    def highlight_part(self, part_id: int) -> None:
        if self.plotter is None or self.mesh is None:
            return
        if "pred_label" not in self.mesh.point_data:
            return
        mask = self.mesh.point_data["pred_label"] == part_id
        if self.highlight_actor is not None:
            self.plotter.remove_actor(self.highlight_actor, render=False)
            self.highlight_actor = None
            
        if mask.any():
            highlighted = self.mesh.extract_points(mask, adjacent_cells=False)
            self.highlight_actor = self.plotter.add_mesh(
                highlighted,
                color="#00e5ff",  # 采用高对比度的亮青色作为高亮，在深色背景下更醒目
                point_size=8,
                render_points_as_spheres=True,
                name="highlight",
            )
        self.plotter.render()

    def _handle_picked_point(self, point) -> None:
        if self.mesh is None or pv is None:
            return
        point_id = int(self.mesh.find_closest_point(point))
        label_array = self.mesh.point_data.get("pred_label")
        if label_array is None:
            return
        part_id = int(label_array[point_id])
        self.highlight_part(part_id)
        part_name = self.part_names.get(part_id, f"part_{part_id:02d}")
        
        if self.on_part_selected is not None:
            length = self.length_map.get(part_id)
            if length is None:
                self.on_part_selected(part_id, part_name)
            else:
                self.on_part_selected(part_id, f"{part_name} | 长度={length:.4f}")