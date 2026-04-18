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
        if QtInteractor is None or pv is None:
            self.placeholder = QLabel("PyVista / PySide6 is not installed. Point cloud preview is unavailable.")
            layout.addWidget(self.placeholder)
            self.plotter = None
            return

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#f7fafc")
        self.plotter.add_text("No point cloud loaded", font_size=10)
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
        self.plotter.add_mesh(
            self.mesh,
            scalars=scalars,
            rgb=scalars is None,
            render_points_as_spheres=True,
            point_size=5,
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
                color="#ff3b30",
                point_size=9,
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
                self.on_part_selected(part_id, f"{part_name} | length={length:.4f}")
