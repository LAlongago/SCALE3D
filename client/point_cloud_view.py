from __future__ import annotations

from pathlib import Path
from typing import Callable

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QLabel, QSizePolicy, QStackedLayout, QVBoxLayout, QWidget
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
        self.main_cloud_actor = None
        self.highlight_actor = None
        self.skeleton_actors = []
        self.show_object_cloud = True
        self.show_skeleton = True
        self.point_size = 4
        self.skeleton_point_size = 7
        self._image_pixmap = None
        self.length_map: dict[int, float | None] = {}
        self.part_names: dict[int, str] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedLayout()
        layout.addLayout(self.stack)

        self.placeholder = QLabel("暂无文件预览")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #888888;")
        self.stack.addWidget(self.placeholder)

        self.image_label = QLabel("暂无图片预览")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #888888; background-color: #111111;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.image_label)

        if QtInteractor is None or pv is None:
            self.plotter = None
            self.pointcloud_placeholder = QLabel("尚未安装 PyVista 或 PySide6，点云预览不可用。")
            self.pointcloud_placeholder.setAlignment(Qt.AlignCenter)
            self.pointcloud_placeholder.setStyleSheet("color: #888888;")
            self.stack.addWidget(self.pointcloud_placeholder)
        else:
            self.plotter = QtInteractor(self)
            self.plotter.set_background("#1e1e1e")
            self._reset_plotter_scene()
            self.plotter.enable_point_picking(
                callback=self._handle_picked_point,
                show_message=False,
                use_picker=True,
                left_clicking=True,
            )
            self.stack.addWidget(self.plotter.interactor)

        self.stack.setCurrentWidget(self.placeholder)

    def clear_view(self) -> None:
        self.mesh = None
        self.main_cloud_actor = None
        self.highlight_actor = None
        self.skeleton_actors = []
        self._image_pixmap = None
        if self.plotter is not None:
            self._reset_plotter_scene()
        self.image_label.clear()
        self.image_label.setText("暂无图片预览")
        self.stack.setCurrentWidget(self.placeholder)

    def preview_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".ply":
            self.load_segmentation_ply(path)
            return
        self.show_image(path)

    def show_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._image_pixmap = None
            self.image_label.setText(f"无法预览图片:\n{path.name}")
        else:
            self._image_pixmap = pixmap
            self._refresh_image_pixmap()
        self.stack.setCurrentWidget(self.image_label)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_image_pixmap()

    def set_lookup(self, part_names: dict[int, str], length_map: dict[int, float | None]) -> None:
        self.part_names = dict(part_names)
        self.length_map = dict(length_map)

    def load_segmentation_ply(self, path: Path) -> None:
        if self.plotter is None or pv is None:
            self.stack.setCurrentWidget(self.pointcloud_placeholder)
            return
        self.plotter.clear()
        self.plotter.show_grid(color="#4a4a4a")
        self.plotter.add_axes(line_width=2, labels_off=True)
        self.mesh = pv.read(str(path))
        render_kwargs = self._build_point_cloud_render_kwargs()
        self.main_cloud_actor = self.plotter.add_mesh(self.mesh, **render_kwargs)
        self.skeleton_actors = []
        self._apply_visibility()
        self.plotter.reset_camera()
        self.stack.setCurrentWidget(self.plotter.interactor)

    def load_skeleton_plys(self, paths: list[Path]) -> None:
        if self.plotter is None or pv is None:
            return
        for actor in self.skeleton_actors:
            self.plotter.remove_actor(actor, render=False)
        self.skeleton_actors = []
        for index, path in enumerate(paths):
            if not path.exists():
                continue
            mesh = pv.read(str(path))
            actor = self.plotter.add_mesh(
                mesh,
                color="#facc15",
                point_size=self.skeleton_point_size,
                render_points_as_spheres=True,
                name=f"skeleton_{index}",
            )
            self.skeleton_actors.append(actor)
        self._apply_visibility()

    def set_object_cloud_visible(self, visible: bool) -> None:
        self.show_object_cloud = visible
        self._apply_visibility()

    def set_skeleton_visible(self, visible: bool) -> None:
        self.show_skeleton = visible
        self._apply_visibility()

    def set_point_size(self, point_size: int) -> None:
        self.point_size = point_size
        self.skeleton_point_size = max(point_size + 3, 5)
        if self.main_cloud_actor is not None:
            self.main_cloud_actor.GetProperty().SetPointSize(self.point_size)
        for actor in self.skeleton_actors:
            actor.GetProperty().SetPointSize(self.skeleton_point_size)
        if self.highlight_actor is not None:
            self.highlight_actor.GetProperty().SetPointSize(max(point_size + 4, 6))
        if self.plotter is not None:
            self.plotter.render()

    def highlight_part(self, part_id: int) -> None:
        if self.plotter is None or self.mesh is None:
            return
        label_name = self._find_label_array_name()
        if label_name is None:
            return

        mask = self.mesh.point_data[label_name] == part_id
        if self.highlight_actor is not None:
            self.plotter.remove_actor(self.highlight_actor, render=False)
            self.highlight_actor = None

        if mask.any():
            highlighted = self.mesh.extract_points(mask, adjacent_cells=False)
            self.highlight_actor = self.plotter.add_mesh(
                highlighted,
                color="#00e5ff",
                point_size=max(self.point_size + 4, 6),
                render_points_as_spheres=True,
                name="highlight",
            )
        self.plotter.render()

    def _handle_picked_point(self, point, *_args) -> None:
        if self.mesh is None or pv is None:
            return
        if point is None:
            return
        point_id = int(self.mesh.find_closest_point(point))
        label_name = self._find_label_array_name()
        if label_name is None:
            return

        label_array = self.mesh.point_data[label_name]
        part_id = int(label_array[point_id])
        self.highlight_part(part_id)
        part_name = self.part_names.get(part_id, f"part_{part_id:02d}")
        if self.on_part_selected is not None:
            length = self.length_map.get(part_id)
            if length is None:
                self.on_part_selected(part_id, part_name)
            else:
                self.on_part_selected(part_id, f"{part_name} | 长度={length:.2f} cm")

    def _refresh_image_pixmap(self) -> None:
        if self._image_pixmap is None or self._image_pixmap.isNull():
            return
        self.image_label.setPixmap(
            self._image_pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _build_point_cloud_render_kwargs(self) -> dict:
        kwargs = {
            "render_points_as_spheres": True,
            "point_size": self.point_size,
            "ambient": 0.2,
            "name": "main_cloud",
        }
        if self.mesh is None:
            return kwargs

        point_data = self.mesh.point_data
        label_name = self._find_label_array_name()
        if label_name is not None:
            kwargs["scalars"] = self._build_label_rgb(point_data[label_name])
            kwargs["rgb"] = True
            return kwargs

        rgb_array_name = self._find_rgb_array_name()
        if rgb_array_name is not None:
            kwargs["scalars"] = rgb_array_name
            kwargs["rgb"] = True
            return kwargs

        if self._has_rgb_components():
            kwargs["scalars"] = self._build_rgb_components()
            kwargs["rgb"] = True
            return kwargs

        kwargs["color"] = "#3b82f6"
        return kwargs

    def _find_rgb_array_name(self) -> str | None:
        if self.mesh is None:
            return None

        point_data = self.mesh.point_data
        preferred_names = (
            "rgb",
            "rgba",
            "RGB",
            "RGBA",
            "colors",
            "Colors",
            "colour",
            "diffuse_colors",
        )
        for name in preferred_names:
            if name in point_data and self._is_rgb_array(point_data[name]):
                return name

        for name in point_data.keys():
            normalized = name.lower()
            if ("rgb" in normalized or "color" in normalized or "colour" in normalized) and self._is_rgb_array(point_data[name]):
                return name
        return None

    def _find_label_array_name(self) -> str | None:
        if self.mesh is None:
            return None
        point_data = self.mesh.point_data
        preferred_names = ("pred_label", "label", "segment", "segmentation")
        lookup = {name.lower(): name for name in point_data.keys()}
        for name in preferred_names:
            actual = lookup.get(name)
            if actual is not None:
                return actual
        return None

    def _has_rgb_components(self) -> bool:
        if self.mesh is None:
            return False
        point_data = self.mesh.point_data
        component_names = {"red", "green", "blue"}
        return component_names.issubset({name.lower() for name in point_data.keys()})

    def _build_rgb_components(self):
        if self.mesh is None or np is None:
            return None
        point_data = self.mesh.point_data
        channel_map = {name.lower(): point_data[name] for name in point_data.keys()}
        red = channel_map["red"]
        green = channel_map["green"]
        blue = channel_map["blue"]
        return np.column_stack((red, green, blue))

    def _build_label_rgb(self, labels):
        if np is None:
            return labels
        palette = np.array(
            [
                [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
                [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
                [210, 245, 60], [250, 190, 190], [0, 128, 128], [230, 190, 255],
                [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
                [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
                [255, 99, 71], [154, 205, 50], [30, 144, 255], [255, 140, 0],
                [186, 85, 211], [0, 206, 209], [255, 20, 147], [124, 252, 0],
                [255, 182, 193], [32, 178, 170], [221, 160, 221], [160, 82, 45],
                [255, 239, 213], [139, 0, 0], [127, 255, 212], [85, 107, 47],
            ],
            dtype=np.uint8,
        )
        label_array = np.asarray(labels, dtype=np.int64)
        return palette[np.mod(label_array, len(palette))]

    @staticmethod
    def _is_rgb_array(array) -> bool:
        shape = getattr(array, "shape", None)
        if shape is None or len(shape) != 2:
            return False
        return shape[1] in {3, 4}

    def _apply_visibility(self) -> None:
        if self.main_cloud_actor is not None:
            self.main_cloud_actor.SetVisibility(self.show_object_cloud)
        for actor in self.skeleton_actors:
            actor.SetVisibility(self.show_skeleton)
        if self.plotter is not None:
            self.plotter.render()

    def _reset_plotter_scene(self) -> None:
        if self.plotter is None:
            return
        self.plotter.clear()
        self.plotter.show_grid(color="#4a4a4a")
        self.plotter.add_axes(line_width=2, labels_off=True)
        self.plotter.add_text("暂无点云数据", font_size=11, color="#888888")
