from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover - UI widgets are not unit-tested
    from PySide6.QtCore import QDir, Qt, Signal
    from PySide6.QtWidgets import (
        QFileSystemModel,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QTreeView,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover
    QWidget = object  # type: ignore[assignment]


class FileBrowserWidget(QWidget):  # pragma: no cover - UI widgets are not unit-tested
    browse_requested = Signal()
    selection_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._configure_model()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.root_label = QLabel("当前目录: 未选择")
        self.root_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.root_label.setMinimumWidth(0)
        self.root_label.setToolTip("当前目录")
        self.browse_button = QPushButton("选择目录")
        self.browse_button.clicked.connect(self.browse_requested.emit)
        header_layout.addWidget(self.root_label, 1)
        header_layout.addWidget(self.browse_button)
        layout.addLayout(header_layout)

        self.tree = QTreeView()
        self.tree.setSelectionMode(QTreeView.ExtendedSelection)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(False)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)
        self.tree.doubleClicked.connect(self._expand_directory)
        layout.addWidget(self.tree, 1)

    def _configure_model(self) -> None:
        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        self.model.setNameFilters(
            [
                "*.png",
                "*.jpg",
                "*.jpeg",
                "*.bmp",
                "*.tif",
                "*.tiff",
                "*.ply",
            ]
        )
        self.model.setNameFilterDisables(False)
        root_path = self._default_root_path()
        self._set_root_label(root_path)
        self.model.setRootPath(root_path)
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(root_path))
        self.tree.selectionModel().selectionChanged.connect(self._emit_selection_changed)

    def set_root_path(self, root_path: Path) -> None:
        resolved = root_path.resolve()
        self._set_root_label(str(resolved))
        self.tree.setRootIndex(self.model.index(str(resolved)))

    def selected_file_paths(self) -> list[Path]:
        paths: list[Path] = []
        seen: set[Path] = set()
        for index in self.tree.selectionModel().selectedRows(0):
            if self.model.isDir(index):
                continue
            path = Path(self.model.filePath(index)).resolve()
            if path not in seen:
                seen.add(path)
                paths.append(path)
        return paths

    def _emit_selection_changed(self) -> None:
        self.selection_changed.emit(self.selected_file_paths())

    def _expand_directory(self, index) -> None:
        if self.model.isDir(index):
            self.tree.setExpanded(index, not self.tree.isExpanded(index))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_full_root_path"):
            self._set_root_label(self._full_root_path)

    def _set_root_label(self, root_path: str) -> None:
        self._full_root_path = root_path
        prefix = "当前目录: "
        metrics = self.root_label.fontMetrics()
        available_width = max(self.root_label.width(), 120)
        elided = metrics.elidedText(
            root_path,
            Qt.TextElideMode.ElideMiddle,
            max(available_width - metrics.horizontalAdvance(prefix), 40),
        )
        self.root_label.setText(f"{prefix}{elided}")
        self.root_label.setToolTip(root_path)

    @staticmethod
    def _default_root_path() -> str:
        return str(Path(__file__).resolve().parents[1])
