from __future__ import annotations

from pathlib import Path

from client.file_browser import FileBrowserWidget
from client.models import JobViewState
from client.point_cloud_view import PointCloudView
from shared.enums import InputType

try:  # pragma: no cover - UI widgets are not unit-tested
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover
    QMainWindow = object  # type: ignore[assignment]


class MainWindow(QMainWindow):  # pragma: no cover - UI widgets are not unit-tested
    browse_directory_requested = Signal()
    submit_requested = Signal()
    refresh_requested = Signal()
    task_selected = Signal(object)
    export_requested = Signal(str)
    part_row_selected = Signal(int)
    point_cloud_part_picked = Signal(int, str)
    file_selection_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._task_items: dict[str, QListWidgetItem] = {}
        self.setWindowTitle("SCALE3D检测系统")
        self.resize(1600, 900)
        self._apply_stylesheet()
        self._build_ui()

    def _apply_stylesheet(self) -> None:
        style = """
        QMainWindow, QWidget {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            border: 1px solid #3c3c3c;
            border-radius: 6px;
            margin-top: 1.5ex;
            padding: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: #4daafc;
        }
        QPushButton {
            background-color: #333333;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 6px 12px;
            color: #cccccc;
        }
        QPushButton:hover {
            background-color: #404040;
            border: 1px solid #4daafc;
        }
        QPushButton:pressed {
            background-color: #2d2d2d;
        }
        QPushButton[class="primary"] {
            background-color: #0066cc;
            color: #ffffff;
            border: 1px solid #005bb5;
        }
        QPushButton[class="primary"]:hover {
            background-color: #0073e6;
        }
        QComboBox, QListWidget, QTextEdit, QTableWidget, QTreeView {
            background-color: #252526;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 4px;
            color: #d4d4d4;
        }
        QComboBox:hover, QListWidget:hover, QTextEdit:hover, QTableWidget:hover, QTreeView:hover {
            border: 1px solid #4daafc;
        }
        QHeaderView::section {
            background-color: #2d2d2d;
            padding: 4px;
            border: none;
            border-right: 1px solid #3c3c3c;
            border-bottom: 1px solid #3c3c3c;
            font-weight: bold;
        }
        QTableWidget::item:selected, QTreeView::item:selected {
            background-color: #094771;
            color: #ffffff;
        }
        QProgressBar {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            text-align: center;
            background-color: #252526;
        }
        QProgressBar::chunk {
            background-color: #0066cc;
            width: 10px;
        }
        QSplitter::handle {
            background-color: #3c3c3c;
            width: 2px;
        }
        """
        self.setStyleSheet(style)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(Qt.Horizontal, central)
        root_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)

        controls = QGroupBox("任务控制")
        controls_layout = QGridLayout(controls)
        controls_layout.setSpacing(10)

        self.product_model_combo = QComboBox()
        self.input_type_combo = QComboBox()
        self.input_type_combo.addItem("图像组", InputType.IMAGE_SET.value)
        self.input_type_combo.addItem("点云文件", InputType.POINT_CLOUD.value)

        self.select_button = QPushButton("打开目录")
        self.submit_button = QPushButton("提交任务")
        self.submit_button.setProperty("class", "primary")
        self.select_button.clicked.connect(self.browse_directory_requested.emit)
        self.submit_button.clicked.connect(self.submit_requested.emit)

        controls_layout.addWidget(QLabel("产品型号"), 0, 0)
        controls_layout.addWidget(self.product_model_combo, 0, 1)
        controls_layout.addWidget(QLabel("输入类型"), 1, 0)
        controls_layout.addWidget(self.input_type_combo, 1, 1)
        controls_layout.addWidget(self.select_button, 2, 0)
        controls_layout.addWidget(self.submit_button, 2, 1)
        left_layout.addWidget(controls)

        browser_label = QLabel("文件选择")
        browser_label.setContentsMargins(0, 10, 0, 5)
        left_layout.addWidget(browser_label)
        self.file_browser = FileBrowserWidget()
        self.file_browser.browse_requested.connect(self.browse_directory_requested.emit)
        self.file_browser.selection_changed.connect(self.file_selection_changed.emit)
        left_layout.addWidget(self.file_browser, 2)

        queue_label = QLabel("任务队列")
        queue_label.setContentsMargins(0, 10, 0, 5)
        left_layout.addWidget(queue_label)
        self.jobs_list = QListWidget()
        self.jobs_list.itemSelectionChanged.connect(self._emit_task_selected)
        left_layout.addWidget(self.jobs_list, 1)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(5, 0, 5, 0)

        view_label = QLabel("文件预览")
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        view_label.setFont(font)
        center_layout.addWidget(view_label)

        self.point_cloud_view = PointCloudView(on_part_selected=self._emit_point_cloud_pick)
        center_layout.addWidget(self.point_cloud_view, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)

        status_box = QGroupBox("当前任务状态")
        status_layout = QGridLayout(status_box)
        self.job_label = QLabel("尚未选择任务")
        self.server_status_label = QLabel("状态: -")
        self.server_stage_label = QLabel("阶段: -")
        self.server_message_label = QLabel("说明: -")
        self.server_message_label.setWordWrap(True)
        self.server_progress_bar = QProgressBar()
        self.server_progress_bar.setRange(0, 100)
        self.server_progress_bar.setFixedHeight(18)
        status_layout.addWidget(self.job_label, 0, 0, 1, 2)
        status_layout.addWidget(self.server_status_label, 1, 0)
        status_layout.addWidget(self.server_stage_label, 1, 1)
        status_layout.addWidget(self.server_message_label, 2, 0, 1, 2)
        status_layout.addWidget(self.server_progress_bar, 3, 0, 1, 2)
        right_layout.addWidget(status_box)

        transfer_box = QGroupBox("传输监控")
        transfer_layout = QGridLayout(transfer_box)
        self.transfer_status_label = QLabel("传输状态: 空闲")
        self.transfer_speed_label = QLabel("实时网速: -")
        self.transfer_bytes_label = QLabel("已传输: -")
        self.transfer_progress_bar = QProgressBar()
        self.transfer_progress_bar.setRange(0, 100)
        self.transfer_progress_bar.setFixedHeight(18)
        transfer_layout.addWidget(self.transfer_status_label, 0, 0, 1, 2)
        transfer_layout.addWidget(self.transfer_speed_label, 1, 0)
        transfer_layout.addWidget(self.transfer_bytes_label, 1, 1)
        transfer_layout.addWidget(self.transfer_progress_bar, 2, 0, 1, 2)
        right_layout.addWidget(transfer_box)

        table_label = QLabel("部件结果")
        table_label.setContentsMargins(0, 10, 0, 5)
        right_layout.addWidget(table_label)
        self.parts_table = QTableWidget(0, 4)
        self.parts_table.setHorizontalHeaderLabels(["部件", "点数", "置信度", "长度"])
        self.parts_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.parts_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.parts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.parts_table.verticalHeader().setVisible(False)
        self.parts_table.itemSelectionChanged.connect(self._emit_part_row_selected)
        right_layout.addWidget(self.parts_table, 1)

        export_box = QGroupBox("数据导出")
        export_layout = QGridLayout(export_box)
        export_layout.setSpacing(10)
        self.export_pdf_button = QPushButton("导出 PDF 报告")
        self.export_json_button = QPushButton("导出 JSON 结果")
        self.export_ply_button = QPushButton("导出分割点云")
        self.refresh_button = QPushButton("刷新全部任务")
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        self.export_pdf_button.clicked.connect(lambda: self.export_requested.emit("inspection_report.pdf"))
        self.export_json_button.clicked.connect(lambda: self.export_requested.emit("inspection_report.json"))
        self.export_ply_button.clicked.connect(lambda: self.export_requested.emit("segmentation_pred.ply"))
        export_layout.addWidget(self.refresh_button, 0, 0, 1, 2)
        export_layout.addWidget(self.export_pdf_button, 1, 0)
        export_layout.addWidget(self.export_json_button, 1, 1)
        export_layout.addWidget(self.export_ply_button, 2, 0, 1, 2)
        right_layout.addWidget(export_box)

        log_box = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_box)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        log_layout.addWidget(self.status_text)
        right_layout.addWidget(log_box, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([420, 800, 420])

    def selected_input_type(self) -> str:
        return self.input_type_combo.currentData()

    def selected_product_model_id(self) -> str | None:
        return self.product_model_combo.currentData()

    def prompt_for_directory(self) -> Path | None:
        selected = QFileDialog.getExistingDirectory(self, "选择资源目录", "")
        return Path(selected) if selected else None

    def prompt_save_path(self, artifact_name: str) -> Path | None:
        target, _ = QFileDialog.getSaveFileName(self, f"导出 {artifact_name}", artifact_name)
        return Path(target) if target else None

    def confirm_image_batch(self, file_count: int) -> bool:
        answer = QMessageBox.question(
            self,
            "确认图像批次",
            f"当前选中了 {file_count} 张图片。\n请确认这些图片属于同一批次，是否继续提交？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def set_browser_root(self, root_path: Path) -> None:
        self.file_browser.set_root_path(root_path)

    def set_product_models(self, models: list[dict]) -> None:
        self.product_model_combo.clear()
        if not models:
            self.product_model_combo.addItem("无可用型号", None)
            return
        for model in models:
            self.product_model_combo.addItem(model["display_name"], model["product_model_id"])

    def set_product_models_loading(self) -> None:
        self.product_model_combo.clear()
        self.product_model_combo.addItem("正在加载...", None)

    def upsert_job(self, state: JobViewState) -> None:
        item = self._task_items.get(state.task_key)
        if item is None:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, state.task_key)
            self.jobs_list.insertItem(0, item)
            self._task_items[state.task_key] = item
        item.setText(self._format_job_item_text(state))
        if self.current_task_key() == state.task_key:
            self.show_job_details(state)

    def select_task(self, task_key: str) -> None:
        item = self._task_items.get(task_key)
        if item is None:
            return
        self.jobs_list.setCurrentItem(item)

    def current_task_key(self) -> str | None:
        selected = self.jobs_list.selectedItems()
        if not selected:
            return None
        return selected[0].data(Qt.UserRole)

    def show_job_details(self, state: JobViewState | None) -> None:
        if state is None:
            self.job_label.setText("尚未选择任务")
            self.server_status_label.setText("状态: -")
            self.server_stage_label.setText("阶段: -")
            self.server_message_label.setText("说明: -")
            self.server_progress_bar.setValue(0)
            self.transfer_status_label.setText("传输状态: 空闲")
            self.transfer_speed_label.setText("实时网速: -")
            self.transfer_bytes_label.setText("已传输: -")
            self.transfer_progress_bar.setValue(0)
            return

        job_ref = state.job_id if state.job_id else state.task_key[:8]
        self.job_label.setText(f"当前任务: {job_ref}")
        self.server_status_label.setText(f"状态: {state.status}")
        self.server_stage_label.setText(f"阶段: {state.stage}")
        self.server_message_label.setText(f"说明: {state.current_message or '-'}")
        self.server_progress_bar.setValue(state.progress)

        transfer = state.transfer
        self.transfer_status_label.setText(f"传输状态: {transfer.status_text}")
        self.transfer_speed_label.setText(f"实时网速: {self._format_speed(transfer.speed_bps)}")
        self.transfer_bytes_label.setText(
            f"已传输: {self._format_bytes(transfer.bytes_transferred)} / {self._format_bytes(transfer.total_bytes)}"
        )
        self.transfer_progress_bar.setValue(transfer.progress_percent)

    def preview_file(self, path: Path | None) -> None:
        if path is None:
            self.point_cloud_view.clear_view()
            return
        self.point_cloud_view.preview_file(path)

    def render_result(self, state: JobViewState | None, segmentation_path: Path | None = None) -> None:
        if state is None or state.result is None:
            self.parts_table.setRowCount(0)
            self.point_cloud_view.clear_view()
            return

        part_names: dict[int, str] = {}
        length_map: dict[int, float | None] = {}
        lengths_by_part = {item["part_id"]: item for item in state.result["lengths"]}
        self.parts_table.setRowCount(0)
        for row_idx, part in enumerate(state.result["segmentation"]):
            length_entry = lengths_by_part.get(part["part_id"])
            part_names[part["part_id"]] = part["part_name"]
            length_map[part["part_id"]] = None if length_entry is None else length_entry["length"]
            self.parts_table.insertRow(row_idx)
            self.parts_table.setItem(row_idx, 0, QTableWidgetItem(part["part_name"]))
            self.parts_table.setItem(row_idx, 1, QTableWidgetItem(str(part["point_count"])))
            confidence_text = "-" if part["confidence"] is None else f"{part['confidence']:.3f}"
            self.parts_table.setItem(row_idx, 2, QTableWidgetItem(confidence_text))
            length_text = "-"
            if length_entry is not None and length_entry["length"] is not None:
                length_text = f"{length_entry['length']:.4f}"
            self.parts_table.setItem(row_idx, 3, QTableWidgetItem(length_text))

        self.point_cloud_view.set_lookup(part_names, length_map)
        if segmentation_path is not None:
            self.point_cloud_view.load_segmentation_ply(segmentation_path)

    def highlight_part(self, part_id: int) -> None:
        self.point_cloud_view.highlight_part(part_id)

    def append_log(self, message: str) -> None:
        self.status_text.append(message)

    def clear_result(self) -> None:
        self.parts_table.setRowCount(0)
        self.point_cloud_view.clear_view()

    def show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def show_info(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def _emit_task_selected(self) -> None:
        self.task_selected.emit(self.current_task_key())

    def _emit_part_row_selected(self) -> None:
        selected = self.parts_table.selectedItems()
        if selected:
            self.part_row_selected.emit(selected[0].row())

    def _emit_point_cloud_pick(self, part_id: int, text: str) -> None:
        self.point_cloud_part_picked.emit(part_id, text)

    def _format_job_item_text(self, state: JobViewState) -> str:
        job_ref = state.job_id[:8] if state.job_id else "本地任务"
        transfer_line = ""
        if state.transfer.active or state.transfer.bytes_transferred:
            transfer_line = (
                f"\n{state.transfer.status_text} | "
                f"{state.transfer.progress_percent}% | "
                f"{self._format_speed(state.transfer.speed_bps)}"
            )
        return (
            f"{state.display_name}\n"
            f"{job_ref} | {state.status} | {state.stage} | {state.progress}%"
            f"{transfer_line}"
        )

    @staticmethod
    def _format_speed(speed_bps: float) -> str:
        if speed_bps <= 0:
            return "-"
        units = ["B/s", "KB/s", "MB/s", "GB/s"]
        value = speed_bps
        unit_index = 0
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1
        return f"{value:.2f} {units[unit_index]}"

    @staticmethod
    def _format_bytes(value: int | None) -> str:
        if value is None:
            return "未知"
        units = ["B", "KB", "MB", "GB"]
        amount = float(value)
        unit_index = 0
        while amount >= 1024 and unit_index < len(units) - 1:
            amount /= 1024
            unit_index += 1
        return f"{amount:.2f} {units[unit_index]}"
