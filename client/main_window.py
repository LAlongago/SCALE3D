from __future__ import annotations

import tempfile
from pathlib import Path

from shared.enums import InputType
from shared.validators import validate_image_paths, validate_pointcloud_path

from client.api_client import ApiClient
from client.point_cloud_view import PointCloudView

try:  # pragma: no cover - UI widgets are not unit-tested
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
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
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self.selected_files: list[Path] = []
        self.current_job_id = None
        self.current_result = None
        self.cache_dir = Path(tempfile.gettempdir()) / "pis-client-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.setWindowTitle("UT 产品点云检测系统")
        self.resize(1600, 900)
        self._build_ui()
        self._load_product_models()

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(2000)
        self.poll_timer.timeout.connect(self.refresh_job_status)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal, central)
        root_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        controls = QGroupBox("任务控制")
        controls_layout = QGridLayout(controls)
        self.product_model_combo = QComboBox()
        self.input_type_combo = QComboBox()
        self.input_type_combo.addItem("图像组", InputType.IMAGE_SET.value)
        self.input_type_combo.addItem("点云文件", InputType.POINT_CLOUD.value)
        self.select_button = QPushButton("选择文件")
        self.submit_button = QPushButton("提交任务")
        self.select_button.clicked.connect(self.select_files)
        self.submit_button.clicked.connect(self.submit_job)
        controls_layout.addWidget(QLabel("产品型号"), 0, 0)
        controls_layout.addWidget(self.product_model_combo, 0, 1)
        controls_layout.addWidget(QLabel("输入类型"), 1, 0)
        controls_layout.addWidget(self.input_type_combo, 1, 1)
        controls_layout.addWidget(self.select_button, 2, 0)
        controls_layout.addWidget(self.submit_button, 2, 1)
        left_layout.addWidget(controls)

        self.selected_files_list = QListWidget()
        left_layout.addWidget(QLabel("已选文件"))
        left_layout.addWidget(self.selected_files_list, 1)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.addWidget(QLabel("点云展示区域"))
        self.point_cloud_view = PointCloudView(on_part_selected=self._show_selected_part_info)
        center_layout.addWidget(self.point_cloud_view, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        status_box = QGroupBox("任务状态")
        status_layout = QVBoxLayout(status_box)
        self.job_label = QLabel("尚未提交任务")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.job_label)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_text)
        right_layout.addWidget(status_box)

        self.parts_table = QTableWidget(0, 4)
        self.parts_table.setHorizontalHeaderLabels(["部件", "点数", "置信度", "长度"])
        self.parts_table.itemSelectionChanged.connect(self._handle_table_selection)
        right_layout.addWidget(QLabel("结果输出栏"))
        right_layout.addWidget(self.parts_table, 1)

        export_box = QGroupBox("导出")
        export_layout = QVBoxLayout(export_box)
        self.export_pdf_button = QPushButton("导出 PDF 报告")
        self.export_json_button = QPushButton("导出 JSON 结果")
        self.export_ply_button = QPushButton("导出分割点云")
        self.refresh_button = QPushButton("手动刷新")
        self.export_pdf_button.clicked.connect(lambda: self.export_artifact("inspection_report.pdf"))
        self.export_json_button.clicked.connect(lambda: self.export_artifact("inspection_report.json"))
        self.export_ply_button.clicked.connect(lambda: self.export_artifact("segmentation_pred.ply"))
        self.refresh_button.clicked.connect(self.refresh_job_status)
        export_layout.addWidget(self.refresh_button)
        export_layout.addWidget(self.export_pdf_button)
        export_layout.addWidget(self.export_json_button)
        export_layout.addWidget(self.export_ply_button)
        right_layout.addWidget(export_box)

        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900, 400])

    def _load_product_models(self) -> None:
        try:
            models = self.api.list_product_models()
        except Exception as exc:
            QMessageBox.critical(self, "连接失败", str(exc))
            return
        self.product_model_combo.clear()
        for model in models:
            self.product_model_combo.addItem(model["display_name"], model["product_model_id"])

    def select_files(self) -> None:
        input_type = self.input_type_combo.currentData()
        if input_type == InputType.IMAGE_SET.value:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "选择图像组",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
            )
        else:
            file, _ = QFileDialog.getOpenFileName(self, "选择点云文件", "", "PLY (*.ply)")
            files = [file] if file else []
        self.selected_files = [Path(item) for item in files if item]
        self.selected_files_list.clear()
        for path in self.selected_files:
            self.selected_files_list.addItem(str(path))

    def submit_job(self) -> None:
        if not self.selected_files:
            QMessageBox.warning(self, "缺少输入", "请先选择图像或点云文件。")
            return
        try:
            if self.input_type_combo.currentData() == InputType.IMAGE_SET.value:
                validate_image_paths(self.selected_files)
            else:
                validate_pointcloud_path(self.selected_files[0])
            response = self.api.create_job(
                product_model_id=self.product_model_combo.currentData(),
                input_type=self.input_type_combo.currentData(),
                file_paths=self.selected_files,
                client_meta={"client": "PySide6 desktop"},
            )
        except Exception as exc:
            QMessageBox.critical(self, "提交失败", str(exc))
            return

        self.current_job_id = response["job_id"]
        self.job_label.setText(f"当前任务: {self.current_job_id}")
        self.status_text.append("任务已提交，开始轮询服务器进度。")
        self.progress_bar.setValue(0)
        self.poll_timer.start()

    def refresh_job_status(self) -> None:
        if not self.current_job_id:
            return
        try:
            payload = self.api.get_job(self.current_job_id)
        except Exception as exc:
            self.status_text.append(f"刷新失败: {exc}")
            return
        self.progress_bar.setValue(payload["current_progress"])
        self.status_text.append(
            f"stage={payload['current_stage']} status={payload['status']} progress={payload['current_progress']}"
        )
        if payload["status"] == "succeeded":
            self.poll_timer.stop()
            self.load_result()
        elif payload["status"] == "failed":
            self.poll_timer.stop()
            QMessageBox.critical(self, "任务失败", payload.get("error") or "Unknown error")

    def load_result(self) -> None:
        if not self.current_job_id:
            return
        try:
            self.current_result = self.api.get_result(self.current_job_id)
            segmentation_target = self.cache_dir / self.current_job_id / "segmentation_pred.ply"
            self.api.download_artifact(self.current_job_id, "segmentation_pred.ply", segmentation_target)
        except Exception as exc:
            QMessageBox.critical(self, "结果加载失败", str(exc))
            return

        self.parts_table.setRowCount(0)
        part_names = {}
        length_map = {}
        lengths_by_part = {item["part_id"]: item for item in self.current_result["lengths"]}
        for row_idx, part in enumerate(self.current_result["segmentation"]):
            length_entry = lengths_by_part.get(part["part_id"])
            length_map[part["part_id"]] = None if length_entry is None else length_entry["length"]
            part_names[part["part_id"]] = part["part_name"]
            self.parts_table.insertRow(row_idx)
            self.parts_table.setItem(row_idx, 0, QTableWidgetItem(part["part_name"]))
            self.parts_table.setItem(row_idx, 1, QTableWidgetItem(str(part["point_count"])))
            confidence_text = "-" if part["confidence"] is None else f"{part['confidence']:.3f}"
            self.parts_table.setItem(row_idx, 2, QTableWidgetItem(confidence_text))
            length_text = "-" if length_entry is None or length_entry["length"] is None else f"{length_entry['length']:.4f}"
            self.parts_table.setItem(row_idx, 3, QTableWidgetItem(length_text))
        self.point_cloud_view.set_lookup(part_names, length_map)
        self.point_cloud_view.load_segmentation_ply(segmentation_target)

    def export_artifact(self, artifact_name: str) -> None:
        if not self.current_job_id:
            QMessageBox.warning(self, "没有任务", "请先完成一个任务。")
            return
        target, _ = QFileDialog.getSaveFileName(self, f"导出 {artifact_name}", artifact_name)
        if not target:
            return
        try:
            self.api.download_artifact(self.current_job_id, artifact_name, Path(target))
            QMessageBox.information(self, "导出成功", f"已保存到 {target}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _handle_table_selection(self) -> None:
        selected = self.parts_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if self.current_result is None:
            return
        part = self.current_result["segmentation"][row]
        self.point_cloud_view.highlight_part(part["part_id"])
        self._show_selected_part_info(part["part_id"], part["part_name"])

    def _show_selected_part_info(self, part_id: int, text: str) -> None:
        self.status_text.append(f"选中部件 {part_id}: {text}")
