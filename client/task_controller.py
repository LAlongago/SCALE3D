from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from client.api_client import ApiClient
from client.background import start_background_task
from client.main_window import MainWindow
from client.models import JobViewState, TransferProgress, TransferState
from shared.enums import InputType
from shared.validators import validate_image_paths, validate_pointcloud_path

try:  # pragma: no cover - UI widgets are not unit-tested
    from PySide6.QtCore import QThreadPool, QTimer
except ImportError:  # pragma: no cover
    QThreadPool = object  # type: ignore[assignment]
    QTimer = object  # type: ignore[assignment]


class ClientTaskController:  # pragma: no cover - UI widgets are not unit-tested
    def __init__(self, window: MainWindow) -> None:
        self.window = window
        self.api = ApiClient()
        self.thread_pool = QThreadPool.globalInstance()
        self.cache_dir = Path(tempfile.gettempdir()) / "pis-client-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.selected_files: list[Path] = []
        self.job_states: dict[str, JobViewState] = {}
        self.selected_task_key: str | None = None

        self.poll_timer = QTimer(self.window)
        self.poll_timer.setInterval(2000)
        self.poll_timer.timeout.connect(self.refresh_all_job_statuses)

        self.window.select_files_requested.connect(self.handle_select_files)
        self.window.submit_requested.connect(self.submit_job)
        self.window.refresh_requested.connect(self.refresh_all_job_statuses)
        self.window.task_selected.connect(self.handle_task_selection)
        self.window.export_requested.connect(self.export_artifact)
        self.window.part_row_selected.connect(self.handle_part_row_selected)
        self.window.point_cloud_part_picked.connect(self.handle_point_cloud_part_picked)

        self.load_product_models()

    def load_product_models(self) -> None:
        self.window.set_product_models_loading()
        start_background_task(
            self.thread_pool,
            lambda _emit: self.api.list_product_models(),
            on_finished=self.window.set_product_models,
            on_error=lambda message: self.window.show_error("连接失败", message),
        )

    def handle_select_files(self) -> None:
        selected = self.window.prompt_for_files(self.window.selected_input_type())
        self.selected_files = selected
        self.window.show_selected_files(selected)

    def submit_job(self) -> None:
        if not self.selected_files:
            self.window.show_warning("缺少输入", "请先选择图像组或点云文件。")
            return

        product_model_id = self.window.selected_product_model_id()
        if not product_model_id:
            self.window.show_warning("缺少型号", "请先选择产品型号。")
            return

        input_type = self.window.selected_input_type()
        try:
            if input_type == InputType.IMAGE_SET.value:
                validate_image_paths(self.selected_files)
            else:
                validate_pointcloud_path(self.selected_files[0])
        except Exception as exc:
            self.window.show_error("本地校验失败", str(exc))
            return

        file_paths = list(self.selected_files)
        file_names = [path.name for path in file_paths]
        display_name = file_names[0] if len(file_names) == 1 else f"{file_names[0]} 等 {len(file_names)} 个文件"
        task_key = uuid4().hex
        state = JobViewState(
            task_key=task_key,
            input_type=input_type,
            product_model_id=product_model_id,
            display_name=display_name,
            status="uploading",
            stage="upload",
            progress=0,
            submitted_files=file_names,
        )
        self.job_states[task_key] = state
        self.window.upsert_job(state)
        self.window.select_task(task_key)
        self.window.append_log(f"开始提交任务: {display_name}")

        def _run(progress_emit):
            return self.api.create_job(
                product_model_id,
                input_type,
                file_paths,
                {"client": "desktop-client"},
                progress_callback=progress_emit,
            )

        start_background_task(
            self.thread_pool,
            _run,
            on_finished=lambda response, task_key=task_key: self._on_job_created(task_key, response),
            on_error=lambda message, task_key=task_key: self._on_submit_failed(task_key, message),
            on_progress=lambda progress, task_key=task_key: self._on_transfer_progress(task_key, progress),
        )

    def _on_transfer_progress(self, task_key: str, progress: object) -> None:
        if not isinstance(progress, TransferProgress):
            return
        state = self.job_states.get(task_key)
        if state is None:
            return

        state.transfer = TransferState.from_progress(progress, active=progress.progress_percent < 100)
        if progress.phase == "upload" and state.job_id is None:
            state.status = "uploading"
            state.stage = "upload"
            state.progress = progress.progress_percent
        self.window.upsert_job(state)

    def _on_submit_failed(self, task_key: str, message: str) -> None:
        state = self.job_states.get(task_key)
        if state is None:
            return
        state.status = "failed"
        state.error = message
        state.transfer.active = False
        state.transfer.status_text = "上传失败"
        self.window.upsert_job(state)
        if self.selected_task_key == task_key:
            self.window.show_error("提交失败", message)

    def _on_job_created(self, task_key: str, response: dict) -> None:
        state = self.job_states[task_key]
        state.job_id = response["job_id"]
        state.status = response["status"]
        state.stage = response["current_stage"]
        state.progress = response["current_progress"]
        state.transfer.active = False
        state.transfer.status_text = "上传完成，等待服务器处理"
        state.transfer.progress_percent = 100
        self.window.upsert_job(state)
        self.window.append_log(f"任务已创建: {state.job_id}")
        if not self.poll_timer.isActive():
            self.poll_timer.start()

    def refresh_all_job_statuses(self) -> None:
        active_states = [
            state
            for state in self.job_states.values()
            if state.job_id is not None and not state.is_terminal and not state.status_in_flight
        ]
        if not active_states and self.poll_timer.isActive():
            self.poll_timer.stop()
            return

        for state in active_states:
            self._refresh_single_job(state.task_key)

    def _refresh_single_job(self, task_key: str) -> None:
        state = self.job_states[task_key]
        if state.job_id is None:
            return
        state.status_in_flight = True
        start_background_task(
            self.thread_pool,
            lambda _emit, job_id=state.job_id: self.api.get_job(job_id),
            on_finished=lambda payload, task_key=task_key: self._on_job_status_loaded(task_key, payload),
            on_error=lambda message, task_key=task_key: self._on_status_failed(task_key, message),
        )

    def _on_status_failed(self, task_key: str, message: str) -> None:
        state = self.job_states.get(task_key)
        if state is None:
            return
        state.status_in_flight = False
        if self.selected_task_key == task_key:
            self.window.append_log(f"刷新任务失败: {message}")

    def _on_job_status_loaded(self, task_key: str, payload: dict) -> None:
        state = self.job_states[task_key]
        state.status_in_flight = False
        state.status = payload["status"]
        state.stage = payload["current_stage"]
        state.progress = payload["current_progress"]
        state.error = payload.get("error")
        if state.status not in {"uploading"} and not state.transfer.active:
            state.transfer.status_text = "等待中" if state.status == "queued" else "服务器处理中"
        self.window.upsert_job(state)

        if self.selected_task_key == task_key:
            self.window.append_log(
                f"任务 {state.job_id}: 状态={state.status} 阶段={state.stage} 进度={state.progress}%"
            )

        if state.status == "succeeded" and state.result is None and not state.result_loading:
            self._load_result(task_key)
        elif state.status == "failed" and self.selected_task_key == task_key:
            self.window.show_error("任务失败", state.error or "Unknown error")

    def _load_result(self, task_key: str) -> None:
        state = self.job_states[task_key]
        if state.job_id is None:
            return
        state.result_loading = True

        def _run(progress_emit):
            result = self.api.get_result(state.job_id)
            segmentation_path = self.cache_dir / task_key / "segmentation_pred.ply"
            self.api.download_artifact(
                state.job_id,
                "segmentation_pred.ply",
                segmentation_path,
                progress_callback=progress_emit,
            )
            return {
                "result": result,
                "segmentation_path": str(segmentation_path),
            }

        start_background_task(
            self.thread_pool,
            _run,
            on_finished=lambda bundle, task_key=task_key: self._on_result_loaded(task_key, bundle),
            on_error=lambda message, task_key=task_key: self._on_result_failed(task_key, message),
            on_progress=lambda progress, task_key=task_key: self._on_transfer_progress(task_key, progress),
        )

    def _on_result_failed(self, task_key: str, message: str) -> None:
        state = self.job_states.get(task_key)
        if state is None:
            return
        state.result_loading = False
        state.error = message
        state.transfer.active = False
        state.transfer.status_text = "结果下载失败"
        self.window.upsert_job(state)
        if self.selected_task_key == task_key:
            self.window.show_error("结果加载失败", message)

    def _on_result_loaded(self, task_key: str, bundle: dict) -> None:
        state = self.job_states[task_key]
        state.result_loading = False
        state.result = bundle["result"]
        state.transfer.active = False
        state.transfer.status_text = "结果下载完成"
        state.transfer.progress_percent = 100
        self.window.upsert_job(state)
        if self.selected_task_key == task_key:
            self.window.render_result(state, Path(bundle["segmentation_path"]))
            self.window.append_log(f"任务 {state.job_id} 的结果已加载。")

    def handle_task_selection(self, task_key: str | None) -> None:
        self.selected_task_key = task_key
        state = self.job_states.get(task_key) if task_key else None
        self.window.show_job_details(state)
        if state is None:
            self.window.clear_result()
            return
        if state.result is not None:
            segmentation_path = self.cache_dir / task_key / "segmentation_pred.ply"
            self.window.render_result(state, segmentation_path if segmentation_path.exists() else None)
        elif state.status == "succeeded" and not state.result_loading:
            self._load_result(task_key)
        else:
            self.window.clear_result()

    def export_artifact(self, artifact_name: str) -> None:
        if not self.selected_task_key:
            self.window.show_warning("没有任务", "请先从任务队列中选择一个任务。")
            return
        state = self.job_states[self.selected_task_key]
        if not state.job_id:
            self.window.show_warning("任务未就绪", "当前任务还没有服务端任务编号，暂时无法导出。")
            return

        target_path = self.window.prompt_save_path(artifact_name)
        if target_path is None:
            return

        start_background_task(
            self.thread_pool,
            lambda progress_emit, job_id=state.job_id: self.api.download_artifact(
                job_id,
                artifact_name,
                target_path,
                progress_callback=progress_emit,
            ),
            on_finished=lambda _path, target=str(target_path): self.window.show_info("导出成功", f"已保存到 {target}"),
            on_error=lambda message: self.window.show_error("导出失败", message),
            on_progress=lambda progress, task_key=self.selected_task_key: self._on_transfer_progress(task_key, progress),
        )

    def handle_part_row_selected(self, row: int) -> None:
        if not self.selected_task_key:
            return
        state = self.job_states.get(self.selected_task_key)
        if state is None or state.result is None:
            return
        part = state.result["segmentation"][row]
        self.window.highlight_part(part["part_id"])
        self.window.append_log(f"选中部件: {part['part_name']}")

    def handle_point_cloud_part_picked(self, part_id: int, text: str) -> None:
        if self.selected_task_key:
            self.window.append_log(f"点选部件 {part_id}: {text}")
