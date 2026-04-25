from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from client.api_client import ApiClient
from client.background import start_background_task
from client.main_window import MainWindow
from client.models import JobViewState, TransferProgress, TransferState
from shared.enums import InputType
from shared.settings import get_settings
from shared.validators import validate_image_paths, validate_pointcloud_path

try:  # pragma: no cover - UI widgets are not unit-tested
    from PySide6.QtCore import QThreadPool, QTimer
except ImportError:  # pragma: no cover
    QThreadPool = object  # type: ignore[assignment]
    QTimer = object  # type: ignore[assignment]


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
POINTCLOUD_SUFFIXES = {".ply"}
JOB_ID_PREFIXES = {
    InputType.IMAGE_SET.value: "Images",
    InputType.POINT_CLOUD.value: "Pointcloud",
}


STATUS_TEXT = {
    "uploaded": "已上传",
    "uploading": "上传中",
    "queued": "排队中",
    "running": "处理中",
    "succeeded": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
}


class ClientTaskController:  # pragma: no cover - UI widgets are not unit-tested
    def __init__(self, window: MainWindow) -> None:
        self.window = window
        self.api = ApiClient()
        self.thread_pool = QThreadPool.globalInstance()
        self.cache_dir = Path(tempfile.gettempdir()) / "scale3d-client-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(__file__).resolve().parents[1] / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selected_files: list[Path] = []
        self.job_states: dict[str, JobViewState] = {}
        self.selected_task_key: str | None = None

        self.poll_timer = QTimer(self.window)
        self.poll_timer.setInterval(2000)
        self.poll_timer.timeout.connect(self.refresh_all_job_statuses)

        self.window.browse_directory_requested.connect(self.handle_browse_directory)
        self.window.file_selection_changed.connect(self.handle_file_selection)
        self.window.submit_requested.connect(self.submit_job)
        self.window.refresh_requested.connect(self.refresh_all_job_statuses)
        self.window.task_selected.connect(self.handle_task_selection)
        self.window.export_requested.connect(self.export_artifact)
        self.window.part_row_selected.connect(self.handle_part_row_selected)

        self.window.append_log("客户端已连接到服务器", status=self.api.base_url)
        self.load_product_models()

    def cleanup(self) -> None:
        self.poll_timer.stop()
        self.selected_files = []
        self.job_states.clear()
        if get_settings().client_cleanup_cache_on_exit:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def load_product_models(self) -> None:
        self.window.set_product_models_loading()
        start_background_task(
            self.thread_pool,
            lambda _emit: self.api.list_product_models(),
            on_finished=self.window.set_product_models,
            on_error=lambda message: self.window.show_error("连接失败", message),
        )

    def handle_browse_directory(self) -> None:
        selected_dir = self.window.prompt_for_directory()
        if selected_dir is not None:
            self.window.set_browser_root(selected_dir)

    def handle_file_selection(self, selected_paths: object) -> None:
        if not isinstance(selected_paths, list):
            return
        self.selected_files = [path for path in selected_paths if isinstance(path, Path)]
        preview_target = self.selected_files[0] if self.selected_files else None
        try:
            self.window.preview_file(preview_target)
        except Exception as exc:
            self.window.clear_result()
            self.window.show_error("预览失败", f"无法预览文件 {preview_target.name if preview_target else ''}: {exc}")

    def submit_job(self) -> None:
        if not self.selected_files:
            self.window.show_warning("缺少输入", "请先在文件选择窗口中选择图像或点云文件。")
            return
        product_model_id = self.window.selected_product_model_id()
        if not product_model_id:
            self.window.show_warning("缺少型号", "请先选择产品型号。")
            return

        input_type = self.window.selected_input_type()
        if input_type == InputType.IMAGE_SET.value:
            image_files = self._validate_image_selection()
            if image_files is None:
                return
            if len(image_files) > 1 and not self.window.confirm_image_batch(len(image_files)):
                self.window.append_log("用户取消图像批次提交", status="已取消")
                return
            self._submit_single_job(product_model_id, input_type, image_files)
            return

        pointcloud_files = self._validate_pointcloud_selection()
        if pointcloud_files is None:
            return
        for pointcloud_path in pointcloud_files:
            self._submit_single_job(product_model_id, input_type, [pointcloud_path])

    def _validate_image_selection(self) -> list[Path] | None:
        invalid = [path for path in self.selected_files if path.suffix.lower() not in IMAGE_SUFFIXES]
        if invalid:
            names = "，".join(path.name for path in invalid[:3])
            self.window.show_error("文件类型错误", f"图像组任务只能选择图片文件，当前包含：{names}")
            return None
        try:
            validate_image_paths(self.selected_files)
        except Exception as exc:
            self.window.show_error("本地校验失败", str(exc))
            return None
        return list(self.selected_files)

    def _validate_pointcloud_selection(self) -> list[Path] | None:
        invalid = [path for path in self.selected_files if path.suffix.lower() not in POINTCLOUD_SUFFIXES]
        if invalid:
            names = "，".join(path.name for path in invalid[:3])
            self.window.show_error("文件类型错误", f"点云任务只能选择 PLY 文件，当前包含：{names}")
            return None
        valid_paths: list[Path] = []
        for pointcloud_path in self.selected_files:
            try:
                validate_pointcloud_path(pointcloud_path)
            except Exception as exc:
                self.window.show_error("点云校验失败", f"{pointcloud_path.name}: {exc}")
                return None
            valid_paths.append(pointcloud_path)
        return valid_paths

    def _submit_single_job(self, product_model_id: str, input_type: str, file_paths: list[Path]) -> None:
        file_names = [path.name for path in file_paths]
        display_name = file_names[0] if len(file_names) == 1 else f"{file_names[0]} 等 {len(file_names)} 个文件"
        task_key = self._build_local_job_id(input_type)
        state = JobViewState(
            task_key=task_key,
            input_type=input_type,
            product_model_id=product_model_id,
            display_name=display_name,
            status="uploading",
            stage="upload",
            progress=0,
            current_message="准备上传文件。",
            submitted_files=file_names,
        )
        self.job_states[task_key] = state
        self.window.upsert_job(state)
        self._update_queue_count()
        self.window.select_task(task_key)
        self.window.append_log("开始提交任务", task_id=task_key, status="上传中")

        def _run(progress_emit):
            return self.api.create_job(
                product_model_id,
                input_type,
                file_paths,
                {
                    "client": "desktop-client",
                    "source_paths": [str(path.resolve()) for path in file_paths],
                },
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
            state.current_message = progress.status_text
        self.window.upsert_job(state)
        self._update_queue_count()

    def _on_submit_failed(self, task_key: str, message: str) -> None:
        state = self.job_states.get(task_key)
        if state is None:
            return
        state.status = "failed"
        state.error = message
        state.current_message = message
        state.transfer.active = False
        state.transfer.status_text = "上传失败"
        self.window.upsert_job(state)
        self._update_queue_count()
        self.window.append_log(message, task_id=state.job_id or task_key, status="提交失败")
        if self.selected_task_key == task_key:
            self.window.show_error("提交失败", message)

    def _on_job_created(self, task_key: str, response: dict) -> None:
        state = self.job_states[task_key]
        local_task_id = state.task_key
        state.job_id = response["job_id"]
        state.status = response["status"]
        state.stage = response["current_stage"]
        state.progress = response["current_progress"]
        state.current_message = response.get("current_message", state.current_message)
        state.transfer.active = False
        state.transfer.status_text = "上传完成，等待服务器处理"
        state.transfer.progress_percent = 100
        self.window.upsert_job(state)
        if state.job_id != local_task_id:
            self.window.replace_log_task_id(local_task_id, state.job_id)
        self._update_queue_count()
        self.window.append_log("任务已创建", task_id=state.job_id, status=self._status_text(state.status))
        if not self.poll_timer.isActive():
            self.poll_timer.start()
        self._refresh_single_job(task_key)

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
        self.window.append_log(message, task_id=state.job_id or task_key, status="刷新任务失败")
        if self.selected_task_key == task_key:
            self.window.show_error("状态刷新失败", message)

    def _on_job_status_loaded(self, task_key: str, payload: dict) -> None:
        state = self.job_states[task_key]
        state.status_in_flight = False
        previous_signature = (state.status, state.stage, state.progress, state.current_message)
        state.status = payload["status"]
        state.stage = payload["current_stage"]
        state.progress = payload["current_progress"]
        state.current_message = payload.get("current_message", state.current_message)
        state.error = payload.get("error")
        if state.status not in {"uploading"} and not state.transfer.active:
            state.transfer.status_text = "等待中" if state.status == "queued" else "服务器处理中"
        self.window.upsert_job(state)
        self._update_queue_count()

        current_signature = (state.status, state.stage, state.progress, state.current_message)
        should_log_progress = (
            self.selected_task_key == task_key
            and current_signature != previous_signature
            and current_signature != state.last_log_signature
        )
        if should_log_progress:
            state.last_log_signature = current_signature
            self.window.append_log(
                f"{self.window._stage_text(state.stage)} | {state.progress}% | {state.current_message}",
                task_id=state.job_id,
                status=self._status_text(state.status),
            )

        if state.status == "succeeded" and state.result is None and not state.result_loading:
            self._load_result(task_key)
        elif state.status == "failed" and self.selected_task_key == task_key:
            self.window.show_error("任务失败", state.error or "未知错误")

    def _load_result(self, task_key: str) -> None:
        state = self.job_states[task_key]
        if state.job_id is None:
            return
        state.result_loading = True
        self.window.append_log("开始加载任务结果", task_id=state.job_id, status=self._status_text(state.status))

        def _run(progress_emit):
            job_status = self.api.get_job(state.job_id)
            result = self.api.get_result(state.job_id)
            artifacts = self.api.list_artifacts(state.job_id)
            task_output_dir = self.output_dir / state.job_id
            artifacts_output_dir = task_output_dir / "artifacts"
            artifacts_output_dir.mkdir(parents=True, exist_ok=True)
            (task_output_dir / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (task_output_dir / "job.json").write_text(
                json.dumps(job_status, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (task_output_dir / "artifacts.json").write_text(
                json.dumps(artifacts, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            segmentation_path = artifacts_output_dir / "segmentation_pred.ply"
            self.api.download_artifact(
                state.job_id,
                "segmentation_pred.ply",
                segmentation_path,
                progress_callback=progress_emit,
            )
            skeleton_paths: list[str] = []
            for artifact in artifacts:
                name = artifact.get("name", "")
                if not (name.startswith("skeleton_") and name.endswith(".ply")):
                    continue
                target_path = artifacts_output_dir / name
                self.api.download_artifact(state.job_id, name, target_path)
                skeleton_paths.append(str(target_path))
            for artifact in artifacts:
                name = artifact.get("name", "")
                if not name or name == "segmentation_pred.ply" or (name.startswith("skeleton_") and name.endswith(".ply")):
                    continue
                self.api.download_artifact(state.job_id, name, artifacts_output_dir / name)
            server_cleanup_error = None
            try:
                self.api.delete_job(state.job_id)
            except Exception as exc:
                server_cleanup_error = str(exc)
            return {
                "result": result,
                "segmentation_path": str(segmentation_path),
                "skeleton_paths": skeleton_paths,
                "output_dir": str(task_output_dir),
                "server_cleanup_error": server_cleanup_error,
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
        state.current_message = message
        state.transfer.active = False
        state.transfer.status_text = "结果下载失败"
        self.window.upsert_job(state)
        self.window.append_log(message, task_id=state.job_id or task_key, status="结果加载失败")
        if self.selected_task_key == task_key:
            self.window.show_error("结果加载失败", message)

    def _on_result_loaded(self, task_key: str, bundle: dict) -> None:
        state = self.job_states[task_key]
        state.result_loading = False
        state.result = bundle["result"]
        state.current_message = "结果已加载完成。"
        state.transfer.active = False
        state.transfer.status_text = "结果下载完成"
        state.transfer.progress_percent = 100
        self.window.upsert_job(state)
        self._update_queue_count()
        self.window.append_log("结果加载完成", task_id=state.job_id or task_key, status=self._status_text(state.status))
        if bundle.get("server_cleanup_error"):
            self.window.append_log(
                f"服务器运行目录清理失败: {bundle['server_cleanup_error']}",
                task_id=state.job_id or task_key,
                status="清理失败",
            )
        else:
            self.window.append_log("服务器运行目录已清理", task_id=state.job_id or task_key, status="已清理")
        if self.selected_task_key == task_key:
            self.window.render_result(
                state,
                Path(bundle["segmentation_path"]),
                [Path(item) for item in bundle.get("skeleton_paths", [])],
            )
            self.window.append_log(f"结果已保存到 {bundle['output_dir']}", task_id=state.job_id, status="完成")

    def handle_task_selection(self, task_key: str | None) -> None:
        self.selected_task_key = task_key
        state = self.job_states.get(task_key) if task_key else None
        self.window.show_job_details(state)
        self.window.update_status_transfer(state)
        if state is None:
            self.window.clear_result()
            preview_target = self.selected_files[0] if self.selected_files else None
            self.window.preview_file(preview_target)
            return
        if state.result is not None:
            output_job_id = state.job_id or task_key
            output_artifacts_dir = self.output_dir / output_job_id / "artifacts"
            segmentation_path = output_artifacts_dir / "segmentation_pred.ply"
            skeleton_paths = sorted(output_artifacts_dir.glob("skeleton_*.ply"))
            self.window.render_result(
                state,
                segmentation_path if segmentation_path.exists() else None,
                skeleton_paths,
            )
        elif state.status == "succeeded" and not state.result_loading:
            if not self._load_local_result_if_available(state):
                self._load_result(task_key)
        else:
            self.window.clear_result()

    def _load_local_result_if_available(self, state: JobViewState) -> bool:
        if state.job_id is None:
            return False
        task_output_dir = self.output_dir / state.job_id
        result_path = task_output_dir / "result.json"
        segmentation_path = task_output_dir / "artifacts" / "segmentation_pred.ply"
        if not result_path.exists() or not segmentation_path.exists():
            return False
        try:
            state.result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.window.append_log(f"本地结果读取失败: {exc}", task_id=state.job_id, status="读取失败")
            return False
        skeleton_paths = sorted((task_output_dir / "artifacts").glob("skeleton_*.ply"))
        self.window.upsert_job(state)
        self.window.render_result(state, segmentation_path, skeleton_paths)
        return True

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
        local_artifact = self.output_dir / state.job_id / "artifacts" / artifact_name
        if local_artifact.exists():
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_artifact, target_path)
                self.window.show_info("导出成功", f"已保存到 {target_path}")
            except Exception as exc:
                self.window.show_error("导出失败", str(exc))
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
        self.window.append_log(part["part_name"], task_id=state.job_id, status="选中部件")

    def _update_queue_count(self) -> None:
        count = sum(1 for state in self.job_states.values() if not state.is_terminal)
        self.window.set_queue_count(count)

    def _build_local_job_id(self, input_type: str) -> str:
        prefix = JOB_ID_PREFIXES.get(input_type, "Job")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_id = f"{prefix}_{timestamp}"
        task_id = base_id
        suffix = 1
        while task_id in self.job_states:
            suffix += 1
            task_id = f"{base_id}_{suffix:02d}"
        return task_id

    @staticmethod
    def _status_text(status: str) -> str:
        return STATUS_TEXT.get(status, status)
