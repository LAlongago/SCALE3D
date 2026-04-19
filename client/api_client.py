from __future__ import annotations

import json
from time import perf_counter
from pathlib import Path
from typing import Callable

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor

from client.models import TransferProgress
from shared.settings import get_settings


class ApiClient:
    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.server_url).rstrip("/")
        self.session = requests.Session()
        if token or settings.api_token:
            self.session.headers["Authorization"] = f"Bearer {token or settings.api_token}"

    def list_product_models(self) -> list[dict]:
        response = self.session.get(f"{self.base_url}/api/v1/product-models", timeout=30)
        response.raise_for_status()
        return response.json()

    def create_job(
        self,
        product_model_id: str,
        input_type: str,
        file_paths: list[Path],
        client_meta: dict | None = None,
        progress_callback: Callable[[TransferProgress], None] | None = None,
    ) -> dict:
        handles = []
        try:
            fields: list[tuple[str, object]] = [
                ("product_model_id", product_model_id),
                ("input_type", input_type),
                ("client_meta", json.dumps(client_meta or {}, ensure_ascii=False)),
            ]
            for path in file_paths:
                handle = path.open("rb")
                handles.append(handle)
                fields.append(("files", (path.name, handle, "application/octet-stream")))

            encoder = MultipartEncoder(fields=fields)
            body = encoder
            headers = {"Content-Type": encoder.content_type}
            start_time = perf_counter()

            if progress_callback is not None:
                def _emit_snapshot(bytes_read: int, status_text: str) -> None:
                    elapsed = max(perf_counter() - start_time, 1e-6)
                    total_bytes = encoder.len
                    progress_percent = int(bytes_read * 100 / total_bytes) if total_bytes else 0
                    progress_callback(
                        TransferProgress(
                            phase="upload",
                            status_text=status_text,
                            bytes_transferred=bytes_read,
                            total_bytes=total_bytes,
                            speed_bps=bytes_read / elapsed,
                            progress_percent=min(progress_percent, 100),
                        )
                    )

                def _on_upload_progress(monitor: MultipartEncoderMonitor) -> None:
                    _emit_snapshot(monitor.bytes_read, "正在上传")

                body = MultipartEncoderMonitor(encoder, _on_upload_progress)
                headers = {"Content-Type": body.content_type}
                _emit_snapshot(0, "准备上传")

            response = self.session.post(
                f"{self.base_url}/api/v1/jobs",
                data=body,
                headers=headers,
                timeout=300,
            )
            response.raise_for_status()
            if progress_callback is not None:
                elapsed = max(perf_counter() - start_time, 1e-6)
                progress_callback(
                    TransferProgress(
                        phase="upload",
                        status_text="上传完成",
                        bytes_transferred=encoder.len,
                        total_bytes=encoder.len,
                        speed_bps=encoder.len / elapsed,
                        progress_percent=100,
                    )
                )
            return response.json()
        finally:
            for handle in handles:
                handle.close()

    def get_job(self, job_id: str) -> dict:
        response = self.session.get(f"{self.base_url}/api/v1/jobs/{job_id}", timeout=30)
        response.raise_for_status()
        return response.json()

    def get_result(self, job_id: str) -> dict:
        response = self.session.get(f"{self.base_url}/api/v1/jobs/{job_id}/result", timeout=30)
        response.raise_for_status()
        return response.json()

    def list_artifacts(self, job_id: str) -> list[dict]:
        response = self.session.get(f"{self.base_url}/api/v1/jobs/{job_id}/artifacts", timeout=30)
        response.raise_for_status()
        return response.json()

    def download_artifact(
        self,
        job_id: str,
        artifact_name: str,
        target_path: Path,
        progress_callback: Callable[[TransferProgress], None] | None = None,
    ) -> Path:
        response = self.session.get(
            f"{self.base_url}/api/v1/jobs/{job_id}/artifacts/{artifact_name}",
            timeout=300,
            stream=True,
        )
        response.raise_for_status()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        total_bytes = response.headers.get("Content-Length")
        total = int(total_bytes) if total_bytes and total_bytes.isdigit() else None
        written = 0
        start_time = perf_counter()

        if progress_callback is not None:
            progress_callback(
                TransferProgress(
                    phase="download",
                    status_text=f"正在下载 {artifact_name}",
                    bytes_transferred=0,
                    total_bytes=total,
                    speed_bps=0.0,
                    progress_percent=0,
                    artifact_name=artifact_name,
                )
            )

        with target_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if chunk:
                    handle.write(chunk)
                    written += len(chunk)
                    if progress_callback is not None:
                        elapsed = max(perf_counter() - start_time, 1e-6)
                        progress_percent = int(written * 100 / total) if total else 0
                        progress_callback(
                            TransferProgress(
                                phase="download",
                                status_text=f"正在下载 {artifact_name}",
                                bytes_transferred=written,
                                total_bytes=total,
                                speed_bps=written / elapsed,
                                progress_percent=min(progress_percent, 100),
                                artifact_name=artifact_name,
                            )
                        )

        if progress_callback is not None:
            elapsed = max(perf_counter() - start_time, 1e-6)
            progress_callback(
                TransferProgress(
                    phase="download",
                    status_text=f"{artifact_name} 下载完成",
                    bytes_transferred=written,
                    total_bytes=total,
                    speed_bps=written / elapsed if written else 0.0,
                    progress_percent=100 if total else 0,
                    artifact_name=artifact_name,
                )
            )
        return target_path
