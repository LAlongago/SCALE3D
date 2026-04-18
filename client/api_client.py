from __future__ import annotations

import json
from pathlib import Path

import requests

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
    ) -> dict:
        data = {
            "product_model_id": product_model_id,
            "input_type": input_type,
            "client_meta": json.dumps(client_meta or {}, ensure_ascii=False),
        }
        files = []
        handles = []
        try:
            for path in file_paths:
                handle = path.open("rb")
                handles.append(handle)
                files.append(("files", (path.name, handle, "application/octet-stream")))
            response = self.session.post(
                f"{self.base_url}/api/v1/jobs",
                data=data,
                files=files,
                timeout=300,
            )
            response.raise_for_status()
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

    def download_artifact(self, job_id: str, artifact_name: str, target_path: Path) -> Path:
        response = self.session.get(
            f"{self.base_url}/api/v1/jobs/{job_id}/artifacts/{artifact_name}",
            timeout=300,
            stream=True,
        )
        response.raise_for_status()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        return target_path
