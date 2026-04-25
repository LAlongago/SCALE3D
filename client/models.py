from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransferProgress:
    phase: str
    status_text: str
    bytes_transferred: int
    total_bytes: int | None
    speed_bps: float
    progress_percent: int
    artifact_name: str | None = None


@dataclass
class TransferState:
    phase: str = "idle"
    status_text: str = "空闲"
    bytes_transferred: int = 0
    total_bytes: int | None = None
    speed_bps: float = 0.0
    progress_percent: int = 0
    artifact_name: str | None = None
    active: bool = False

    @classmethod
    def from_progress(cls, progress: TransferProgress, *, active: bool = True) -> "TransferState":
        return cls(
            phase=progress.phase,
            status_text=progress.status_text,
            bytes_transferred=progress.bytes_transferred,
            total_bytes=progress.total_bytes,
            speed_bps=progress.speed_bps,
            progress_percent=progress.progress_percent,
            artifact_name=progress.artifact_name,
            active=active,
        )


@dataclass
class JobViewState:
    task_key: str
    input_type: str
    product_model_id: str
    display_name: str
    job_id: str | None = None
    status: str = "uploaded"
    stage: str = "upload"
    progress: int = 0
    current_message: str = ""
    error: str | None = None
    result: dict | None = None
    result_loading: bool = False
    status_in_flight: bool = False
    last_log_signature: tuple[str, str, int, str] | None = None
    submitted_files: list[str] = field(default_factory=list)
    transfer: TransferState = field(default_factory=TransferState)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "cancelled"}
