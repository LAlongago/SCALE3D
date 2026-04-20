from __future__ import annotations

from shared.settings import get_settings
from workers.tasks import local_dispatcher, run_job_task


def dispatch_job(job_id: str) -> None:
    settings = get_settings()
    if settings.use_celery:
        run_job_task.delay(job_id)
    else:
        local_dispatcher.submit_pipeline(job_id)


def shutdown_dispatcher() -> None:
    settings = get_settings()
    if not settings.use_celery:
        local_dispatcher.shutdown()
