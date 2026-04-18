from __future__ import annotations

from celery import Celery

from shared.settings import get_settings


def create_celery() -> Celery:
    settings = get_settings()
    app = Celery(
        "product_inspection_system",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )
    app.conf.task_default_queue = "segmentation_gpu"
    app.conf.task_routes = {
        "workers.tasks.run_job_task": {"queue": "segmentation_gpu"},
    }
    return app


celery_app = create_celery()
