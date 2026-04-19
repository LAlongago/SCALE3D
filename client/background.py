from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - UI widgets are not unit-tested
    from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
except ImportError:  # pragma: no cover
    QObject = object  # type: ignore[assignment]
    QRunnable = object  # type: ignore[assignment]
    QThreadPool = object  # type: ignore[assignment]

    def Signal(*_args, **_kwargs):  # type: ignore[no-redef]
        return None


class WorkerSignals(QObject):  # pragma: no cover - UI widgets are not unit-tested
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(object)


class BackgroundTask(QRunnable):  # pragma: no cover - UI widgets are not unit-tested
    def __init__(self, fn: Callable[[Callable[[object], None]], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn(self.signals.progress.emit)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        else:
            self.signals.finished.emit(result)


def start_background_task(
    thread_pool: QThreadPool,
    fn: Callable[[Callable[[object], None]], Any],
    *,
    on_finished: Callable[[Any], None],
    on_error: Callable[[str], None],
    on_progress: Callable[[object], None] | None = None,
) -> None:
    task = BackgroundTask(fn)
    task.signals.finished.connect(on_finished)
    task.signals.error.connect(on_error)
    if on_progress is not None:
        task.signals.progress.connect(on_progress)
    thread_pool.start(task)
