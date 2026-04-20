from __future__ import annotations

import sys


def main() -> None:  # pragma: no cover - client entrypoint is not unit-tested
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit("PySide6 is required to run the desktop client.") from exc

    from client.main_window import MainWindow
    from client.task_controller import ClientTaskController

    app = QApplication(sys.argv)
    window = MainWindow()
    controller = ClientTaskController(window)
    window._controller = controller  # type: ignore[attr-defined]
    app.aboutToQuit.connect(controller.cleanup)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
