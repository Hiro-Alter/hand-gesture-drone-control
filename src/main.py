from __future__ import annotations

import sys

from src.utils.config_loader import load_app_config


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        print("PySide6 no esta instalado. Ejecuta: pip install -r requirements.txt")
        return 1

    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    config = load_app_config()
    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

