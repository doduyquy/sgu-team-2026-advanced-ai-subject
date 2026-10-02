"""
Research Workbench Application Entrypoint (Gate 7.5B).
"""

import os
import sys

from PySide6.QtWidgets import QApplication

from src.workbench.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Research Workbench V1")
    app.setOrganizationName("SGU Team 2026")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
