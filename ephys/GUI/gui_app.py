from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QRunnable,
    Slot,
    QObject,
    Signal,
)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
)

import pyqtgraph as pg

from .styles import apply_style
from ephys.classes.experiment_objects import ExpData
from ephys.GUI.mainwindow import MainWindow

pg.setConfigOptions(antialias=True)


class DataLoader(QRunnable):
    def __init__(self, file_path, experimenter_name):
        super().__init__()
        self.file_path = file_path
        self.experimenter_name = experimenter_name
        self.data: ExpData | None = None
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        self.data = ExpData(
            file_path=self.file_path, experimenter=self.experimenter_name
        )

        try:
            results = self.get_data()
        except Exception as e:
            self.signals.error_occurred.emit(str(e))
        else:
            if results is not None:
                self.signals.data_ready.emit(results)
            else:
                self.signals.error_occurred.emit("No data loaded")
        finally:
            self.signals.finished.emit()

    def get_data(self) -> None | ExpData:
        if self.data is None:
            print("Data could not be loaded")
            return None
        return self.data


class WorkerSignals(QObject):
    """
    Custom signal class to emit data from the worker thread.
    """

    finished = Signal()
    data_ready = Signal(ExpData)
    error_occurred = Signal(str)


def run_app():
    """
    Function to run the application.
    This is useful for testing purposes.
    """
    app = QApplication(sys.argv)
    theme = "dark"
    style: str = apply_style(theme=theme)
    app.setStyleSheet(style)

    window = MainWindow()
    window.show()

    app.exec()


run_app()
