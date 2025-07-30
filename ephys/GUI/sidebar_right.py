import __future__
from typing import TYPE_CHECKING
from PySide6.QtGui import QCursor, QFont, QIcon, QPalette, QRegion
from PySide6.QtWidgets import (
    QSizePolicy,
    QVBoxLayout,
    QFrame,
)
from PySide6.QtCore import QLocale, QPoint, QRect, QSize, Qt, QDate

if TYPE_CHECKING:
    from ephys.GUI.gui_app import MainWindow

# from ephys.GUI.mainwindow import MainWindow
from ephys.GUI.windowwidget import WindowSelection


class SideBarRight(QFrame):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self.main_window = main_window
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setLineWidth(2)
        self.setMidLineWidth(2)
        self.setContentsMargins(5, 5, 5, 5)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(5)

        self.setMinimumSize(QSize(300, 300))
        self.setMaximumSize(QSize(300, 1200))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self.window_frame = WindowSelection(self.main_window)
        self.main_layout.addWidget(self.window_frame)
        self.main_layout.addStretch()
