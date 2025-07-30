import __future__
from typing import TYPE_CHECKING
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QWidget,
    QFileDialog,
    QFrame,
)
from PySide6.QtCore import Qt, QDate

if TYPE_CHECKING:
    from ephys.GUI.gui_app import MainWindow


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

        self.setMinimumWidth(150)
        self.setMaximumWidth(300)
        self.setMinimumHeight(150)
        self.setMaximumHeight(1200)

        # Add widgets to the sidebar
        self.window_selection = QHBoxLayout()
        self.add_window_button = QPushButton("Add Window")
        self.add_window_button.setMaximumWidth(150)
        # self.add_window_button.clicked.connect(self.main_window.add_window)
        self.window_selection.addWidget(self.add_window_button)
        self.main_layout.addLayout(self.window_selection)
        self.window_start = QDoubleSpinBox()
        self.window_start.setMaximumWidth(100)
        self.window_start.setSingleStep(0.1)
        self.window_start.setValue(0.0)
        self.window_start.setSuffix(" s")
        self.window_start.setPrefix("Start: ")
        self.window_selection.addWidget(self.window_start)
        self.window_end = QDoubleSpinBox()
        self.window_end.setMaximumWidth(100)
        self.window_end.setSingleStep(0.1)
        self.window_end.setValue(1.0)
        self.window_end.setSuffix(" s")
        self.window_end.setPrefix("End: ")
        self.window_selection.addWidget(self.window_end)
        self.window_selection.addStretch()
        self.window_selection.setAlignment(Qt.AlignmentFlag.AlignTop)
