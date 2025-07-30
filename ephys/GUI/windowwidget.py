from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtWidgets import (
    QFrame,
    QWidget,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QDoubleSpinBox,
    QPushButton,
    QLabel,
)
from PySide6.QtCore import Qt, QSize

if TYPE_CHECKING:
    from ephys.GUI.gui_app import MainWindow


class WindowSelection(QWidget):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self.main_window = main_window
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumSize(QSize(150, 150))
        self.setMaximumSize(QSize(300, 200))
        self.setContentsMargins(0, 0, 0, 0)

        self.window_frame = QFrame(self)
        self.window_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.window_frame.setLineWidth(2)
        self.window_frame.setMidLineWidth(2)
        self.window_frame.setContentsMargins(5, 5, 5, 5)
        self.window_frame.setMinimumWidth(150)
        self.window_frame.setMaximumWidth(300)
        self.window_frame.setMinimumHeight(50)
        self.window_frame.setMaximumHeight(100)

        # Add widgets to the sidebar
        self.window_group = QVBoxLayout(self.window_frame)
        self.button_layout = QVBoxLayout()
        widget_title = QLabel("Window Selection")
        widget_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        widget_title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        self.window_group.addWidget(widget_title)
        self.window_selection = QHBoxLayout()
        self.window_spin = QVBoxLayout()
        self.add_window_button = QPushButton("+")
        self.add_window_button.setMaximumWidth(150)
        self.add_window_button.focusPolicy()
        self.remove_window_button = QPushButton("-")
        self.remove_window_button.setMaximumWidth(150)
        self.remove_window_button.focusPolicy()
        # self.add_window_button.clicked.connect(self.main_window.add_window)
        self.button_layout.addWidget(self.add_window_button)
        self.button_layout.addWidget(self.remove_window_button)
        self.window_selection.addLayout(self.button_layout)
        self.window_selection.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.window_start = QDoubleSpinBox()
        self.window_start.setMaximumWidth(100)
        self.window_start.setSingleStep(0.1)
        self.window_start.setValue(0.0)
        self.window_start.setSuffix(" s")
        self.window_start.setPrefix("Start: ")

        self.window_spin.addWidget(self.window_start)
        self.window_end = QDoubleSpinBox()
        self.window_end.setMaximumWidth(100)
        self.window_end.setSingleStep(0.1)
        self.window_end.setValue(1.0)
        self.window_end.setSuffix(" s")
        self.window_end.setPrefix("End: ")
        self.window_spin.addWidget(self.window_end)
        self.window_spin.addStretch()
        self.window_spin.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.window_selection.addLayout(self.window_spin)
        self.window_selection.addStretch()
        self.window_group.addLayout(self.window_selection)
        self.window_frame.setLayout(self.window_group)
