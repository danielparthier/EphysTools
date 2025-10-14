# Add menu to change visual trace color and antialiasing (Fast mode)
from __future__ import annotations
from typing import TYPE_CHECKING
import ephys.GUI.GUI_config as config
from ephys.GUI.style_sheets.dark_theme import dark_vars
from ephys.GUI.style_sheets.light_theme import light_vars
from PySide6.QtWidgets import (
    QMenuBar,
    QCheckBox,
    QToolBar,
    QWidget,
    QHBoxLayout,
)
from PySide6.QtGui import QAction

import pyqtgraph as pg

if TYPE_CHECKING:
    from ephys.GUI.mainwindow import MainWindow
    from ephys.GUI.trace_view import TracePlotWindow


class PerformanceMenu(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.color_mode_action = QCheckBox("Color Mode")
        self.color_mode_action.setChecked(config.USE_COLOR_MODE)
        self.color_mode_action.toggled.connect(self.toggle_color_mode)

        self.antialiasing_action = QCheckBox("Antialiasing")
        self.antialiasing_action.setChecked(config.USE_ANTIALIASING)
        self.antialiasing_action.toggled.connect(self.toggle_antialiasing)

        layout.addWidget(self.color_mode_action)
        layout.addWidget(self.antialiasing_action)
        self.setLayout(layout)

    def toggle_color_mode(self):
        config.USE_COLOR_MODE = not config.USE_COLOR_MODE
        # change color according to theme
        if config.USE_COLOR_MODE:
            config.CURVE_COLOR = "viridis"
        else:
            if config.theme == "dark":
                config.CURVE_COLOR = dark_vars["color"]
            else:
                config.CURVE_COLOR = light_vars["color"]

    def toggle_antialiasing(self):
        config.USE_ANTIALIASING = not config.USE_ANTIALIASING
        print(f"Antialiasing set to {config.USE_ANTIALIASING}")
        pg.setConfigOptions(antialias=config.USE_ANTIALIASING)
