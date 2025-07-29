from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QPushButton,
    QCalendarWidget,
)
from PySide6.QtCore import Qt, QDate


class MetaDataWindow(QMainWindow):
    def __init__(self, on_confirm=None) -> None:
        self.on_confirm = on_confirm
        super().__init__()
        self.setWindowTitle("MetaData")
        self.setGeometry(100, 100, 200, 300)
        self.setMaximumHeight(500)
        self.setMaximumWidth(300)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        meta_data_layout = QVBoxLayout()
        # Create a calendar widget
        calendar = QCalendarWidget()
        calendar.setMaximumWidth(180)
        calendar.setMaximumHeight(190)
        calendar.setGridVisible(True)
        calendar.setVerticalHeaderFormat(
            QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader
        )
        calendar.setHorizontalHeaderFormat(
            QCalendarWidget.HorizontalHeaderFormat.SingleLetterDayNames
        )

        btn_today = QPushButton("Today")
        btn_today.setMaximumWidth(120)
        btn_today.clicked.connect(lambda: calendar.setSelectedDate(QDate.currentDate()))
        meta_data_layout.addWidget(btn_today, alignment=Qt.AlignmentFlag.AlignCenter)

        meta_data_layout.addWidget(calendar, alignment=Qt.AlignmentFlag.AlignCenter)
        # add calendar output to the window
        self.calendar: QCalendarWidget = calendar

        # Create a widget to type in experimenter name
        experimenter_name = QLineEdit()
        # align box to center
        experimenter_name.setMaximumWidth(200)
        experimenter_name.setPlaceholderText("Experimenter Name")
        meta_data_layout.addWidget(
            experimenter_name, alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.experimenter_name: QLineEdit = experimenter_name

        # Set the layout to a central widget so widgets show up
        central_widget = QWidget()
        central_widget.setLayout(meta_data_layout)
        self.setCentralWidget(central_widget)

        # add button to confirm metadata and calendar selection then close window
        confirm_button = QPushButton("Confirm")
        confirm_button.setMaximumWidth(150)
        confirm_button.setCheckable(False)
        confirm_button.clicked.connect(self.confirm_and_close)
        meta_data_layout.addWidget(
            confirm_button, alignment=Qt.AlignmentFlag.AlignJustify
        )
        # add button to close window
        cancel_button = QPushButton("Cancel")
        cancel_button.setMaximumWidth(150)
        cancel_button.setCheckable(False)
        cancel_button.clicked.connect(self.close)
        meta_data_layout.addWidget(
            cancel_button, alignment=Qt.AlignmentFlag.AlignCenter
        )
        # add button to close window
        self.setLayout(meta_data_layout)

    def confirm_and_close(self) -> None:
        if self.on_confirm:
            self.on_confirm(self.experimenter_name.text(), self.calendar.selectedDate())
        self.close()
