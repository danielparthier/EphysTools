from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLineEdit, QPushButton, QCalendarWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

class MetaDataWindow(QMainWindow):
    def __init__(self, on_confirm=None):
        self.on_confirm = on_confirm
        super().__init__()
        self.setWindowTitle("MetaData")
        self.setGeometry(100, 100, 400, 300)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        # self.setAttribute(Qt.WA_DeleteOnClose)
        # self.setAttribute(Qt.WA_QuitOnClose, False)
        # self.setAttribute(Qt.WA_ShowWithoutActivating)
        # self.setAttribute(Qt.WA_AlwaysShowToolTips)
        # self.setAttribute(Qt.WA_AlwaysStackOnTop)
     
        meta_data_layout = QVBoxLayout()
        # Create a calendar widget
        calendar = QCalendarWidget()
        calendar.setMaximumWidth(150)
        calendar.setMaximumHeight(180)

        calendar.setGridVisible(True)
        calendar.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        calendar.setHorizontalHeaderFormat(QCalendarWidget.HorizontalHeaderFormat.SingleLetterDayNames)
    
        meta_data_layout.addWidget(calendar)
        # add calendar output to the window
        self.calendar = calendar

        # Create a widget to type in experimenter name
        experimenter_name = QLineEdit()
        experimenter_name.setMaximumWidth(150)
        experimenter_name.setPlaceholderText("Experimenter Name")
        meta_data_layout.addWidget(experimenter_name)
        self.experimenter_name = experimenter_name

        # Set the layout to a central widget so widgets show up
        central_widget = QWidget()
        central_widget.setLayout(meta_data_layout)
        self.setCentralWidget(central_widget)

        # add button to confirm metadata and calendar selection then close window
        confirm_button = QPushButton("Confirm")
        confirm_button.setMaximumWidth(150)
        confirm_button.setCheckable(False)
        confirm_button.clicked.connect(self.confirm_and_close)
        meta_data_layout.addWidget(confirm_button)
        # add button to close window
        cancel_button = QPushButton("Cancel")
        cancel_button.setMaximumWidth(150)
        cancel_button.setCheckable(False)
        cancel_button.clicked.connect(self.close)
        meta_data_layout.addWidget(cancel_button)
        # add button to close window
        self.setLayout(meta_data_layout)


    def confirm_and_close(self):
        if self.on_confirm:
            self.on_confirm(
                self.experimenter_name.text(),
                self.calendar.selectedDate()
            )
        self.close()