from __future__ import annotations
from typing import Any, TYPE_CHECKING
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLineEdit, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt
from ephys.labfolder.classes.labfolder_access import labfolder_login, LabFolderUserInfo

class LabfolderLogingWindow(QMainWindow):
    def __init__(self, on_confirm=None):
        self.on_confirm = on_confirm
        self.username = None
        self.password = None
        self.clicked_status = False
        super().__init__()
        self.setWindowTitle("Labfolder Login")
        self.setGeometry(100, 100, 300, 100)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_AlwaysShowToolTips)
        self.setAttribute(Qt.WA_AlwaysStackOnTop)

        # Create a layout for the login window
        login_layout = QVBoxLayout()
        # Create a QLineEdit for the username
        username_input = QLineEdit()
        username_input.setPlaceholderText("user.name@email.com")
        login_layout.addWidget(username_input)
        # Create a QLineEdit for the password
        password_input = QLineEdit()
        password_input.setPlaceholderText("Password")
        password_input.setEchoMode(QLineEdit.Password)
        login_layout.addWidget(password_input)
        # Create a button to confirm login
        button_layout = QHBoxLayout()
        login_button = QPushButton("Login")
        login_button.setCheckable(False)
        # Button as default button (enter key)
        login_button.setDefault(True)
        login_button.clicked.connect(self.confirm_and_close)
        button_layout.addWidget(login_button)
        # Create a button to cancel login
        cancel_button = QPushButton("Cancel")
        cancel_button.setCheckable(False)
        cancel_button.clicked.connect(self.close)
        button_layout.addWidget(cancel_button)
        login_layout.addLayout(button_layout)
        # Create a central widget to hold the layout
        central_widget = QWidget()
        central_widget.setLayout(login_layout)
        self.setCentralWidget(central_widget)
        # Set the layout to the central widget
        self.setLayout(login_layout)

        self.username_input = username_input
        self.password_input = password_input

    def confirm_and_close(self):
        if self.on_confirm:
            self.on_confirm(
                self.username_input.text(),
                self.password_input.text(),
            )
        self.close()