from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
)
from PySide6.QtCore import Qt
from ephys.labfolder.classes.labfolder_access import LabFolderUserInfo, labfolder_login
from ephys.labfolder.labfolder_config import labfolder_url

if TYPE_CHECKING:
    from ephys.GUI.sidemenu import SideMenu


class LabfolderLoginWindow(QMainWindow):
    def __init__(self, on_confirm=None) -> None:
        self.on_confirm = on_confirm
        self.username = None
        self.password = None
        self.clicked_status = False
        super().__init__()
        self.setWindowTitle("Labfolder Login")
        self.setGeometry(100, 100, 300, 100)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop)

        # Create a layout for the login window
        login_layout = QVBoxLayout()
        # Create a QLineEdit for the username
        username_input = QLineEdit()
        username_input.setPlaceholderText("user.name@email.com")
        login_layout.addWidget(username_input)
        # Create a QLineEdit for the password
        password_input = QLineEdit()
        password_input.setPlaceholderText("Password")
        password_input.setEchoMode(QLineEdit.EchoMode.Password)
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

        self.username_input: QLineEdit = username_input
        self.password_input: QLineEdit = password_input

    def confirm_and_close(self) -> None:
        if self.on_confirm:
            self.on_confirm(
                self.username_input.text(),
                self.password_input.text(),
            )
        self.close()


class LabfolderWindow(QPushButton):
    def __init__(self, side_menu: SideMenu):
        super().__init__("Labfolder Login")
        self.side_menu: SideMenu = side_menu
        self.username = ""

        self.setMaximumWidth(150)
        self.setCheckable(False)
        self.clicked.connect(self.login_labfolder)

    def login_labfolder(self) -> None:
        # Open the LabfolderLoginWindow as a separate window and keep it open
        self.labfolder_login_window = LabfolderLoginWindow(
            on_confirm=self.get_labfolder_auth
        )
        self.labfolder_login_window.show()
        # make sure that content of labfolder_login_window is transferred to the main window
        print("Labfolder login window opened")

    def get_labfolder_auth(self, username: str, password: str) -> None:
        self.username = username
        self.labfolder_auth: LabFolderUserInfo = labfolder_login(
            labfolder_url=labfolder_url,
            user=username,
            password=password,
            allow_input=False,
        )
        if len(self.labfolder_auth.auth_token) == 0:
            print("Labfolder authentication failed")
            return None

        popup_message = QMessageBox(self)
        popup_message.setWindowTitle("Labfolder Login Successful")
        popup_message.setText("Hello " + self.labfolder_auth.first_name + "!")
        popup_message.setWindowOpacity(0.5)
        popup_message.exec()
        self.current_user = (
            self.labfolder_auth.first_name + " " + self.labfolder_auth.last_name
        )
        self.side_menu.current_user_label.setPlaceholderText(self.current_user)
        # add popup message that login was successful
        # Close the login window
        self.labfolder_login_window.close()
