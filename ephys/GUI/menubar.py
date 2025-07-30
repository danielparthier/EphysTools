import ephys.GUI.GUI_config as config
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QMainWindow,
    QVBoxLayout,
    QTabWidget,
    QMenuBar,
    QWidget,
    QToolBar,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QObject, Signal, Slot

from ephys.GUI.sidemenu import FileSelector


class FileMenu(QMenuBar):
    def __init__(self) -> None:
        super().__init__()
        # Add a menu bar
        file_menu = self.addMenu("File")
        edit_menu = self.addMenu("Edit")
        view_menu = self.addMenu("View")
        help_menu = self.addMenu("Help")

        # Add actions to the file menu
        new_action = QAction("New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)
        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        save_as_action = QAction("Save As", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_file_as)
        file_menu.addAction(save_as_action)
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        # exit_action.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(exit_action)

    def new_file(self):
        """Create a new file."""
        print("New file created")

    def open_file(self):
        """Open an existing file."""
        file_selector = FileSelector()
        file_selector.exec()

    def save_file(self):
        """Save the current file."""
        print("File saved")

    def save_file_as(self):
        """Save the current file with a new name."""
        print("File saved as a new name")
