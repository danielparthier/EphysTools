import __future__
from typing import TYPE_CHECKING
from PySide6.QtWidgets import (
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QWidget,
    QFileDialog,
    QLabel,
)

from PySide6.QtGui import QPainter, QPixmap, QBrush, QColor
from PySide6.QtSvg import QSvgRenderer

from PySide6.QtCore import Qt, QDate, QSize, QRect
from ephys.GUI.meta_data import MetaDataWindow
from ephys.GUI.labfolder import LabfolderWindow
from ephys.GUI.styles import apply_style
from ephys.GUI.trace_view import TracePlotWindow
from ephys.classes.experiment_objects import ExpData
import ephys.GUI.GUI_config as config

if TYPE_CHECKING:
    from ephys.GUI.gui_app import MainWindow


class FileSelector(QFileDialog):
    def __init__(self):
        super().__init__()
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)
        self.setNameFilter("WCP files (*.wcp);;ABF files (*.abf)")
        self.setViewMode(QFileDialog.ViewMode.List)
        self.setOption(QFileDialog.Option.ReadOnly, True)
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.setDirectory("data/WCP/WholeCell")
        self.setWindowTitle("Select WCP files")
        self.setModal(True)
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop)


class SideMenu(QVBoxLayout):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self.main_window = main_window
        self.theme = config.theme
        if "dark" in self.theme:
            self.theme_title = "Light Mode"
        if "light" in self.theme:
            self.theme_title = "Dark Mode"

        # set up the side menu layout
        self.setContentsMargins(10, 10, 10, 10)
        self.setSpacing(10)
        self.setAlignment(Qt.AlignmentFlag.AlignTop)

        title_label = QLabel("Experiment Info")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        self.addWidget(title_label)

        # Current User
        self.current_user_label = QLineEdit()
        self.current_user_label.setMaximumWidth(150)
        self.current_user_label.setPlaceholderText("Current User")
        self.current_user_label.setReadOnly(False)

        # Meta Data
        self.meta_data_window = MetaDataWindow()
        self.meta_data_button = QPushButton("MetaData")
        self.meta_data_button.setMaximumWidth(150)
        self.meta_data_button.setCheckable(False)

        # File Selector
        self.file_selector_button = QPushButton("Select Files")
        self.file_selector_button.setMaximumWidth(150)
        self.file_selector_button.setCheckable(False)

        # Labfolder Login
        self.labfolder_login_button = LabfolderWindow(self)

        # File List
        combobox = QComboBox()
        combobox.setMaximumWidth(150)
        combobox.setPlaceholderText("Select File")
        combobox.addItems(self.main_window.session_info.file_list)
        self.combobox = combobox

        # Add widgets to the layout
        self.addWidget(self.current_user_label)
        self.addWidget(self.meta_data_button)
        self.addWidget(self.file_selector_button)
        self.addWidget(self.labfolder_login_button)
        self.addWidget(self.combobox)

        self.connect_to_main_window()

        # add toggle to change style
        self.style_switch = QCheckBox(self.theme_title)
        self.style_switch.setChecked("dark" in self.theme)
        self.main_window.session_info.theme = self.theme
        self.style_switch.stateChanged.connect(self.change_style)
        self.addWidget(self.style_switch)

    def change_style(self):
        """
        Change the style of the application.
        """
        if "dark" in self.theme:
            self.theme = "light"
            self.theme_title = "Dark Mode"
            config.theme = "light"
            # change color of switch to light
            # self.style_switch.setStyleSheet("color: black;")
        else:
            self.theme = "dark"
            self.theme_title = "Light Mode"
            # self.style_switch.setStyleSheet("color: white;")
            config.theme = "dark"
        self.style_switch.setText(self.theme_title)
        style_sheet = apply_style(theme=self.theme)
        self.main_window.setStyleSheet(style_sheet)
        print(config.theme)
        self.main_window.session_info.theme = self.theme

        print(f"Style changed to {self.theme}")

    def connect_to_main_window(self):
        self.main_window.session_info.current_user = self.current_user_label.text()
        self.meta_data_button.clicked.connect(self.add_meta_data)
        self.file_selector_button.clicked.connect(self.choose_file)

    #  self.labfolder_login_button.clicked.connect(self.login_labfolder)

    def file_loading(
        self,
        file_path: list | str = "data/WCP/WholeCell/180207_005.cc_step_40_comp.1.wcp",
    ):
        self.clicked_status = True
        experimenter_name = self.main_window.session_info.experimenter_name_val
        print(f"Experimenter name: {experimenter_name}")
        print("Date of experiment: ", self.main_window.session_info.exp_date)

        if self.main_window.data is None:
            data = ExpData(file_path=file_path, experimenter=experimenter_name)
        else:
            print("Data already loaded, adding file to existing data")
            print(self.main_window.data.meta_data_summary())
            data = self.main_window.data
            data.add_file(file_path, experimenter_name)

        # data = DataLoader(file_path=file_path, experimenter_name=experimenter_name)
        # self.main_window.threadpool.start(data)
        if data is None:
            print("Data could not be loaded")
        self.combobox.clear()

        # data = data.get_data()
        self.file_list = [
            file_info["file_name"] for file_info in data.meta_data.file_info
        ]
        self.main_window.data = data
        print("Data loaded successfully")
        print(f"File list: {self.file_list}")
        print(f"Clicked status: {self.clicked_status}")
        if isinstance(self.main_window.data, ExpData):
            print(self.main_window.data.meta_data_summary())

        # make for every file a new tab on the main window
        for i, file_name in enumerate(self.file_list):
            print(f"File {i}: {file_name}")
            # Create a new tab for each file

            trace_plot = TracePlotWindow(self.main_window, file_name)

            trace_plot.add_trace_plot(trace=self.main_window.data.protocols[i])

        self.refresh_file_list()

    def choose_file(self):
        # Store the dialog as an instance attribute to prevent it from being deleted
        self.get_file = FileSelector()
        # self.get_file.exec()
        file_path = self.get_file.getOpenFileNames(
            self.main_window,
            "Select WCP files",
            "data/WCP/WholeCell",
            "WCP files (*.wcp)",
        )[0]
        # if not file_path:
        #     print("No files selected")
        #     return None
        print(f"Selected files: {file_path}")
        self.file_loading(file_path)
        if isinstance(file_path, str):
            file_path = [file_path]
        self.main_window.session_info.file_list = file_path

    def add_meta_data(self):
        # Open the MetaDataWindow as a separate window and keep it open
        self.meta_data_window = MetaDataWindow(on_confirm=self.get_meta_data)
        self.meta_data_window.show()
        # make sure that content of meta_data_window is transferred to the main window
        print("MetaData window opened")

    def get_meta_data(self, experimenter_name: str, selected_date: QDate):
        self.main_window.session_info.experimenter_name_val = experimenter_name
        self.main_window.session_info.exp_date = selected_date.toString(
            Qt.DateFormat.ISODate
        )
        self.main_window.session_info.current_user = (
            self.main_window.session_info.experimenter_name_val
        )
        self.current_user_label.setPlaceholderText(
            self.main_window.session_info.current_user
        )
        print(
            f"Experimenter name: {self.main_window.session_info.experimenter_name_val}"
        )
        print(f"Selected date: " + self.main_window.session_info.exp_date)

    def refresh_file_list(self):
        """
        Refresh the file list in the combobox.
        """
        self.combobox.clear()
        if self.main_window.data is not None:
            self.combobox.addItems(
                self.main_window.data.meta_data_summary()["file_name"]
            )
        print("File list refreshed")


class SideMenuContainer(QWidget):
    def __init__(self):
        super().__init__(parent=None)
        self.svg_renderer = None
        self.setFixedSize(150, 150)

    def set_background_svg(self, svg_path):
        self.svg_renderer = QSvgRenderer(svg_path)

    def paintEvent(self, event):
        if self.svg_renderer and self.svg_renderer.isValid():
            painter = QPainter(self)
            self.svg_renderer.render(painter, self.rect())
        else:
            super().paintEvent(event)
