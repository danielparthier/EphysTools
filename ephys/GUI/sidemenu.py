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

from PySide6.QtGui import QPainter, QColor
from PySide6.QtSvg import QSvgRenderer

from PySide6.QtCore import Qt, QDate, Signal, QObject, QRunnable, Slot
from ephys.GUI.labfolder import LabfolderWindow
from ephys.GUI.meta_data import MetaDataWindow
from ephys.GUI.styles import apply_style
from ephys.GUI.trace_view import TracePlotWindow
from ephys.classes.experiment_objects import ExpData, MetaData
import ephys.GUI.GUI_config as config
import numpy as np

if TYPE_CHECKING:
    from ephys.GUI.gui_app import MainWindow


class FileSelector(QFileDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)
        self.setNameFilter(
            "WCP/ABF files (*.wcp | *.abf);;WCP files (*.wcp);;ABF files (*.abf)"
        )
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
        from ephys.GUI.meta_data import MetaDataWindow

        super().__init__()
        self.main_window: MainWindow = main_window
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
        self.labfolder_login_button = LabfolderWindow(side_menu=self)

        # File List
        combobox = QComboBox()
        combobox.setMaximumWidth(150)
        combobox.setPlaceholderText("Select File")
        combobox.addItems(self.main_window.session_info.file_list)
        self.combobox: QComboBox = combobox

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

    def change_style(self) -> None:
        """
        Change the style of the application.
        """
        if "dark" in self.theme:
            self.theme = "light"
            self.theme_title = "Dark Mode"
            config.theme = "light"
            self.main_window.side_menu_container.set_background_svg(
                svg_path="logos/Ephys_letters.svg"
            )

            # change color of switch to light
            # self.style_switch.setStyleSheet("color: black;")
        else:
            self.theme = "dark"
            self.theme_title = "Light Mode"
            self.main_window.side_menu_container.set_background_svg(
                svg_path="logos/Ephys_letters_gray.svg"
            )
            # self.style_switch.setStyleSheet("color: white;")
            config.theme = "dark"
        self.style_switch.setText(self.theme_title)
        style_sheet = apply_style(theme=self.theme)
        self.main_window.setStyleSheet(style_sheet)
        if hasattr(self.main_window, "trace_plot"):
            for i in range(self.main_window.trace_plot.count()):
                tab = self.main_window.trace_plot.widget(i)
                if isinstance(tab, TracePlotWindow):
                    tab.update_theme(self.theme)
        if isinstance(self.meta_data_window, MetaDataWindow):
            self.meta_data_window.setStyleSheet(style_sheet)
            # change calendar header background
            header_format = self.meta_data_window.calendar.headerTextFormat()
            header_format.setBackground(QColor("blue"))
            header_format.setForeground(QColor("white"))
        self.main_window.session_info.theme = self.theme
        self.main_window.highlight_switch.setChecked(False)

    def connect_to_main_window(self) -> None:
        self.main_window.session_info.current_user = self.current_user_label.text()
        self.meta_data_button.clicked.connect(self.add_meta_data)
        self.file_selector_button.clicked.connect(self.choose_file)

    #  self.labfolder_login_button.clicked.connect(self.login_labfolder)

    def file_loading(
        self,
        file_path: list | str = "",
    ):
        if file_path == "":
            print("No files selected")
            return None
        self.clicked_status = True
        experimenter_name = self.main_window.session_info.experimenter_name_val
        print(f"Experimenter name: {experimenter_name}")
        print("Date of experiment: ", self.main_window.session_info.exp_date)
        # if isinstance(self.main_window.session_info.file_path, list):
        # new_files = np.array(file_path)[
        #     np.invert(np.isin(file_path, self.main_window.session_info.file_list))
        # ].tolist()
        print(f"File: {file_path}")
        if self.main_window.data is None:
            # to do add worker with slot
            data_worker = DataLoader(
                file_path=file_path, experimenter=experimenter_name
            )
            data_worker.signals.result.connect(self.on_data_loaded)
            self.main_window.threadpool.start(data_worker)
            print("DataWorker is an instance of DataLoader")
            # data = ExpData(file_path=file_path, experimenter=experimenter_name)
        else:
            print("Data already loaded, adding file to existing data")
            new_files = self.main_window.data.new_file_paths(file_path)
            print("New files to add: ", new_files)
            self.main_window.data.add_file(
                file_path=new_files, experimenter=experimenter_name
            )
        if isinstance(self.main_window.data, ExpData):
            self.main_window.session_info.sync_metadata(self.main_window.data.meta_data)

        # data = DataLoader(file_path=file_path, experimenter_name=experimenter_name)
        # self.main_window.threadpool.start(data)
        # The rest of the logic is now handled in on_data_loaded for new data
        if self.main_window.data is not None and isinstance(
            self.main_window.data, ExpData
        ):
            # does clearing not work?
            self.combobox.clear()
            self.file_list = self.main_window.data.meta_data.get_file_name()
            self.file_paths = self.main_window.data.meta_data.get_file_path()

            print("Data loaded successfully")
            print(f"File list: {self.file_list}")
            print("File_path:", file_path)
            if isinstance(self.main_window.data, ExpData):
                print(self.main_window.data.meta_data_summary())

            # make for every file a new tab on the main window
            # check if tab already exists
            self.update_plot_tab()
            self.refresh_file_list()

    def check_new_tabs(self, file_list: list[str] | str) -> list[str]:
        """Check which files in file_list do not have a tab yet.
        Args:
            file_list (list[str] | str): List of file names to check.
        Returns:
            list[str]: List of file names that do not have a tab yet.
        """
        if isinstance(file_list, str):
            file_list = [file_list]
        tabl_list = [
            self.main_window.trace_plot.tabText(tab_index)
            for tab_index in range(self.main_window.trace_plot.count())
        ]
        return list(set(file_list) - set(tabl_list))

    def choose_file(self) -> None:
        # Store the dialog as an instance attribute to prevent it from being deleted
        self.get_file = FileSelector()
        # self.get_file.exec()
        file_path = self.get_file.getOpenFileNames(
            self.main_window,
            "Select WCP/ABF files",
            "data/WCP/WholeCell",
            "All supported (*.wcp and *.abf);;WCP files (*.wcp);;ABF files (*.abf)",
        )[0]

        print("choose files: ", file_path)
        self.file_loading(file_path)

    def add_meta_data(self) -> None:
        from ephys.GUI.meta_data import MetaDataWindow

        # Open the MetaDataWindow as a separate window and keep it open
        self.meta_data_window = MetaDataWindow(on_confirm=self.get_meta_data)
        self.meta_data_window.show()
        # make sure that content of meta_data_window is transferred to the main window
        print("MetaData window opened")

    def get_meta_data(self, experimenter_name: str, selected_date: QDate) -> None:
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

    def refresh_file_list(self) -> None:
        """
        Refresh the file list in the combobox.
        """
        self.combobox.clear()
        if self.main_window.data is not None:
            self.combobox.addItems(self.main_window.data.meta_data.get_file_name())

        # get current tab
        self.combobox.setCurrentText(
            self.main_window.trace_plot.tabText(
                self.main_window.trace_plot.currentIndex()
            )
        )

        print("File list refreshed")

    def on_data_loaded(self, data: ExpData) -> None:
        # find why it is duplicate
        """
        Slot to handle the result from DataLoader.
        """

        self.main_window.data = data
        self.combobox.clear()
        self.file_list = data.meta_data.get_file_name()
        self.file_paths = data.meta_data.get_file_path()
        print("Data loaded successfully")
        print(f"File list: {self.file_list}")
        print(f"Clicked status: {getattr(self, 'clicked_status', None)}")
        if isinstance(self.main_window.data, ExpData):
            print(self.main_window.data.meta_data_summary())
            self.update_plot_tab()
        self.refresh_file_list()

    def update_plot_tab(self) -> None:
        if isinstance(self.main_window.data, ExpData):
            new_tabs = self.check_new_tabs(self.file_list)
            for i, file_name in enumerate(self.file_list):
                if file_name not in new_tabs:
                    continue
                print(f"File {i}: {file_name}")
                # Create a new tab for each file
                print(self.file_paths[i])
                trace_plot = TracePlotWindow(
                    main_window=self.main_window, file_name=self.file_paths[i]
                )
                trace_plot.add_trace_plot(
                    trace=self.main_window.data.protocols[i],
                    color=config.CURVE_COLOR,
                    antialiasing=config.USE_ANTIALIASING,
                )
                self.main_window.trace_plot.addTab(trace_plot, file_name)
        else:
            print("No data loaded yet.")


class SideMenuContainer(QWidget):
    def __init__(self) -> None:
        super().__init__(parent=None)
        self.svg_renderer = None
        self.setFixedSize(120, 48)

    def set_background_svg(self, svg_path) -> None:
        self.svg_renderer = QSvgRenderer(svg_path)

    def paintEvent(self, event) -> None:
        if self.svg_renderer and self.svg_renderer.isValid():
            painter = QPainter(self)
            self.svg_renderer.render(painter, self.rect())
        else:
            super().paintEvent(event)


class DataLoaderSignals(QObject):

    finished = Signal()
    error = Signal(object)
    result = Signal(object)
    progress = Signal(int)


class DataLoader(QRunnable):

    def __init__(self, file_path: str | list[str], experimenter: str):
        super().__init__()
        self.file_path = file_path
        self.experimenter = experimenter
        self.signals = DataLoaderSignals()

    @Slot()
    def run(self):
        # Load data from the file
        try:
            data = self.load_data(self.file_path, self.experimenter)
        except Exception as e:
            self.signals.error.emit(e)
        else:
            self.signals.result.emit(data)
        finally:
            self.signals.finished.emit()

    def load_data(self, file_path: str | list[str], experimenter: str) -> object:
        # Implement the data loading logic here
        return ExpData(file_path=file_path, experimenter=experimenter)
