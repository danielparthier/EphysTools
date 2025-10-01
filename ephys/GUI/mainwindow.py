from PySide6.QtCore import Qt, QThreadPool, QSize
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QFrame,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QSpinBox,
    QCheckBox,
)

from ephys.classes.experiment_objects import ExpData, MetaData
from ephys.GUI.sessioninfo import SessionInfo
from ephys.GUI.sidemenu import SideMenu, SideMenuContainer
from ephys.GUI.sidebar_right import SideBarRight
from ephys.GUI.menubar import FileMenu
from ephys.GUI.trace_view import TracePlotWindow
from PySide6.QtWidgets import QWidget, QHBoxLayout


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.clicked_status = False
        self.data: ExpData | None = None
        self.session_info = SessionInfo()

        self.threadpool = QThreadPool()
        thread_count = self.threadpool.maxThreadCount()
        print(f"Multithreading enabled with {thread_count} threads.")

        self.setWindowTitle("EphysTools")
        self.setGeometry(100, 100, 1200, 800)

        file_menu = FileMenu()
        self.setMenuBar(file_menu)

        icon = QIcon("logos/Logo_short.svg")
        icon_action = QAction(icon, "", self)

        self.trace_list = []

        # Add a status bar
        self.status_bar = self.statusBar()
        self.status_bar.addAction(icon_action)
        self.status_bar.showMessage("Ready")

        # Add sweep selector to the status bar using a QWidget wrapper

        sweep_selector_container = QWidget()
        sweep_selector_layout = QHBoxLayout(sweep_selector_container)
        sweep_selector_layout.setContentsMargins(0, 0, 0, 0)

        self.sweep_selector = QSpinBox()
        self.sweep_selector.setSingleStep(1)

        self.highlight_switch = QCheckBox("Highlight")
        self.highlight_switch.setChecked(False)
        self.highlight_switch.stateChanged.connect(self.change_highlight)
        self.status_bar.addPermanentWidget(self.highlight_switch)

        sweep_selector_layout.addWidget(self.sweep_selector)
        self.sweep_selector.valueChanged.connect(self.sweep_highlight)

        self.status_bar.addPermanentWidget(sweep_selector_container)

        # Create a splitter
        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)
        # Create a frame

        sidebar_frame = QFrame()
        sidebar_frame.setFrameShape(QFrame.Shape.StyledPanel)
        sidebar_frame.setMinimumSize(QSize(150, 150))
        sidebar_frame.setMaximumSize(QSize(300, 1200))

        self.side_menu_container = SideMenuContainer()
        self.side_menu_container.setParent(sidebar_frame)
        self.side_menu_container.set_background_svg(
            svg_path="logos/Ephys_letters_gray.svg"
        )
        # splitter.addWidget(sidebar_frame)
        # Create a layout for the frame
        self.menu_layout_left = SideMenu(self)
        self.menu_layout_left.combobox.currentTextChanged.connect(
            self.select_file_from_list
        )

        sidebar_frame.setLayout(self.menu_layout_left)

        splitter.addWidget(sidebar_frame)
        # splitter.addWidget(menu_layout_left)

        self.trace_plot = QTabWidget()
        self.trace_plot.setTabsClosable(True)
        self.trace_plot.setMovable(True)
        self.trace_plot.setMinimumSize(800, 600)
        self.trace_plot.currentChanged.connect(self.connect_sweep_selector)
        self.trace_plot.tabCloseRequested.connect(self.close_tab)

        splitter.addWidget(self.trace_plot)

        # Add a sidebar on the right side
        sidebar_right = SideBarRight(self)
        sidebar_right.setMinimumSize(QSize(180, 150))
        sidebar_right.setMaximumSize(QSize(300, 1200))
        splitter.addWidget(sidebar_right)

        self.menu_layout_left.addWidget(self.side_menu_container)
        # Add the session info widget

        # decouple tab_widget from the splitter

        self.show()

    def closeEvent(self, event) -> None:
        print("Window closed")
        # Perform any necessary cleanup here
        if isinstance(self.data, ExpData):
            self.data = None
        event.accept()

    def close_tab(self, index: int) -> None:
        """Close the tab at the given index."""
        # Store a reference to the widget before removing the tab
        tab_selected = self.trace_plot.widget(index)
        tab_title = self.trace_plot.tabText(index)

        # Disconnect the tabCloseRequested signal temporarily to prevent cascading closes
        old_signal = self.trace_plot.tabCloseRequested.disconnect()
        if isinstance(self.data, ExpData):
            self.data.remove_file(index=index)
            self.session_info.set_file_path(self.data.meta_data.get_file_path())
            self.session_info.set_file_list(self.data.meta_data.get_file_name())
        # Remove just this one tab
        self.trace_plot.removeTab(index)
        print(f"Closed tab '{tab_title}' at index {index}")

        # Reconnect the signal
        self.trace_plot.tabCloseRequested.connect(self.close_tab)
        self.menu_layout_left.refresh_file_list()
        # Clean up the tab's resources safely
        if isinstance(tab_selected, TracePlotWindow):
            try:
                tab_selected.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {e}")

            # Schedule widget for deletion
            tab_selected.deleteLater()

    def on_color_selected(self, color) -> None:
        # Handle the selected color
        print(f"Selected color: {color.name()}")

    def _sweep_highlight(self, sweep_number: int | None) -> None:
        """Highlight a specific sweep in the plot."""
        current_tab_index = self.trace_plot.currentIndex()
        current_trace_tab = self.trace_plot.widget(0)
        sweep_index = None
        if isinstance(current_trace_tab, TracePlotWindow):
            current_plot = current_trace_tab.trace_list[current_tab_index]
            if sweep_number is None:
                sweep_index = None
            if isinstance(sweep_number, int):
                sweep_index = sweep_number - 1
            current_plot.sweep_highlight(sweep_index)

    def sweep_highlight(self, sweep_number: int | None = None) -> None:
        if self.highlight_switch.isChecked():
            self._sweep_highlight(sweep_number)
        else:
            self._sweep_highlight(sweep_number=None)

    def connect_sweep_selector(self) -> None:
        current_tab_index: int = self.trace_plot.currentIndex()
        if current_tab_index >= 0 and len(self.trace_list) > 0:
            if self.trace_list[current_tab_index].trace.sweep_count == 1:
                self.sweep_selector.setEnabled(False)
            else:
                self.sweep_selector.setEnabled(True)
                self.sweep_selector.setRange(
                    1, self.trace_list[current_tab_index].trace.sweep_count
                )
            if self.trace_list[current_tab_index].highlight["sweep_index"] is None:
                self.sweep_selector.setValue(1)
            else:
                self.sweep_selector.setValue(
                    self.trace_list[current_tab_index].highlight["sweep_index"]
                )

    def change_highlight(self) -> None:
        if not self.highlight_switch.isChecked():
            self.sweep_highlight(sweep_number=None)
        else:
            self.sweep_highlight(sweep_number=self.sweep_selector.value())

    def select_file_from_list(self, s):
        print("text change: ", s)

    # def close_tab(self, index: int) -> None:
    #     """Close the tab at the given index."""
    #     tab_selected: TracePlotWindow = cast(
    #         TracePlotWindow, self.trace_plot.widget(index)
    #     )
    #     print(index, tab_selected)
    #     if index >= 0:
    #         self.trace_plot.removeTab(index)
    #         print(f"Closed tab at index {index}")
    #         if hasattr(tab_selected, "cleanup") and callable(
    #             getattr(tab_selected, "cleanup", None)
    #         ):
    #             tab_selected.cleanup()
    #     if tab_selected is not None:
    #         # Ensure the tab is properly deleted
    #         if hasattr(tab_selected, "deleteLater"):
    #             tab_selected.deleteLater()
    #         else:
    #             print("Tab selected does not have deleteLater method")

    # def choose_file(self):
    #     # Store the dialog as an instance attribute to prevent it from being deleted
    #     self.get_file = FileSelector()
    #     # self.get_file.exec()
    #     file_path = self.get_file.getOpenFileNames(
    #         self, "Select WCP files", "data/WCP/WholeCell", "WCP files (*.wcp)"
    #     )[0]
    #     # if not file_path:
    #     #     print("No files selected")
    #     #     return None
    #     print(f"Selected files: {file_path}")
    #     self.the_button_was_clicked(file_path)
    #     if isinstance(file_path, str):
    #         file_path = [file_path]
    #     self.file_list = file_pathlen(self.trace_list)
