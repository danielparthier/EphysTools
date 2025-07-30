from typing import cast

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QFrame, QMainWindow, QSplitter, QTabWidget

from ephys.classes.experiment_objects import ExpData
from ephys.GUI.sessioninfo import SessionInfo
from ephys.GUI.sidemenu import SideMenu, SideMenuContainer
from ephys.GUI.sidebar_right import SideBarRight
from ephys.GUI.menubar import FileMenu
from ephys.GUI.trace_view import TracePlotWindow


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

        # Add a status bar
        self.status_bar = self.statusBar()
        self.status_bar.addAction(icon_action)
        self.status_bar.showMessage("Ready")

        # Create a splitter
        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)
        splitter.setMaximumHeight(2400)
        self.setCentralWidget(splitter)
        # Create a frame

        sidebar_frame = QFrame()
        sidebar_frame.setFrameShape(QFrame.Shape.StyledPanel)
        sidebar_frame.setMinimumWidth(150)
        sidebar_frame.setMaximumWidth(300)
        sidebar_frame.setMinimumHeight(150)
        sidebar_frame.setMaximumHeight(1200)

        self.side_menu_container = SideMenuContainer()
        self.side_menu_container.setParent(sidebar_frame)
        self.side_menu_container.set_background_svg(svg_path="logos/Logo_short.svg")
        # splitter.addWidget(sidebar_frame)
        # Create a layout for the frame
        menu_layout_left = SideMenu(self)

        sidebar_frame.setLayout(menu_layout_left)

        splitter.addWidget(sidebar_frame)
        # splitter.addWidget(menu_layout_left)

        self.trace_plot = QTabWidget()
        self.trace_plot.setTabsClosable(True)
        self.trace_plot.setMovable(True)
        self.trace_plot.setMinimumWidth(800)
        self.trace_plot.setMinimumHeight(600)
        self.trace_plot.setMaximumWidth(2400)
        self.trace_plot.setMaximumHeight(1200)
        self.trace_plot.setTabsClosable(True)
        self.trace_plot.setMovable(True)
        self.trace_plot.tabCloseRequested.connect(self.close_tab)
        splitter.addWidget(self.trace_plot)

        # Add a sidebar on the right side
        sidebar_right = SideBarRight(self)
        sidebar_right.setMinimumWidth(180)
        sidebar_right.setMaximumWidth(300)
        sidebar_right.setMinimumHeight(150)
        sidebar_right.setMaximumHeight(1200)
        splitter.addWidget(sidebar_right)

        menu_layout_left.addWidget(self.side_menu_container)
        # Add the session info widget

        # decouple tab_widget from the splitter

        self.show()

    def closeEvent(self, event):
        print("Window closed")
        event.accept()

    def close_tab(self, index: int) -> None:
        """Close the tab at the given index."""
        tab_selected: TracePlotWindow = cast(
            TracePlotWindow, self.trace_plot.widget(index)
        )
        print(index, tab_selected)
        if index >= 0:
            self.trace_plot.removeTab(index)
            print(f"Closed tab at index {index}")
            if hasattr(tab_selected, "cleanup") and callable(
                getattr(tab_selected, "cleanup", None)
            ):
                tab_selected.cleanup()
        if tab_selected is not None:
            # Ensure the tab is properly deleted
            if hasattr(tab_selected, "deleteLater"):
                tab_selected.deleteLater()
            else:
                print("Tab selected does not have deleteLater method")

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
    #     self.file_list = file_path
