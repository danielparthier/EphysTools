from __future__ import annotations
from typing import TYPE_CHECKING
import sys

from PySide6.QtWidgets import QApplication, QMainWindow,QSplitter, QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtWidgets import QPushButton, QFrame, QComboBox, QLineEdit, QFileDialog
from PySide6.QtWidgets import QCalendarWidget, QTabWidget, QMessageBox
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


from .meta_data import MetaDataWindow
from .labfolder import LabfolderLogingWindow
from ephys.classes.experiment_objects import ExpData
from ephys.labfolder.labfolder_config import labfolder_url as labfolder_url
from ephys.labfolder.classes.labfolder_access import labfolder_login, LabFolderUserInfo
from .styles import apply_style
# filepath: /home/daniel/Work/RETAIN/Code/MossyFibre/ephys/GUI/gui_app.py


class FileSelector(QFileDialog):
    def __init__(self):
        super().__init__()
        self.setFileMode(QFileDialog.ExistingFiles)
        self.setNameFilter("WCP files (*.wcp);;ABF files (*.abf)")
        self.setViewMode(QFileDialog.List)
        self.setOption(QFileDialog.ReadOnly, True)
        self.setOption(QFileDialog.DontUseNativeDialog, True)
        self.setDirectory("data/WCP/WholeCell")
        self.setWindowTitle("Select WCP files")
        self.setModal(True)
        self.setAcceptMode(QFileDialog.AcceptOpen)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_AlwaysShowToolTips)
        self.setAttribute(Qt.WA_AlwaysStackOnTop)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.clicked_status = False
        self.data = None
        self.experimenter_name_val = ""
        self.meta_data_window = None
        self.exp_date = ""
        self.file_path = ""
        self.file_list = [""]
        self.current_user = ""
        self.setWindowTitle("EphysTools")
        self.setGeometry(100, 100, 1200, 800)

        # Add a menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        edit_menu = menu_bar.addMenu("Edit")
        view_menu = menu_bar.addMenu("View")
        help_menu = menu_bar.addMenu("Help")
        
        icon = QIcon("logos/Logo_short.svg")
        icon_action = QAction(icon, "", self)

        # Add a status bar
        self.status_bar = self.statusBar()
        self.status_bar.addAction(icon_action)
        self.status_bar.showMessage("Ready")

        # Create a splitter
        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        self.setCentralWidget(splitter)
        # Create a frame
        frame = QFrame()
        frame.setMinimumWidth(150)
        frame.setMaximumWidth(300)
        frame.setMinimumHeight(150)
        frame.setMaximumHeight(300)

        frame.setFrameShape(QFrame.StyledPanel)
        splitter.addWidget(frame)
        # Create a layout for the frame
        layout = QVBoxLayout(frame)

        meta_data_button = QPushButton("MetaData")
        meta_data_button.setMaximumWidth(130)
        meta_data_button.setCheckable(False)
        meta_data_button.clicked.connect(self.add_meta_data)
        layout.addWidget(meta_data_button)



        # add button which opens new window with file selector
        file_selector_button = QPushButton("Select Files")
        file_selector_button.setMaximumWidth(130)
        file_selector_button.setCheckable(False)
        file_selector_button.clicked.connect(self.choose_file)
        layout.addWidget(file_selector_button)

        # # Create a splitter for the file selector
        # file_selector_splitter = QSplitter()
        # file_selector_splitter.setOrientation(Qt.Vertical)
        # file_selector_splitter.addWidget(file_selector)
        # splitter.addWidget(file_selector_splitter)

        # Create a button
        button = QPushButton("Labfolder Login")
        button.setMaximumWidth(150)
        button.setCheckable(False)
        button.clicked.connect(self.login_labfolder)
        layout.addWidget(button)

        # Show current user as label
        self.current_user_label = QLineEdit()
        self.current_user_label.setMaximumWidth(150)
        self.current_user_label.setPlaceholderText("Current User")
        self.current_user_label.setReadOnly(False)
        layout.addWidget(self.current_user_label)


        # Create a combobox
        combobox = QComboBox()
        combobox.setMaximumWidth(150)
        combobox.setPlaceholderText("Select File")
        combobox.addItems(self.file_list)
        self.combobox = combobox
        layout.addWidget(self.combobox)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.setMinimumWidth(800)
        self.tab_widget.setMinimumHeight(600)
        splitter.addWidget(self.tab_widget)

        self.show()

    def the_button_was_clicked(self, file_path: list|str = "data/WCP/WholeCell/180207_005.cc_step_40_comp.1.wcp"):
        self.clicked_status = True
        experimenter_name = self.experimenter_name_val
        print(f"Experimenter name: {experimenter_name}")
        print("Date of experiment: ", self.exp_date)

        data = ExpData(file_path=file_path, experimenter=experimenter_name)
        self.combobox.clear()

        self.file_list =  [file_info["file_name"] for file_info in data.meta_data.file_info]
        self.data = data
        self.combobox.addItems(self.file_list)
        print("Data loaded successfully")
        print(f"Data: {self.data}")

        print("Button was clicked")
        print(f"Clicked status: {self.clicked_status}")

        # make for every file a new tab on the main window
        for i, file_name in enumerate(self.file_list):
            print(f"File {i}: {file_name}")
            # Create a new tab for each file
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab.setLayout(tab_layout)
            self.tab_widget.addTab(tab, file_name)

            # Create a frame for the plot
            plot_frame = QFrame()
            plot_frame.setFrameShape(QFrame.StyledPanel)
            plot_frame.setMinimumWidth(800)
            plot_frame.setMinimumHeight(600)
            tab_layout.addWidget(plot_frame)

            # Create a layout for the plot frame
            plot_layout = QVBoxLayout(plot_frame)
            plot_frame.setLayout(plot_layout)

            # Create a plot area
            plot_area = QFrame()
            plot_area.setFrameShape(QFrame.StyledPanel)
            plot_area.setMinimumWidth(800)
            plot_area.setMinimumHeight(600)
            plot_layout.addWidget(plot_area)

            # Create a layout for the plot area
            plot_area_layout = QVBoxLayout(plot_area)
            plot_area.setLayout(plot_area_layout)
            # Create a plot widget
            fig, axs = self.data.protocols[i].plot(return_fig=True, show=False)
            # Add the plot to the plot area
            plot_area_layout.addWidget(fig.canvas)

            # Set the plot title
            fig.suptitle(file_name)
            # Set the plot size
            fig.set_size_inches(10, 6)
            # Set the plot layout

        # for i, protocol in enumerate(self.data.protocols):
        #     file_name = self.data.meta_data.file_info[i]["file_name"]
        #     print(f"Protocol {i}: {protocol}")
        #     fig, axs = protocol.plot(return_fig=True, show=False)
        #     # figure name
        #     fig.canvas.manager.set_window_title(file_name)
        #     fig.show()
                #print(f"Protocol meta_data: {protocol.meta_data.__dict__}")
        #self.data.plot()
        # make figure of length of protocols and put single plot in each subplot
        # fig = plt.figure(figsize=(10, 10))
        # gs = gridspec.GridSpec(len(self.data.protocols), 2)
        # for i, protocol in enumerate(self.data.protocols):
        #     fig_plot, channel_ax = protocol.plot(show=False, return_fig=True)
        #     fig.add_subplot(gs[i], sharex=channel_ax, sharey=channel_ax)
        #     channel_ax.plot(protocol.time, protocol.data, label=protocol.meta_data.channel_name)
        #     channel_ax.set_title(protocol.meta_data.channel_name)
        # fig.show()


        #[protocol.plot() for protocol in self.data.protocols]

    def login_labfolder(self):
        # Open the LabfolderLoginWindow as a separate window and keep it open
        self.labfolder_login_window = LabfolderLogingWindow(on_confirm=self.get_labfolder_auth)
        self.labfolder_login_window.show()
        # make sure that content of labfolder_login_window is transferred to the main window
        print("Labfolder login window opened")

    def get_labfolder_auth(self, username: str, password: str):
        self.username = username
        self.password = password
        self.labfolder_auth = labfolder_login(
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
        self.current_user = self.labfolder_auth.first_name + " " + self.labfolder_auth.last_name
        self.current_user_label.setPlaceholderText(self.current_user)
        # add popup message that login was successful


        # Close the login window
        
        self.labfolder_login_window.close()


    def closeEvent(self, event):
        print("Window closed")
        event.accept()

    def choose_file(self):
        # Store the dialog as an instance attribute to prevent it from being deleted
        self.get_file = FileSelector()
        #self.get_file.exec()
        file_path = self.get_file.getOpenFileNames(self, "Select WCP files", "data/WCP/WholeCell", "WCP files (*.wcp)")[0]
       # if not file_path:
       #     print("No files selected")
       #     return None
        print(f"Selected files: {file_path}")
        self.the_button_was_clicked(file_path)
        if isinstance(file_path, str):
            file_path = [file_path]
        self.file_list = file_path

    def add_meta_data(self):
        # Open the MetaDataWindow as a separate window and keep it open
        self.meta_data_window = MetaDataWindow(on_confirm=self.get_meta_data)
        self.meta_data_window.show()
        # make sure that content of meta_data_window is transferred to the main window
        print("MetaData window opened")
    def get_meta_data(self, experimenter_name: str, selected_date: Qt.ISODate):
        self.experimenter_name_val = experimenter_name
        self.exp_date = selected_date.toString(Qt.ISODate)
        self.current_user = self.experimenter_name_val
        self.current_user_label.setPlaceholderText(self.current_user)
        print(f"Experimenter name: {self.experimenter_name_val}")
        print(f"Selected date: " + self.exp_date)
        # Get the experimenter name from the QLineEdit
        
        #self.experimenter_name_val = self.meta_data_window.experimenter_name.text()
        #print(f"Experimenter name: {self.experimenter_name_val}")
        # Get the selected date from the calendar widget
        #selected_date = self.meta_data_window.calendar.selectedDate()
        #print(f"Selected date: {selected_date.toString(Qt.ISODate)}")
        # You can use this date for further processing
        # Close the metadata window
        # self.meta_data_window = MetaDataWindow()
        # self.meta_data_window.show()

app = QApplication(sys.argv)
style = apply_style("light")
app.setStyleSheet(style)

window = MainWindow()
window.show()

app.exec()
      