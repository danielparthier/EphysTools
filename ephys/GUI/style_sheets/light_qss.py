light_qss = """
QMainWindow {
    background-color: #f3f3f3;
    color: #222222;
}
QWidget {
    background-color: #f3f3f3;
    color: #222222;
    font-family: 'Segoe UI', 'Liberation Sans', Arial, sans-serif;
    font-size: 12pt;
}
QPushButton {
    background-color: #e6e6e6;
    color: #222222;
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 6px 12px;
}
QPushButton:hover {
    background-color: #d0e7fa;
    border: 1px solid #007acc;
}
QPushButton:pressed {
    background-color: #b3d6f7;
}
QFrame, QSplitter {
    background-color: #f9f9f9;
    border: none;
}
QLabel {
    color: #222222;
}

QCalendarWidget {
    background-color: #f3f3f3;
    color: #222222;
}
QCalendarWidget QHeaderView {
    background-color: #e6e6e6;
    color: #222222;
}
QCalendarWidget QHeaderView::section {
    background-color: #e6e6e6;
    color: #222222;
    padding: 4px;
}
QCalendarWidget QCalendarGridLine {
    background-color: #e6e6e6;
}
QCalendarWidget QCalendarGridLine::section {
    background-color: #e6e6e6;
    color: #222222;
}

QLineEdit {
    background-color: #ffffff;
    color: #222222;
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 4px 8px;
}
QLineEdit:focus {
    border: 1px solid #007acc;
}

QComboBox {
    background-color: #ffffff;
    color: #222222;
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 4px 8px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #222222;
    selection-background-color: #b3d6f7;
    selection-color: #222222;
}

QCheckBox {
    color: #222222;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QCheckBox::indicator:unchecked {
    border: 1px solid #cccccc;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    border: 1px solid #007acc;
    background: #b3d6f7;
}

QRadioButton {
    color: #222222;
    spacing: 6px;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
}
QRadioButton::indicator:unchecked {
    border: 1px solid #cccccc;
    background: #ffffff;
}
QRadioButton::indicator:checked {
    border: 1px solid #007acc;
    background: #b3d6f7;
}

QScrollBar:vertical {
    background: #f9f9f9;
    width: 12px;
    margin: 16px 0 16px 0;
    border: none;
}
QScrollBar::handle:vertical {
    background: #cccccc;
    min-height: 20px;
    border-radius: 6px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    background: none;
    border: none;
}
QScrollBar:horizontal {
    background: #f9f9f9;
    height: 12px;
    margin: 0 16px 0 16px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #cccccc;
    min-width: 20px;
    border-radius: 6px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    background: none;
    border: none;
}

QTabWidget::pane {
    border: 1px solid #cccccc;
    background: #f9f9f9;
}
QTabBar::tab {
    background: #e6e6e6;
    color: #222222;
    border: 1px solid #cccccc;
    border-bottom: none;
    padding: 6px 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #ffffff;
    border-color: #007acc;
    color: #222222;
}
QTabBar::tab:!selected {
    margin-top: 2px;
}

QMenuBar {
    background-color: #f9f9f9;
    color: #222222;
}
QMenuBar::item:selected {
    background: #b3d6f7;
    color: #222222;
}
QMessageBox {
    background-color: #f9f9f9;
    color: #222222;
    border: 1px solid #cccccc;
}
QMenu {
    background-color: #f9f9f9;
    color: #222222;
    border: 1px solid #cccccc;
}
QMenu::item:selected {
    background-color: #b3d6f7;
    color: #222222;
}

QToolBar {
    background: #f9f9f9;
    border-bottom: 1px solid #cccccc;
}

QStatusBar {
    background: #f9f9f9;
    color: #222222;
    border-top: 1px solid #cccccc;
}

QSlider::groove:horizontal {
    border: 1px solid #cccccc;
    height: 6px;
    background: #e6e6e6;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #007acc;
    border: 1px solid #cccccc;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::groove:vertical {
    border: 1px solid #cccccc;
    width: 6px;
    background: #e6e6e6;
    border-radius: 3px;
}
QSlider::handle:vertical {
    background: #007acc;
    border: 1px solid #cccccc;
    height: 14px;
    margin: 0 -4px;
    border-radius: 7px;
}

QProgressBar {
    background-color: #f9f9f9;
    border: 1px solid #cccccc;
    border-radius: 4px;
    text-align: center;
    color: #222222;
}
QProgressBar::chunk {
    background-color: #007acc;
    border-radius: 4px;
}

QTableView, QListView, QTreeView {
    background-color: #ffffff;
    color: #222222;
    border: 1px solid #cccccc;
    selection-background-color: #b3d6f7;
    selection-color: #222222;
    gridline-color: #cccccc;
}
QHeaderView::section {
    background-color: #e6e6e6;
    color: #222222;
    border: 1px solid #cccccc;
    padding: 4px;
}
"""