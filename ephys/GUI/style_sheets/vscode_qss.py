vscode_qss = """
QMainWindow {
    background-color: #1e1e1e;
    color: #d4d4d4;
}
QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Segoe UI', 'Liberation Sans', Arial, sans-serif;
    font-size: 12pt;
}
QPushButton {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 6px 12px;
}
QPushButton:hover {
    background-color: #37373d;
    border: 1px solid #007acc;
}
QPushButton:pressed {
    background-color: #094771;
}
QFrame, QSplitter {
    background-color: #232323;
    border: none;
}
QLabel {
    color: #d4d4d4;
}

QCalendarWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
}
QCalendarWidget QHeaderView {
    background-color: #2d2d2d;
    color: #d4d4d4;
}
QCalendarWidget QHeaderView::section {
    background-color: #2d2d2d;
    color: #d4d4d4;
    padding: 4px;
}
QCalendarWidget QCalendarGridLine {
    background-color: #2d2d2d;
}
QCalendarWidget QCalendarGridLine::section {
    background-color: #2d2d2d;
    color: #d4d4d4;
}

QLineEdit {
    background-color: #232323;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 4px 8px;
}
QLineEdit:focus {
    border: 1px solid #007acc;
}

QComboBox {
    background-color: #232323;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 4px 8px;
}
QComboBox QAbstractItemView {
    background-color: #232323;
    color: #d4d4d4;
    selection-background-color: #094771;
    selection-color: #ffffff;
}

QCheckBox {
    color: #d4d4d4;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QCheckBox::indicator:unchecked {
    border: 1px solid #3c3c3c;
    background: #232323;
}
QCheckBox::indicator:checked {
    border: 1px solid #007acc;
    background: #094771;
}

QRadioButton {
    color: #d4d4d4;
    spacing: 6px;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
}
QRadioButton::indicator:unchecked {
    border: 1px solid #3c3c3c;
    background: #232323;
}
QRadioButton::indicator:checked {
    border: 1px solid #007acc;
    background: #094771;
}

QScrollBar:vertical {
    background: #232323;
    width: 12px;
    margin: 16px 0 16px 0;
    border: none;
}
QScrollBar::handle:vertical {
    background: #3c3c3c;
    min-height: 20px;
    border-radius: 6px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    background: none;
    border: none;
}
QScrollBar:horizontal {
    background: #232323;
    height: 12px;
    margin: 0 16px 0 16px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #3c3c3c;
    min-width: 20px;
    border-radius: 6px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    background: none;
    border: none;
}

QTabWidget::pane {
    border: 1px solid #3c3c3c;
    background: #232323;
}
QTabBar::tab {
    background: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-bottom: none;
    padding: 6px 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #232323;
    border-color: #007acc;
    color: #ffffff;
}
QTabBar::tab:!selected {
    margin-top: 2px;
}

QMessageBox {
    background-color: #232323;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
}
QMessageBox QPushButton {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 6px 12px;
}
QMessageBox QPushButton:hover {
    background-color: #37373d;
    border: 1px solid #007acc;
}
QMessageBox QPushButton:pressed {
    background-color: #094771;
}
QMessageBox::icon {
    background-color: #232323;
    color: #d4d4d4;
}
QMessageBox::text {
    background-color: #232323;
    color: #d4d4d4;
}
QMessageBox::button {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 6px 12px;
}
QMessageBox::button:hover {
    background-color: #37373d;
    border: 1px solid #007acc;
}
QMessageBox::button:pressed {
    background-color: #094771;
}

QMenuBar {
    background-color: #232323;
    color: #d4d4d4;
}
QMenuBar::item:selected {
    background: #094771;
    color: #ffffff;
}
QMenu {
    background-color: #232323;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
}
QMenu::item:selected {
    background-color: #094771;
    color: #ffffff;
}

QToolBar {
    background: #232323;
    border-bottom: 1px solid #3c3c3c;
}

QStatusBar {
    background: #232323;
    color: #d4d4d4;
    border-top: 1px solid #3c3c3c;
}

QSlider::groove:horizontal {
    border: 1px solid #3c3c3c;
    height: 6px;
    background: #2d2d2d;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #007acc;
    border: 1px solid #3c3c3c;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::groove:vertical {
    border: 1px solid #3c3c3c;
    width: 6px;
    background: #2d2d2d;
    border-radius: 3px;
}
QSlider::handle:vertical {
    background: #007acc;
    border: 1px solid #3c3c3c;
    height: 14px;
    margin: 0 -4px;
    border-radius: 7px;
}

QProgressBar {
    background-color: #232323;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    text-align: center;
    color: #d4d4d4;
}
QProgressBar::chunk {
    background-color: #007acc;
    border-radius: 4px;
}

QTableView, QListView, QTreeView {
    background-color: #232323;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    selection-background-color: #094771;
    selection-color: #ffffff;
    gridline-color: #3c3c3c;
}
QHeaderView::section {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    padding: 4px;
}
"""