def apply_style(theme: str) -> str:
    theme_vars = None
    if theme not in ["dark", "light"]:
        raise ValueError("Theme must be 'dark' or 'light'.")
    if theme == "dark":
        from ephys.GUI.style_sheets.dark_theme import dark_vars as theme_vars
    if theme == "light":
        from ephys.GUI.style_sheets.light_theme import light_vars as theme_vars

    if isinstance(theme_vars, dict):
        # Ensure all required keys are present in the theme_vars dictionary
        required_keys = [
            "background_color",
            "color",
            "background_button",
            "background_button_hover",
            "background_button_pressed",
            "border_color",
            "background_scrollbar",
            "background_scrollbar_handle",
            "border_color_hover",
            "background_scrollbar",
            "background_scrollbar_handle",
            "background_scrollbar",
            "arrow_color",
        ]
        for key in required_keys:
            if key not in theme_vars.keys():
                raise KeyError(f"Missing key in theme_vars: {key}")
    else:
        raise TypeError("theme_vars must be a dictionary containing style variables.")

    border_radius = "3px"
    font_family = "'Segoe UI', 'Liberation Sans', Arial, sans-serif"
    font_size = "10pt"
    padding_big = "3px 8px"
    padding_small = "2px 4px"
    font_size_spinbox = "8pt"

    # Generate the style sheet using theme_vars
    theme_out: str = f"""
    QMainWindow {{
    background-color: {theme_vars['background_color']};
    color: {theme_vars['color']};
    }}
    QWidget {{
    background-color: {theme_vars['background_color']};
    color: {theme_vars['color']};
    font-family: {font_family};
    font-size: {font_size};
    }}
    QPushButton {{
    background-color: {theme_vars['background_button']};
    color: {theme_vars['color']};
    border: 1px solid {theme_vars['border_color']};
    border-radius: {border_radius};
    padding: {padding_big};
    }}
    QPushButton:hover {{
    background-color: {theme_vars['background_button_hover']};
    border: 1px solid {theme_vars['border_color_hover']};
    }}
    QPushButton:pressed {{
    background-color: {theme_vars['background_button_pressed']};
    }}
    QFrame, QSplitter {{
    background-color: {theme_vars['background_scrollbar']};
    border: none;
    }}
    QLabel {{
    color: {theme_vars['color']};
    }}

    QCalendarWidget {{
    background-color: {theme_vars['background_color']};
    color: {theme_vars['color']};
    }}
    QCalendarWidget QHeaderView {{
    background-color: {theme_vars['background_button']};
    color: {theme_vars['color']};
    }}
    QCalendarWidget QHeaderView::section {{
    background-color: {theme_vars['background_button']};
    color: {theme_vars['color']};
    padding: 4px;
    }}
    QCalendarWidget QCalendarGridLine {{
    background-color: {theme_vars['background_button']};
    }}
    QCalendarWidget QCalendarGridLine::section {{
    background-color: {theme_vars['background_button']};
    color: {theme_vars['color']};
    }}

    QLineEdit {{
    background-color: {theme_vars['background_scrollbar']};
    color: {theme_vars['color']};
    border: 1px solid {theme_vars['border_color']};
    border-radius: {border_radius};
    padding: {padding_small};
    }}
    QLineEdit:focus {{
    border: 1px solid {theme_vars['background_button_pressed']};
    }}

    QComboBox {{
    background-color: {theme_vars['background_scrollbar']};
    color: {theme_vars['color']};
    border: 1px solid {theme_vars['border_color']};
    border-radius: {border_radius};
    padding: {padding_small};
    }}
    QComboBox QAbstractItemView {{
    background-color: {theme_vars['background_scrollbar']};
    color: {theme_vars['color']};
    selection-background-color: {theme_vars['background_button_pressed']};
    selection-color: {theme_vars['color']};
    }}

    QCheckBox {{
    color: {theme_vars['color']};
    spacing: 6px;
    }}
    QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    }}
    QCheckBox::indicator:unchecked {{
    border: 1px solid {theme_vars['border_color']};
    background: {theme_vars['background_button']};
    }}
    QCheckBox::indicator:checked {{
    border: 1px solid {theme_vars['border_color']};
    background: {theme_vars['background_button_pressed']};
    }}

    QRadioButton {{
    color: {theme_vars['color']};
    spacing: 6px;
    }}
    QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    }}
    QRadioButton::indicator:unchecked {{
    border: 1px solid {theme_vars['border_color']};
    background: {theme_vars['background_button']};
    }}
    QRadioButton::indicator:checked {{
    border: 1px solid {theme_vars['border_color']};
    background: {theme_vars['background_button_pressed']};
    }}

    QScrollBar:vertical {{
    background: {theme_vars['background_scrollbar']};
    width: 12px;
    margin: 16px 0 16px 0;
    border: none;
    }}
    QScrollBar::handle:vertical {{
    background: {theme_vars['background_scrollbar_handle']};
    min-height: 20px;
    border-radius: 6px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background: none;
    border: none;
    }}
    QScrollBar:horizontal {{
    background: {theme_vars['background_scrollbar']};
    height: 12px;
    margin: 0 16px 0 16px;
    border: none;
    }}
    QScrollBar::handle:horizontal {{
    background: {theme_vars['background_scrollbar_handle']};
    min-width: 20px;
    border-radius: 6px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    background: none;
    border: none;
    }}

    QTabWidget::pane {{
    border: 1px solid {theme_vars['border_color']};
    background: {theme_vars['background_scrollbar']};
    }}
    QTabBar::tab {{
    background: {theme_vars['background_color']};
    color: {theme_vars['color']};
    border: 1px solid {theme_vars['border_color']};
    border-bottom: none;
    padding: {padding_big};
    border-top-left-radius: {border_radius};
    border-top-right-radius: {border_radius};
    }}
    QTabBar::tab:selected {{
    background: {theme_vars['background_button']};
    border-color: {theme_vars['border_color']};
    color: {theme_vars['color']};
    }}
    QTabBar::tab:!selected {{
    margin-top: 2px;
    }}
    QTabBar::tab:hover {{
    background: {theme_vars['background_button_hover']};
    border: 1px solid {theme_vars['border_color_hover']};
    }}
    QTabBar::close-button:hover {{
    background: {theme_vars['background_button_hover']};
    }}
    QTabBar::close-button:pressed {{
    background: {theme_vars['background_button_pressed']};
    }}

    QMenuBar {{
    background-color: {theme_vars['background_scrollbar']};
    color: {theme_vars['color']};
    }}
    QMenuBar::item:selected {{
    background: {theme_vars['background_button_pressed']};
    color: {theme_vars['color']};
    }}
    QMessageBox {{
    background-color: {theme_vars['background_scrollbar']};
    color: {theme_vars['color']};
    border: 1px solid {theme_vars['border_color']};
    }}
    QMenu {{
    background-color: {theme_vars['background_scrollbar']};
    color: {theme_vars['color']};
    border: 1px solid {theme_vars['border_color']};
    }}
    QMenu::item:selected {{
    background-color: {theme_vars['background_button_pressed']};
    color: {theme_vars['color']};
    }}

    QToolBar {{
    background: {theme_vars['background_scrollbar']};
    border-bottom: 1px solid {theme_vars['border_color']};
    }}

    QStatusBar {{
    background: {theme_vars['background_scrollbar']};
    color: {theme_vars['color']};
    border-top: 1px solid {theme_vars['border_color']};
    }}

    QSlider::groove:horizontal {{
    border: 1px solid {theme_vars['border_color']};
    height: 6px;
    background: {theme_vars['background_button']};
    border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
    background: {theme_vars['background_scrollbar_handle']};
    border: 1px solid {theme_vars['border_color']};
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
    }}
    QSlider::groove:vertical {{
    border: 1px solid {theme_vars['border_color']};
    width: 6px;
    background: {theme_vars['background_button']};
    border-radius: 3px;
    }}
    QSlider::handle:vertical {{
    background: {theme_vars['background_scrollbar_handle']};
    border: 1px solid {theme_vars['border_color']};
    height: 14px;
    margin: 0 -4px;
    border-radius: 7px;
    }}

    QProgressBar {{
    background-color: {theme_vars['background_scrollbar']};
    border: 1px solid {theme_vars['border_color']};
    border-radius: {border_radius};
    text-align: center;
    color: {theme_vars['color']};
    }}
    QProgressBar::chunk {{
    background-color: #007acc;
    border-radius: {border_radius};
    }}

    QTableView, QListView, QTreeView {{
    background-color: {theme_vars['background_scrollbar']};
    color: {theme_vars['color']};
    border: 1px solid {theme_vars['border_color']};
    selection-background-color: {theme_vars['background_button_pressed']};
    selection-color: {theme_vars['color']};
    gridline-color: {theme_vars['border_color']};
    }}
    QHeaderView::section {{
    background-color: {theme_vars['background_button']};
    color: {theme_vars['color']};
    border: 1px solid {theme_vars['border_color']};
    padding: 4px;
    }}

    QToolTip {{
    background-color: {theme_vars['background_button']};
    color: {theme_vars['color']};
    border: 1px solid {theme_vars['border_color']};
    border-radius: {border_radius};
    padding: {padding_small};
    font-family: {font_family};
    font-size: {font_size};
    }}

    QSpinBox {{
    font-size: {font_size_spinbox};
    }}
    QDoubleSpinBox {{
    font-size: {font_size_spinbox};
    }}

    """
    return theme_out
