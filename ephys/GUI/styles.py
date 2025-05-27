
def apply_style(style: str) -> str:
    """
    Apply the selected style to the application.

    Args:
        style (str): The style to be applied. Options are "dark" or "light".

    Returns:
        str: The QSS style sheet corresponding to the selected style.
    """
    if style not in ["dark", "light"]:
        raise ValueError("Invalid style selected. Choose 'dark' or 'light'.")
    if style == "dark":
        from .style_sheets.vscode_qss import vscode_qss as style_sheet
    elif style == "light":
        from .style_sheets.light_qss import light_qss as style_sheet
    return style_sheet
