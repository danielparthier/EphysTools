"""
This module provides utility functions for pattern matching in strings.
It includes functions to check if elements in a list or numpy array match a
given pattern.
"""

from typing import Any
from re import compile as compile_regex
from matplotlib import colormaps
from matplotlib.colors import is_color_like, to_rgb
from pyqtgraph import mkColor
from pyqtgraph.Qt.QtGui import QColor
import numpy as np


def color_picker(
    length: int, index: int, color: str = "black", alpha: float = 1.0
) -> str | np.ndarray:
    """
    Selects a color from a colormap or validates a given color string.

    Parameters:
    length (int): Number of colors in the colormap.
    index (int): Index of the color to select.
    color (str): Name of the colormap or a color string. Defaults to 'black'.
    alpha (float): Alpha value for the color. Defaults to 1.0.

    Returns:
    str | np.ndarray: The selected color.

    Notes:
    - Defaults to 'viridis' colormap if the color is invalid.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha value must be between 0 and 1")
    if color in colormaps.keys():
        color_map = colormaps[color]
        color = color_map(np.linspace(0, 1, length), alpha)[index]
    elif is_color_like(color):
        pass
    else:
        print("Invalid color. Default to 'viridis'.")
        color_map = colormaps["viridis"]
        color = color_map(np.linspace(0, 1, length), alpha)[index]
    return color


def color_picker_qcolor(
    length: int, index: int, color: str = "black", alpha: float = 1.0
) -> QColor:
    """
    Selects a color from a colormap or validates a given color string and returns
    it as a QColor object.
    Args:
        length (int): Number of colors in the colormap.
        index (int): Index of the color to select.
        color (str): Name of the colormap or a color string. Defaults to 'black'.
        alpha (float): Alpha value for the color. Defaults to 1.0.
    Returns:
        QColor: The selected color as a QColor object.
    """
    selected_color: str | np.ndarray = color_picker(length, index, color, alpha)
    rgba_val: tuple = (0, 0, 0, 255)  # Default RGBA value
    if isinstance(selected_color, np.ndarray):
        rgba_val = tuple(int(c * 255) for c in selected_color)
    elif isinstance(selected_color, str):
        rgba_val = tuple(int(c * 255) for c in to_rgb(selected_color) + (alpha,))
    return mkColor(rgba_val)


def trace_color(
    traces: np.ndarray, index: int, color: str = "black"
) -> str | np.ndarray:
    """
    Returns the color for a specific trace in a given colormap.

    Parameters:
        traces (np.ndarray): The array of traces.
        index (int): The index of the trace.
        color (str): The name of the colormap or a specific color. Default is 'black'.

    Returns:
        str or np.ndarray: The color for the specified trace.
    """
    return color_picker(traces.shape[0], index, color)


def string_match(pattern: Any, string_list: Any) -> np.ndarray:
    """
    Check if the given pattern matches any element in the input list.

    Parameters:
    pattern (str, list, or numpy array): The pattern to match against.
    It can be a string, a list of strings, or a numpy array of strings.
    input (str, list, or numpy array): The input list to check for matches.
    It can be a string, a list of strings, or a numpy array of strings.

    Returns:
    np.ndarray: A boolean numpy array indicating whether each element in the
    string_input list matches the pattern.
    """

    def single_match(pattern_string: str, input_string: str):
        regex_pattern = compile_regex(pattern_string)
        return bool(regex_pattern.match(input_string))

    if isinstance(pattern, str):
        pattern = [pattern]
    elif isinstance(pattern, np.ndarray):
        pattern = pattern.tolist()
    if not isinstance(pattern, list):
        raise ValueError("Pattern must be a string, list, or numpy array of strings.")
    if isinstance(string_list, np.ndarray):
        string_list = string_list.tolist()
    elif isinstance(string_list, str):
        string_list = [string_list]
    if not isinstance(string_list, list):
        raise ValueError("Input list must be a str, list or numpy array of strings.")
    pattern_string = "|".join(pattern)
    return np.array([single_match(pattern_string, i) for i in string_list])
