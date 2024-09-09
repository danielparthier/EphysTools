"""
This module provides utility functions for pattern matching in strings.
It includes functions to check if elements in a list or numpy array match a
given pattern.
"""

from re import compile as compile_regex
from matplotlib import colormaps
from matplotlib.colors import is_color_like
import numpy as np


def trace_color(
    traces: np.ndarray, index: int, color: str = "black"
) -> str | np.ndarray:
    """
    Returns the color for a specific trace in a given colormap.

    Parameters:
        traces (np.ndarray): The array of traces.
        index (int): The index of the trace.
        color (str): The name of the colormap or a specific color. Default is "black".

    Returns:
        str or np.ndarray: The color for the specified trace.
    """
    if color in colormaps.keys():
        color_map = colormaps[color]
        color = color_map(np.linspace(0, 1, traces.shape[1]))[index]
    elif is_color_like(color):
        pass
    else:
        print("Invalid color. Default to 'viridis'.")
        color_map = colormaps["viridis"]
        color = color_map(np.linspace(0, 1, traces.shape[1]))[index]
    return color


def string_match(pattern: any, string_list: any) -> np.ndarray:
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
