from __future__ import annotations
from typing import Any
import numpy as np
import re as re


def string_match(pattern: Any, input_list: Any) -> np.ndarray:
    """
    Check if elements in the input list match the given pattern.

    Args:
        pattern (str or list): The pattern to match against. It can be a single string or a list of strings.
        input_list (list or np.array): The list of elements to check for a match. It can be a list or a numpy array.

    Returns:
        np.array: A boolean numpy array indicating whether each element in the input list matches the pattern.
    """
    if type(pattern) == str:
        pattern = [pattern]
    if type(pattern) != list:
        raise TypeError("Pattern must be a string or list of strings.")
        return None
    if type(input_list) == list:
        input_list = np.array(input_list)
    if type(input_list) != np.ndarray:
        raise TypeError("Input list must be a list or numpy array.")
        return None
    pattern_string = '|'.join(pattern)
    r = re.compile(pattern_string)
    vmatch = np.vectorize(lambda x:bool(r.match(x)))
    return vmatch(input_list)

def test_string_match():
    # Test case 1: Single pattern, single element match
    pattern = 'apple'
    input_list = ['apple', 'banana', 'cherry']
    expected_output = np.array([True, False, False])
    assert np.array_equal(string_match(pattern, input_list), expected_output)

    # Test case 2: Single pattern, multiple element match
    pattern = 'a'
    input_list = ['apple', 'banana', 'cherry']
    expected_output = np.array([True, True, False])
    assert np.array_equal(string_match(pattern, input_list), expected_output)

    # Test case 3: Multiple patterns, single element match
    pattern = ['apple', 'banana']
    input_list = ['apple', 'banana', 'cherry']
    expected_output = np.array([True, True, False])
    assert np.array_equal(string_match(pattern, input_list), expected_output)

    # Test case 4: Multiple patterns, multiple element match
    pattern = ['a', 'b']
    input_list = ['apple', 'banana', 'cherry']
    expected_output = np.array([True, True, False])
    assert np.array_equal(string_match(pattern, input_list), expected_output)

    # Test case 5: Empty pattern, empty input list
    pattern = []
    input_list = []
    expected_output = np.array([])
    assert np.array_equal(string_match(pattern, input_list), expected_output)

    # Test case 6: Invalid pattern type
    pattern = 123
    input_list = ['apple', 'banana', 'cherry']
    try:
        string_match(pattern, input_list)
    except ValueError as e:
        assert str(e) == "Pattern must be a string or list of strings."

    # Test case 7: Invalid input list type
    pattern = 'apple'
    input_list = 123
    try:
        string_match(pattern, input_list)
    except ValueError as e:
        assert str(e) == "Input list must be a list or numpy array."

    print("All test cases pass")

test_string_match()