"""
This module provides function to check and import ephys objects and generates metadata.
"""
from re import findall
from math import isclose
from numpy.lib.stride_tricks import sliding_window_view
import neo
import numpy as np


def _type_check(data):
    """
    Check the type of data and determine the type of each channel.

    Parameters:
    - data: The data to be checked.

    Returns:
    - A list containing three lists: channel_list, type_out, and signal_type.
      - channel_list: A list of channel indices.
      - type_out: A list of channel types (either "field" or "cell").
      - signal_type: A list of signal_type types (either "voltage" or "current").
    """

    type_out = []
    channel_list = []
    signal_type = []
    clamp_type = []
    channel_groups = []
    channel_index = 0

    if isinstance(data, neo.io.winwcpio.WinWcpIO):
        Analogsignals = data.read_block().segments[0].analogsignals
        for i in data.header["signal_channels"]:
            if len(findall("Vm\(AC\)", i[0])) == 1:
                type_out.append("field")
                signal_type.append("voltage")
            elif len(findall("Vm", i[0])) == 1:
                type_out.append("cell")
                signal_type.append("voltage")
            elif len(findall("Im", i[0])) == 1:
                type_out.append("cell")
                signal_type.append("current")
            channel_groups.append(i[7].astype(int).tolist())
            clamp_type.append(
                _is_clamp(Analogsignals[channel_index].magnitude.squeeze())
            )
            channel_index += 1
            channel_list.append(channel_index)
    return [channel_list, type_out, signal_type, clamp_type, channel_groups]


def wcp_trace(trace, file_path):
    """
    Reads data from a WinWcp file and populates the given `trace` object with the data.

    Parameters:
        trace (Trace): The Trace object to populate with the data.
        file_path (str): The path to the WinWcp file.

    Returns:
        None

    Raises:
        None
    """
    reader = neo.WinWcpIO(file_path)
    data_block = reader.read_block()
    segment_len = len(data_block.segments)
    channel_information = _signal_check(data_block)
    channel_count = channel_information.max(axis=0)[2:4]
    trace_len = len(data_block.segments[0].analogsignals[0])
    trace.voltage = np.zeros((channel_count[0], segment_len, trace_len))
    trace.current = np.zeros((channel_count[1], segment_len, trace_len))
    trace.time = data_block.segments[0].analogsignals[0].times
    trace.sampling_rate = data_block.segments[0].analogsignals[0].sampling_rate
    trace.channel_information = channel_information
    trace.channel_type = _type_check(reader)
    for segment in enumerate(data_block.segments):
        j = 0
        for voltage_channel in trace.channel_information[segment[0]][0]:
            trace.voltage[j, segment[0], :] = (
                segment[1].analogsignals[voltage_channel].magnitude[:, 0]
            )
            j += 1
        j = 0
        for current_channel in trace.channel_information[segment[0]][1]:
            trace.current[j, segment[0], :] = (
                segment[1].analogsignals[current_channel].magnitude[:, 0]
            )
            j += 1


def _signal_check(data):
    """
    Check the signals in the given data and count the number of voltage and current channels.

    Parameters:
    data (object): The data object containing segments and analog signals.

    Returns:
    numpy.ndarray: An array containing the count of voltage and current channels for each segment.

    """
    channel_count = []
    for segment in enumerate(data.segments):
        v_count = []
        c_count = []
        for signal_idx in enumerate(segment[1].analogsignals):
            unit_i = str(signal_idx[1].units)
            if "V" in unit_i:
                v_count.append(signal_idx[0])
            elif "A" in unit_i:
                c_count.append(signal_idx[0])
            else:
                pass
        channel_count.append(
            [np.array(v_count), np.array(c_count), len(v_count), len(c_count)]
        )
    return np.asarray(channel_count, dtype=object)


def _is_clamp(trace: np.array, window_len: int = 100, tol=1e-20) -> bool:
    """
    Check if the given trace represents a clamp.

    Parameters:
    - trace (np.array): The input trace.
    - window_len (int): The length of the sliding window used for median filtering. Default is 100.
    - tol (float): The tolerance value for comparing the standard deviation to zero. Default is 1e-20.

    Returns:
    - bool: True if the trace represents a clamp, False otherwise.
    """
    if not isinstance(trace, np.ndarray):
        assert isinstance(trace, np.ndarray), "Invalid input. Must be a numpy array."
    trace_median = np.median(sliding_window_view(trace, window_len), axis=1)
    return isclose(
        np.median(np.std(sliding_window_view(trace_median, window_len), axis=1)),
        0.0,
        abs_tol=tol,
    )
