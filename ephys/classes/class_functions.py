"""
This module provides function to check and import ephys objects and generates metadata.
"""
from re import findall
from math import isclose
from quantities import Quantity
from numpy.lib.stride_tricks import sliding_window_view
import neo
import numpy as np

def wcp_trace(trace, file_path: str) -> None:
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
    import ephys.classes.experiment_objects as ephys_class
    reader = neo.WinWcpIO(file_path)
    data_block = reader.read_block()
    segment_len = reader.segment_count(0)
    trace_len = len(data_block.segments[0].analogsignals[0])
    trace.sampling_rate = data_block.segments[0].analogsignals[0].sampling_rate
    trace.channel_information = ephys_class.ChannelInformation(reader)
    channel_count = trace.channel_information.count()
    trace.voltage = np.zeros((channel_count["signal_type"]["voltage"],
                              segment_len, trace_len))
    trace.current = np.zeros((channel_count["signal_type"]["current"],
                              segment_len, trace_len))
    trace.time = np.zeros((segment_len, trace_len))
    voltage_channels = np.where(trace.channel_information.signal_type == "voltage")[0]
    current_channels = np.where(trace.channel_information.signal_type == "current")[0]
    for index, segment in enumerate(data_block.segments):
        if index == 0:
            time_unit = segment.analogsignals[0].times.units
        j = 0
        for voltage_channel in voltage_channels:
            trace.voltage[j, index, :] = (
                segment.analogsignals[voltage_channel].magnitude[:, 0]
            )
            j += 1
        j = 0
        for current_channel in current_channels:
            trace.current[j, index, :] = (
                segment.analogsignals[current_channel].magnitude[:, 0]
            )
            j += 1
        trace.time[index,:] = segment.analogsignals[0].times
    trace.time = Quantity(trace.time, units=time_unit)

# TODO: add the function to read ABF files
# NOTE: abf file does not always save protocol trace - find setting/reason and/or alternative needed
# NOTE: requires pyabf package - evaluate how often protocol can be read

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
        units = []
        for signal_idx in enumerate(segment[1].analogsignals):
            unit_i = str(signal_idx[1].units)
            if "V" in unit_i:
                v_count.append(signal_idx[0])
            elif "A" in unit_i:
                c_count.append(signal_idx[0])
            else:
                pass
            units.append(signal_idx[1].units)
        channel_count.append(
            [np.array(v_count), np.array(c_count), len(v_count), len(c_count), np.array(units)]
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

def _get_time_index(time: Quantity, time_point: float) -> any:
    """
    Get the index of the time point in the given time array.

    Parameters:
    - time (Quantity): The time array.
    - time_point (float): The time point to find in the time array.

    Returns:
    - int: The index of the time point in the time array.
    """
    if time.magnitude.ndim == 2:
        return np.argmin(np.abs(time.magnitude - time_point), axis=1)
    return np.argmin(np.abs(time.magnitude - time_point))
