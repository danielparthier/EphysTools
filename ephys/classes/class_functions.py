"""
This module provides function to check and import ephys objects and generates metadata.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from math import isclose
from quantities import Quantity
from numpy.lib.stride_tricks import sliding_window_view
from neo import WinWcpIO, AxonIO
import numpy as np

if TYPE_CHECKING:
    from ephys.classes.trace import Trace
    from ephys.classes.experiment_objects import get_exp_date
    from ephys.classes.voltage import VoltageTrace
    from ephys.classes.current import CurrentTrace
    from ephys.classes.channels import Channel


def wcp_trace(trace, file_path: str, quick_check: bool = True) -> None:
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

    from ephys.classes.voltage import VoltageTrace, VoltageClamp
    from ephys.classes.current import CurrentTrace, CurrentClamp
    from ephys.classes.channels import ChannelInformation
    from ephys.classes.experiment_objects import get_exp_date

    reader = WinWcpIO(file_path)
    data_block = reader.read_block()
    segment_len = reader.segment_count(0)
    trace_len = len(data_block.segments[0].analogsignals[0])
    trace.sampling_rate = data_block.segments[0].analogsignals[0].sampling_rate
    trace.channel_information = ChannelInformation(reader)
    channel_count = len(reader.header["signal_channels"])
    trace.channel = []
    time_unit = "s"
    trace.sweep_count = segment_len
    trace.time = np.zeros((segment_len, trace_len))

    for channel_index in range(channel_count):
        unit_string = str(reader.header["signal_channels"][channel_index]["units"])
        if unit_string.find("V") != -1:
            trace_insert = VoltageTrace(segment_len, trace_len, unit_string)
            trace_insert.channel_number = channel_index + 1
            trace_insert.sampling_rate = trace.sampling_rate
        elif unit_string.find("A") != -1:
            trace_insert = CurrentTrace(segment_len, trace_len, unit_string)
            trace_insert.channel_number = channel_index + 1
            trace_insert.sampling_rate = trace.sampling_rate
        else:
            raise ValueError("Signal type not recognized")
        for segment_index, segment in enumerate(data_block.segments):
            trace_insert.insert_data(
                segment.analogsignals[channel_index], segment_index
            )
            if channel_index == 0:
                if segment_index == 0:
                    time_unit = segment.analogsignals[0].times.units
                trace.time[segment_index, :] = segment.analogsignals[0].times
        trace_insert.check_clamp(quick_check=quick_check, warnings=False)
        if trace_insert.clamped:
            if isinstance(trace_insert, VoltageTrace):
                trace_insert = VoltageClamp(channel=trace_insert)
            elif isinstance(trace_insert, CurrentTrace):
                trace_insert = CurrentClamp(channel=trace_insert)
        trace_insert.starting_time = Quantity(trace.time[:, 0], time_unit)
        trace_insert.rec_datetime = get_exp_date(
            file_path
        )  # change once updated on neo side
        trace.channel.append(trace_insert)
    if quick_check:
        print("Warning: Quick clamp check might not be accurate.")
    trace.time = Quantity(trace.time, units=time_unit)


def abf_trace(trace, file_path: str, quick_check: bool = True) -> None:
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
    from ephys.classes.voltage import VoltageTrace, VoltageClamp
    from ephys.classes.current import CurrentTrace, CurrentClamp
    from ephys.classes.channels import ChannelInformation

    reader = AxonIO(file_path)
    data_block = reader.read_block()
    segment_len = reader.segment_count(0)
    trace_len = len(data_block.segments[0].analogsignals[0])
    trace.sampling_rate = data_block.segments[0].analogsignals[0].sampling_rate
    trace.channel_information = ChannelInformation(reader)
    channel_count = len(reader.header["signal_channels"])
    trace.channel = []
    time_unit = "s"
    trace.sweep_count = segment_len
    trace.time = np.zeros((segment_len, trace_len))

    for channel_index in range(channel_count):
        unit_string = str(reader.header["signal_channels"][channel_index]["units"])
        if unit_string.find("V") != -1:
            trace_insert = VoltageTrace(segment_len, trace_len, unit_string)
            trace_insert.channel_number = channel_index + 1
            trace_insert.sampling_rate = trace.sampling_rate
        elif unit_string.find("A") != -1:
            trace_insert = CurrentTrace(segment_len, trace_len, unit_string)
            trace_insert.channel_number = channel_index + 1
            trace_insert.sampling_rate = trace.sampling_rate
        else:
            raise ValueError("Signal type not recognized")
        for segment_index, segment in enumerate(data_block.segments):
            trace_insert.insert_data(
                segment.analogsignals[channel_index], segment_index
            )
            if channel_index == 0:
                if segment_index == 0:
                    time_unit = segment.analogsignals[0].times.units
                trace.time[segment_index, :] = segment.analogsignals[0].times
        trace_insert.check_clamp(quick_check=quick_check, warnings=False)
        if trace_insert.clamped:
            if isinstance(trace_insert, VoltageTrace):
                trace_insert = VoltageClamp(channel=trace_insert)
            elif isinstance(trace_insert, CurrentTrace):
                trace_insert = CurrentClamp(channel=trace_insert)
        trace_insert.rec_datetime = data_block.rec_datetime
        trace.channel.append(trace_insert)
    if quick_check:
        print("Warning: Quick clamp check might not be accurate.")
    trace.time = Quantity(trace.time, units=time_unit)


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
            [
                np.array(v_count),
                np.array(c_count),
                len(v_count),
                len(c_count),
                np.array(units),
            ]
        )
    return np.asarray(channel_count, dtype=object)


def _is_clamp(trace: np.ndarray, window_len: int = 100, tol=1e-20) -> bool:
    """
    Check if the given trace represents a clamp.

    Parameters:
    - trace (np.array): The input trace.
    - window_len (int): The length of the sliding window used for median
      filtering. Default is 100.
    - tol (float): The tolerance value for comparing the standard deviation
      to zero. Default is 1e-20.

    Returns:
    - bool: True if the trace represents a clamp, False otherwise.
    """

    if not isinstance(trace, np.ndarray):
        assert isinstance(trace, np.ndarray), "Invalid input. Must be a numpy array."
    trace_median = np.median(sliding_window_view(trace, window_len, axis=0), axis=1)
    return isclose(
        np.median(
            np.std(sliding_window_view(trace_median, window_len, axis=0), axis=1)
        ),
        0.0,
        abs_tol=tol,
    )


def check_clamp(
    trace: Channel | VoltageTrace | CurrentTrace,
    quick_check: bool = False,
    warnings: bool = True,
) -> None:
    """
    Check if the given trace is clamped.

    Parameters:
    trace (VoltageTrace or CurrentTrace): The trace object to check.
    It should have a 'trace' attribute which is a list of values.
    quick_check (bool, optional): If True, only the first value of the trace is
    checked. Defaults to False.
    warnings (bool, optional): If True, prints a warning message when using
    quick_check. Defaults to True.

    Returns:
    bool: True if the trace is clamped, False otherwise.

    Notes:
    - If quick_check is True, the function sets the 'clamped' attribute of the
    trace based on the first value of the trace.
    - If quick_check is False, the function checks all values in the trace and
    sets the 'clamped' attribute based on the consistency of the clamp status.
    - If the clamp status is not consistent, a warning is printed.
    """

    if quick_check:
        trace.clamped = _is_clamp(trace.data[0])
        if warnings:
            print("Warning: Quick clamp check might not be accurate.")
    else:
        clamp_check = [_is_clamp(trace_check) for trace_check in trace.data]
        if len(np.unique(clamp_check)) == 1:
            trace.clamped = clamp_check[0]
        else:
            print("Clamp status is not consistent.")


def _get_time_index(time: Quantity, time_point: float) -> Any:
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


def moving_average(input_array: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute the moving average of a 1D array.

    Parameters:
    input_array (numpy.ndarray): The input array.
    window_size (int): The size of the moving window.

    Returns:
    numpy.ndarray: The moving averages.
    """

    if len(input_array) == 0:
        raise ValueError("Input array is empty. Cannot compute moving average.")
    if window_size <= 0:
        if len(input_array) > 1:
            window_size = len(input_array) // 2
    if window_size > len(input_array):
        raise ValueError("Window size cannot be larger than the input array length.")

    padded_input_array = np.pad(input_array, (window_size // 2), mode="edge")
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(padded_input_array, window, "same")[
        (window_size // 2) : (window_size // 2 + len(input_array))
    ]


def _get_sweep_subset(array: np.ndarray, sweep_subset: Any) -> np.ndarray:
    """
    Get a subset of sweeps from the given trace.

    Parameters:
    array (np.ndarray): The array containing the data.
    sweep_subset (np.ndarray): An array of indices specifying the subset of sweeps to retrieve.
                               If None, all sweeps are included.

    Returns:
    np.ndarray: An array of unique indices specifying the subset of sweeps.
    """

    if sweep_subset is None:
        sweep_subset = np.r_[range(array.shape[0])]
    else:
        sweep_subset = np.unique(np.r_[sweep_subset])
    return sweep_subset
