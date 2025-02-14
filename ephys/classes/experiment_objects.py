"""
This module provides classes for representing experimental data and metadata.
"""
import os
import time
from copy import deepcopy
from re import findall
import neo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from quantities import Quantity
from ephys.classes.class_functions import _get_time_index, _is_clamp, _get_sweep_subset
from ephys import utils


class ChannelInformation:
    """
    A class to extract and store channel information from electrophysiological data.

    Attributes:
    -----------
    channel_number : np.ndarray or None
        Array of channel numbers.
    array_index : np.ndarray or None
        Array of array indices.
    recording_type : np.ndarray or None
        Array of recording types (e.g., 'field', 'cell').
    signal_type : np.ndarray or None
        Array of signal types (e.g., 'voltage', 'current').
    clamped : np.ndarray or None
        Array indicating the clamp type.
    channel_grouping : np.ndarray or None
        Array of channel groupings.
    unit : np.ndarray or None
        Array of units for each channel.

    Methods:
    --------
    __init__(data: any) -> dict:
        Initializes the ChannelInformation object with data from a neo.io object.

    to_dict() -> dict:
        Converts the channel information to a dictionary format.

    count() -> dict:
        Counts the occurrences of unique values in the channel information attributes.
    """

    def __init__(self, data: any) -> dict:
        type_out = []
        channel_list = []
        signal_type = []
        clamp_type = []
        channel_groups = []
        channel_unit = []
        array_index = []
        channel_index = 0
        voltage_index = 0
        current_index = 0

        if isinstance(data, neo.io.winwcpio.WinWcpIO):
            analogsignals = data.read_block().segments[0].analogsignals
            for i in data.header["signal_channels"]:
                if len(findall(r"Vm\(AC\)", i[0])) == 1:
                    type_out.append("field")
                    signal_type.append("voltage")
                    array_index.append(voltage_index)
                    voltage_index += 1
                elif len(findall(r"Vm", i[0])) == 1:
                    type_out.append("cell")
                    signal_type.append("voltage")
                    array_index.append(voltage_index)
                    voltage_index += 1
                elif len(findall(r"Im", i[0])) == 1:
                    type_out.append("cell")
                    signal_type.append("current")
                    array_index.append(current_index)
                    current_index += 1
                channel_groups.append(i["stream_id"].astype(int).tolist())
                clamp_type.append(
                    _is_clamp(analogsignals[channel_index].magnitude.squeeze())
                )
                channel_index += 1
                channel_list.append(channel_index)
                channel_unit.append(str(i["units"]))
        elif isinstance(data, neo.io.axonio.AxonIO):
            analogsignals = data.read_block().segments[0].analogsignals
            for i in data.header["signal_channels"]:
                if len(findall(r"V", i[4])) == 1:
                    type_out.append("cell")
                    signal_type.append("voltage")
                    array_index.append(voltage_index)
                    voltage_index += 1
                elif len(findall(r"A", i[4])) == 1:
                    type_out.append("cell")
                    signal_type.append("current")
                    array_index.append(current_index)
                    current_index += 1
                channel_groups.append(i["stream_id"].astype(int).tolist())
                clamp_type.append(
                    _is_clamp(analogsignals[channel_index].magnitude.squeeze())
                )
                channel_index += 1
                channel_list.append(channel_index)
                channel_unit.append(str(i["units"]))
        elif isinstance(data, neo.io.igorproio.IgorIO):
            pass
        if len(channel_list) > 0:
            self.channel_number = np.array(channel_list)
            self.array_index = np.array(array_index)
            self.recording_type = np.array(type_out)
            self.signal_type = np.array(signal_type)
            self.clamped = np.array(clamp_type)
            self.channel_grouping = np.array(channel_groups)
            self.unit = np.array(channel_unit)
        else:
            self.channel_number = None
            self.array_index = None
            self.recording_type = None
            self.signal_type = None
            self.clamped = None
            self.channel_grouping = None
            self.unit = None
            print("No channel information found.")

    def to_dict(self) -> dict:
        """
        Return the channel information as dictionary.

        Returns:
            dict: A dictionary containing the following key-value pairs:
                - 'channel_number' (int): The channel number.
                - 'array_index' (int): The index of the array.
                - 'recording_type' (str): The type of recording.
                - 'signal_type' (str): The type of signal.
                - 'clamped' (bool): Indicates if the signal is clamped.
                - 'channel_grouping' (str): The grouping of the channel.
                - 'unit' (str): The unit of measurement.
        """
        return {
            "channel_number": self.channel_number,
            "array_index": self.array_index,
            "recording_type": self.recording_type,
            "signal_type": self.signal_type,
            "clamped": self.clamped,
            "channel_grouping": self.channel_grouping,
            "unit": self.unit,
        }

    def count(self) -> dict:
        """
        Return the count of unique values in the channel information attributes.

        Returns:
            dict: A dictionary containing the attributes and their counts.
                - 'signal_type' (dict): The count of unique signal types.
                - 'array_index' (dict): The count of unique array indices.
                - 'recording_type' (dict): The count of unique recording types.
                - 'clamped' (dict): The count of unique clamp types.
                - 'channel_grouping' (dict): The count of unique channel groupings.
                - 'unit' (dict): The count of unique units.
        """
        signal_type, signal_type_count = np.unique(self.signal_type, return_counts=True)
        array_index, array_index_count = np.unique(self.array_index, return_counts=True)
        recording_type, recording_type_count = np.unique(
            self.recording_type, return_counts=True
        )
        clamped, clamped_count = np.unique(self.clamped, return_counts=True)
        channel_grouping, channel_grouping_count = np.unique(
            self.channel_grouping, return_counts=True
        )
        unit, unit_idx = np.unique(self.unit, return_counts=True)
        return {
            "signal_type": dict(zip(signal_type, signal_type_count)),
            "array_index": dict(zip(array_index, array_index_count)),
            "recording_type": dict(zip(recording_type, recording_type_count)),
            "clamped": dict(zip(clamped, clamped_count)),
            "channel_grouping": dict(zip(channel_grouping, channel_grouping_count)),
            "unit": dict(zip(unit, unit_idx)),
        }


class VoltageTrace:
    """
    A class to represent a voltage trace in electrophysiology experiments.

    Attributes:
    -----------
    sweep_count : int
        The number of sweeps in the voltage trace.
    sweep_length : int
        The length of each sweep.
    unit : str
        The unit of the voltage trace.
    data : Quantity
        The voltage trace data.
    clamped : None or bool
        The clamped state of the voltage trace.

    Methods:
    --------
    __init__(self, sweep_count: int, sweep_length: int, unit: str):
        Initializes the VoltageTrace with the given sweep count, sweep length, and unit.

    insert_data(self, data, sweep_count: int):
        Inserts data into the trace at the specified sweep count.

    append_data(self, data):
        Appends data to the trace.

    load_data_block(self, data_block, channel_index):
        Loads a block of data into the trace (currently a placeholder).

    change_unit(self, unit):
        Changes the unit of the voltage trace to the specified unit if it is a voltage unit.

    check_clamp(self, quick_check: bool = False, warnings: bool = True):
        Checks if the voltage trace is clamped using the check_clamp function.
    """

    def __init__(self, sweep_count: int, sweep_length: int, unit: str):
        self.sweep_count = sweep_count
        self.sweep_length = sweep_length
        self.unit = unit
        self.data = Quantity(np.zeros((self.sweep_count, self.sweep_length)), unit)
        self.average = None
        self.clamped = None

    def insert_data(self, data, sweep_count: int) -> None:
        """
        Inserts sweep data into the trace at the specified sweep count.

        Parameters:
        data (numpy.ndarray): The data to be inserted.
        sweep_count (int): The index at which the data should be inserted.

        Returns:
        None
        """
        self.data[sweep_count] = data.flatten()

    def append_data(self, data) -> None:
        """
        Appends new data to the existing trace.

        Parameters:
        data (numpy.ndarray): The data to be appended. It should be a numpy array that can be
        flattened and stacked with the existing trace.

        Returns:
        None
        """
        self.data = Quantity(np.vstack((self.data, data.flatten())), self.unit)

    def load_data_block(self, data_block, channel_index) -> None:
        # TODO: Implement this method --> not sure if it is necessary
        pass

    def change_unit(self, unit: str) -> None:
        """
        Change the unit of the trace to the specified voltage unit.

        Parameters:
        unit (str): The new unit to rescale the trace to. Must be a voltage unit.

        Returns:
        None

        Raises:
        ValueError: If the unit does not end with 'V'.
        """
        if unit.endswith("V"):
            if unit.find("µ") == 0:
                unit = unit.replace("µ", "u")
            self.data = self.data.rescale(unit)
            self.unit = unit
        else:
            raise ValueError("Unit must be voltage.")

    def check_clamp(self, quick_check: bool = False, warnings: bool = True) -> None:
        """
        Checks the clamp status of the experiment.

        This method uses the `check_clamp` function from the `ephys.classes.class_functions` module
        to verify the clamp status of the experiment object.

        Args:
            quick_check (bool, optional): If True, performs a quick check. Defaults to False.
            warnings (bool, optional): If True, displays warnings. Defaults to True.

        Returns:
            None
        """
        from ephys.classes.class_functions import check_clamp  # pylint: disable=C

        check_clamp(self, quick_check, warnings)
    
    def channel_average(self,
                        sweep_subset: any = None) -> None:
        """
        Calculate the channel average for a given subset of sweeps.

        Parameters:
            sweep_subset (any, optional): A subset of sweeps to be averaged. If None, 
            the entire set of sweeps will be used. Defaults to None.

        Returns:
            None: The result is stored in the `self.average` attribute as a ChannelAverage object.
        """
        sweep_subset = _get_sweep_subset(self.data, sweep_subset)
        self.average = ChannelAverage(self, sweep_subset)

class CurrentTrace:
    """
    A class to represent a current trace in electrophysiology experiments.

    Attributes:
    -----------
    sweep_count : int
        The number of sweeps in the current trace.
    sweep_length : int
        The length of each sweep.
    unit : str
        The unit of the current trace.
    data : Quantity
        The current trace data.
    clamped : None or bool
        The clamped state of the current trace.

    Methods:
    --------
    __init__(self, sweep_count: int, sweep_length: int, unit: str):
        Initializes the CurrentTrace with the given sweep count, sweep length, and unit.

    insert_data(self, data, sweep_count: int):
        Inserts data into the trace at the specified sweep count.

    append_data(self, data):
        Appends data to the trace.

    load_data_block(self, data_block, channel_index):
        Loads a block of data into the trace (currently a placeholder).

    change_unit(self, unit):
        Changes the unit of the current trace to the specified unit if it is a current unit.

    check_clamp(self, quick_check: bool = False, warnings: bool = True):
        Checks if the current trace is clamped using the check_clamp function.
    """

    def __init__(self, sweep_count: int, sweep_length: int, unit: str):
        self.sweep_count = sweep_count
        self.sweep_length = sweep_length
        self.unit = unit
        self.data = Quantity(np.zeros((self.sweep_count, self.sweep_length)), unit)
        self.average = None
        self.clamped = None

    def insert_data(self, data, sweep_count: int) -> None:
        """
        Inserts sweep data into the trace at the specified sweep count.

        Parameters:
        data (numpy.ndarray): The data to be inserted.
        sweep_count (int): The index at which the data should be inserted.

        Returns:
        None
        """
        self.data[sweep_count] = data.flatten()

    def append_data(self, data) -> None:
        """
        Appends new data to the existing trace.

        Parameters:
        data (numpy.ndarray): The data to be appended. It should be a numpy array that can be
        flattened and stacked with the existing trace.

        Returns:
        None
        """
        self.data = Quantity(np.vstack((self.data, data.flatten())), self.unit)

    def load_data_block(self, data_block, channel_index):
        # TODO: Implement this method --> not sure if it is necessary
        pass

    def change_unit(self, unit: str) -> None:
        """
        Change the unit of the trace to the specified voltage unit.

        Parameters:
        unit (str): The new unit to rescale the trace to. Must be a voltage unit.

        Returns:
        None

        Raises:
        ValueError: If the unit does not end with 'V'.
        """
        if unit.endswith("A"):
            if unit.find("µ") == 0:
                unit = unit.replace("µ", "u")
            self.data = self.data.rescale(unit)
            self.unit = unit
        else:
            raise ValueError("Unit must be current.")

    def check_clamp(self, quick_check: bool = False, warnings: bool = True) -> None:
        """
        Checks the clamp status of the experiment.

        This method uses the `check_clamp` function from the `ephys.classes.class_functions` module
        to verify the clamp status of the experiment object.

        Args:
            quick_check (bool, optional): If True, performs a quick check. Defaults to False.
            warnings (bool, optional): If True, displays warnings. Defaults to True.

        Returns:
            None
        """
        from ephys.classes.class_functions import check_clamp  # pylint: disable=C

        check_clamp(self, quick_check, warnings)
    
    def channel_average(self,
                        sweep_subset: any = None) -> None:
        """
        Calculate the channel average for a given subset of sweeps.

        Parameters:
            sweep_subset (any, optional): A subset of sweeps to be averaged. If None, 
            the entire set of sweeps will be used. Defaults to None.

        Returns:
            None: The result is stored in the `self.average` attribute as a ChannelAverage object.
        """
        sweep_subset = _get_sweep_subset(self.data, sweep_subset)
        self.average = ChannelAverage(self, sweep_subset)



class Trace:
    """
    Represents a trace object.

    Args:
        file_path (str): The file path of the trace.

    Attributes:
        file_path (str): The file path of the trace.

    Methods:
        copy() -> any:
            Returns a deep copy of the Trace object.

        subset(channels: any = all channels, can be a list,
               signal_type: any = 'voltage' and 'current',
               rec_type: any = all rec_types) -> any:
            Returns a subset of the Trace object based on the specified channels, signal_type, and
            rec_type.

        average_trace(channels: any = all channels, can be a list,
                      signal_type: any = 'voltage' and 'current', can be a list,
                      rec_type: any = all rec_types) -> any:
            Returns the average trace of the Trace object based on the specified channels,
            signal_type, and rec_type.

        plot(signal_type: str, channels: list, average: bool = False, color: str ='k',
            alpha: float = 0.5, avg_color: str = 'r'):
            Plots the trace data based on the specified signal_type, channels, and other optional
            parameters.
    """

    def __init__(self, file_path: str, quick_check: bool = True) -> None:
        self.file_path = file_path
        self.voltage = None
        self.current = None
        self.time = None
        self.sampling_rate = None
        self.channel = None
        self.channel_information = None
        if file_path.endswith(".wcp"):
            from ephys.classes.class_functions import (
                wcp_trace_old,
                wcp_trace_new,
            )  # pylint: disable=C

            wcp_trace_old(self, file_path)
            wcp_trace_new(self, file_path, quick_check)
        elif file_path.endswith(".abf"):
            from ephys.classes.class_functions import abf_trace  # pylint: disable=C

            abf_trace(self, file_path, quick_check)
        else:
            print("File type not supported")

    def copy(self) -> any:
        """
        Returns a deep copy of the Trace object.
        """
        return deepcopy(self)

    def subset(
        self,
        channels: any = None,
        signal_type: any = None,
        rec_type: any = "",
        clamp_type: any = None,
        channel_groups: any = None,
        sweep_subset: any = None,
        subset_index_only: bool = False,
        in_place: bool = False,
    ) -> any:
        """
        Subset the experiment object based on specified criteria.

        Args:
            channels (any, optional): Channels to include in the subset.
                                      Defaults to all channels.
            signal_type (any, optional): Types of signal_type to include in the subset.
                                   Defaults to ['voltage', 'current'].
            rec_type (any, optional): Recording types to include in the subset. Defaults to ''.
            clamp_type (any, optional): Clamp types to include in the subset. Defaults to None.
            channel_groups (any, optional): Channel groups to include in the subset. Defaults to None.
            sweep_subset (any, optional): Sweeps to include in the subset. Possible inputs can be list,
                                    arrays or slice(). Defaults to None.
            subset_index_only (bool, optional): If True, returns only the subset index. Defaults to False.
            in_place (bool, optional): If True, modifies the object in place. Defaults to False.

        Returns:
            any: Subset of the experiment object.

        """
        if channels is None and signal_type is None and rec_type == "" and clamp_type is None and channel_groups is None:
            if subset_index_only:
                return self.channel_information
            else:
                return self
        sweep_subset = _get_sweep_subset(self.time, sweep_subset)
        if in_place:
            subset_trace = self
        else:
            subset_trace = self.copy()
        rec_type_get = utils.string_match(
            rec_type, self.channel_information.recording_type
        )
        if clamp_type is None:
            clamp_type = np.array([True, False])
        clamp_type_get = np.isin(self.channel_information.clamped, np.array(clamp_type))
        if channel_groups is None:
            channel_groups = self.channel_information.channel_grouping
        channel_groups_get = np.isin(
            self.channel_information.channel_grouping, channel_groups
        )
        if signal_type is None:
            signal_type = np.array(["voltage", "current"])
        signal_type_get = utils.string_match(
            signal_type, self.channel_information.signal_type
        )
        if isinstance(channels, int):
            channels = np.array([channels])
        if channels is None:
            channels = self.channel_information.channel_number
        else:
            channels = np.array(channels)
        channels_get = np.isin(self.channel_information.channel_number, channels)
        # TODO: switch to boolean indexing and np.array inside dict as default
        combined_index = np.logical_and.reduce(
            (
                rec_type_get,
                signal_type_get,
                channels_get,
                clamp_type_get,
                channel_groups_get,
            )
        )

        if len(combined_index) > 0:
            signal_type = self.channel_information.signal_type[combined_index]
            voltage_index = 0
            current_index = 0
            array_index = []
            for type_test in signal_type:
                if type_test == "voltage":
                    array_index.append(voltage_index)
                    voltage_index += 1
                elif type_test == "current":
                    array_index.append(current_index)
                    current_index += 1
            subset_trace.channel_information.channel_number = (
                self.channel_information.channel_number[combined_index]
            )
            subset_trace.channel_information.array_index = (
                self.channel_information.array_index[combined_index]
            )
            subset_trace.channel_information.recording_type = (
                self.channel_information.recording_type[combined_index]
            )
            subset_trace.channel_information.signal_type = signal_type
            subset_trace.channel_information.clamped = self.channel_information.clamped[
                combined_index
            ]
            subset_trace.channel_information.channel_grouping = (
                self.channel_information.channel_grouping[combined_index]
            )
            subset_trace.channel_information.unit = self.channel_information.unit[
                combined_index
            ]
            for channel_index, channel in enumerate(subset_trace.channel):
                if combined_index[channel_index]:
                    subset_trace.channel[channel_index].data = channel.data[sweep_subset, :]
                else:
                    subset_trace.channel.pop(channel_index)
            subset_trace.time = subset_trace.time[sweep_subset, :]
        else:
            subset_trace.channel_information.channel_number = np.array([])
            subset_trace.channel_information.array_index = np.array([])
            subset_trace.channel_information.recording_type = np.array([])
            subset_trace.channel_information.signal_type = np.array([])
            subset_trace.channel_information.clamped = np.array([])
            subset_trace.channel_information.channel_grouping = np.array([])
            subset_trace.channel_information.unit = np.array([])
        if subset_index_only:
            return subset_trace.channel_information
        else:
            return subset_trace

    def set_time(
        self,
        align_to_zero: bool = True,
        cumulative: bool = False,
        stimulus_interval: float = 0.0,
        overwrite_time: bool = True,
    ) -> any:
        """
        Set the time axis for the given trace data.

        Parameters:
        - trace_data (Trace): The trace data object.
        - align_to_zero (bool): If True, align the time axis to zero. Default is True.
        - cumulative (bool): If True, set the time axis to cumulative. Default is False.
        - stimulus_interval (float): The stimulus interval. Default is 0.0.

        Returns:
        - Trace or None
        """
        tmp_time = deepcopy(self.time)
        time_unit = tmp_time.units
        start_time = Quantity(0, time_unit)
        stimulus_interval = Quantity(stimulus_interval, time_unit)
        sampling_interval = (1 / self.sampling_rate).rescale(time_unit)

        for sweep_index, sweep in enumerate(tmp_time):
            if align_to_zero:
                start_time = Quantity(np.min(sweep.magnitude), time_unit)
            if cumulative:
                if sweep_index > 0:
                    start_time = (
                        Quantity(
                            np.min(sweep.magnitude)
                            - np.max(tmp_time[sweep_index - 1].magnitude),
                            time_unit,
                        )
                        - stimulus_interval
                        - sampling_interval
                    )
            tmp_time[sweep_index] -= start_time
        if overwrite_time:
            self.time = tmp_time
            return None
        return tmp_time

    def rescale_time(self, time_unit: str = "s") -> None:
        """
        Rescale the time axis for the given trace data.

        Parameters:
        - trace_data (Trace): The trace data object.
        - time_unit (str): The time unit. Default is 's'.

        Returns:
        - None
        """
        self.time = self.time.rescale(time_unit)

    def subtract_baseline(
        self,
        window: tuple = (0, 0.1),
        channels: any = None,
        signal_type: any = None,
        rec_type: str = "",
        median: bool = False,
        overwrite: bool = False,
        sweep_subset: any = None,
    ) -> any:
        """
        Subtracts the baseline from the signal within a specified time window.

        Parameters:
        self : object
            The instance of the class containing the signal data.
        window : tuple, optional
            A tuple specifying the start and end of the time window for baseline
            calculation (default is (0, 0.1)).
        channels : any, optional
            The channels to be processed. If None, all channels are processed
            (default is None).
        signal_type : any, optional
            The type of signal to be processed (e.g., 'voltage' or 'current').
            If None, all signal types are processed (default is None).
        rec_type : str, optional
            The type of recording (default is an empty string).
        median : bool, optional
            If True, the median value within the window is used as the baseline.
            If False, the mean value is used (default is False).
        overwrite : bool, optional
            If True, the baseline-subtracted data will overwrite the original data.
            If False, a copy of the data with the baseline subtracted will be
            returned (default is False).

        Returns:
        any
            If overwrite is False, returns a copy of the data with the baseline
            subtracted. If overwrite is True, returns None.
        """
        if not overwrite:
            trace_copy = deepcopy(self)
        else:
            trace_copy = self
        subset_channels = self.subset(
            channels=channels,
            signal_type=signal_type,
            rec_type=rec_type,
            subset_index_only=True,
        )
        trace_copy.set_time(
            align_to_zero=True,
            cumulative=False,
            stimulus_interval=0.0,
            overwrite_time=True,
        )
        window_start_index = _get_time_index(trace_copy.time[0, :], window[0])
        window_end_index = _get_time_index(trace_copy.time[0, :], window[1])
        sweep_subset = _get_sweep_subset(trace_copy.time, sweep_subset)
        for subset_channel in trace_copy.channel:
            if median:
                subset_channel.data.magnitude[
                    sweep_subset, :
                ] = subset_channel.data.magnitude[sweep_subset, :] - np.median(
                    subset_channel.data.magnitude[
                        sweep_subset, window_start_index:window_end_index
                    ],
                    axis=1,
                    keepdims=True,
                )
            else:
                subset_channel.data.magnitude[
                    sweep_subset, :
                ] = subset_channel.data.magnitude[sweep_subset, :] - np.mean(
                    subset_channel.data.magnitude[
                        sweep_subset, window_start_index:window_end_index
                    ],
                    axis=1,
                    keepdims=True,
                )
        # FIXME: remove this section after adjust downstream functions to new format
        for subset_index, signal_type_subset in enumerate(subset_channels.signal_type):
            channel_index = subset_channels.array_index[subset_index]
            if signal_type_subset == "voltage":
                for sweep_index in range(0, trace_copy.voltage.shape[1]):
                    if median:
                        baseline = np.median(
                            trace_copy.voltage[
                                channel_index,
                                sweep_index,
                                window_start_index:window_end_index,
                            ]
                        )
                    else:
                        baseline = np.mean(
                            trace_copy.voltage[
                                channel_index,
                                sweep_index,
                                window_start_index:window_end_index,
                            ]
                        )
                    trace_copy.voltage[channel_index, sweep_index, :] -= baseline
            elif signal_type_subset == "current":
                for sweep_index in range(0, trace_copy.current.shape[1]):
                    if median:
                        baseline = np.median(
                            trace_copy.current[
                                channel_index,
                                sweep_index,
                                window_start_index:window_end_index,
                            ]
                        )
                    else:
                        baseline = np.mean(
                            trace_copy.current[
                                channel_index,
                                sweep_index,
                                window_start_index:window_end_index,
                            ]
                        )
                    trace_copy.current[channel_index, sweep_index, :] -= baseline
        if not overwrite:
            return trace_copy
        return None

    def window_function(
        self,
        window: list = [(0, 0)],
        channels: any = None,
        signal_type: any = None,
        rec_type: str = "",
        function: str = "mean",
        label: str = "",
        sweep_subset: any = None,
        return_output: bool = False,
        plot=False
    ) -> any:
        """
        Apply a specified function to a subset of channels within given time windows.

        Parameters:
        -----------
        window : list, optional
            List of tuples specifying the start and end of each window. Default is [(0, 0)].
        channels : any, optional
            Channels to be included in the subset. Default is None.
        signal_type : any, optional
            Type of signal to be included in the subset. Default is None.
        rec_type : str, optional
            Type of recording to be included in the subset. Default is an empty string.
        function : str, optional
            Function to apply to the data. Supported functions are 'mean', 'median', 'max',
            'min', 'min_avg'. Default is 'mean'.
        return_output : bool, optional
            If True, the function returns the output. Default is False.
        plot : bool, optional
            If True, the function plots the output. Default is False.

        Returns:
        --------
        any
            The output of the applied function if return_output is True, otherwise None.

        Notes:
        ------
        The function updates the `window_summary` attribute of the class with the output.
        """
        if function not in ["mean", "median", "max", "min", "min_avg"]:
            print("Function not supported")
            return None
        sweep_subset = _get_sweep_subset(self.time, sweep_subset)
        subset_channels = self.subset(
            channels=channels, signal_type=signal_type, rec_type=rec_type, sweep_subset=sweep_subset
        )
        output = np.ndarray((len(window), self.time.shape[0]))
        output = FunctionOutput(function)
        for channel_index, channel in enumerate(
            subset_channels.channel_information.channel_number
        ):
            for window_subset in window:
                output.append(
                    trace=subset_channels,
                    window=window_subset,
                    channels=channel,
                    signal_type=subset_channels.channel_information.signal_type[
                        channel_index
                    ],
                    rec_type=subset_channels.channel_information.recording_type[
                        channel_index
                    ],
                    label=label,
                )
        if plot:
            subset_channels.plot(trace=subset_channels, show=True, window_data=output)
        if return_output:
            return output
        if hasattr(self, "window_summary"):
            self.window_summary.merge(output)
        else:
            setattr(self, "window_summary", output)

    def average_trace(
        self,
        channels: any = None,
        signal_type: any = None,
        rec_type: any = "",
        sweep_subset: any = None,
        in_place: bool = True
    ) -> any:
        """
        Calculates the average trace for the given channels, signal_type types, and recording type.

        Parameters:
        - channels (any): The channels to calculate the average trace for.
          If None, uses the first channel type.
        - signal_type (any): The signal_type types to calculate the average trace for.
          Defaults to ['voltage', 'current'].
        - rec_type (any): The recording type to calculate the average trace for.

        Returns:
        - any: The average trace object.

        """
        if channels is None:
            channels = self.channel_information.channel_number
        if signal_type is None:
            signal_type = ["voltage", "current"]
        sweep_subset = _get_sweep_subset(self.time, sweep_subset)
        if in_place:
            avg_trace = self
        else:
            avg_trace = self.copy()
        avg_trace.subset(signal_type=signal_type, rec_type=rec_type, sweep_subset=sweep_subset,in_place=True)
        for channel in avg_trace.channel:
            channel.channel_average()
        # FIXME: remove this section after adjust downstream functions to new format
        if utils.string_match("current", signal_type).any():
            avg_trace.current = avg_trace.current.mean(axis=1)
        if utils.string_match("voltage", signal_type).any():
            avg_trace.voltage = avg_trace.voltage.mean(axis=1)
        # until here
        if in_place:
            return None
        else:
            return avg_trace

    def plot(
        self,
        signal_type: str = None,
        channels: list = None,
        average: bool = False,
        color: str = "black",
        alpha: float = 0.5,
        avg_color: str = "red",
        align_onset: bool = True,
        sweep_subset: any = None,
        window: tuple = (0, 0),
        show: bool = True,
        return_fig: bool = False
    ) -> None | Figure:
        """
        Plots the traces for the specified channels.

        Args:
            signal_type (str): The type of signal_type to use. Must be either 'current' or
            'voltage'.
            channels (list, optional): The list of channels to plot. If None, all channels
            will be plotted. Defaults to None.
            average (bool, optional): Whether to plot the average trace. Defaults to False.
            color (str, optional): The color of the individual traces. Defaults to 'black'.
                                   Can be a colormap.
            alpha (float, optional): The transparency of the individual traces. Defaults to 0.5.
            avg_color (str, optional): The color of the average trace. Defaults to 'red'.
            align_onset (bool, optional): Whether to align the traces on the onset. Defaults to True.
            sweep_subset (any, optional): The subset of sweeps to plot. Defaults to None.
            window (tuple, optional): The time window to plot. Defaults to (0, 0).
            show (bool, optional): Whether to display the plot. Defaults to True.
            return_fig (bool, optional): Whether to return the figure. Defaults to False.

        Returns:
            None or Figure: If show is True, returns None. If return_fig is True,
            returns the figure.
        """
        if channels is None:
            channels = self.channel_information.channel_number
        sweep_subset = _get_sweep_subset(self.time, sweep_subset)
        trace_select = self.subset(channels=channels, signal_type=signal_type, sweep_subset=sweep_subset)
        fig, channel_axs = plt.subplots(
            len(trace_select.channel), 1, sharex=True
        )

        if len(trace_select.channel) == 0:
            print("No traces found.")
            return None

        if align_onset:
            time_array = trace_select.set_time(
                align_to_zero=True,
                cumulative=False,
                stimulus_interval=0.0,
                overwrite_time=False,
            )
        else:
            time_array = trace_select.time
        
        for channel_index, channel in enumerate(trace_select.channel):
            if len(trace_select.channel) == 1:
                tmp_axs = channel_axs
            else:
                tmp_axs = channel_axs[channel_index]
            for i in range(0, channel.data.shape[0]):
                tmp_axs.plot(
                    time_array[i, :],
                    channel.data[i, :],
                    color=utils.trace_color(traces=channel.data, index=i, color=color),
                    alpha=alpha,
                )
            if window != (0, 0):
                tmp_axs.axvspan(xmin=window[0], xmax=window[1], color="gray", alpha=0.1)
            if average:
                channel.channel_average(sweep_subset=sweep_subset, in_place=True)
                tmp_axs.plot(time_array[0, :], channel.average.trace, color=avg_color)
        if show:
            plt.show()
            return None
        if return_fig:
            return fig
        
    def plot_summary(
        self,
        show_trace: bool = True,
        align_onset: bool = True,
        label_filter: list | str = None,
        color="black",
    ) -> None:
        """
        Plots a summary of the experiment data.

        Parameters:
        -----------
        show_trace : bool, optional
            If True, includes the trace in the plot. Default is True.
        align_onset : bool, optional
            If True, aligns the plot on the onset. Default is True.
        label_filter : list or str, optional
            A filter to apply to the labels. Default is None.
        color : str, optional
            The color to use for the trace plot. Default is 'black'.

        Returns:
        --------
        None
        """
        if label_filter is None:
            label_filter = []
        if hasattr(self, "window_summary"):
            if show_trace:
                self.window_summary.plot(
                    trace=self,
                    align_onset=align_onset,
                    show=True,
                    label_filter=label_filter,
                    color=color,
                )
            else:
                self.window_summary.plot(
                    align_onset=align_onset, show=True, label_filter=label_filter
                )
        else:
            print("No summary data found")

class ChannelAverage:
    """
    A class to calculate the average trace from a set of voltage or current traces.

    Attributes:
    -----------
    trace : np.ndarray
        The average trace calculated from the provided data.
    sweeps_used : np.ndarray
        The indices of the sweeps used to calculate the average trace.

    Methods:
    --------
    __init__(trace: VoltageTrace | CurrentTrace, sweep_subset: any = None) -> None
        Initializes the ChannelAverage object and calculates the average trace.

    Parameters:
    -----------
    trace : VoltageTrace | CurrentTrace
        An object containing the voltage or current trace data.
    sweep_subset : any, optional
        A subset of sweeps to use for calculating the average trace. If None, all sweeps are used.

    Raises:
    -------
    ValueError
        If the provided trace data is empty.
    """
    def __init__(self, trace: VoltageTrace | CurrentTrace,
        sweep_subset: any = None) -> None:
        self.trace = None
        self.sweeps_used = None
        if len(trace.data) == 0:
            raise ValueError('No data available to calculate the average trace.')
        sweep_subset = _get_sweep_subset(trace.data, sweep_subset)
        self.trace = np.mean(trace.data[sweep_subset,:], axis=0)
        self.sweeps_used = sweep_subset
        trace.average = self

class FunctionOutput:
    """A class to handle the output of various functions applied to electrophysiological trace data.

    Attributes:
        function_name (str): The name of the function to be applied to the trace data.
        measurements (np.ndarray): An array to store the measurements obtained from the trace data.
        location (np.ndarray): An array to store the locations corresponding to the measurements.
        sweep (np.ndarray): An array to store the sweep indices.
        channel (np.ndarray): An array to store the channel numbers.
        signal_type (np.ndarray): An array to store the types of signals (e.g., current, voltage).
        window (np.ndarray): An array to store the time windows used for measurements.
        label (np.ndarray): An array to store the labels associated with the measurements.
        time (np.ndarray): An array to store the time points corresponding to the measurements.

    Methods:
        __init__(self, function_name: str) -> None:
            Initializes the FunctionOutput object with the given function name.

        append(self, trace: Trace, window: tuple, channels: any = None, signal_type: any = None,
               rec_type: str = '', avg_window_ms: float = 1.0, label: str = '') -> None:
            Appends measurements and related information from the given trace data to the
            FunctionOutput object.

        merge(self, window_summary, remove_duplicates=False) -> None:
            Merges the measurements, location, sweep, window, signal_type, and channel attributes
            from the given window_summary object into the current object. Optionally removes
            duplicates from these attributes after merging.

        label_diff(self, labels: list = [], new_name: str = '', time_label: str = '') -> None:
            Calculates the difference between two sets of measurements and appends the result.

        plot(self, trace: Trace = None, show: bool = True, align_onset: bool = True,
             label_filter: list | str = [], color='black') -> None:
            Plots the trace and/or summary measurements.

        to_dict(self) -> dict:
            Converts the experiment object to a dictionary representation.

        to_dataframe(self) -> pd.DataFrame:
            Converts the experiment object to a pandas DataFrame.
    """

    def __init__(self, function_name: str) -> None:
        self.function_name = function_name
        self.measurements = np.array([])
        self.location = np.array([])
        self.sweep = np.array([])
        self.channel = np.array([])
        self.signal_type = np.array([])
        self.window = np.ndarray(dtype=object, shape=(0, 2))
        self.label = np.array([])
        self.time = np.array([])

    def append(
        self,
        trace: Trace,
        window: tuple,
        channels: any = None,
        signal_type: any = None,
        rec_type: str = "",
        avg_window_ms: float = 1.0,
        label: str = "",
    ) -> None:
        """
        Appends measurements from a given trace within a specified time window.

        Parameters:
        -----------
        trace : Trace
            The trace object containing the data to be analyzed.
        window : tuple
            A tuple specifying the start and end times of the window for measurement.
        channels : any, optional
            The channels to be included in the subset of the trace. Default is None.
        signal_type : any, optional
            The type of signal to be included in the subset of the trace. Default is None.
        rec_type : str, optional
            The recording type to be included in the subset of the trace. Default is an
            empty string.
        avg_window_ms : float, optional
            The averaging window size in milliseconds for the 'min_avg' function. Default
            is 1.0 ms.
        label : str, optional
            A label to be associated with the measurements. Default is an empty string.

        Returns:
        --------
        None
        """
        trace_subset = trace.subset(
            channels=channels, signal_type=signal_type, rec_type=rec_type
        )
        actual_time = deepcopy(trace_subset.time)
        trace_subset.set_time(
            align_to_zero=True,
            cumulative=False,
            stimulus_interval=0.0,
            overwrite_time=True,
        )
        for channel_index, channel in enumerate(
            trace_subset.channel_information.channel_number
        ):
            array_index = trace_subset.channel_information.array_index[channel_index]
            time_window_size = Quantity(avg_window_ms, "ms")
            channel_signal_type = trace_subset.channel_information.signal_type[
                channel_index
            ]
            window_start_index = _get_time_index(trace_subset.time, window[0])
            window_end_index = _get_time_index(trace_subset.time, window[1])
            for sweep_index in range(0, trace_subset.time.shape[0]):
                try:
                    window_start_index = _get_time_index(trace_subset.time, window[0])
                    window_end_index = _get_time_index(trace_subset.time, window[1])
                except:
                    continue
                if channel_signal_type == "current":
                    trace_array = trace_subset.current
                elif channel_signal_type == "voltage":
                    trace_array = trace_subset.voltage
                else:
                    print("Signal type not found")
                    return None
                if self.function_name == "mean":
                    self.measurements = np.append(
                        self.measurements,
                        np.mean(
                            trace_array[
                                array_index,
                                sweep_index,
                                window_start_index[sweep_index] : window_end_index[
                                    sweep_index
                                ],
                            ]
                        ),
                    )
                    self.location = np.append(self.location, np.mean(window))
                elif self.function_name == "median":
                    self.measurements = np.append(
                        self.measurements,
                        np.median(
                            trace_array[
                                array_index,
                                sweep_index,
                                window_start_index[sweep_index] : window_end_index[
                                    sweep_index
                                ],
                            ]
                        ),
                    )
                    self.location = np.append(self.location, np.mean(window))
                elif self.function_name == "max":
                    self.measurements = np.append(
                        self.measurements,
                        np.max(
                            trace_array[
                                array_index,
                                sweep_index,
                                window_start_index[sweep_index] : window_end_index[
                                    sweep_index
                                ],
                            ]
                        ),
                    )
                    self.location = np.append(
                        self.location,
                        trace_subset.time[
                            sweep_index,
                            np.argmin(
                                trace_array[
                                    array_index,
                                    sweep_index,
                                    window_start_index[sweep_index] : window_end_index[
                                        sweep_index
                                    ],
                                ]
                            )
                            + window_start_index[sweep_index],
                        ],
                    )
                elif self.function_name == "min":
                    self.measurements = np.append(
                        self.measurements,
                        np.min(
                            trace_array[
                                array_index,
                                sweep_index,
                                window_start_index[sweep_index] : window_end_index[
                                    sweep_index
                                ],
                            ]
                        ),
                    )
                    self.location = np.append(
                        self.location,
                        trace_subset.time[
                            sweep_index,
                            np.argmin(
                                trace_array[
                                    array_index,
                                    sweep_index,
                                    window_start_index[sweep_index] : window_end_index[
                                        sweep_index
                                    ],
                                ]
                            )
                            + window_start_index[sweep_index],
                        ],
                    )
                elif self.function_name == "min_avg":
                    min_time = trace_subset.time[
                        sweep_index,
                        np.argmin(
                            trace_array[
                                array_index,
                                sweep_index,
                                window_start_index[sweep_index] : window_end_index[
                                    sweep_index
                                ],
                            ]
                        )
                        + window_start_index[sweep_index],
                    ]
                    window_start_index[sweep_index] = _get_time_index(
                        trace_subset.time[sweep_index, :],
                        Quantity(
                            min_time - time_window_size, trace_subset.time.units
                        ).magnitude,
                    )
                    window_end_index[sweep_index] = _get_time_index(
                        trace_subset.time[sweep_index, :],
                        Quantity(
                            min_time + time_window_size, trace_subset.time.units
                        ).magnitude,
                    )
                    self.location = np.append(self.location, min_time)
                    self.measurements = np.append(
                        self.measurements,
                        np.mean(
                            trace_array[
                                array_index,
                                sweep_index,
                                window_start_index[sweep_index] : window_end_index[
                                    sweep_index
                                ],
                            ]
                        ),
                    )
                self.sweep = np.append(self.sweep, sweep_index + 1)
                self.window = np.append(self.window, [window], axis=0)
                self.signal_type = np.append(self.signal_type, channel_signal_type)
                self.channel = np.append(self.channel, channel)
                self.label = np.append(self.label, label)
                self.time = np.append(
                    self.time,
                    actual_time[
                        sweep_index,
                        _get_time_index(
                            trace_subset.time[sweep_index], self.location[-1]
                        ),
                    ],
                )

    def merge(self, window_summary, remove_duplicates=False) -> None:
        """
        Merges the measurements, location, sweep, window, signal_type, and channel attributes
        from the given window_summary object into the current object. Optionally removes duplicates
        from these attributes after merging.

        Args:
            window_summary (object): An object containing measurements, location, sweep, window,
                         signal_type, and channel attributes to be merged.
            remove_duplicates (bool, optional): If True, removes duplicate entries from the merged
                            attributes. Defaults to True.

        Returns:
            None
        """
        self.measurements = np.append(self.measurements, window_summary.measurements)
        self.location = np.append(self.location, window_summary.location)
        self.sweep = np.append(self.sweep, window_summary.sweep)
        self.window = np.append(self.window, window_summary.window)
        self.signal_type = np.append(self.signal_type, window_summary.signal_type)
        self.channel = np.append(self.channel, window_summary.channel)
        self.label = np.append(self.label, window_summary.label)
        self.time = np.append(self.time, window_summary.time)
        if remove_duplicates:
            np.unique(self.measurements)
            self.measurements = np.unique(self.measurements)
            self.location = np.unique(self.location)
            self.sweep = np.unique(self.sweep)
            self.window = np.unique(self.window)
            self.signal_type = np.unique(self.signal_type)
            self.channel = np.unique(self.channel)
            self.label = np.unique(self.label)

    def label_diff(
        self, labels: list = None, new_name: str = "", time_label: str = ""
    ) -> None:
        """
        Calculate the difference between two sets of measurements and append the result.

        Parameters:
        labels (list): Labels whose measurements will be used to calculate the difference.
        new_name (str): Label name for the new set of measurements.
        time_label (str): Label to identify the time points for the new measurements.

        Returns:
        None
        """
        if labels is None:
            labels = []
        unique_labels = np.unique(self.label)
        if not all(label in unique_labels for label in labels):
            print("Labels not found in data")
            return None
        label_index_1 = np.where(self.label == labels[0])
        label_index_2 = np.where(self.label == labels[1])
        time_label_index = np.where(self.label == time_label)
        diff = self.measurements[label_index_1] - self.measurements[label_index_2]
        self.measurements = np.append(self.measurements, diff)
        self.location = np.append(self.location, self.location[time_label_index])
        self.sweep = np.append(self.sweep, self.sweep[time_label_index])
        self.window = np.append(self.window, self.window[time_label_index])
        self.signal_type = np.append(
            self.signal_type, self.signal_type[time_label_index]
        )
        # self.channel = np.append(self.channel, self.channel[time_label_index])
        self.label = np.append(
            self.label, np.repeat(new_name, len(time_label_index[0]))
        )
        self.time = np.append(self.time, self.time[time_label_index])

    def plot(
        self,
        trace: Trace = None,
        show: bool = True,
        align_onset: bool = True,
        label_filter: list | str = None,
        color="black",
    ) -> None:
        """
        Plots the trace and/or summary measurements.

        Parameters:
        trace (Trace, optional): The trace object to be plotted. If None, only the summary
                                 measurements are plotted. Default is None.
        show (bool, optional): If True, the plot will be displayed. Default is True.
        summary_only (bool, optional): If True, only the summary measurements will be
                                      plotted. Default is True.

        Returns:
        None
        """
        from ephys.classes.class_functions import moving_average  # pylint: disable=C

        if label_filter is None:
            label_filter = []
        if trace is not None:
            self.channel = np.unique(np.array(self.channel))
            trace_select = trace.subset(
                channels=self.channel, signal_type=self.signal_type
            )
            trace_select.plot(show=False, align_onset=align_onset, color=color)
        # TODO: make sure to plot dots on right channel
        unique_labels = np.unique(self.label)
        if align_onset:
            x_axis = self.location
        else:
            x_axis = self.time
        for color_index, label in enumerate(unique_labels):
            if len(label_filter) > 0:
                if label not in label_filter:
                    continue
            label_idx = np.where(self.label == label)
            label_colors = utils.color_picker(
                length=len(unique_labels), index=color_index, color="gist_rainbow"
            )
            if not align_onset:
                y_smooth = moving_average(
                    self.measurements[label_idx], len(label_idx[0]) // 10
                )
                plt.plot(
                    x_axis[label_idx], y_smooth, color=label_colors, alpha=0.4, lw=2
                )

            plt.plot(
                x_axis[label_idx],
                self.measurements[label_idx],
                "o",
                color=label_colors,
                alpha=0.5,
                label=label,
            )
        # plt.xlabel('Time (' + self.time.dimensionality.latex + ')')
        plt.legend(loc="upper left")

        if show:
            plt.show()

    def to_dict(self):
        """
        Convert the experiment object to a dictionary representation.

        Returns:
            dict: A dictionary containing the following keys:
                - 'measurements': The measurements associated with the experiment.
                - 'location': The location of the experiment.
                - 'sweep': The sweep information of the experiment.
                - 'window': The window information of the experiment.
                - 'signal_type': The type of signal used in the experiment.
                - 'channel': The channel information of the experiment.
        """
        return {
            "measurements": self.measurements,
            "location": self.location,
            "sweep": self.sweep,
            "window": self.window,
            "signal_type": self.signal_type,
            "channel": self.channel,
            "label": self.label,
            "time": self.time,
        }

    def to_dataframe(self):
        """
        Convert the experiment object to a pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing the measurements, location,
            sweep, window, signal_type, and channel information.
        """
        return pd.DataFrame(self.to_dict())


class Events:
    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.events = []
        self.event_times = []


class MetaData:
    """
    A class representing metadata for experiment files.

    Args:
        file_path (str | list): The path(s) of the file(s) to be added.
        experimenter (str | list, optional): The name(s) of the experimenter(s).
                                             Defaults to 'unknown'.

    Attributes:
        file_info (numpy.ndarray): An array containing information about the file(s).
        experiment_info (numpy.ndarray): An array containing information about the experiment(s).

    Methods:
        __init__(self, file_path: str | list, experimenter: str | list = 'unknown') -> None:
            Initializes the MetaData object.

        add_file_info(self, file_path: str | list, experimenter: str | list = 'unknown',
                      add: bool = True) -> None:
            Adds file information to the MetaData object.

        remove_file_info(self, file_path: str | list) -> None:
            Removes file information from the MetaData object.
    """

    def __init__(
        self,
        file_path: str | list,
        experimenter: str | list = "unknown",
    ) -> None:
        self.file_info = None
        self.add_file_info(file_path, experimenter, add=False)

    def add_file_info(
        self,
        file_path: str | list,
        experimenter: str | list = "unknown",
        add: bool = True,
    ) -> None:
        """
        Adds file information to the MetaData object.

        Args:
            file_path (str | list): The path(s) of the file(s) to be added.
            experimenter (str | list, optional): The name(s) of the experimenter(s).
                                                 Defaults to 'unknown'.
            add (bool, optional): Whether to append the information to existing data.
                                  Defaults to True.
        """
        if isinstance(file_path, str):
            file_path = [file_path]
        file_list = []
        experiment_list = []
        for file in file_path:
            time_created = time.ctime(os.path.getctime(file))
            time_modified = time.ctime(os.path.getmtime(file))
            estimated_exp_date = (
                time_created if time_created < time_modified else time_modified
            )
            # NOTE: abf files have date of experiment in the header
            file_list.append(
                {
                    "data_of_creation": time_created,
                    "last_modified": time_modified,
                    "file_name": os.path.basename(file),
                    "file_path": file,
                }
            )
            experiment_list.append(
                {"date_of_experiment": estimated_exp_date, "experimenter": experimenter}
            )
            print("Date of Experiment estimated. Please check for correct date.")
        if add:
            self.file_info = np.append(self.file_info.tolist(), file_list)
            self.experiment_info = np.append(
                self.experiment_info.tolist(), experiment_list
            )
        else:
            self.file_info = np.array(file_list)
            self.experiment_info = np.array(experiment_list)

    def remove_file_info(self, file_path: str | list) -> None:
        """
        Removes file information from the MetaData object.

        Args:
            file_path (str | list): The path(s) of the file(s) to be removed.
        """
        if isinstance(file_path, str):
            file_path = [file_path]
        files_to_remove = [os.path.basename(file) for file in file_path]
        files_exist = [
            entry["file_name"] for entry in self.file_info if isinstance(entry, dict)
        ]
        self.file_info = self.file_info[
            utils.string_match(files_to_remove, files_exist)
        ]


class ExpData:
    """
    A class representing experimental data.

    Args:
        file_path (str | list): The path(s) to the file(s) containing the data.
        experimenter (str, optional): The name of the experimenter. Defaults to 'unknown'.

    Attributes:
        protocols (list): A list of Trace objects representing the protocols.
        file_info: Information about the file.
        meta_data: An instance of the MetaData class.

    """

    def __init__(self, file_path: str | list, experimenter: str = "unknown") -> None:
        self.protocols = []
        self.meta_data = []
        self.file_info = []
        if isinstance(file_path, str):
            self.protocols.append(Trace(file_path))
        elif isinstance(file_path, list):
            for file in file_path:
                self.protocols.append(Trace(file))
        self.meta_data = MetaData(file_path, experimenter)

    # TODO: function to add different protocols to the same experiment
    # TODO: get summary outputs for the experiment
    # TODO: get summary plots for the experiment
