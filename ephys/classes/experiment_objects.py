"""
This module provides classes for representing experimental data and metadata.
"""
import os
import time
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from quantities import Quantity
from ephys.classes.class_functions import wcp_trace
from ephys.classes.class_functions import _get_time_index
from ephys import utils


class Trace:
    """
    Represents a trace object.

    Args:
        file_path (str): The file path of the trace.

    Attributes:
        file_path (str): The file path of the trace.

    Methods:
        self_init() -> any:
            Returns a deep copy of the Trace object.

        subset(channels: any = all channels, can be a list,
               signal_type: any = "voltage" and "current",
               rec_type: any = all rec_types) -> any:
            Returns a subset of the Trace object based on the specified channels, signal_type, and
            rec_type.

        average_trace(channels: any = all channels, can be a list,
                      signal_type: any = "voltage" and "current", can be a list,
                      rec_type: any = all rec_types) -> any:
            Returns the average trace of the Trace object based on the specified channels,
            signal_type, and rec_type.

        plot(signal_type: str, channels: list, average: bool = False, color: str ='k',
            alpha: float = 0.5, avg_color: str = 'r'):
            Plots the trace data based on the specified signal_type, channels, and other optional
            parameters.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.voltage = None
        self.current = None
        self.time = None
        self.sampling_rate = None
        self.channel_information = np.asarray(
            [np.array([]), np.array([]), 0, 0, np.array([])], dtype=object
        )
        self.channel_type = None
        if file_path.endswith(".wcp"):
            wcp_trace(self, file_path)
        else:
            print("File type not supported")

    def _self_init(self) -> any:
        return deepcopy(self)

    def subset(
        self,
        channels: any = None,
        signal_type: any = None,
        rec_type: any = "",
        clamp_type: any = None,
        channel_groups: any = None,
        subset_index_only: bool = False
    ) -> any:
        """
        Subset the experiment object based on specified criteria.

        Args:
            channels (any, optional): Channels to include in the subset.
                                      Defaults to all channels.
            signal_type (any, optional): Types of signal_type to include in the subset.
                                   Defaults to ["voltage", "current"].
            rec_type (any, optional): Recording types to include in the subset. Defaults to "".

        Returns:
            any: Subset of the experiment object.

        """
        subset_trace = self._self_init()
        rec_type_get = utils.string_match(rec_type, self.channel_type["recording_type"])
        if clamp_type is None:
            clamp_type = np.array([True, False])
        clamp_type_get = np.isin(np.array(self.channel_type["clamped"]), np.array(clamp_type))
        if channel_groups is None:
            channel_groups = np.array(self.channel_type["channel_grouping"])
        channel_groups_get = np.isin(np.array(self.channel_type["channel_grouping"]), np.array(channel_groups))
        if signal_type is None:
            signal_type = ["voltage", "current"]
        signal_type_get = utils.string_match(signal_type, self.channel_type["recording_configuration"])
        if isinstance(channels, int):
            channels = [channels]
        if channels is None:
            channels = np.array(self.channel_type["channel_number"])
        else:
            channels = np.array(channels)
        channels_get = np.isin(np.array(self.channel_type["channel_number"]), channels)
        combined_index = (
            np.array(self.channel_type["channel_number"])[
                np.logical_and.reduce((
                    rec_type_get,
                    signal_type_get,
                    channels_get,
                    clamp_type_get,
                    channel_groups_get
                ))
            ]
            - 1
        )
        voltage_selection = np.isin(self.channel_information[0][0], combined_index)
        current_selection = np.isin(self.channel_information[0][1], combined_index)
        if len(voltage_selection) > 0:
            subset_trace.voltage = self.voltage[voltage_selection]
        else:
            subset_trace.voltage = np.zeros(self.voltage.shape)
        if len(current_selection) > 0:
            subset_trace.current = self.current[current_selection]
        else:
            subset_trace.current = np.zeros(self.current.shape)
        if len(combined_index) > 0:
            recording_configuration =  np.array(self.channel_type["recording_configuration"])[combined_index].tolist()
            voltage_index = 0
            current_index = 0
            array_index = []
            for type_test in recording_configuration:
                if type_test == "voltage":
                    array_index.append(voltage_index)
                    voltage_index += 1
                elif type_test == "current":
                    array_index.append(current_index)
                    current_index += 1
            subset_trace.channel_type = {
                "channel_number": np.array(self.channel_type["channel_number"])[combined_index].tolist(),
                "array_index": array_index,
                "recording_type": np.array(self.channel_type["recording_type"])[combined_index].tolist(),
                "recording_configuration": recording_configuration,
                "clamped": np.array(self.channel_type["clamped"])[combined_index].tolist(),
                "channel_grouping": np.array(self.channel_type["channel_grouping"])[combined_index].tolist(),
                "unit": np.array(self.channel_type["unit"])[combined_index].tolist()
                }
        else:
            subset_trace.channel_type = {"channel_number": [],
                                         "array_index": [],
                                         "recording_type": [],
                                         "recording_configuration": [],
                                         "clamped": [],
                                         "channel_grouping": [],
                                         "unit": []}
        if subset_index_only:
            return subset_trace.channel_type
        return subset_trace
    
    def set_time(self,
                 align_to_zero: bool = True,
                 cumulative: bool = False,
                 stimulus_interval: float = 0.0,
                 overwrite_time: bool = True) -> any:
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
        sampling_interval = (1/self.sampling_rate).rescale(time_unit)
        
        for sweep_index, sweep in enumerate(tmp_time):
            if align_to_zero:
                start_time = Quantity(np.min(sweep.magnitude), time_unit)
            if cumulative:
                if sweep_index > 0:
                    start_time = Quantity(np.min(sweep.magnitude) - np.max(tmp_time[sweep_index-1].magnitude), time_unit) - stimulus_interval - sampling_interval
            tmp_time[sweep_index] -= start_time
        if overwrite_time:
            self.time = tmp_time
            return None
        else:
            return tmp_time
    def rescale_time(self,
                     time_unit: str = "s") -> None:
        """
        Rescale the time axis for the given trace data.

        Parameters:
        - trace_data (Trace): The trace data object.
        - time_unit (str): The time unit. Default is "s".

        Returns:
        - None
        """
        self.time = self.time.rescale(time_unit)

    def subtract_baseline(self,
                          window: tuple = (0, 0.1),
                          channels: any = None,
                          signal_type: any = None,
                          rec_type: str = "",
                          median: bool = False,
                          overwrite: bool = False) -> any:
        if not overwrite:
            trace_copy = deepcopy(self)
        else:
            trace_copy = self
        subset_channels = self.subset(channels=channels,signal_type=signal_type, rec_type=rec_type, subset_index_only=True)
        trace_copy.set_time(align_to_zero=True, cumulative=False, stimulus_interval=0.0, overwrite_time=True)
        window_start_index = _get_time_index(trace_copy.time[0,:], window[0])
        window_end_index = _get_time_index(trace_copy.time[0,:], window[1])
        for subset_index, signal_type_subset in enumerate(subset_channels[2]):
            channel_index = subset_channels[4][subset_index]
            if signal_type_subset == "voltage":
                for sweep_index in range(0, trace_copy.voltage.shape[1]):
                    if median:
                        baseline = np.median(trace_copy.voltage[channel_index, sweep_index, window_start_index:window_end_index])
                    else:
                        baseline = np.mean(trace_copy.voltage[channel_index, sweep_index, window_start_index:window_end_index])
                    trace_copy.voltage[channel_index, sweep_index, :] -= baseline
            elif signal_type_subset == "current":
                for sweep_index in range(0, trace_copy.current.shape[1]):
                    if median:
                        baseline = np.median(trace_copy.voltage[channel_index, sweep_index, window_start_index:window_end_index])
                    else:
                        baseline = np.mean(trace_copy.voltage[channel_index, sweep_index, window_start_index:window_end_index])
                    trace_copy.current[channel_index, sweep_index, :] -= baseline
        if not overwrite:
            return trace_copy

    def window_function(self,
                        window: list = [(0, 0)],
                        channels: any = None,
                        signal_type: any = None,
                        rec_type: str = "",
                        function: str = "mean",
                        plot=False) -> any:
        if not function in ["mean", "median", "max", "min", "min_avg"]:
            print("Function not supported")
            return None
        subset_channels = self.subset(channels=channels, signal_type=signal_type, rec_type=rec_type)
        output = np.ndarray((len(window), self.time.shape[0]))
        output = FunctionOutput(function)          
        for i, window_subset in enumerate(window):
            for subset_index, signal_type_subset in enumerate(subset_channels.channel_type["recording_configuration"]):
                # BUG: channel_index is incorrect, has to be added to dictionary and during subset
                channel_index = subset_channels.channel_type["channel_grouping"][subset_index]
                output.append(trace=subset_channels, window=window_subset, channels=channel_index, signal_type=signal_type, rec_type=rec_type)
        if plot:
            subset_channels.plot(show=False)
            for window_subset in window:
                plt.axvspan(xmin=window_subset[0], xmax=window_subset[1], color='gray', alpha=0.1)
                plt.plot(output.location, output.measurements, "o", color="orange")
            plt.show()
        return output

    def average_trace(
        self,
        channels: any = None,
        signal_type: any = None,
        rec_type: any = "",
    ) -> any:
        """
        Calculates the average trace for the given channels, signal_type types, and recording type.

        Parameters:
        - channels (any): The channels to calculate the average trace for.
          If None, uses the first channel type.
        - signal_type (any): The signal_type types to calculate the average trace for.
          Defaults to ["voltage", "current"].
        - rec_type (any): The recording type to calculate the average trace for.

        Returns:
        - any: The average trace object.

        """
        if channels is None:
            channels = np.array(self.channel_type["channel_number"])
        if signal_type is None:
            signal_type = ["voltage", "current"]
        avg_trace = self.subset(channels, signal_type, rec_type)
        if utils.string_match("current", signal_type).any():
            avg_trace.current = avg_trace.current.mean(axis=1)
        if utils.string_match("voltage", signal_type).any():
            avg_trace.voltage = avg_trace.voltage.mean(axis=1)
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
        window: tuple = (0, 0),
        show: bool = True
    ):
        """
        Plots the traces for the specified channels.

        Args:
            signal_type (str): The type of signal_type to use. Must be either "current" or
            "voltage".
            channels (list, optional): The list of channels to plot. If None, all channels
            will be plotted. Defaults to None.
            average (bool, optional): Whether to plot the average trace. Defaults to False.
            color (str, optional): The color of the individual traces. Defaults to "black".
                                   Can be a colormap.
            alpha (float, optional): The transparency of the individual traces. Defaults to 0.5.
            avg_color (str, optional): The color of the average trace. Defaults to "red".
        """
        if channels is None:
            channels = np.array(self.channel_type["channel_number"])
        fig, channel_axs = plt.subplots(len(channels), 1, sharex=True)
        if signal_type is None:
            #channel_type_subset = self.subset(
            #    channels=channels
            #).channel_type
            #signal_type = channel_type_subset[2][np.array(channel_type_subset[3]) == False]
            signal_type = "voltage"
        trace_select = trace_select = self.subset(
                channels=channels, signal_type=signal_type
            )
        if signal_type == "current":
            trace_signal = trace_select.current
        if signal_type == "voltage":
            trace_signal = trace_select.voltage
        if trace_signal.shape[0] == 0:
            print("No traces found.")
            return None
        if window != (0, 0):
            plt.axvspan(xmin=window[0], xmax=window[1], color='gray', alpha=0.1)
        if align_onset:
            time_array = self.set_time(align_to_zero=True, cumulative=False,stimulus_interval=0.0, overwrite_time=False)
        else:
            time_array = self.time
            trace_select.channel_type
        for index, array_index in enumerate(trace_select.channel_type["channel_number"]):
            for i in range(0, trace_signal.shape[1]):
                #  if(signal_type == 'current'):
                channel_axs[index].plot(
                    time_array[i, :],
                    trace_signal[array_index, i, :],
                    color=utils.trace_color(traces=trace_signal, index=i, color=color),
                    alpha=alpha
                )
            if average:
                trace_select_avg = self.average_trace(
                    channels=channels, signal_type=signal_type
                )
                if signal_type == "current":
                    channel_axs[index].plot(time_array[0, :], trace_select_avg.current[0], color=avg_color)
                if signal_type == "voltage":
                    channel_axs[index].plot(time_array[0, :], trace_select_avg.voltage[0], color=avg_color)
            channel_axs[index].set_xlabel("Time (" + time_array.dimensionality.latex + ")")
       # print(trace_select.channel_type[5])
            for unit_title in  np.unique(trace_select.channel_type["unit"]):
                channel_axs[index].set_ylabel(signal_type.title() + " (" + unit_title + ")")
        if show:
            plt.show()

class FunctionOutput:
    def __init__(self,
                 function_name: str) -> None:
        self.function_name = function_name
        self.measurements = []
        self.location = []
        self.sweep = []
        self.window = []
    def append(self,
               trace: Trace,
               window: tuple,
               channels: any = None,
               signal_type: any = None,
               rec_type: str = "",
               avg_window_ms: float = 1.0) -> None:
        for channel, channel_index in enumerate(trace_subset.channel_type["channel_number"]):
            trace_subset = trace.subset(channels=channels+1, signal_type=signal_type, rec_type=rec_type)
            time_window_size = Quantity(avg_window_ms, "ms")
            window_start_index = _get_time_index(trace_subset.time, window[0])
            window_end_index = _get_time_index(trace_subset.time, window[1])
            for sweep_index in range(0, trace_subset.time.shape[0]):
                try:
                    window_start_index = _get_time_index(trace_subset.time, window[0])
                    window_end_index = _get_time_index(trace_subset.time, window[1])
                except:
                    continue
                if signal_type == "current":
                    trace_array = trace_subset.current
                elif signal_type == "voltage":
                    trace_array = trace_subset.voltage
                else:
                    print("Signal type missing: default to voltage")
                    trace_array = trace_subset.voltage

                if self.function_name == "mean":
                        self.measurements.append(np.mean(trace_array[:,sweep_index,window_start_index:window_end_index]))
                        self.location.append(np.mean(window))
                        self.sweep.append(sweep_index+1)
                        self.window.append(window)
                elif self.function_name == "median":
                        self.measurements.append(np.median(trace_array[:,sweep_index,window_start_index:window_end_index]))
                        self.location.append(np.mean(window))
                        self.sweep.append(sweep_index+1)
                        self.window.append(window)
                elif self.function_name == "max":
                        self.measurements.append(np.max(trace_array[:,sweep_index,window_start_index:window_end_index]))
                        self.location.append(trace_subset.time[sweep_index,np.argmin(trace_array[:,sweep_index,window_start_index:window_end_index])+window_start_index])
                        self.sweep.append(sweep_index+1)
                        self.window.append(window)
                elif self.function_name == "min":
                        self.measurements.append(np.min(trace_array[:,sweep_index,window_start_index:window_end_index]))
                        self.location.append(trace_subset.time[sweep_index,np.argmin(trace_array[:,sweep_index,window_start_index:window_end_index])+window_start_index])
                        self.sweep.append(sweep_index+1)
                        self.window.append(window)
                elif self.function_name == "min_avg":
                        min_time = trace_subset.time[:,sweep_index,np.argmin(trace_array[:,sweep_index,window_start_index:window_end_index])+window_start_index]
                        window_start_index = _get_time_index(trace_subset.time[sweep_index,:], (min_time-time_window_size).magnitude)
                        window_end_index = _get_time_index(trace_subset.time[sweep_index,:], (min_time+time_window_size).magnitude)
                        self.location.append(min_time)
                        self.measurements.append(np.mean(trace_array[:,sweep_index,window_start_index:window_end_index]))
                        self.sweep.append(sweep_index+1)
                        self.window.append(window)
            
    def plot(self, show=False):
        plt.plot(self.location, self.measurements, "o")
        if show:
            plt.show()

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
                                             Defaults to "unknown".

    Attributes:
        file_info (numpy.ndarray): An array containing information about the file(s).
        experiment_info (numpy.ndarray): An array containing information about the experiment(s).

    Methods:
        __init__(self, file_path: str | list, experimenter: str | list = "unknown") -> None:
            Initializes the MetaData object.

        add_file_info(self, file_path: str | list, experimenter: str | list = "unknown",
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
                                                 Defaults to "unknown".
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
        experimenter (str, optional): The name of the experimenter. Defaults to "unknown".

    Attributes:
        protocols (list): A list of Trace objects representing the protocols.
        file_info: Information about the file.
        meta_data: An instance of the MetaData class.

    """

    def __init__(self, file_path: str | list, experimenter: str = "unknown") -> None:
        self.protocols = []
        self.file_info = None
        if isinstance(file_path, str):
            self.protocols.append(Trace(file_path))
        elif isinstance(file_path, list):
            for file in file_path:
                self.protocols.append(Trace(file))
        self.meta_data = MetaData(file_path, experimenter)
    # TODO: function to add different protocols to the same experiment
    # TODO: get summary outputs for the experiment
    # TODO: get summary plots for the experiment

