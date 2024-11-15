"""
This module provides classes for representing experimental data and metadata.
"""
import os
import time
import neo
from copy import deepcopy
from re import findall
import numpy as np
import matplotlib.pyplot as plt
from quantities import Quantity
from ephys.classes.class_functions import wcp_trace
from ephys.classes.class_functions import _get_time_index
from ephys.classes.class_functions import _is_clamp
from ephys import utils

class ChannelInformation:
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
            Analogsignals = data.read_block().segments[0].analogsignals
            for i in data.header["signal_channels"]:
                if len(findall("Vm\(AC\)", i[0])) == 1:
                    type_out.append("field")
                    signal_type.append("voltage")
                    array_index.append(voltage_index)
                    voltage_index += 1
                elif len(findall("Vm", i[0])) == 1:
                    type_out.append("cell")
                    signal_type.append("voltage")
                    array_index.append(voltage_index)
                    voltage_index += 1
                elif len(findall("Im", i[0])) == 1:
                    type_out.append("cell")
                    signal_type.append("current")
                    array_index.append(current_index)
                    current_index += 1
                channel_groups.append(i["stream_id"].astype(int).tolist())
                clamp_type.append(
                    _is_clamp(Analogsignals[channel_index].magnitude.squeeze())
                )
                channel_index += 1
                channel_list.append(channel_index)
                channel_unit.append(str(i["units"]))
        elif isinstance(data, neo.io.abfio.ABFIO):
            pass
        elif isinstance(data, neo.io.igorproio.IgorProIO):
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

    def to_dict(self):
        return {
            "channel_number": self.channel_number,
            "array_index": self.array_index,
            "recording_type": self.recording_type,
            "signal_type": self.signal_type,
            "clamped": self.clamped,
            "channel_grouping": self.channel_grouping,
            "unit": self.unit,
        }
    
    def count(self) -> None:
        signal_type, signal_type_count = np.unique(self.signal_type, return_counts=True)
        array_index, array_index_count = np.unique(self.array_index, return_counts=True)
        recording_type, recording_type_count = np.unique(self.recording_type, return_counts=True)
        clamped, clamped_count = np.unique(self.clamped, return_counts=True)
        channel_grouping, channel_grouping_count = np.unique(self.channel_grouping, return_counts=True)
        unit, unit_idx = np.unique(self.unit, return_counts=True)
        return {
            "signal_type": dict(zip(signal_type, signal_type_count)),
            "array_index": dict(zip(array_index, array_index_count)),
            "recording_type": dict(zip(recording_type, recording_type_count)),
            "clamped": dict(zip(clamped, clamped_count)),
            "channel_grouping": dict(zip(channel_grouping, channel_grouping_count)),
            "unit": dict(zip(unit, unit_idx)),
        }


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
        self.channel_information = None
        if file_path.endswith(".wcp"):
            wcp_trace(self, file_path)
        else:
            print("File type not supported")

    def copy(self) -> any:
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
        subset_trace = self.copy()
        rec_type_get = utils.string_match(rec_type, self.channel_information.recording_type)
        if clamp_type is None:
            clamp_type = np.array([True, False])
        clamp_type_get = np.isin(self.channel_information.clamped, np.array(clamp_type))
        if channel_groups is None:
            channel_groups = self.channel_information.channel_grouping
        channel_groups_get = np.isin(self.channel_information.channel_grouping, channel_groups)
        if signal_type is None:
            signal_type = np.array(["voltage", "current"])
        signal_type_get = utils.string_match(signal_type, self.channel_information.signal_type)
        if isinstance(channels, int):
            channels = np.array([channels])
        if channels is None:
            channels = self.channel_information.channel_number
        else:
            channels = np.array(channels)
        channels_get = np.isin(self.channel_information.channel_number, channels)
        # TODO: switch to boolean indexing and np.array inside dict as default
        combined_index = np.logical_and.reduce((
            rec_type_get,
            signal_type_get,
            channels_get,
            clamp_type_get,
            channel_groups_get
            ))
        voltage_selection = self.channel_information.array_index[(self.channel_information.signal_type=="voltage") & combined_index]
        current_selection = self.channel_information.array_index[(self.channel_information.signal_type=="current") & combined_index]
        if len(voltage_selection) > 0:
            subset_trace.voltage = self.voltage[voltage_selection]
        else:
            subset_trace.voltage = np.zeros(self.voltage.shape)
        if len(current_selection) > 0:
            subset_trace.current = self.current[current_selection]
        else:
            subset_trace.current = np.zeros(self.current.shape)
        if len(combined_index) > 0:
            signal_type =  self.channel_information.signal_type[combined_index]
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
            subset_trace.channel_information.channel_number = self.channel_information.channel_number[combined_index]
            subset_trace.channel_information.array_index = self.channel_information.array_index[combined_index]
            subset_trace.channel_information.recording_type = self.channel_information.recording_type[combined_index]
            subset_trace.channel_information.signal_type = signal_type
            subset_trace.channel_information.clamped = self.channel_information.clamped[combined_index]
            subset_trace.channel_information.channel_grouping = self.channel_information.channel_grouping[combined_index]
            subset_trace.channel_information.unit = self.channel_information.unit[combined_index]
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
        for subset_index, signal_type_subset in enumerate(subset_channels["signal_type"]):
            channel_index = subset_channels["array_index"][subset_index]
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
                        baseline = np.median(trace_copy.current[channel_index, sweep_index, window_start_index:window_end_index])
                    else:
                        baseline = np.mean(trace_copy.current[channel_index, sweep_index, window_start_index:window_end_index])
                    trace_copy.current[channel_index, sweep_index, :] -= baseline
        if not overwrite:
            return trace_copy

    def window_function(self,
                        window: list = [(0, 0)],
                        channels: any = None,
                        signal_type: any = None,
                        rec_type: str = "",
                        function: str = "mean",
                        return_output: bool = False,
                        plot=False) -> any:
        if not function in ["mean", "median", "max", "min", "min_avg"]:
            print("Function not supported")
            return None
        subset_channels = self.subset(channels=channels, signal_type=signal_type, rec_type=rec_type)
        output = np.ndarray((len(window), self.time.shape[0]))
        output = FunctionOutput(function)
        for channel_index, channel in enumerate(subset_channels.channel_information.channel_number):
            for window_index, window_subset in enumerate(window):
                output.append(trace=subset_channels, window=window_subset,
                              channels=channel,
                              signal_type=subset_channels.channel_information.signal_type[channel_index],
                              rec_type=subset_channels.channel_information.recording_type[channel_index])
        if plot:
            subset_channels.plot(show=True, window_data=output)
        if return_output:
            return output
        try:
            self.window_summary = self.window_summary.merge(output)
        except:
                setattr(self, "window_summary", output)

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
            channels = self.channel_information.channel_number
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
        show: bool = True,
        return_fig: bool = False
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
            channels = self.channel_information.channel_number
        trace_select = self.subset(
                channels=channels, signal_type=signal_type
            )
        fig, channel_axs = plt.subplots(len(trace_select.channel_information.channel_number), 1, sharex=True)
        if (trace_select.voltage.shape[0] == 0) and (trace_select.current.shape[0] == 0):
            print("No traces found.")
            return None
        if window != (0, 0):
            plt.axvspan(xmin=window[0], xmax=window[1], color='gray', alpha=0.1)
        if align_onset:
            time_array = self.set_time(align_to_zero=True, cumulative=False,stimulus_interval=0.0, overwrite_time=False)
        else:
            time_array = self.time
        for index, array_index in enumerate(trace_select.channel_information.array_index):
            # NOTE: add here all the windows for the channel if window is existing
            if trace_select.channel_information.signal_type[index]=="voltage":
                trace_signal = trace_select.voltage
            if trace_select.channel_information.signal_type[index]=="current":
                trace_signal = trace_select.current
            if len(trace_select.channel_information.array_index) == 1:
                tmp_axs = channel_axs
            else:
                tmp_axs = channel_axs[index]
            for i in range(0, trace_signal.shape[1]):
                tmp_axs.plot(
                    time_array[i, :],
                    trace_signal[array_index, i, :],
                    color=utils.trace_color(traces=trace_signal, index=i, color=color),
                    alpha=alpha
                )
            if average:
                trace_select_avg = self.average_trace(channels=trace_select.channel_information.channel_number[index], signal_type=trace_select.channel_information.signal_type[index])
                if trace_select.channel_information.signal_type[index]=="voltage":
                    tmp_axs.plot(time_array[0, :], trace_select_avg.voltage[0], color=avg_color)
                if trace_select.channel_information.signal_type[index]=="current":
                    tmp_axs.plot(time_array[0, :], trace_select_avg.current[0], color=avg_color)
            tmp_axs.set_xlabel("Time (" + time_array.dimensionality.latex + ")")
            tmp_axs.set_ylabel(trace_select.channel_information.signal_type[index].title() + " (" + trace_select.channel_information.unit[index] + ")")
        if return_fig:
                fig_out = deepcopy(fig)
        if show:
            fig.show()
        if return_fig:
            return fig_out
    def plot_summary(self):
        try:
            self.window_summary.plot(self, show=True)
        except:
            print("No summary data found")
            return None

class FunctionOutput:
    def __init__(self,
                 function_name: str) -> None:
        self.function_name = function_name
        self.measurements = np.array([])
        self.location = np.array([])
        self.sweep = np.array([])
        self.channel = np.array([])
        self.signal_type = np.array([])
        self.window = np.array([])
    def append(self,
               trace: Trace,
               window: tuple,
               channels: any = None,
               signal_type: any = None,
               rec_type: str = "",
               avg_window_ms: float = 1.0) -> None:
        trace_subset = trace.subset(channels=channels, signal_type=signal_type, rec_type=rec_type)
        for channel_index, channel in enumerate(trace_subset.channel_information.channel_number):
            array_index = trace_subset.channel_information.array_index[channel_index]
            time_window_size = Quantity(avg_window_ms, "ms")
            channel_signal_type = trace_subset.channel_information.signal_type[channel_index]
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
                        self.measurements = np.append(self.measurements, np.mean(trace_array[array_index,sweep_index,window_start_index[sweep_index]:window_end_index[sweep_index]]))
                        self.location = np.append(self.location, np.mean(window))
                elif self.function_name == "median":
                        self.measurements = np.append(self.measurements, np.median(trace_array[array_index,sweep_index,window_start_index[sweep_index]:window_end_index[sweep_index]]))
                        self.location = np.append(self.location, np.mean(window))
                elif self.function_name == "max":
                        self.measurements = np.append(self.measurements, np.max(trace_array[array_index,sweep_index,window_start_index[sweep_index]:window_end_index[sweep_index]]))
                        self.location = np.append(self.location, trace_subset.time[sweep_index,np.argmin(trace_array[array_index,sweep_index,window_start_index[sweep_index]:window_end_index[sweep_index]])+window_start_index[sweep_index]])
                elif self.function_name == "min":
                        self.measurements = np.append(self.measurements, np.min(trace_array[array_index,sweep_index,window_start_index[sweep_index]:window_end_index[sweep_index]]))
                        self.location = np.append(self.location, trace_subset.time[sweep_index,np.argmin(trace_array[array_index,sweep_index,window_start_index[sweep_index]:window_end_index[sweep_index]])+window_start_index[sweep_index]])
                elif self.function_name == "min_avg":
                        min_time = trace_subset.time[:,sweep_index,np.argmin(trace_array[array_index,sweep_index,window_start_index[sweep_index]:window_end_index[sweep_index]])+window_start_index[sweep_index]]
                        window_start_index = _get_time_index(trace_subset.time[sweep_index,:], (min_time-time_window_size).magnitude)
                        window_end_index = _get_time_index(trace_subset.time[sweep_index,:], (min_time+time_window_size).magnitude)
                        self.location = np.append(self.location, min_time)
                        self.measurements = np.append(self.measurements, np.mean(trace_array[array_index,sweep_index,window_start_index[sweep_index]:window_end_index[sweep_index]]))
                self.sweep = np.append(self.sweep, sweep_index+1)
                self.window = np.append(self.window, (window))
                self.signal_type = np.append(self.signal_type,channel_signal_type)
                self.channel = np.append(self.channel, channel)
    def merge(self, window_summary, remove_duplicates=True) -> None:
        self.measurements = np.append(self.measurements, window_summary.measurements)
        self.location = np.append(self.location, window_summary.location)
        self.sweep = np.append(self.sweep, window_summary.sweep)
        self.window = np.append(self.window, window_summary.window)
        self.signal_type = np.append(self.signal_type, window_summary.signal_type)
        self.channel = np.append(self.channel, window_summary.channel)
        if remove_duplicates:
            np.unique(self.measurements)
            self.measurements = np.unique(self.measurements)
            self.location = np.unique(self.location)
            self.sweep = np.unique(self.sweep)
            self.window = np.unique(self.window)
            self.signal_type = np.unique(self.signal_type)
            self.channel = np.unique(self.channel)
    def plot(self, trace: Trace = None, show: bool = True) -> None:
        if trace is not None:
            self.channel = np.unique(np.array(self.channel))
            trace_select = trace.subset(channels=self.channel, signal_type=self.signal_type)
    def to_dict(self):
        return {
            "measurements": self.measurements,
            "location": self.location,
            "sweep": self.sweep,
            "window": self.window,
            "signal_type": self.signal_type,
            "channel": self.channel,
        }


#        window_data_select = np.logical_and(np.array(self.channel) == trace_select.channel_information.channel_number[index],
 #                                                       np.array(window_data.sweep) == i)
  #                  tmp_axs.plot(np.array(window_data.location)[window_data_select], np.array(window_data.measurements)[window_data_select], "o", color="orange")
   #                 tmp_axs.axvspan(xmin=window_data.window[window_data_select][0][0], xmax=window_data.window[window_data_select][0][1], color='gray', alpha=0.1)


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

