"""
This module defines the Trace class, which represents electrophysiological trace data.
It provides methods for manipulating, analyzing, and visualizing trace data, including
subsetting, averaging, baseline subtraction, and plotting.

Classes:
    Trace: Represents a trace object with methods for data manipulation and analysis.

Functions:
    None

Dependencies:
    - numpy
    - matplotlib
    - quantities
    - ephys.utils
    - ephys.classes.class_functions
"""

from typing import Any
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from quantities import Quantity

from ephys import utils
from ephys.classes.class_functions import (
    _get_sweep_subset,
    _get_time_index,
    wcp_trace,
    abf_trace,
)  # pylint: disable=import-outside-toplevel
from ephys.classes.channels import ChannelInformation
from ephys.classes.window_functions import FunctionOutput


class Trace:
    """
    Represents a trace object.

    Args:
        file_path (str): The file path of the trace.

    Attributes:
        file_path (str): The file path of the trace.

    Methods:
        copy() -> Any:
            Returns a deep copy of the Trace object.

        subset(channels: Any = all channels, can be a list,
               signal_type: Any = 'voltage' and 'current',
               rec_type: Any = all rec_types) -> Any:
            Returns a subset of the Trace object based on the specified channels, signal_type, and
            rec_type.

        average_trace(channels: Any = all channels, can be a list,
                      signal_type: Any = 'voltage' and 'current', can be a list,
                      rec_type: Any = all rec_types) -> Any:
            Returns the average trace of the Trace object based on the specified channels,
            signal_type, and rec_type.

        plot(signal_type: str, channels: list, average: bool = False, color: str ='k',
            alpha: float = 0.5, avg_color: str = 'r'):
            Plots the trace data based on the specified signal_type, channels, and other optional
            parameters.
    """

    def __init__(self, file_path: str, quick_check: bool = True) -> None:
        self.file_path = file_path
        self.time = Quantity(np.array([]), units="s")
        self.sampling_rate = None
        self.channel = np.array([])
        self.channel_information = ChannelInformation()
        self.sweep_count = None
        self.window_summary = FunctionOutput()
        if file_path.endswith(".wcp"):
            wcp_trace(self, file_path, quick_check)
        elif file_path.endswith(".abf"):
            abf_trace(self, file_path, quick_check)
        else:
            print("File type not supported")

    def copy(self) -> Any:
        """
        Returns a deep copy of the Trace object.
        """
        return deepcopy(self)

    def subset(
        self,
        channels: Any = None,
        signal_type: Any = None,
        rec_type: Any = "",
        clamp_type: Any = None,
        channel_groups: Any = None,
        sweep_subset: Any = None,
        subset_index_only: bool = False,
        in_place: bool = False,
    ) -> Any:
        """
        Subset the experiment object based on specified criteria.

        Args:
            channels (Any, optional): Channels to include in the subset.
                Defaults to all channels.
            signal_type (Any, optional): Types of signal_type to include in the subset.
                Defaults to ['voltage', 'current'].
            rec_type (Any, optional): Recording types to include in the subset.
                Defaults to ''.
            clamp_type (Any, optional): Clamp types to include in the subset.
                Defaults to None.
            channel_groups (Any, optional): Channel groups to include in the subset.
                Defaults to None.
            sweep_subset (Any, optional): Sweeps to include in the subset. Possible inputs can be
                list, arrays or slice(). Defaults to None.
            subset_index_only (bool, optional): If True, returns only the subset index.
                Defaults to False.
            in_place (bool, optional): If True, modifies the object in place.
                Defaults to False.

        Returns:
            Any: Subset of the experiment object.
        """

        if (
            channels is None
            and signal_type is None
            and rec_type == ""
            and clamp_type is None
            and channel_groups is None
            and sweep_subset is None
        ):
            if subset_index_only:
                return self.channel_information
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
            subset_trace.channel_information.channel_number = (
                self.channel_information.channel_number[combined_index]
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
                    channel.data = channel.data[sweep_subset, :]
                else:
                    subset_trace.channel = np.delete(
                        subset_trace.channel, channel_index
                    )
            subset_trace.time = subset_trace.time[sweep_subset, :]
        else:
            subset_trace.channel_information.channel_number = np.array([])
            subset_trace.channel_information.recording_type = np.array([])
            subset_trace.channel_information.signal_type = np.array([])
            subset_trace.channel_information.clamped = np.array([])
            subset_trace.channel_information.channel_grouping = np.array([])
            subset_trace.channel_information.unit = np.array([])
        if subset_index_only:
            return subset_trace.channel_information
        subset_trace.sweep_count = subset_trace.time.shape[0] + 1
        return subset_trace

    def set_time(
        self,
        align_to_zero: bool = True,
        cumulative: bool = False,
        stimulus_interval: float = 0.0,
        overwrite_time: bool = True,
    ) -> Any:
        """
        Set the time axis for the given trace data.

        Parameters:
        - trace_data (Trace): The trace data object.
        - align_to_zero (bool): If True, align the time axis to zero. Default is True.
        - cumulative (bool): If True, set the time axis to cumulative. Default is False.
        - stimulus_interval (float): The stimulus interval. Default is 0.0 (s).

        Returns:
        - Trace or None
        """

        tmp_time = deepcopy(self.time)
        time_unit = tmp_time.units
        start_time = Quantity(0, time_unit)
        if self.sampling_rate is None:
            raise ValueError(
                "Sampling rate is not set."
                "Please set 'self.sampling_rate' before calling this method."
            )
        sampling_interval = (1 / self.sampling_rate).rescale(time_unit).magnitude

        for sweep_index, sweep in enumerate(tmp_time):
            sweep = Quantity(sweep, time_unit)
            if align_to_zero:
                start_time = Quantity(np.min(sweep.magnitude), time_unit)
            if cumulative:
                if sweep_index > 0:
                    start_time = Quantity(
                        Quantity(
                            np.min(sweep.magnitude)
                            - np.max(tmp_time[sweep_index - 1].magnitude),
                            time_unit,
                        ).magnitude
                        - stimulus_interval
                        - sampling_interval,
                        time_unit,
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
        channels: Any = None,
        signal_type: Any = None,
        rec_type: str = "",
        median: bool = False,
        overwrite: bool = False,
        sweep_subset: Any = None,
    ) -> Any | None:
        """
        Subtracts the baseline from the signal within a specified time window.

        Parameters:
        self : object
            The instance of the class containing the signal data.
        window : tuple, optional
            A tuple specifying the start and end of the time window for baseline
            calculation (default is (0, 0.1)).
        channels : Any, optional
            The channels to be processed. If None, all channels are processed
            (default is None).
        signal_type : Any, optional
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
        Any
            If overwrite is False, returns a copy of the data with the baseline
            subtracted. If overwrite is True, returns None.
        """

        if not overwrite:
            trace_copy = deepcopy(self)
        else:
            trace_copy = self
        subset_channels = trace_copy.subset(
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
        for subset_channel_index, subset_channel in enumerate(trace_copy.channel):
            if not np.isin(
                trace_copy.channel_information.channel_number[subset_channel_index],
                subset_channels.channel_number,
            ):
                continue
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

        if not overwrite:
            return trace_copy
        return None

    def window_function(
        self,
        window: list | None = None,
        channels: Any = None,
        signal_type: Any = None,
        rec_type: str = "",
        function: str = "mean",
        label: str = "",
        sweep_subset: Any = None,
        return_output: bool = False,
        plot=False,
    ) -> Any | None:
        """
        Apply a specified function to a subset of channels within given time windows.

        Parameters:
        -----------
        window : list, optional
            List of tuples specifying the start and end of each window. Default is [(0, 0)].
        channels : Any, optional
            Channels to be included in the subset. Default is None.
        signal_type : Any, optional
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
        Any
            The output of the applied function if return_output is True, otherwise None.

        Notes:
        ------
        The function updates the `window_summary` attribute of the class with the output.
        """

        if window is None:
            window = [(0, 0)]
        if function not in ["mean", "median", "max", "min", "min_avg"]:
            print("Function not supported")
        if not isinstance(window, list):
            window = [window]
        sweep_subset = _get_sweep_subset(self.time, sweep_subset)
        subset_channels = self.subset(
            channels=channels,
            signal_type=signal_type,
            rec_type=rec_type,
            sweep_subset=sweep_subset,
        )
        # output = np.ndarray((len(window), self.time.shape[0]))
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
                    unit=subset_channels.channel_information.unit[channel_index],
                )
        if plot:
            subset_channels.plot(trace=subset_channels, show=True, window_data=output)
        if return_output:
            return output
        self.window_summary.merge(output)
        return None

    def average_trace(
        self,
        channels: Any = None,
        signal_type: Any = None,
        rec_type: Any = "",
        sweep_subset: Any = None,
        in_place: bool = True,
    ) -> Any:
        """
        Calculates the average trace for the given channels, signal_type types, and recording type.

        Parameters:
        - channels (Any): The channels to calculate the average trace for.
          If None, uses the first channel type.
        - signal_type (Any): The signal_type types to calculate the average trace for.
          Defaults to ['voltage', 'current'].
        - rec_type (Any): The recording type to calculate the average trace for.

        Returns:
        - Any: The average trace object.
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
        avg_trace.subset(
            signal_type=signal_type,
            rec_type=rec_type,
            sweep_subset=sweep_subset,
            in_place=True,
        )

        for channel in avg_trace.channel:
            channel.channel_average()

        if in_place:
            return None
        return avg_trace

    def plot(
        self,
        signal_type: str = "",
        channels: np.ndarray = np.array([], dtype=np.int64),
        average: bool = False,
        color: str = "black",
        alpha: float = 0.5,
        avg_color: str = "red",
        align_onset: bool = True,
        sweep_subset: Any = None,
        window: tuple = (0, 0),
        xlim: tuple = (),
        show: bool = True,
        return_fig: bool = False,
    ) -> None | tuple:
        """
        Plots the traces for the specified channels.

        Args:
            signal_type (str): The type of signal_type to use. Must be either 'current' or
            'voltage'.
            channels (list, optional): The list of channels to plot. If None, all channels
            will be plotted.
                Defaults to None.
            average (bool, optional): Whether to plot the average trace.
                Defaults to False.
            color (str, optional): The color of the individual traces. Can be a colormap.
                Defaults to 'black'.
            alpha (float, optional): The transparency of the individual traces.
                Defaults to 0.5.
            avg_color (str, optional): The color of the average trace.
                Defaults to 'red'.
            align_onset (bool, optional): Whether to align the traces on the onset.
                Defaults to True.
            sweep_subset (Any, optional): The subset of sweeps to plot.
                Defaults to None.
            window (tuple, optional): The time window to plot.
                Defaults to (0, 0).
            show (bool, optional): Whether to display the plot.
                Defaults to True.
            return_fig (bool, optional): Whether to return the figure.
                Defaults to False.

        Returns:
            None or Figure: If show is True, returns None. If return_fig is True,
            returns the figure.
        """

        if len(channels) == 0:
            channels = self.channel_information.channel_number
        sweep_subset = _get_sweep_subset(self.time, sweep_subset)
        trace_select = self.subset(
            channels=channels, signal_type=signal_type, sweep_subset=sweep_subset
        )

        fig, channel_axs = plt.subplots(len(trace_select.channel), 1, sharex=True)

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

        tmp_axs: Axes | None = None
        for channel_index, channel in enumerate(trace_select.channel):
            if len(trace_select.channel) == 1:
                if isinstance(channel_axs, Axes):
                    tmp_axs = channel_axs
            else:
                if isinstance(channel_axs, np.ndarray):
                    if isinstance(channel_axs[channel_index], Axes):
                        tmp_axs = channel_axs[channel_index]
            if tmp_axs is None:
                pass
            else:
                for i in range(channel.data.shape[0]):
                    tmp_axs.plot(
                        time_array[i, :],
                        channel.data[i, :],
                        color=utils.trace_color(
                            traces=channel.data, index=i, color=color
                        ),
                        alpha=alpha,
                    )
                if window != (0, 0):
                    tmp_axs.axvspan(
                        xmin=window[0], xmax=window[1], color="gray", alpha=0.1
                    )
                if average:
                    channel.channel_average(sweep_subset=sweep_subset)
                    tmp_axs.plot(
                        time_array[0, :], channel.average.trace, color=avg_color
                    )
                tmp_axs.set_ylabel(
                    f"Channel {trace_select.channel_information.channel_number[channel_index]} "
                    f"({trace_select.channel_information.unit[channel_index]})"
                )
        #   tmp_axs.set_ylabel(f'Channel')
        if isinstance(tmp_axs, Axes):
            tmp_axs.set_xlabel(
                f"Time ({trace_select.time.units.dimensionality.string})"
            )
            if len(xlim) > 0:
                tmp_axs.set_xlim(xlim[0], xlim[1])
        plt.tight_layout()
        if show:
            plt.show()
        #            return None
        if return_fig:
            return fig, channel_axs
        return None

    def plot_summary(
        self,
        show_trace: bool = True,
        align_onset: bool = True,
        label_filter: list | str = "",
        color="black",
        show=True,
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

        if label_filter == "":
            label_filter = []
        if self.window_summary is not None:
            if show_trace:
                self.window_summary.plot(
                    trace=self,
                    align_onset=align_onset,
                    show=show,
                    label_filter=label_filter,
                    color=color,
                )
            else:
                self.window_summary.plot(
                    align_onset=align_onset, show=show, label_filter=label_filter
                )
        else:
            print("No summary data found")
