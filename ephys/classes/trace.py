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

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
from copy import deepcopy
from datetime import datetime
from uuid import uuid4
import numpy as np
from quantities import Quantity
import pyqtgraph as pg

from ephys import utils
from ephys.classes.class_functions import (
    _get_sweep_subset,
    _get_time_index,
    wcp_trace,
    abf_trace,
)  # pylint: disable=import-outside-toplevel
from ephys.classes.channels import ChannelInformation
from ephys.classes.window_functions import FunctionOutput

if TYPE_CHECKING:
    from ephys.classes.plot.plot_trace import TracePlotPyQt, TracePlotMatplotlib


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

    def __init__(self, file_path: str = "", quick_check: bool = True) -> None:
        self.file_path: str = file_path
        self.time: Quantity = Quantity(np.array([]), units="s")
        self.sampling_rate: Quantity | None = None
        self.rec_datetime: Optional[datetime] = None
        self.channel: np.ndarray = np.array([])
        self.channel_information: ChannelInformation = ChannelInformation()
        self.sweep_count: int | None = None
        self.object_id: str = str(uuid4())
        self.window_summary: FunctionOutput = FunctionOutput()
        self.window: None | list = None
        if self.file_path and len(self.file_path) > 0:
            self.load(file_path=self.file_path, quick_check=quick_check)

    def load(self, file_path: str, quick_check: bool = True) -> None:
        """
        Load the trace data from a file.

        Args:
            file_path (str): The path to the file to load.
            quick_check (bool, optional): If True, performs a quick check of the file.
        """
        if file_path.endswith(".wcp"):
            wcp_trace(trace=self, file_path=file_path, quick_check=quick_check)
        elif file_path.endswith(".abf"):
            abf_trace(trace=self, file_path=file_path, quick_check=quick_check)
        else:
            print("File type not supported")
        if self.sampling_rate is not None:
            self.rec_datetime = self.channel[0].rec_datetime

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

        sweep_subset = _get_sweep_subset(array=self.time, sweep_subset=sweep_subset)
        if in_place:
            subset_trace = self
        else:
            subset_trace = self.copy()
        rec_type_get: np.ndarray = utils.string_match(
            pattern=rec_type, string_list=self.channel_information.recording_type
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

        tmp_time: Quantity = deepcopy(self.time)
        time_unit: Quantity = tmp_time.units
        start_time = Quantity(data=0, units=time_unit)
        if self.sampling_rate is None:
            raise ValueError(
                "Sampling rate is not set."
                "Please set 'self.sampling_rate' before calling this method."
            )
        sampling_interval = (1 / self.sampling_rate).rescale(time_unit).magnitude

        for sweep_index, sweep in enumerate(tmp_time):
            sweep = Quantity(data=sweep, units=time_unit)
            if align_to_zero:
                start_time = Quantity(data=np.min(sweep.magnitude), units=time_unit)
            if cumulative:
                if sweep_index > 0:
                    start_time = Quantity(
                        data=Quantity(
                            data=np.min(sweep.magnitude)
                            - np.max(tmp_time[sweep_index - 1].magnitude),
                            units=time_unit,
                        ).magnitude
                        - stimulus_interval
                        - sampling_interval,
                        units=time_unit,
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

        self.time = self.time.rescale(units=time_unit)

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

    def get_window(
        self,
        index: int | None = None,
    ) -> tuple | list | None:
        """
        Get the current window of the trace.

        Args:
            index (int, optional): The index of the window to retrieve. If None, returns the
                entire window. Defaults to None.

        Returns:
            tuple or list: The current window of the trace.
        """
        if self.window is None:
            print("No window set for the trace.")
            return None
        if index is None:
            return self.window
        if self.window is not None and (index < 0 or index >= len(self.window)):
            raise IndexError("Index out of range for the window list.")
        if isinstance(self.window, list):
            return self.window[index]

    def add_window(
        self,
        window: tuple | list,
    ) -> None:
        """
        Add a window for the trace.

        Args:
            window (tuple or list): The window to set for the trace.
        """
        if isinstance(window, tuple):
            if self.window is None:
                self.window = [window]
            if isinstance(self.window, list):
                self.window.append(window)
        elif isinstance(window, list):
            for win in window:
                if not isinstance(win, tuple) or len(win) != 2:
                    raise TypeError("Each window must be a tuple of length 2.")
            if self.window is None:
                self.window = window
            if isinstance(self.window, list):
                self.window.extend(window)
        else:
            raise ValueError("Window must be a tuple or a list.")

    def remove_window(
        self,
        index: int | None = None,
        all: bool = False,
    ) -> None:
        """
        Remove a window from the trace.

        Args:
            index (int, optional): The index of the window to remove. If None, removes the last
                window. Defaults to None.
        """
        if self.window is None:
            return None
        if index is None and not all:
            index = -1
        if all:
            self.window = None
        if isinstance(self.window, list):
            if index is not None and 0 <= index < len(self.window):
                del self.window[index]
            else:
                self.window.pop()

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
        self.window_summary.merge(window_summary=output)
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
        sweep_subset = _get_sweep_subset(array=self.time, sweep_subset=sweep_subset)
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
        self, backend: str = "matplotlib", **kwargs
    ) -> None | pg.GraphicsLayoutWidget | tuple:
        """
        Plots the traces using the specified backend.

        Args:
            backend (str): The plotting backend to use. Options are 'matplotlib' or 'pyqt'.
            **kwargs: Additional keyword arguments for the plotting function.

        Returns:
            None or pg.GraphicsLayoutWidget: If using pyqtgraph, returns the plot widget.
        """
        from ephys.classes.plot.plot_trace import TracePlotPyQt, TracePlotMatplotlib

        if backend == "matplotlib":
            plot_out = TracePlotMatplotlib(trace=self, **kwargs)
            return plot_out.plot()
        if backend == "pyqt":
            plot_out = TracePlotPyQt(trace=self, **kwargs)
            return plot_out.plot()
        raise ValueError("Unsupported backend. Use 'matplotlib' or 'pyqt'.")

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
