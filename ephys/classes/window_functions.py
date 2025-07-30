"""
This module defines the FunctionOutput class, which is used to handle the output of
various functions applied to electrophysiological trace data. It includes methods for
appending measurements, merging data, calculating differences and ratios, plotting
results, and converting to dictionary or DataFrame formats.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TYPE_CHECKING
import numpy as np
import pandas as pd
from quantities import Quantity
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ephys import utils
from ephys.classes.plot.plot_params import PlotParams
from ephys.classes.class_functions import _get_time_index

if TYPE_CHECKING:
    from ephys.classes.trace import Trace


class FunctionOutput:
    """
    A class to handle the output of various functions applied to electrophysiological
    trace data.

    Args:
        function_name (str): The name of the function to be applied to the trace data.

    Attributes:
        function_name (str): The name of the function to be applied to the trace data.
        measurements (np.ndarray): An array to store the measurements obtained from the
            trace data.
        location (np.ndarray): An array to store the locations corresponding to the
            measurements.
        sweep (np.ndarray): An array to store the sweep indices.
        channel (np.ndarray): An array to store the channel numbers.
        signal_type (np.ndarray): An array to store the types of signals (e.g., current,
            voltage).
        window (np.ndarray): An array to store the time windows used for measurements.
        label (np.ndarray): An array to store the labels associated with the measurements.
        time (np.ndarray): An array to store the time points corresponding to the
            measurements.

    Methods:
        __init__(self, function_name: str) -> None:
            Initializes the FunctionOutput object with the given function name.

        append(self, trace: Trace, window: tuple, channels: Any = None,
               signal_type: Any = None, rec_type: str = '', avg_window_ms: float = 1.0,
               label: str = '') -> None:
            Appends measurements and related information from the given trace data to the
            FunctionOutput object.

        merge(self, window_summary, remove_duplicates=False) -> None:
            Merges the measurements, location, sweep, window, signal_type, and channel
            attributes from the given window_summary object into the current object.
            Optionally removes duplicates from these attributes after merging.

        label_diff(self, labels: list = [], new_name: str = '', time_label: str = '')
            -> None:
            Calculates the difference between two sets of measurements and appends the
            result.

        plot(self, trace: Trace = None, show: bool = True, align_onset: bool = True,
             label_filter: list | str = [], color='black') -> None:
            Plots the trace and/or summary measurements.

        to_dict(self) -> dict:
            Converts the experiment object to a dictionary representation.

        to_dataframe(self) -> pd.DataFrame:
            Converts the experiment object to a pandas DataFrame.

        delete_label(self, label: str | list) -> None:
            Deletes a label from the measurements.
    """

    def __init__(self, function_name: str = "") -> None:
        self.trace: Trace | None = None
        self.function_name = function_name
        self.function = np.array([], dtype=str)
        self.measurements = np.array([])
        self.location = np.array([])
        self.sweep = np.array([])
        self.channel = np.array([])
        self.signal_type = np.array([])
        self.window = np.ndarray(dtype=object, shape=(0, 2))
        self.label = np.array([])
        self.time = np.array([])
        self.unit = np.array([])

    def append(
        self,
        trace: Trace,
        window: tuple,
        channels: Any = None,
        signal_type: Any = None,
        rec_type: str = "",
        avg_window_ms: float = 1.0,
        label: str = "",
    ) -> None:
        """
        Appends measurements from a given trace within a specified time window.

        Args:
            trace (Trace): The trace object containing the data to be analyzed.
            window (tuple): A tuple specifying the start and end times of the window
            for measurement.
            channels (Any, optional): The channels to be included in the subset of the
            trace. Default is None.
            signal_type (Any, optional): The type of signal to be included in the
            subset of the trace. Default is None.
            rec_type (str, optional): The recording type to be included in the subset
            of the trace. Default is an empty string.
            avg_window_ms (float, optional): The averaging window size in milliseconds
            for the 'min_avg' function. Default is 1.0 ms.
            label (str, optional): A label to be associated with the measurements.
            Default is an empty string.

        Returns:
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
        self.trace = trace
        for channel_index, channel in enumerate(
            trace_subset.channel_information.channel_number
        ):
            tmp_location = np.array([])
            time_window_size = Quantity(avg_window_ms, "ms")
            channel_signal_type = trace_subset.channel_information.signal_type[
                channel_index
            ]
            sweep_dim = trace_subset.time.shape[0]
            window_start_index = _get_time_index(trace_subset.time, window[0])
            window_end_index = _get_time_index(trace_subset.time, window[1])

            if self.function_name == "min_avg":
                window_index = int(
                    (
                        time_window_size.rescale(trace_subset.time.units)
                        / np.diff(trace_subset.time.flatten()[0:2])
                    ).magnitude
                )
                window_start_index -= window_index
                window_end_index += window_index
            else:
                window_index = 0

            windowed_time = Quantity(
                np.array(
                    [
                        row[start:end]
                        for row, start, end in zip(
                            trace_subset.time, window_start_index, window_end_index
                        )
                    ]
                ),
                trace_subset.time.units,
            )
            if channel_signal_type in ["voltage", "current"]:
                windowed_trace = np.array(
                    [
                        row[start:end]
                        for row, start, end in zip(
                            trace_subset.channel[channel_index].data.magnitude,
                            window_start_index,
                            window_end_index,
                        )
                    ]
                )
            else:
                print("Signal type not found")
                return None
            if self.function_name == "mean":
                self.measurements = np.append(
                    self.measurements,
                    np.mean(windowed_trace, axis=1),
                )
                tmp_location = np.repeat(np.mean(window), sweep_dim)

            elif self.function_name == "median":
                self.measurements = np.append(
                    self.measurements,
                    np.median(windowed_trace, axis=1),
                )
                tmp_location = np.repeat(np.mean(window), sweep_dim)
            elif self.function_name == "max":
                self.measurements = np.append(
                    self.measurements,
                    np.max(windowed_trace, axis=1),
                )
                tmp_location = np.array(
                    [
                        windowed_time[row, col]
                        for row, col in enumerate(windowed_trace.argmax(axis=1))
                    ]
                )
            elif self.function_name == "min":
                self.measurements = np.append(
                    self.measurements,
                    np.min(windowed_trace, axis=1),
                )
                tmp_location = np.array(
                    [
                        windowed_time[row, col]
                        for row, col in enumerate(windowed_trace.argmin(axis=1))
                    ]
                )
            elif self.function_name == "min_avg":
                windowed_time_local = Quantity(
                    np.array(
                        [
                            row[(start + window_index) : (end - window_index)]
                            for row, start, end in zip(
                                trace_subset.time, window_start_index, window_end_index
                            )
                        ]
                    ),
                    trace_subset.time.units,
                )
                windowed_trace_local = np.array(
                    [
                        row[(start + window_index) : (end - window_index)]
                        for row, start, end in zip(
                            trace_subset.channel[channel_index].data.magnitude,
                            window_start_index,
                            window_end_index,
                        )
                    ]
                )

                tmp_location = Quantity(
                    np.array(
                        [
                            windowed_time_local[row, col]
                            for row, col in enumerate(
                                windowed_trace_local.argmin(axis=1)
                            )
                        ]
                    ),
                    windowed_time.units,
                )
                # calculate how many indexes to go back and forward
                # to get the average
                time_window_size = Quantity(avg_window_ms, "ms")
                time_window_size = time_window_size.rescale(windowed_time.units)
                # convert to indexes
                avg_start = (
                    np.array(
                        [
                            _get_time_index(
                                windowed_time[i, :],
                                float(
                                    (min_time_i).rescale(windowed_time.units).magnitude
                                ),
                            )
                            for i, min_time_i in enumerate(tmp_location)
                        ]
                    )
                    - window_index
                )
                avg_end = (
                    np.array(
                        [
                            _get_time_index(
                                windowed_time[i, :],
                                float(
                                    (min_time_i).rescale(windowed_time.units).magnitude
                                ),
                            )
                            for i, min_time_i in enumerate(tmp_location)
                        ]
                    )
                    + window_index
                )
                self.measurements = np.append(
                    self.measurements,
                    np.array(
                        [
                            row[start:end]
                            for row, start, end in zip(
                                windowed_trace, avg_start, avg_end
                            )
                        ]
                    ).mean(axis=1),
                )
            if label == "":
                # find the function_name in the self.function
                label = self.function_name

            self.function = np.append(
                self.function, np.repeat(self.function_name, sweep_dim)
            )
            self.location = np.append(self.location, tmp_location)
            self.sweep = np.append(self.sweep, np.arange(1, sweep_dim + 1))
            self.window = np.vstack((self.window, np.tile(window, (sweep_dim, 1))))
            self.label = np.append(self.label, np.repeat(label, sweep_dim))
            self.signal_type = np.append(
                self.signal_type,
                np.repeat(
                    trace_subset.channel_information.signal_type[channel_index],
                    sweep_dim,
                ),
            )
            self.channel = np.append(
                self.channel,
                np.repeat(
                    channel,
                    sweep_dim,
                ),
            )
            self.unit = np.append(
                self.unit,
                np.repeat(
                    trace_subset.channel_information.unit[channel_index], sweep_dim
                ),
            )

            self.time = np.append(
                self.time,
                [
                    actual_time[
                        sweep_index,
                        _get_time_index(
                            trace_subset.time[sweep_index],
                            float(tmp_location[sweep_index]),
                        ),
                    ]
                    for sweep_index in range(0, sweep_dim)
                ],
            )
        return None

    def merge(self, window_summary, remove_duplicates=False) -> None:
        """
        Merges the measurements, location, sweep, window, signal_type, and channel
        attributes from the given window_summary object into the current object.
        Optionally removes duplicates from these attributes after merging.

        Args:
            window_summary (object): An object containing measurements, location,
            sweep, window, signal_type, and channel attributes to be merged.
            remove_duplicates (bool, optional): If True, removes duplicate entries
            from the merged attributes. Defaults to True.

        Returns:
            None
        """

        self.measurements = np.append(self.measurements, window_summary.measurements)
        self.location = np.append(self.location, window_summary.location)
        self.sweep = np.append(self.sweep, window_summary.sweep)
        self.window = np.vstack((self.window, window_summary.window))
        self.signal_type = np.append(self.signal_type, window_summary.signal_type)
        self.channel = np.append(self.channel, window_summary.channel)
        self.label = np.append(
            self.label, utils.unique_label_name(self.label, window_summary.label)
        )
        self.function = np.append(self.function, window_summary.function)
        self.unit = np.append(self.unit, window_summary.unit)
        self.time = np.append(self.time, window_summary.time)
        self.trace = window_summary.trace

        if remove_duplicates:
            np.unique(self.measurements)
            self.measurements = np.unique(self.measurements)
            self.location = np.unique(self.location)
            self.sweep = np.unique(self.sweep)
            self.window = np.unique(self.window)
            self.signal_type = np.unique(self.signal_type)
            self.channel = np.unique(self.channel)
            self.label = np.unique(self.label)
            self.function = np.unique(self.function)
            self.unit = np.unique(self.unit)

    def label_diff(
        self, labels: list | None = None, new_name: str = "", time_label: str = ""
    ) -> None:
        """
        Calculate the difference between two sets of measurements and append the result.

        Args:
            labels (list): Labels whose measurements will be used to calculate the
            difference.
            new_name (str): Label name for the new set of measurements.
            time_label (str): Label to identify the time points for the new
            measurements.

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
        self.function = np.append(self.function, np.repeat("diff", len(diff)))

        self.measurements = np.append(self.measurements, diff)
        self.location = np.append(self.location, self.location[time_label_index])
        self.sweep = np.append(self.sweep, self.sweep[time_label_index])
        self.window = np.vstack((self.window, self.window[time_label_index]))
        self.signal_type = np.append(
            self.signal_type, self.signal_type[time_label_index]
        )
        self.channel = np.append(self.channel, self.channel[time_label_index])
        self.label = np.append(
            self.label, np.repeat(new_name, len(time_label_index[0]))
        )
        self.time = np.append(self.time, self.time[time_label_index])
        return None

    def label_ratio(
        self, labels: list | None = None, new_name: str = "", time_label: str = ""
    ) -> None:
        """
        Calculate the ratio between two sets of measurements and append the result.

        Args:
            labels (list): Labels whose measurements will be used to calculate the ratio.
            new_name (str): Label name for the new set of measurements.
            time_label (str): Label to identify the time points for the new
            measurements.

        Returns:
            None
        """

        if labels is None:
            labels = []
        unique_labels = np.unique(self.label)
        if not all(label in unique_labels for label in labels):
            print("Labels not found in data")
        else:
            label_index_1 = np.where(self.label == labels[0])
            label_index_2 = np.where(self.label == labels[1])
            time_label_index = np.where(self.label == time_label)
            ratio = self.measurements[label_index_1] / self.measurements[label_index_2]
            self.measurements = np.append(self.measurements, ratio)
            self.location = np.append(self.location, self.location[time_label_index])
            self.sweep = np.append(self.sweep, self.sweep[time_label_index])
            self.window = np.vstack((self.window, self.window[time_label_index]))
            self.signal_type = np.append(
                self.signal_type, self.signal_type[time_label_index]
            )
            self.channel = np.append(self.channel, self.channel[time_label_index])
            self.label = np.append(
                self.label, np.repeat(new_name, len(time_label_index[0]))
            )
            self.time = np.append(self.time, self.time[time_label_index])

    def plot(
        self,
        plot_trace: bool = True,
        trace: Trace | None = None,
        label_filter: list | str | None = None,
        backend: str = "matplotlib",  # remove default after setting up pyqt
        **kwargs,
    ) -> None | tuple[Figure, Axes | np.ndarray] | None:
        """
        Plot the trace and/or summary measurements.

        Args:
            plot_trace (bool, optional): If True, plot the trace data.
            Default is True.
            trace (Trace or None, optional): The trace object to be plotted.
            If None, uses the trace stored in the FunctionOutput object.
            label_filter (list, str, or None, optional): Labels to filter the
            measurements for plotting.
            backend (str, optional): The plotting backend to use
            ('matplotlib' or 'pyqt'). Default is 'matplotlib'.
            **kwargs: Additional keyword arguments for plotting.

        Returns:
            FunctionOutputPyQt or FunctionOutputMatplotlib: The plot output object.

        Raises:
            ValueError: If an unsupported backend is specified.
            TypeError: If the trace is not an instance of Trace when plot_trace is True.
        """

        # pylint:disable=import-outside-toplevel
        from ephys.classes.trace import Trace

        plot_params = PlotParams()
        plot_params.update_params(**kwargs)
        if plot_trace:
            if trace is not None:
                if trace is None:
                    trace = self.trace
                if not isinstance(trace, Trace):
                    raise TypeError("trace must be an instance of Trace.")
        else:
            trace = None
        if backend == "matplotlib":
            # pylint:disable=import-outside-toplevel
            from ephys.classes.plot.plot_window_functions import (
                FunctionOutputMatplotlib,
            )

            plot_output = FunctionOutputMatplotlib(
                function_output=self, **plot_params.__dict__
            )
        elif backend == "pyqt":
            # pylint:disable=import-outside-toplevel
            from ephys.classes.plot.plot_window_functions import FunctionOutputPyQt

            plot_output = FunctionOutputPyQt(
                function_output=self, **plot_params.__dict__
            )
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Use 'matplotlib' or 'pyqt'."
            )

        plot_output.plot(
            trace=trace,
            label_filter=label_filter,
        )

    def to_dict(self):
        """
        Convert the FunctionOutput object to a dictionary representation.

        Returns:
            dict: A dictionary containing the function output attributes.
        """

        return {
            "function": self.function,
            "measurements": self.measurements,
            "location": self.location,
            "sweep": self.sweep,
            "window": self.window,
            "signal_type": self.signal_type,
            "channel": self.channel,
            "label": self.label,
            "unit": self.unit,
            "time": self.time,
        }

    def to_dataframe(self):
        """
        Convert the FunctionOutput object to a pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing the function output attributes.
        """

        tmp_dictionary = self.to_dict()
        window_tmp = [tuple(row) for row in tmp_dictionary["window"]]
        tmp_dictionary["window"] = window_tmp
        return pd.DataFrame(tmp_dictionary)

    def delete_label(self, label: str | list | None = None) -> None:
        """
        Delete a label from the measurements.

        Parameters:
        label (str | list): The label(s) to be deleted.

        Returns:
        None
        """
        if label is None:
            self.function = np.array([], dtype=str)
            self.measurements = np.array([])
            self.location = np.array([])
            self.sweep = np.array([])
            self.channel = np.array([])
            self.signal_type = np.array([])
            self.window = np.ndarray(dtype=object, shape=(0, 2))
            self.label = np.array([])
            self.time = np.array([])
            self.unit = np.array([])
            return None
        if isinstance(label, str):
            label = [label]
        for label_i in label:
            label_index = np.where(self.label == label_i)
            self.function = np.delete(self.function, label_index)
            self.measurements = np.delete(self.measurements, label_index)
            self.location = np.delete(self.location, label_index)
            self.sweep = np.delete(self.sweep, label_index)
            self.window = np.delete(self.window, label_index, axis=0)
            self.signal_type = np.delete(self.signal_type, label_index)
            self.channel = np.delete(self.channel, label_index)
            self.label = np.delete(self.label, label_index)
            self.time = np.delete(self.time, label_index)
            self.unit = np.delete(self.unit, label_index)
        return None
