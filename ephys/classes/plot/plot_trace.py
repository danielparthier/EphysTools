"""This module contains classes for plotting traces using different backends.
It includes support for both PyQtGraph and Matplotlib, allowing for flexible
visualization of trace data
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import pyqtgraph as pg
import matplotlib.colors as mcolors

from ephys.classes.trace import Trace
from ephys.classes.class_functions import _get_sweep_subset
from ephys import utils


@dataclass
class PlotParams:
    """Parameters for plotting traces.
    Attributes:
        signal_type (str): Type of signal to plot ('current' or 'voltage').
        channels (np.ndarray): Channels to plot.
        color (str): Color or colormap name for individual traces.
        alpha (float): Transparency for individual traces.
        average (bool): Whether to plot the average trace.
        avg_color (str): Color for the average trace.
        align_onset (bool): Whether to align traces on onset.
        sweep_subset (Any): Subset of sweeps to plot.
        bg_color (str): Background color for the plot.
        axis_color (str): Color for axes.
        window (list[tuple[float, float]]): Time windows for plotting.
        window_color (str): Color for window regions.
        xlim (tuple[float, float]): X-axis limits.
        show (bool): Whether to show the plot.
        return_fig (bool): Whether to return the figure object.
    """

    signal_type: str = ""  # 'current' or 'voltage'
    channels: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    color: str = "white"  # Color or colormap name
    alpha: float = 1  # Transparency for individual traces
    average: bool = False  # Whether to plot average trace
    avg_color: str = "red"  # Color for average trace
    align_onset: bool = True  # Align traces on onset
    sweep_subset: Any = None  # Subset of sweeps to plot
    bg_color: str = "black"  # Background color for the plot
    axis_color: str = "white"  # Color for axes
    window: list[tuple[float, float]] = field(
        default_factory=lambda: [(0, 0)]
    )  # Time windows
    window_color: str = "gray"  # Color for window regions
    xlim: tuple[float, float] = field(default_factory=tuple)  # X-axis limits
    show: bool = True  # Whether to show the plot
    return_fig: bool = False  # Whether to return the figure object

    if theme := "dark":
        color = "white"
        bg_color = "black"
        axis_color = "white"
        window_color = "gray"
    elif theme := "light":
        color = "black"
        bg_color = "white"
        axis_color = "black"
        window_color = "lightgray"
    else:
        theme = "dark"  # Default to dark theme if not specified
        color = "white"
        bg_color = "black"
        axis_color = "white"
        window_color = "gray"


class TracePlot:
    """Base class for plotting traces."""

    def __init__(self, trace: Trace, backend: str = "pyqt", **kwargs) -> None:
        """
        Args:
            trace (Trace): The trace object to plot.
            backend (str): The plotting backend to use ('pyqt' or 'matplotlib').
            **kwargs: Additional parameters for plotting.
        """

        # check backend
        if backend not in ["pyqt", "matplotlib"]:
            raise ValueError(
                f"Unsupported backend: {backend}. Choose 'pyqt' or 'matplotlib'."
            )
        self.params = PlotParams(**kwargs)
        self.backend = backend
        self.trace = trace

    def update_params(self, **kwargs) -> None:
        """Update the plotting parameters with the provided keyword arguments.
        Args:
            **kwargs: Keyword arguments to update the parameters.
        Raises:
            ValueError: If an invalid parameter is provided.
        """
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")


class TracePlotMatplotlib(TracePlot):
    """Class for plotting traces using Matplotlib."""

    def __init__(self, trace: Trace, backend: str = "matplotlib", **kwargs) -> None:
        super().__init__(trace, backend=backend, **kwargs)

    def plot(
        self,
        **kwargs,
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
        if kwargs:
            self.update_params(**kwargs)
        if len(self.params.channels) == 0:
            self.params.channels = self.trace.channel_information.channel_number
        sweep_subset = _get_sweep_subset(self.trace.time, self.params.sweep_subset)
        trace_select = self.trace.subset(
            channels=self.params.channels,
            signal_type=self.params.signal_type,
            sweep_subset=self.params.sweep_subset,
        )

        fig, channel_axs = plt.subplots(len(trace_select.channel), 1, sharex=True)
        # color background and axis

        # change color of all axes
        set_axs_color(trace_plot=self, input_axs=channel_axs)

        fig.set_facecolor(self.params.bg_color)
        if isinstance(channel_axs, Axes):
            channel_axs.set_facecolor(self.params.bg_color)
        elif isinstance(channel_axs, np.ndarray):
            for axs in channel_axs:
                axs.set_facecolor(self.params.bg_color)
        else:
            raise TypeError("channel_axs must be an Axes or np.ndarray of Axes.")
        if len(trace_select.channel) == 0:
            print("No traces found.")
            return None

        if self.params.align_onset:
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
                            traces=channel.data, index=i, color=self.params.color
                        ),
                        alpha=self.params.alpha,
                    )
                if self.params.window != [(0, 0)]:
                    if isinstance(self.params.window, tuple):
                        self.trace.window = [self.params.window]
                    elif isinstance(self.params.window, list):
                        self.trace.window = self.params.window
                    else:
                        raise TypeError("Window must be a tuple or list of tuples.")
                    if (
                        isinstance(self.trace.window, list)
                        and len(self.trace.window) == 0
                    ):
                        raise ValueError("Window list is empty.")
                    for win in self.trace.window:
                        if isinstance(tmp_axs, Axes):
                            tmp_axs.axvspan(
                                xmin=win[0],
                                xmax=win[1],
                                color=self.params.window_color,
                                alpha=0.1,
                            )
                if self.params.average:
                    channel.channel_average(sweep_subset=sweep_subset)
                    tmp_axs.plot(
                        time_array[0, :],
                        channel.average.trace,
                        color=self.params.avg_color,
                    )
                tmp_axs.set_ylabel(
                    f"Channel {trace_select.channel_information.channel_number[channel_index]} "
                    f"({trace_select.channel_information.unit[channel_index]})"
                )
        if isinstance(tmp_axs, Axes):
            tmp_axs.set_xlabel(
                f"Time ({trace_select.time.units.dimensionality.string})"
            )
            if len(self.params.xlim) == 2:
                tmp_axs.set_xlim(self.params.xlim[0], self.params.xlim[1])
        plt.tight_layout()
        if self.params.show:
            plt.show()
        if self.params.return_fig:
            return fig, channel_axs
        return None


class TracePlotPyQt(TracePlot):
    """Class for plotting traces using PyQtGraph."""

    def __init__(self, trace: Trace, backend: str = "pyqt", **kwargs):
        super().__init__(trace, backend=backend, **kwargs)

    def plot(
        self,
        **kwargs: Any,
    ) -> None | pg.GraphicsLayoutWidget:
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
        if kwargs:
            self.update_params(**kwargs)

        def sync_channels(source_region, channel_items, window_index=0):
            # Get region bounds from the source region
            min_val, max_val = source_region.getRegion()

            # Update all other regions
            for r in channel_items:
                if r is not source_region:
                    r.blockSignals(True)
                    r.setRegion((min_val, max_val))
                    r.blockSignals(False)

            # Update the trace window property
            if isinstance(self.trace.window, list):
                self.trace.window[window_index] = (min_val, max_val)
            elif isinstance(self.trace.window, tuple):
                self.trace.window = (min_val, max_val)
            else:
                raise TypeError("Window must be a tuple or list of tuples.")

        def make_region_callback(region_obj, channel_items, window_index=0):
            return lambda: sync_channels(
                region_obj, channel_items, window_index=window_index
            )

        if len(self.params.channels) == 0:
            self.params.channels = self.trace.channel_information.channel_number
        trace_select = self.trace.subset(
            channels=self.params.channels,
            signal_type=self.params.signal_type,
            sweep_subset=self.params.sweep_subset,
        )

        if self.params.align_onset:
            time_array = trace_select.set_time(
                align_to_zero=True,
                cumulative=False,
                stimulus_interval=0.0,
                overwrite_time=False,
            )
        else:
            time_array = trace_select.time

        if len(self.params.xlim) > 2:
            raise ValueError("xlim must be a tuple of two values.")
        if len(self.params.xlim) < 2:
            self.params.xlim = (
                np.min(time_array.magnitude),
                np.max(time_array.magnitude),
            )

        win = pg.GraphicsLayoutWidget(show=self.params.show, title="Trace Plot")
        win.setBackground(self.params.bg_color)
        window_fill = pg.mkBrush(
            color=tuple(
                np.round(color_val * 255)
                for color_val in mcolors.to_rgba(self.params.window_color, alpha=0.5)
            )
        )
        window_fill_hover = pg.mkBrush(
            color=tuple(
                np.round(color_val * 255)
                for color_val in mcolors.to_rgba(self.params.window_color, alpha=0.8)
            )
        )
        # Handle window regions for interactive selection
        if self.params.window is not None:
            if isinstance(self.params.window, tuple):
                self.trace.window = [self.params.window]
            elif isinstance(self.params.window, list):
                self.trace.window = self.params.window
            else:
                raise TypeError("Window must be a tuple or list of tuples.")
            window_items: list[list[pg.LinearRegionItem]] = [
                [] for _ in range(len(self.trace.window))
            ]
        else:
            self.trace.window = [(0, 0)]
            window_items: list[list[pg.LinearRegionItem]] = [[]]

        region: pg.LinearRegionItem | None = None
        channel_0: pg.PlotItem | None = None
        for channel_index, channel in enumerate(trace_select.channel):
            channel_tmp = win.addPlot(row=channel_index, col=0)  # type: ignore
            if channel_index == 0:
                channel_0 = channel_tmp
            channel_tmp.setXLink(channel_0)
            channel_tmp.setLabel(
                "left",
                f"Channel {trace_select.channel_information.channel_number[channel_index]} "
                f"({trace_select.channel_information.unit[channel_index]})",
                color=self.params.axis_color,
            )

            if channel_index == len(trace_select.channel) - 1:
                channel_tmp.setLabel("bottom", "Time (s)", color=self.params.axis_color)

            channel_tmp.setLabel(
                "bottom",
                f"Time ({time_array.units.dimensionality.string})",
                color=self.params.axis_color,
            )
            channel_tmp.setDownsampling(mode="subsample", auto=True)
            channel_tmp.setClipToView(True)
            channel_box = channel_tmp.getViewBox()
            channel_box.setXRange(self.params.xlim[0], self.params.xlim[1])

            for i in range(channel.data.shape[0]):
                qt_color = utils.color_picker_QColor(
                    length=channel.data.shape[0],
                    index=i,
                    color=self.params.color,
                    alpha=self.params.alpha,
                )
                channel_tmp.plot(
                    time_array[i],
                    channel.data[i],
                    pen=pg.mkPen(
                        color=qt_color,
                        # width=1,
                    ),
                )
            if self.params.window != [(0, 0)]:
                for win_index, win_item in enumerate(window_items):
                    if isinstance(window_items, list) and len(window_items) > 0:
                        region = pg.LinearRegionItem(
                            values=self.trace.window[win_index],
                            pen=pg.mkPen(color=self.params.window_color),
                            brush=window_fill,
                            hoverBrush=window_fill_hover,
                        )
                    elif isinstance(window_items, tuple):
                        region = pg.LinearRegionItem(
                            values=self.trace.window,
                            pen=pg.mkPen(color=self.params.window_color),
                            brush=window_fill,
                            hoverBrush=window_fill_hover,
                        )
                    else:
                        continue
                    win_item.append(region)
                    region.sigRegionChanged.connect(
                        make_region_callback(region, win_item, window_index=win_index)
                    )
                    region.setZValue(10 + win_index)
                    channel_tmp.addItem(region)

            if self.params.average:
                channel.channel_average(sweep_subset=self.params.sweep_subset)
                channel_tmp.plot(
                    time_array[0, :],
                    channel.average.trace,
                    pen=pg.mkPen(color=self.params.avg_color, width=2),
                )

        return win


def set_axs_color(
    trace_plot: TracePlotMatplotlib, input_axs: Axes | np.ndarray
) -> None:
    """Set the background and axis color for the given axes."""
    if isinstance(input_axs, Axes):
        input_axs.set_facecolor(trace_plot.params.bg_color)
        input_axs.spines["bottom"].set_color(trace_plot.params.axis_color)
        input_axs.spines["left"].set_color(trace_plot.params.axis_color)
        # remove top and right spines
        input_axs.spines["top"].set_visible(False)
        input_axs.spines["right"].set_visible(False)
        input_axs.tick_params(axis="x", colors=trace_plot.params.axis_color)
        input_axs.tick_params(axis="y", colors=trace_plot.params.axis_color)
        # title color
        input_axs.title.set_color(trace_plot.params.axis_color)
        input_axs.xaxis.label.set_color(trace_plot.params.axis_color)
        input_axs.yaxis.label.set_color(trace_plot.params.axis_color)
    elif isinstance(input_axs, np.ndarray):
        for axs in input_axs:
            axs.set_facecolor(trace_plot.params.bg_color)
            axs.spines["bottom"].set_color(trace_plot.params.axis_color)
            axs.spines["left"].set_color(trace_plot.params.axis_color)
            # remove top and right spines
            axs.spines["top"].set_visible(False)
            axs.spines["right"].set_visible(False)
            axs.tick_params(axis="x", colors=trace_plot.params.axis_color)
            axs.tick_params(axis="y", colors=trace_plot.params.axis_color)
            # title color
            axs.title.set_color(trace_plot.params.axis_color)
            axs.xaxis.label.set_color(trace_plot.params.axis_color)
            axs.yaxis.label.set_color(trace_plot.params.axis_color)
    else:
        raise TypeError("channel_axs must be an Axes or np.ndarray of Axes.")
