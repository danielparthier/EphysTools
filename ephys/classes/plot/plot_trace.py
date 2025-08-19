"""
This module contains classes for plotting traces using different backends.
It includes support for both PyQtGraph and Matplotlib, allowing for flexible
visualization of trace data
"""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import pyqtgraph as pg
from pyqtgraph.Qt.QtGui import QColor
import quantities

from ephys.classes.trace import Trace
from ephys.classes.class_functions import _get_sweep_subset
from ephys import utils
from ephys.classes.plot.plot_params import PlotParams, _set_axs_color

# switch off antialiasing for pyqtgraph
# pg.setConfigOptions(antialias=False)


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

    def handle_windows(self) -> list:
        """Handle window interaction between plot parameters and trace"""

        # Initialize windows to display
        windows_to_display = []

        # Case 1: Add the plot windows to trace.window
        if self.params.window_mode == "add_to_trace":
            # Convert input to proper format
            if isinstance(self.params.window, tuple):
                plot_windows = [self.params.window]
            elif isinstance(self.params.window, list):
                plot_windows = self.params.window
            else:
                raise TypeError("Window must be a tuple or list of tuples.")

            # Initialize trace.window if it doesn't exist
            if self.trace.window is None:
                self.trace.window = []

            # Add new windows from plot parameters
            if plot_windows != [(0, 0)]:
                # Add each new window to trace.window
                for win in plot_windows:
                    if win not in self.trace.window:
                        self.trace.window.append(win)

            # Use the updated trace.window for display
            windows_to_display = self.trace.window

        # Case 2: Use existing trace.window
        elif self.params.window_mode == "use_trace":
            # Use existing trace.window for display
            if self.trace.window is None or len(self.trace.window) == 0:
                windows_to_display = []
            else:
                windows_to_display = self.trace.window

        # Case 3: Use windows from plot without modifying trace.window
        else:  # self.params.window_mode == "use_plot"
            if isinstance(self.params.window, tuple):
                windows_to_display = [self.params.window]
            elif isinstance(self.params.window, list):
                windows_to_display = self.params.window
            else:
                raise TypeError("Window must be a tuple or list of tuples.")
        return windows_to_display

    def _prepare_time_array(self, trace_select, **kwargs) -> quantities.Quantity:
        """Prepare time array based on alignment settings"""
        if kwargs:
            self.align_onset = kwargs.get("align_onset", self.params.align_onset)
            self.cumulative = kwargs.get("cumulative", self.params.cumulative)
            self.stimulus_interval = kwargs.get(
                "stimulus_interval", self.params.stimulus_interval
            )
        if self.params.align_onset:
            return trace_select.set_time(
                align_to_zero=self.params.align_onset,
                cumulative=self.params.cumulative,
                stimulus_interval=self.params.stimulus_interval,
                overwrite_time=False,
            )
        return trace_select.time


class TracePlotMatplotlib(TracePlot):
    """Class for plotting traces using Matplotlib."""

    def __init__(self, trace: Trace, backend: str = "matplotlib", **kwargs) -> None:
        super().__init__(trace=trace, backend=backend, **kwargs)

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
            self.params.update_params(**kwargs)
        if len(self.params.channels) == 0:
            self.params.channels = self.trace.channel_information.channel_number
        sweep_subset = _get_sweep_subset(
            array=self.trace.time, sweep_subset=self.params.sweep_subset
        )
        trace_select = self.trace.subset(
            channels=self.params.channels,
            signal_type=self.params.signal_type,
            sweep_subset=self.params.sweep_subset,
        )

        fig, channel_axs = plt.subplots(len(trace_select.channel), 1, sharex=True)
        # color background and axis

        # change color of all axes
        self._set_axs_color(input_axs=channel_axs)

        fig.set_facecolor(self.params.bg_color)
        if isinstance(channel_axs, Axes):
            channel_axs.set_facecolor(color=self.params.bg_color)
        elif isinstance(channel_axs, np.ndarray):
            for axs in channel_axs:
                axs.set_facecolor(self.params.bg_color)
        else:
            raise TypeError("channel_axs must be an Axes or np.ndarray of Axes.")
        if len(trace_select.channel) == 0:
            print("No traces found.")
            return None

        time_array = self._prepare_time_array(trace_select, **kwargs)

        windows_to_display = self.handle_windows()

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

                if windows_to_display != [(0, 0)]:
                    if (
                        isinstance(windows_to_display, list)
                        and len(windows_to_display) != 0
                    ):
                        for win in windows_to_display:
                            if isinstance(tmp_axs, Axes):
                                tmp_axs.axvspan(
                                    xmin=win[0],
                                    xmax=win[1],
                                    color=self.params.window_color,
                                    alpha=0.5,
                                )
                if self.params.average:
                    channel.channel_average(sweep_subset=sweep_subset)
                    tmp_axs.plot(
                        time_array[0, :],
                        channel.average.trace,
                        color=self.params.avg_color,
                    )
                tmp_axs.set_ylabel(
                    ylabel=(
                        f"Channel {trace_select.channel_information.channel_number[channel_index]} "
                        f"({trace_select.channel_information.unit[channel_index]})"
                    )
                )
        if isinstance(tmp_axs, Axes):
            tmp_axs.set_xlabel(
                xlabel=f"Time ({trace_select.time.units.dimensionality.string})"
            )
            if len(self.params.xlim) == 2:
                if self.params.xlim[0] < self.params.xlim[1]:
                    tmp_axs.set_xlim(
                        left=self.params.xlim[0], right=self.params.xlim[1]
                    )
                else:
                    tmp_axs.set_xlim(left=time_array.min(), right=time_array.max())
        plt.tight_layout()
        if self.params.show:
            plt.show()
        if self.params.return_fig:
            return fig, channel_axs
        return None

    def _set_axs_color(self, input_axs: Axes | np.ndarray) -> None:
        _set_axs_color(params=self.params, input_axs=input_axs)


class TracePlotPyQt(TracePlot):
    """Class for plotting traces using PyQtGraph."""

    def __init__(self, trace: Trace, backend: str = "pyqt", **kwargs) -> None:
        super().__init__(trace=trace, backend=backend, **kwargs)
        self.win = pg.GraphicsLayoutWidget(show=self.params.show, title="Trace Plot")
        self.highlight: dict[str, int | None | QColor] = {
            "sweep_index": None,
            "color": QColor(),
        }

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
                Defaults to 1.
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
            return_fig (bool, optional): Whetherupdate_params to return the figure.
                Defaults to False.

        Returns:
            None or Figure: If show is True, returns None. If return_fig is True,
            returns the figure.
        """
        if kwargs:
            self.params.update_params(**kwargs)

        def sync_channels(source_region, channel_items, window_index=0):
            # Get region bounds from the source region
            min_val, max_val = source_region.getRegion()

            # Update all other regions
            for r in channel_items:
                if r is not source_region:
                    r.blockSignals(True)
                    r.setRegion((min_val, max_val))
                    r.blockSignals(False)
            # Update the trace window property only if we're not in "use_plot" mode
            if self.params.window_mode != "use_plot":
                if isinstance(self.trace.window, list) and window_index < len(
                    self.trace.window
                ):
                    self.trace.window[window_index] = (min_val, max_val)
                else:
                    # Handle case where window_index is out of bounds or trace.window is not a list
                    pass

        def make_region_callback(region_obj, channel_items, window_index=0):
            return lambda: sync_channels(
                source_region=region_obj,
                channel_items=channel_items,
                window_index=window_index,
            )

        if len(self.params.channels) == 0:
            self.params.channels = self.trace.channel_information.channel_number
        trace_select = self.trace.subset(
            channels=self.params.channels,
            signal_type=self.params.signal_type,
            sweep_subset=self.params.sweep_subset,
        )

        time_array = self._prepare_time_array(trace_select, **kwargs)

        if len(self.params.xlim) > 2:
            raise ValueError("xlim must be a tuple of two values.")
        if len(self.params.xlim) < 2 or self.params.xlim == (0, 0):
            self.params.xlim = (
                np.min(time_array.magnitude),
                np.max(time_array.magnitude),
            )

        self.win = pg.GraphicsLayoutWidget(show=self.params.show, title="Trace Plot")
        self.win.setBackground(self.params.bg_color)
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
        windows_to_display = self.handle_windows()
        if windows_to_display is not None:
            window_items: list[list[pg.LinearRegionItem]] = [
                [] for _ in range(len(windows_to_display))
            ]
        else:
            window_items: list[list[pg.LinearRegionItem]] = [[]]

        region: pg.LinearRegionItem | None = None
        channel_0: pg.PlotItem | None = None
        for channel_index, channel in enumerate(trace_select.channel):
            channel_tmp = self.win.addPlot(row=channel_index, col=0)  # type: ignore
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
                qt_color = utils.color_picker_qcolor(
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
                        width=self.params.line_width,
                    ),
                )
            if windows_to_display != [(0, 0)]:
                for win_index, win_item in enumerate(window_items):
                    if isinstance(window_items, list) and len(window_items) > 0:
                        region = pg.LinearRegionItem(
                            values=windows_to_display[win_index],
                            pen=pg.mkPen(color=self.params.window_color),
                            brush=window_fill,
                            hoverBrush=window_fill_hover,
                        )
                    elif isinstance(window_items, tuple):
                        region = pg.LinearRegionItem(
                            values=windows_to_display,
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
                    for line in region.lines:
                        line.setPen(self.params.window_color)
                        line.setHoverPen(self.params.window_color_hover)
                    channel_tmp.addItem(region)

            if self.params.average:
                channel.channel_average(sweep_subset=self.params.sweep_subset)
                channel_tmp.plot(
                    time_array[0, :],
                    channel.average.trace,
                    pen=pg.mkPen(color=self.params.avg_color, width=2),
                )

    # return self.win

    def update_theme(self, theme: str, **kwargs) -> None:
        """Update the theme of the plot."""
        self.params.apply_theme(theme)
        self.params.update_params(**kwargs)
        # Update the background and axis colors
        self.win.setBackground(self.params.bg_color)
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

        # reset highlights
        self.highlight["sweep_index"] = None
        self.highlight["color"] = QColor()
        window_items = self.handle_windows()
        if self.trace.window is not None:
            # Update existing regions with new theme colors
            for i, region in enumerate(window_items):
                if isinstance(region, pg.LinearRegionItem):
                    region.setBrush(window_fill)
                    region.setHoverBrush(window_fill_hover)
                    for line in region.lines:
                        line.setPen(pg.mkPen(color=self.params.window_color))
                        line.setHoverPen(pg.mkPen(color=self.params.window_color_hover))
        for plot_item in self.win.items():
            if isinstance(plot_item, pg.PlotItem):
                plot_item.setLabel(
                    "left",
                    color=self.params.axis_color,
                    text=plot_item.getAxis("left").labelText,
                )
                plot_item.setLabel(
                    "bottom",
                    color=self.params.axis_color,
                    text=plot_item.getAxis("bottom").labelText,
                )
                item_list = plot_item.listDataItems()
                for i, curve in enumerate(item_list):
                    curve.setPen(
                        color=utils.color_picker_qcolor(
                            length=len(item_list),
                            index=i,
                            alpha=self.params.alpha,
                            color=self.params.color,
                        )
                    )

    def sweep_highlight(
        self, sweep_index: int | None = None, color="red", alpha=0.4
    ) -> None:
        """Highlight the specified sweep in the plot."""
        for plot_item in self.win.items():
            if isinstance(plot_item, pg.PlotItem):
                item_list = plot_item.items
                if isinstance(self.highlight["sweep_index"], int):
                    item_list[self.highlight["sweep_index"]].setPen(
                        color=self.highlight["color"]
                    )
                    item_list[self.highlight["sweep_index"]].setZValue(0)
                if alpha < self.params.alpha:
                    local_alpha = alpha
                    if sweep_index is None:
                        local_alpha = self.params.alpha
                    for sweep in item_list:
                        sweep.setAlpha(local_alpha, True)
        for plot_item in self.win.items():
            if isinstance(plot_item, pg.PlotItem):
                item_list = plot_item.items

                if isinstance(sweep_index, int):
                    if sweep_index < len(item_list):
                        self.highlight["sweep_index"] = sweep_index
                        self.highlight["color"] = (
                            item_list[self.highlight["sweep_index"]].opts["pen"].color()
                        )
                        item_list[sweep_index].setPen(pg.mkPen(color=color, width=2))
                        item_list[sweep_index].setAlpha(1.0, True)
                        item_list[sweep_index].setZValue(1)
                else:
                    self.highlight["sweep_index"] = None
                    self.highlight["color"] = QColor()

    def show(self) -> None:
        """Show the plot window."""
        self.win.show()
