"""
Plotting functions for analyzing and visualizing electrophysiological data.
This module provides classes and functions for plotting traces and summary
measurements using both PyQtGraph and Matplotlib backends.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pyqtgraph as pg
import pandas as pd

from ephys import utils
from ephys.classes.plot.plot_params import PlotParams

from ephys.classes.class_functions import moving_average

if TYPE_CHECKING:
    from ephys.classes.window_functions import FunctionOutput
    from ephys.classes.plot.plot_trace import TracePlotPyQt
    from ephys.classes.trace import Trace


class FunctionOutputPlot:
    """Class for plotting traces and summary measurements with matplotlib."""

    def __init__(
        self, function_output: FunctionOutput, trace: Trace | None = None, **kwargs: Any
    ) -> None:
        """
        Initializes the FunctionOutputPlot class with function_output and
        additional arguments.

        Args:
            function_output (FunctionOutput): The function output to be plotted.
            **kwargs: Additional keyword arguments for plot parameters.
        """
        # self.trace = function_output.trace
        self.function_output = function_output
        self.params = PlotParams()
        if kwargs:
            self.params.update_params(**kwargs)


class FunctionOutputPyQt(FunctionOutputPlot):
    """Class for plotting traces and summary measurements with PyQtGraph."""

    def __init__(self, function_output: FunctionOutput, **kwargs: Any) -> None:
        """
        Init FunctionOutputPyQt with function_output and params.

        Args:
            function_output (FunctionOutput): Output to plot.
            **kwargs: Plot params.
        """
        super().__init__(function_output, **kwargs)
        self.params.update_params(**kwargs)

    def plot(
        self,
        trace: Trace | None = None,
        label_filter: list | str | None = None,
        **kwargs: Any,
    ) -> FunctionOutputPyQt | None:
        """
        Plots the trace and/or summary measurements.

        Args:
            label_filter (list | str | None): Labels to filter for plotting.
            **kwargs: Additional plot parameters.

        Returns:
            None
        """
        from ephys.classes.trace import Trace

        self.params.update_params(**kwargs)

        align_onset: bool = self.params.__dict__.get("align_onset", True)
        show: bool = self.params.__dict__.get("show", True)
        trace_channels: list[pg.PlotItem] | None = None
        # trace = self.function_output.trace
        if self.function_output.measurements.size == 0:
            print("No measurements to plot")
            return None
        window_groups = self.function_output.to_dataframe().groupby(["unit", "channel"])
        channel_plot_dict: dict[str, pg.PlotItem] = {}
        self.win = pg.GraphicsLayoutWidget(show=show, title="Summary Plot")

        if label_filter is None:
            label_filter = []
        if isinstance(trace, Trace):
            from ephys.classes.plot.plot_trace import TracePlotPyQt

            trace_select: Trace = trace.subset(
                channels=self.function_output.channel,
                signal_type=self.function_output.signal_type,
            )
            trace_plot_params = deepcopy(self.params.__dict__)
            trace_plot_params["show"] = False
            trace_plot_params["return_fig"] = True
            trace_plot = trace_select.plot(backend="pyqt", **trace_plot_params)
            if isinstance(trace_plot, TracePlotPyQt):
                x_range = trace_plot.params.xlim
                trace_channels = [
                    item
                    for item in trace_plot.win.items()
                    if isinstance(item, pg.PlotItem)
                ]
                self.win: pg.GraphicsLayoutWidget = trace_plot.win
                channel_plot_dict = {
                    channel.getAxis("left").labelText: channel
                    for channel in trace_channels
                }
        self.win.setBackground(self.params.bg_color)

        channel_0: pg.PlotItem | None = None
        from ephys.classes.class_functions import moving_average

        if self.params.align_onset:
            x_axis = self.function_output.location
        else:
            if trace is not None:
                x_axis = self.function_output.location + np.array(
                    [
                        trace.time[int(sweep - 1), 0]
                        for sweep in self.function_output.sweep
                    ]
                )
            else:
                x_axis = self.function_output.time
            x_range = (np.min(x_axis), np.max(x_axis))
        for channel_index, ((unit, channel), group) in enumerate(window_groups):
            if isinstance(trace_channels, list):
                channel_plot = channel_plot_dict[f"Channel {int(channel)} " f"({unit})"]
            else:
                channel_plot: pg.PlotItem = self.win.addPlot(row=channel_index, col=0)  # type: ignore
            if channel_index == 0:
                channel_0 = channel_plot
                channel_plot.addLegend()
            channel_plot.setXLink(channel_0)  # type: ignore
            channel_box = channel_plot.getViewBox()
            channel_box.setXRange(x_range[0], x_range[1])  # type: ignore
            subgroups = group.groupby("label")
            label_count = len(subgroups)
            for i, (label, subgroup) in enumerate(subgroups):
                label_colors = utils.color_picker_qcolor(
                    length=label_count, index=i, color="gist_rainbow", alpha=0.5
                )
                if not align_onset:
                    y_smooth = moving_average(
                        subgroup["measurements"].to_numpy(),
                        subgroup.index.size // 10,
                    )
                    channel_plot.plot(
                        x_axis[subgroup.index],
                        y_smooth,
                        pen=label_colors,
                        alpha=0.4,
                        width=2.0,
                    )
                channel_plot.setLabel("left", f"Channel {int(channel)} " f"({unit})")
                channel_plot.scatterPlot(
                    x_axis[subgroup.index],
                    subgroup["measurements"].values,
                    pen=label_colors,
                    symbol="o",
                    symbolSize=10,
                    symbolPen="w",
                    symbolBrush=label_colors,
                    name=f"{label}",
                )
        if show:
            self.show()
        return self

    def show(self) -> pg.GraphicsLayoutWidget | None:
        if self.win is not None:
            self.win.show()
            return self.win


class FunctionOutputMatplotlib(FunctionOutputPlot):
    """Class for plotting traces and summary measurements with Matplotlib."""

    def __init__(self, function_output: FunctionOutput, **kwargs: Any) -> None:
        """
        Initializes the FunctionOutputMatplotlib class.

        Args:
            function_output (FunctionOutput): The function output to plot.
            theme (str, optional): Plot theme ('dark' or 'light'). Default 'dark'.
            **kwargs: Additional plot parameters.
        """
        super().__init__(function_output, **kwargs)
        self.params.update_params(**kwargs)

    def plot(
        self,
        trace: Trace | None = None,
        label_filter: list | str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray] | None:
        """
        Plots the trace and/or summary measurements.

        Args:
            trace (Trace, optional): Trace object to plot. If None, only summary
            measurements are plotted. Default is None.
            label_filter (list | str | None, optional): Labels to filter for plotting.
            **kwargs: Additional plot parameters.

        Returns:
            None
        """
        self.params.update_params(**kwargs)

        align_onset: bool = self.params.__dict__.get("align_onset", True)
        show: bool = self.params.__dict__.get("show", True)
        return_fig: bool = self.params.__dict__.get("return_fig", False)

        fig_out: Figure | None = None
        channel_axs_out: Axes | np.ndarray | None = None

        if self.function_output.measurements.size == 0:
            print("No measurements to plot")
            return None
        if label_filter is None:
            label_filter = []
        if trace is not None:  # plot_output.plot(
            #     trace=trace,
            #     label_filter=label_filter,
            # )
            trace_select: Trace = trace.subset(
                channels=self.function_output.channel,
                signal_type=self.function_output.signal_type,
            )
            trace_plot_params = deepcopy(self.params.__dict__)
            trace_plot_params["show"] = False
            trace_plot_params["return_fig"] = True
            tmp = trace_select.plot(
                **trace_plot_params,
            )
            if isinstance(tmp, tuple):
                fig, channel_axs = tmp
            else:
                raise TypeError(
                    "Trace plot did not return a valid figure or axes object."
                )
        else:
            fig, channel_axs = plt.subplots(
                np.unique(self.function_output.channel).size, 1, sharex=True
            )
        channel_count = np.unique(self.function_output.channel).size
        unique_labels = np.unique(self.function_output.label)
        if align_onset:
            x_axis = self.function_output.location.copy()
        else:
            if trace is not None:
                x_axis = self.function_output.location + np.array(
                    [
                        trace.time[int(sweep - 1), 0]
                        for sweep in self.function_output.sweep
                    ]
                )
            else:
                x_axis = self.function_output.time.copy()
        for color_index, label in enumerate(unique_labels):
            # add section to plot on channel by channel basis
            for channel_index, channel_number in enumerate(
                np.unique(self.function_output.channel)
            ):
                tmp_axs: Axes | None = None
                if channel_count > 1:
                    if isinstance(channel_axs, np.ndarray):
                        tmp_axs = channel_axs[channel_index]
                else:
                    if isinstance(channel_axs, Axes):
                        tmp_axs = channel_axs
                if len(label_filter) > 0:
                    if label not in label_filter:
                        continue
                label_idx = np.where(
                    (self.function_output.label == label)
                    & (self.function_output.channel == channel_number)
                )
                label_colors = utils.color_picker(
                    length=len(unique_labels), index=color_index, color="gist_rainbow"
                )
                if not align_onset:
                    y_smooth = moving_average(
                        self.function_output.measurements[label_idx],
                        len(label_idx[0]) // 10,
                    )
                    if tmp_axs is not None:
                        tmp_axs.plot(
                            x_axis[label_idx],
                            y_smooth,
                            color=label_colors,
                            alpha=0.4,
                            lw=2,
                        )
                if tmp_axs is not None:
                    tmp_axs.plot(
                        x_axis[label_idx],
                        self.function_output.measurements[label_idx],
                        "o",
                        color=label_colors,
                        alpha=0.5,
                        label=label,
                    )
        if trace is None:
            if isinstance(channel_axs, np.ndarray):
                for channel_index, channel_number in enumerate(
                    np.unique(self.function_output.channel)
                ):
                    if channel_index == len(channel_axs) - 1:
                        channel_axs[channel_index].set_xlabel("Time (s)")
                    channel_unit = np.unique(
                        self.function_output.unit[
                            self.function_output.channel == channel_number
                        ]
                    )
                    channel_axs[channel_index].set_ylabel(
                        f"Channel {int(channel_number)} " f"({channel_unit[0]})"
                    )
            else:
                if isinstance(channel_axs, Axes):
                    channel_axs.set_xlabel("Time (s)", color=self.params.axis_color)
                    channel_unit = np.unique(self.function_output.unit)
                    channel_number = np.unique(self.function_output.channel)
                    channel_axs.set_ylabel(
                        f"Channel {int(channel_number)} " f"({channel_unit[0]})",
                        color=self.params.axis_color,
                    )
        if isinstance(channel_axs, np.ndarray):
            channel_axs[0].legend(loc="best")
            for single_axs in channel_axs:
                single_axs.set_facecolor(self.params.bg_color)
                single_axs.spines[["top", "right"]].set_visible(False)
                single_axs.spines["bottom"].set_color(self.params.axis_color)
                single_axs.spines["left"].set_color(self.params.axis_color)
                single_axs.xaxis.label.set_color(
                    self.params.axis_color,
                )
                single_axs.yaxis.label.set_color(
                    self.params.axis_color,
                )
                single_axs.tick_params(
                    axis="x",
                    colors=self.params.axis_color,
                )
                single_axs.tick_params(
                    axis="y",
                    colors=self.params.axis_color,
                )
        else:
            if isinstance(channel_axs, Axes):
                channel_axs.set_facecolor(self.params.bg_color)
                channel_axs.spines[["top", "right"]].set_visible(False)
                channel_axs.spines["bottom"].set_color(self.params.axis_color)
                channel_axs.spines["left"].set_color(self.params.axis_color)
                channel_axs.xaxis.label.set_color(
                    self.params.axis_color,
                )
                channel_axs.yaxis.label.set_color(
                    self.params.axis_color,
                )
                channel_axs.tick_params(
                    axis="x",
                    colors=self.params.axis_color,
                )
                channel_axs.tick_params(
                    axis="y",
                    colors=self.params.axis_color,
                )
                channel_axs.legend(loc="best")
        fig.set_facecolor(self.params.bg_color)

        if return_fig:
            fig_out = deepcopy(fig)
            channel_axs_out = deepcopy(channel_axs)
        if show:
            plt.show()
        if return_fig and fig_out is not None and channel_axs_out is not None:
            return fig_out, channel_axs_out
        return None
