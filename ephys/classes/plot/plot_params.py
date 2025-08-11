"""
This module defines the PlotParams class, which encapsulates configurable
parameters for plotting electrophysiological traces. It provides options for
customizing signal type, channels, colors, transparency, averaging, alignment,
sweep selection, plot appearance, time windows, axis limits, and theme. The
class supports validation, parameter updates, theme application, and conversion
to dictionary format for flexible and robust plotting workflows.
"""

from typing import Any
import numpy as np


class PlotParams:
    """Parameters for plotting traces.

    Args:
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
        window_mode (str): Mode for handling windows
            ('use_plot', 'use_trace', 'add_to_trace').
        theme (str): Theme for the plot ('dark' or 'light').
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize PlotParams with default values or provided kwargs."""
        self.theme = kwargs.get("theme", "dark")
        self.apply_theme(self.theme)
        self.signal_type = kwargs.get("signal_type", "")
        self.channels = kwargs.get("channels", np.array([], dtype=np.int64))
        self.color = kwargs.get("color", self.color)
        self.alpha = kwargs.get("alpha", 1.0)
        self.average = kwargs.get("average", False)
        self.avg_color = kwargs.get("avg_color", "red")
        self.align_onset = kwargs.get("align_onset", True)
        self.cumulative = kwargs.get("cumulative", False)
        self.stimulus_interval = kwargs.get("stimulus_interval", 0.0)
        self.sweep_subset = kwargs.get("sweep_subset", None)
        self.bg_color = kwargs.get("bg_color", self.bg_color)
        self.axis_color = kwargs.get("axis_color", self.axis_color)
        self.window = kwargs.get("window", [(0, 0)])
        self.window_color = kwargs.get("window_color", self.window_color)
        self.xlim = kwargs.get("xlim", (0, 0))
        self.show = kwargs.get("show", True)
        self.return_fig = kwargs.get("return_fig", False)
        self.window_mode = kwargs.get(
            "window_mode", "add_to_trace"
        )  # Default mode for handling windows
        self.line_width = kwargs.get("line_width", 1.0)

    def apply_theme(self, theme="dark") -> None:
        """Apply the specified theme to the plot parameters.

        Args:
            theme (str): Theme for the plot ('dark' or 'light').
        """
        self.theme = theme
        if self.theme == "dark":
            self.color = "#d0d0d0ff"
            self.bg_color = "#292929"
            self.axis_color = "white"
            self.window_color = "#8A8A8A"
            self.window_color_hover = "#EEEEEE"
        elif self.theme == "light":
            self.color = "#1D1D1DFF"
            self.bg_color = "#FFFFFF"
            self.axis_color = "#000000"
            self.window_color = "#ACACAC"
            self.window_color_hover = "#5E5E5E"
        else:
            raise ValueError(f"Invalid theme: {self.theme}. Must be 'dark' or 'light'.")

    def update_params(self, **kwargs: Any) -> None:
        """Update the plot parameters with provided keyword arguments."""
        if "theme" in kwargs:
            self.apply_theme(kwargs["theme"])
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"PlotParams has no attribute '{key}'.")

        self.validate()

    def validate(self) -> None:
        """Validate the plot parameters."""
        if not isinstance(self.signal_type, str):
            raise TypeError("signal_type must be a string.")
        if not isinstance(self.channels, np.ndarray):
            raise TypeError("channels must be a numpy ndarray.")
        if not isinstance(self.window, list) or not all(
            isinstance(win, tuple) and len(win) == 2 for win in self.window
        ):
            raise TypeError("window must be a list of tuples (start, end).")
        if (
            self.xlim is not None
            and not isinstance(self.xlim, tuple)
            or len(self.xlim) > 2
            or len(self.xlim) == 1
        ):
            raise TypeError("xlim must be a tuple of two values (min, max).")
        if not isinstance(self.show, bool):
            raise TypeError("show must be a boolean value.")
        if not isinstance(self.return_fig, bool):
            raise TypeError("return_fig must be a boolean value.")
        if self.window_mode not in ["use_plot", "use_trace", "add_to_trace"]:
            raise ValueError(
                f"Invalid window_mode: {self.window_mode}. "
                "Must be 'use_plot', 'use_trace', or 'add_to_trace'."
            )
        if not isinstance(self.align_onset, bool):
            raise TypeError("align_onset must be a boolean value.")
        if not isinstance(self.cumulative, bool):
            raise TypeError("cumulative must be a boolean value.")
        if not isinstance(self.stimulus_interval, (int, float)):
            raise TypeError("stimulus_interval must be a number (int or float).")
        if not isinstance(self.line_width, (int, float)):
            raise TypeError("line_width must be a number (int or float).")
        if not isinstance(self.theme, str) and self.theme not in ["dark", "light"]:
            raise TypeError("theme must be a string and either 'dark' or 'light'.")

    def to_dict(self) -> dict:
        """Convert the plot parameters to a dictionary."""
        return {
            "signal_type": self.signal_type,
            "channels": self.channels.tolist(),
            "color": self.color,
            "alpha": self.alpha,
            "average": self.average,
            "avg_color": self.avg_color,
            "align_onset": self.align_onset,
            "cumulative": self.cumulative,
            "stimulus_interval": self.stimulus_interval,
            "line_width": self.line_width,
            "sweep_subset": self.sweep_subset,
            "bg_color": self.bg_color,
            "axis_color": self.axis_color,
            "window": self.window,
            "window_color": self.window_color,
            "xlim": self.xlim,
            "show": self.show,
            "return_fig": self.return_fig,
            "window_mode": self.window_mode,
            "theme": self.theme,
        }

    def __iter__(self):
        """Iterate over the plot parameters."""
        for key, value in self.to_dict().items():
            yield key, value
