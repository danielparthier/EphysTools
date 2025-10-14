from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

from PySide6.QtCore import QObject, QRunnable, Signal, Slot
from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from ephys.GUI.gui_app import MainWindow
    from ephys.classes.trace import Trace
    from ephys.classes.plot.plot_trace import TracePlotPyQt


class PlotWorker(QRunnable):
    """Worker thread for creating plots."""

    class WorkerSignals(QObject):
        """Signals for the worker thread."""

        finished = Signal(object)  # Emits the plot widget when done
        error = Signal(str)  # Emits error message if fails

    def __init__(self, trace, **kwargs):
        super().__init__()
        self.trace = trace
        self.kwargs = kwargs
        self.signals = self.WorkerSignals()

    @Slot()
    def run(self):
        """Create the plot in a background thread."""
        try:
            # Call the plotting function
            plot_result = self.trace.plot(backend="pyqt", **self.kwargs)
            # Emit the result
            self.signals.finished.emit(plot_result)
        except Exception as e:
            self.signals.error.emit(str(e))


class TracePlotWindow(QWidget):
    def __init__(self, main_window: MainWindow, file_name: str) -> None:
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle(f"Trace Plot - {file_name}")
        tab_layout = QVBoxLayout(self)
        self.setLayout(tab_layout)
        self.main_window.trace_plot.addTab(self, file_name)
        self.main_window.trace_plot.setTabsClosable(True)
        self.main_window.trace_plot.setMovable(True)
        self.main_window.trace_plot.tabCloseRequested.connect(
            self.main_window.close_tab
        )

        # Trace List
        if isinstance(self.main_window.trace_list, list):
            self.trace_list = self.main_window.trace_list
        else:
            self.main_window.trace_list = []
            self.trace_list = []

        # Create a frame for the plot
        plot_frame = QFrame()
        plot_frame.setFrameShape(QFrame.Shape.StyledPanel)
        plot_frame.setMinimumWidth(800)
        plot_frame.setMinimumHeight(600)
        plot_frame.setMaximumHeight(1200)
        plot_frame.setMaximumWidth(2400)
        tab_layout.addWidget(plot_frame)

        # Create a layout for the plot frame
        self.plot_layout = QVBoxLayout(plot_frame)
        plot_frame.setLayout(self.plot_layout)

        # Create a plot area
        plot_area = QFrame()
        plot_area.setFrameShape(QFrame.Shape.StyledPanel)
        plot_area.setMinimumWidth(800)
        plot_area.setMinimumHeight(600)
        plot_area.setMaximumHeight(1200)
        plot_area.setMaximumWidth(2400)
        self.plot_layout.addWidget(plot_area)

        # Create a layout for the plot area
        self.plot_area_layout = QVBoxLayout(plot_area)
        plot_area.setLayout(self.plot_area_layout)

    def add_trace_plot(self, trace: Trace, color="viridis", **kwargs) -> None:
        from ephys.classes.plot.plot_trace import TracePlotPyQt

        """Add a trace plot widget to the plot area."""
        trace_plot: None | TracePlotPyQt | tuple[Figure, Axes] = trace.plot(
            backend="pyqt",
            alpha=kwargs.get("alpha", 1.0),
            color=color,
            show=False,
            return_fig=True,
            theme=self.main_window.session_info.theme,
            **kwargs,
        )

        if isinstance(trace_plot, TracePlotPyQt):
            self.trace_list.append(trace_plot)
            self.plot_area_layout.addWidget(trace_plot.win)

    def update_theme(self, theme: str) -> None:
        """Update the theme of the plot area."""
        for trace_plot in self.trace_list:
            if hasattr(trace_plot, "update_theme"):
                trace_plot.update_theme(theme)
            else:
                print(f"Trace plot {trace_plot} does not have update_theme method.")

    def cleanup(self):
        """Clean up resources before widget is destroyed"""
        # Release any held resources, disconnect signals, etc.
        for i in reversed(range(self.plot_area_layout.count())):
            item = self.plot_area_layout.itemAt(i)
            if item.widget():
                widget = item.widget()
                self.plot_area_layout.removeWidget(widget)
                widget.setParent(None)
