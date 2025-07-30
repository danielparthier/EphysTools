from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtWidgets import QVBoxLayout, QFrame, QWidget
from pyqtgraph import GraphicsLayoutWidget
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from pyqtgraph.widgets.GraphicsLayoutWidget import GraphicsLayoutWidget
from PySide6.QtCore import QObject, Signal, Slot, QRunnable
from PySide6.QtWidgets import QLabel


if TYPE_CHECKING:
    from ephys.GUI.gui_app import MainWindow
    from ephys.classes.trace import Trace


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

    # def add_trace_plot(self, trace: Trace, **kwargs) -> None:
    #     """Add a trace plot widget to the plot area using a worker thread."""
    #     # Show loading indicator
    #     loading_label = QLabel("Creating plot...")
    #     self.plot_area_layout.addWidget(loading_label)

    #     # Create worker
    #     worker = PlotWorker(
    #         trace, alpha=0.5, theme=self.main_window.session_info.theme, **kwargs
    #     )

    #     # Connect signals
    #     worker.signals.finished.connect(self._handle_plot_finished)
    #     worker.signals.error.connect(self._handle_plot_error)

    #     # Store reference to loading label
    #     worker.loading_label = loading_label

    #     # Start the worker
    #     self.main_window.threadpool.start(worker)

    # def _handle_plot_finished(self, result):
    #     """Handle completed plot from worker thread."""
    #     # Get the sender
    #     worker = self.sender().parent()

    #     # Remove loading indicator
    #     if hasattr(worker, "loading_label"):
    #         self.plot_area_layout.removeWidget(worker.loading_label)
    #         worker.loading_label.deleteLater()

    #     # Add the plot widget
    #     if isinstance(result, GraphicsLayoutWidget):
    #         self.plot_area_layout.addWidget(result)

    # def _handle_plot_error(self, error_message):
    #     """Handle error from worker thread."""
    #     # Get the sender
    #     worker = self.sender().parent()

    #     # Remove loading indicator
    #     if hasattr(worker, "loading_label"):
    #         self.plot_area_layout.removeWidget(worker.loading_label)
    #         worker.loading_label.deleteLater()

    #     # Show error message
    #     error_label = QLabel(f"Error creating plot: {error_message}")
    #     error_label.setStyleSheet("color: red")
    #     self.plot_area_layout.addWidget(error_label)

    def add_trace_plot(self, trace: Trace, **kwargs) -> None:
        """Add a trace plot widget to the plot area."""
        trace_plot: None | GraphicsLayoutWidget | tuple[Figure, Axes] = trace.plot(
            backend="pyqt",
            alpha=0.5,
            theme=self.main_window.session_info.theme,
            **kwargs,
        )
        if isinstance(trace_plot, GraphicsLayoutWidget):
            self.plot_area_layout.addWidget(trace_plot)

    def cleanup(self):
        """Clean up resources before widget is destroyed"""
        # Release any held resources, disconnect signals, etc.
        for i in reversed(range(self.plot_area_layout.count())):
            item = self.plot_area_layout.itemAt(i)
            if item.widget():
                widget = item.widget()
                self.plot_area_layout.removeWidget(widget)
                widget.setParent(None)
