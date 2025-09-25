"""
This module defines the CurveArray class, which is used to store and manage
2D curves represented by their x and y coordinates. Using CurveArray increases
the efficiency of curve handling in pyqtgraph.
"""

import numpy as np
from pyqtgraph import PlotCurveItem


class CurveArray:
    """Class to store 2D curves for efficient plotting in pyqtgraph."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.array([])
        self.y = np.array([])
        self.start_index = np.array([], dtype="int64")
        self.end_index = np.array([], dtype="int64")
        self.sweep_index = np.array([], dtype="int64")
        self.add_array(x, y)

    def add_array(self, x: np.ndarray, y: np.ndarray) -> None:
        x_nan = np.array([])
        if y.ndim == 1:
            y_nan = np.append(y, np.nan)
            if x.ndim == 1:
                x_nan = np.append(x, np.nan)
            else:
                print("x is not matching dimensions of y.")
                return None
        else:
            y_nan = np.hstack((y, np.full((y.shape[0], 1), np.nan))).flatten()
            if x.ndim == 1:
                x_nan = np.tile(np.append(x, np.nan), y.shape[0])
            if x.shape == y.shape:
                x_nan = np.hstack((x, np.full((x.shape[0], 1), np.nan))).flatten()
            if y_nan.shape != x_nan.shape:
                print("x and y shapes do not match.")
                return None
        starting_point = self.x.size - 1
        if starting_point < 0:
            starting_point = 0

        self.sweep_index = np.append(
            self.sweep_index, np.arange(y.shape[0]) + self.sweep_index.size
        )
        self.x = np.append(self.x, x_nan)
        self.y = np.append(self.y, y_nan)
        start_index = np.append(0, np.where(np.isnan(x_nan))[0])
        end_index = np.append(start_index[1:], x_nan.size - 1)
        self.start_index = np.append(
            self.start_index, start_index + self.start_index.size
        )
        self.end_index = np.append(self.end_index, end_index + self.end_index.size)


class HighlightCurve(PlotCurveItem):
    def __init__(
        self, trace_curve: "TraceCurve", sweep_index: int | None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sweep_index = sweep_index
        self.trace_curve = trace_curve
        self.update_data(sweep_index=sweep_index)

    def update_data(
        self, sweep_index: int | None, sweep_trace: "TraceCurve |  None" = None
    ) -> None:
        if sweep_trace is not None:
            self.trace_curve = sweep_trace
        if isinstance(self.trace_curve, TraceCurve):
            data_xy = self.trace_curve.return_sweep_data(sweep_index)
            if data_xy is not None:
                self.setData(x=data_xy[0], y=data_xy[1])
                self.sweep_index = sweep_index
            else:
                self.setData(x=np.array([]), y=np.array([]))
                self.sweep_index = None


class TraceCurve(PlotCurveItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compressed = False
        self.sweep_index = None
        self.start_index = None
        self.end_index = None

    def add_compressed_data(self, data: CurveArray) -> None:
        self.setData(x=data.x, y=data.y)
        self.compressed = True
        self.sweep_index = data.sweep_index
        self.start_index = data.start_index
        self.end_index = data.end_index

    def add_sweep_data(self, x: np.ndarray, y: np.ndarray, sweep_index: int) -> None:
        self.setData(x=x, y=y)
        self.compressed = False
        self.sweep_index = sweep_index
        self.start_index = np.array([0])
        self.end_index = np.array([x.size - 1])

    def return_sweep_data(
        self, sweep_index: int | None, silent=True
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if (
            self.compressed
            and self.start_index is not None
            and self.end_index is not None
            and self.xData is not None
            and self.yData is not None
            and self.sweep_index is not None
        ):
            idx = np.where(sweep_index == self.sweep_index)[0]
            if idx.size > 0:
                start = self.start_index[idx[0]]
                end = self.end_index[idx[0]]
                return self.xData[start:end], self.yData[start:end]
            else:
                if not silent:
                    print(f"Sweep index {sweep_index} not found in compressed data.")
                return None
        elif (
            not self.compressed
            and self.sweep_index == sweep_index
            and self.xData is not None
            and self.yData is not None
        ):
            return self.xData, self.yData
        else:
            if not silent:
                print(f"Sweep index {sweep_index} not found in uncompressed data.")
            return None

    def return_sweep_highlight(
        self, sweep_index: int | None, *args, **kwargs
    ) -> HighlightCurve | None:
        data_xy = self.return_sweep_data(sweep_index)
        if data_xy is not None:
            highlight_trace = HighlightCurve(
                trace_curve=self, sweep_index=sweep_index, *args, **kwargs
            )
            return highlight_trace
        else:
            return None
