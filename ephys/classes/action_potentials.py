"""
Action Potentials Class
This module defines the ActionPotentials class, which is used to handle action
potentials in voltage traces.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from scipy import signal
import numpy as np
from quantities import Quantity
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd

if TYPE_CHECKING:
    from ephys.classes.events import Events
    from ephys.classes.voltage import VoltageTrace


def _ap_extract(
    sweep: Quantity,
    window_len: float = 8,
    detection_threshold: float = 0,
    sampling_rate: float = 20000,
) -> tuple[Quantity, np.ndarray]:
    window_len_samples = int(window_len / 1000 * sampling_rate)
    half_window_len_samples = window_len_samples // 2
    window_len_samples = half_window_len_samples * 2
    peak_index, _ = signal.find_peaks(sweep.magnitude, height=detection_threshold)
    if len(peak_index) == 0:
        return Quantity(np.array([]), sweep.units), np.array([])
    ap_array = np.full(
        (len(peak_index), half_window_len_samples * 2), fill_value=np.nan
    )
    for i, i_peak in enumerate(peak_index):
        ap_array[i] = sweep.magnitude[
            (i_peak - half_window_len_samples) : (i_peak + half_window_len_samples)
        ]
    ap_diff = np.diff(ap_array, axis=1, prepend=np.nan)
    max_threshold = np.nanmax(ap_diff, axis=1) * 0.05
    ap_arr_onset = ap_diff > max_threshold[:, np.newaxis]
    ap_arr_onset_neg = np.flip(ap_diff < -max_threshold[:, np.newaxis], axis=1)

    def find_onset(x, half_window_len_samples=half_window_len_samples):
        index = np.argwhere(x).reshape(-1)
        diff_index = np.diff(index, prepend=index[0])
        index = index[np.where((diff_index > 1) & (index > half_window_len_samples))]
        if index.size == 0:
            return len(x)
        index = index[0]
        return index

    onset_backwards_indices = -(
        np.apply_along_axis(
            find_onset, 1, ap_arr_onset_neg, half_window_len_samples
        ).flatten()
        - window_len_samples
    )
    onset_indices = np.apply_along_axis(
        find_onset, 1, ap_arr_onset, half_window_len_samples
    ).flatten()
    action_potentials = np.full_like(ap_array, fill_value=np.nan)
    for i, ap_ind in enumerate(peak_index):
        action_potentials[
            i, onset_backwards_indices[i] : onset_indices[i]
        ] = sweep.magnitude[
            (ap_ind - half_window_len_samples + onset_backwards_indices[i]) : (
                ap_ind + onset_indices[i] - half_window_len_samples
            )
        ]
    return Quantity(action_potentials, sweep.units), peak_index


def _find_threshold(ap_array: np.ndarray) -> tuple[np.int64, np.float64]:
    ap_diff = np.diff(ap_array, prepend=np.nan)
    max_index = np.nanargmax(ap_diff)
    threshold_05 = ap_diff[max_index] * 0.05
    indices = np.argwhere(ap_diff < threshold_05)
    if indices.min() > max_index:
        print("No threshold found")
        print("Minimum diff.: ", np.nanmin(ap_diff[:max_index]))
        print("Diff.-threshold: ", threshold_05)
        return (np.int64(0), np.float64(0))
    index = np.flip(indices[indices < max_index])[0]
    threshold_value = np.float64(ap_array[index])
    return (index, threshold_value)


def _fwhm(
    action_potentials: Quantity,
    peak_amplitude: Quantity,
    thresholds: np.ndarray,
    time: Quantity,
) -> Quantity:
    fwhm = np.full_like(peak_amplitude, np.nan)
    half_height = (peak_amplitude + thresholds) / 2
    for i, ap in enumerate(action_potentials):
        higher_val = np.where(ap > half_height[i])
        first = higher_val[0][0]
        last = higher_val[0][-1]
        fwhm[i] = (
            last
            + np.abs(ap[last].magnitude - half_height[i])
            / (
                np.abs(ap[last].magnitude - half_height[i])
                + np.abs(ap[last + 1].magnitude - half_height[i])
            )
            - first
            - 1
            + np.abs(ap[first - 1].magnitude - half_height[i])
            / (
                np.abs(ap[first].magnitude - half_height[i])
                + np.abs(ap[first - 1].magnitude - half_height[i])
            )
        )
    return Quantity(fwhm * np.diff(time[:2].magnitude), time.units)


class ActionPotentials:
    """
    A class to handle action potentials in electrophysiological data.
    Attributes:
        trace (Trace): The trace object associated with the action potentials.
        events (Events): An Events object to store event data.
        action_potentials (list): A list to store action potential data.
        sampling_rate (Quantity): The sampling rate of the trace.
    """

    def __init__(
        self,
        voltage_channel: VoltageTrace,
        detection_threshold: float,
        window_len: float = 8,
    ) -> None:
        self.sampling_rate = voltage_channel.sampling_rate
        self.detection_threshold = detection_threshold
        self.window_len = window_len
        self.action_potentials = Quantity(np.array([]), voltage_channel.unit)
        self.action_potentials_diff = Quantity(np.array([]), voltage_channel.unit)
        self.peak_index = np.array([])
        self.threshold = {
            "threshold": np.array([]),
            "index": np.array([], dtype=np.int64),
        }
        self.peak_amplitude = Quantity(np.array([]), voltage_channel.unit)
        self.sweep_numbers = np.array([])
        self.ap_numbers = np.array([])
        self.channel = np.array([])
        self.peak_location = Quantity(np.array([]), voltage_channel.starting_time.units)
        self.max_pos_slope = np.array([])
        self.max_neg_slope = np.array([])
        self.ahp = Quantity(np.array([]), voltage_channel.unit)
        # self.ahp = np.array([])
        window_len_half = int(window_len / 1000 * self.sampling_rate.magnitude) // 2
        self.time = Quantity(
            (np.arange(start=0, stop=window_len_half * 2) - window_len_half)
            / self.sampling_rate.magnitude,
            "s",
        ).rescale("ms")

        self.fwhm = Quantity(np.array([]), "ms")
        self.add_action_potentials(voltage_channel, detection_threshold, window_len)

    def add_action_potentials(
        self,
        voltage_channel: VoltageTrace,
        detection_threshold: float | None = None,
        window_len: float = 8,
    ) -> None:
        """
        Add action potentials to the ActionPotentials object.
        Args:
            voltage_channel (VoltageTrace): The voltage channel to extract action potentials from.
            detection_threshold (float, optional): The threshold for detecting action potentials.
            window_len (float, optional): The length of the window for extracting action potentials.
        """
        from ephys.classes.voltage import VoltageTrace
        if not isinstance(voltage_channel, VoltageTrace):
            raise TypeError("voltage_channel must be a VoltageTrace object")
        if detection_threshold is None:
            detection_threshold = self.detection_threshold
        for sweep_index, sweep in enumerate(voltage_channel.data):
            action_potentials, peak_index = _ap_extract(
                sweep,
                window_len=window_len,
                detection_threshold=detection_threshold,
                sampling_rate=float(voltage_channel.sampling_rate.magnitude),
            )
            window_len_half = self.time.size // 2
            if len(action_potentials) == 0:
                continue
            time = Quantity(
                np.linspace(
                    voltage_channel.starting_time.magnitude[sweep_index],
                    len(sweep),
                    len(sweep),
                ),
                voltage_channel.starting_time[sweep_index].units,
            )
            if len(self.action_potentials) == 0:
                self.action_potentials = action_potentials
                self.action_potentials_diff = Quantity(
                    np.diff(action_potentials.magnitude, axis=1, prepend=np.nan),
                    action_potentials.units,
                )
                self.peak_index = peak_index
                self.sweep_numbers = np.full_like(peak_index, sweep_index + 1)
                self.ap_numbers = np.arange(len(peak_index))
                ap_threshold_index = np.apply_along_axis(
                    _find_threshold, 1, action_potentials.magnitude
                )
                self.threshold["threshold"] = ap_threshold_index[:, 1]
                self.threshold["index"] = ap_threshold_index[:, 0].astype(np.int64)
                self.peak_location = Quantity(time[peak_index].magnitude, time.units)
                tmp_peak_amplitude = np.nanmax(action_potentials.magnitude, axis=1)
                self.peak_amplitude = Quantity(
                    tmp_peak_amplitude - ap_threshold_index[:, 1],
                    action_potentials.units,
                )
                self.fwhm = _fwhm(
                    action_potentials,
                    tmp_peak_amplitude,
                    ap_threshold_index[:, 1],
                    self.time,
                )
                self.ahp = Quantity(
                    (
                        np.nanmin(
                            action_potentials.magnitude[:, window_len_half:], axis=1
                        )
                        - ap_threshold_index[:, 1]
                    ),
                    action_potentials.units,
                )

            else:
                self.action_potentials = Quantity(
                    np.concatenate(
                        (self.action_potentials.magnitude, action_potentials.magnitude),
                        axis=0,
                    ),
                    self.action_potentials.units,
                )
                self.action_potentials_diff = Quantity(
                    np.concatenate(
                        (
                            self.action_potentials_diff.magnitude,
                            np.diff(
                                action_potentials.magnitude, axis=1, prepend=np.nan
                            ),
                        ),
                        axis=0,
                    ),
                    self.action_potentials_diff.units,
                )
                self.peak_index = np.concatenate(
                    (self.peak_index, peak_index), axis=0
                ).astype(np.int64)
                self.sweep_numbers = np.concatenate(
                    (self.sweep_numbers, np.full_like(peak_index, sweep_index + 1)),
                    axis=0,
                )
                self.ap_numbers = np.concatenate(
                    (self.ap_numbers, np.arange(len(peak_index))), axis=0
                )
                ap_threshold_index = np.apply_along_axis(
                    _find_threshold, 1, action_potentials.magnitude
                )
                self.threshold["threshold"] = np.concatenate(
                    (self.threshold["threshold"], ap_threshold_index[:, 1]), axis=0
                )
                self.threshold["index"] = np.concatenate(
                    (self.threshold["index"], ap_threshold_index[:, 0]), axis=0
                ).astype(np.int64)
                self.peak_location = Quantity(
                    np.concatenate(
                        (self.peak_location.magnitude, time[peak_index].magnitude),
                        axis=0,
                    ),
                    time.units,
                )
                self.peak_amplitude = Quantity(
                    np.concatenate(
                        (
                            self.peak_amplitude.magnitude,
                            np.nanmax(action_potentials.magnitude, axis=1)
                            - ap_threshold_index[:, 1],
                        ),
                        axis=0,
                    ),
                    action_potentials.units,
                )
                fwhm = _fwhm(
                    action_potentials,
                    np.nanmax(action_potentials.magnitude, axis=1),
                    ap_threshold_index[:, 1],
                    self.time,
                )
                self.fwhm = Quantity(
                    np.concatenate((self.fwhm.magnitude, fwhm.magnitude), axis=0),
                    fwhm.units,
                )
                ahp = (
                    np.nanmin(action_potentials.magnitude[:, window_len_half:], axis=1)
                    - ap_threshold_index[:, 1]
                )
                self.ahp = Quantity(
                    np.concatenate((self.ahp.magnitude, ahp), axis=0),
                    action_potentials.units,
                )
        self.peak_amplitude = np.nanmax(self.action_potentials.magnitude, axis=1)
        self.max_pos_slope = np.nanmax(self.action_potentials_diff.magnitude, axis=1)
        self.max_neg_slope = np.nanmin(self.action_potentials_diff.magnitude, axis=1)
        if len(self.channel) == 0:
            self.channel = np.full_like(
                self.peak_index, voltage_channel.channel_number, dtype=np.int64
            )
        else:
            self.channel = np.concatenate(
                (
                    self.channel,
                    np.full_like(
                        self.peak_index, voltage_channel.channel_number, dtype=np.int64
                    ),
                ),
                axis=0,
            )
        self.remove_duplicates()

    def remove_duplicates(self):
        """
        Remove duplicate action potentials from the ActionPotentials object.
        """
        if len(self.action_potentials) == 0:
            return None
        _, unique_action_potentials = np.unique(
            np.vstack((self.peak_index, self.sweep_numbers, self.channel)),
            return_index=True,
            axis=1,
        )
        self.action_potentials = Quantity(
            self.action_potentials[unique_action_potentials],
            self.action_potentials.units,
        )
        self.action_potentials_diff = Quantity(
            self.action_potentials_diff[unique_action_potentials],
            self.action_potentials_diff.units,
        )
        self.peak_index = self.peak_index[unique_action_potentials]
        self.sweep_numbers = self.sweep_numbers[unique_action_potentials]
        self.ap_numbers = self.ap_numbers[unique_action_potentials]
        self.channel = self.channel[unique_action_potentials]
        self.threshold["threshold"] = self.threshold["threshold"][
            unique_action_potentials
        ]
        self.threshold["index"] = self.threshold["index"][unique_action_potentials]
        self.peak_location = self.peak_location[unique_action_potentials]
        self.peak_amplitude = self.peak_amplitude[unique_action_potentials]
        self.max_pos_slope = self.max_pos_slope[unique_action_potentials]
        self.max_neg_slope = self.max_neg_slope[unique_action_potentials]
        self.peak_amplitude = self.peak_amplitude[unique_action_potentials]
        self.fwhm = self.fwhm[unique_action_potentials]
        self.ahp = self.ahp[unique_action_potentials]
        return None

    def plot(
        self,
        show: bool = True,
        save: bool = False,
        sweep_numbers: np.ndarray | list | int | None = None,
        align_threshold: bool = True,
        threshold: bool = False,
        save_path: str = "action_potentials.png",
    ) -> None:
        """
        Plot the action potentials.
        Args:
            show (bool): Whether to show the plot.
            save (bool): Whether to save the plot.
            sweep_numbers (np.ndarray | list | int | None): The sweep numbers to plot.
            save_path (str): The path to save the plot.
        """
        if sweep_numbers is None:
            sweep_numbers = np.unique(self.sweep_numbers)
        else:
            if isinstance(sweep_numbers, int):
                if sweep_numbers not in self.sweep_numbers:
                    print("Sweep number out of range")
                    return None
                sweep_numbers = [sweep_numbers]
            elif isinstance(sweep_numbers, list):
                if sweep_numbers not in self.sweep_numbers:
                    print("Sweep number out of range")
                    return None
                sweep_numbers = np.array(sweep_numbers)
            elif not isinstance(sweep_numbers, np.ndarray):
                print("Sweep numbers must be a list or numpy array")
                return None
            if len(sweep_numbers) > len(self.sweep_numbers):
                print("Sweep numbers out of range")
                return None
            sweep_numbers = np.unique(sweep_numbers)
        mask = np.isin(self.sweep_numbers, sweep_numbers)
        filtered_action_potentials = self.action_potentials[mask]
        sweep_vals = self.sweep_numbers[mask]
        # ap_numbers = self.ap_numbers[mask]
        # action_potentials per sweep
        unique_sweeps = np.unique(sweep_vals)

        # make a plot for each channel
        channels = np.unique(self.channel[mask])
        fig, axs = plt.subplots(
            len(channels), 1, figsize=(8, 4 * len(channels)), squeeze=False
        )
        axs = axs.flatten()
        for idx, ch in enumerate(channels):
            ch_mask = self.channel[mask] == ch
            ch_action_potentials = filtered_action_potentials[ch_mask]
            ch_sweeps = sweep_vals[ch_mask]
            ch_threshold_index = self.threshold["index"][ch_mask]
            for i, action_potential in enumerate(ch_action_potentials):
                sweep_id = ch_sweeps[i]
                sweep_color = cm.get_cmap("gist_rainbow")(
                    (sweep_id - unique_sweeps.min())
                    / (np.ptp(unique_sweeps) if np.ptp(unique_sweeps) else 1)
                )
                if align_threshold:
                    time_scale = (
                        self.time.magnitude - self.time.magnitude[ch_threshold_index[i]]
                    )
                else:
                    time_scale = self.time.magnitude
                axs[idx].plot(
                    time_scale, action_potential, color=sweep_color, alpha=0.2
                )
            axs[idx].set_title(f"Action Potentials - Channel {ch}")
            axs[idx].set_xlabel(f"Time ({self.time.dimensionality.string})")
            axs[idx].set_ylabel(
                f"Amplitude ({self.action_potentials.dimensionality.string})"
            )
        if threshold:
            for idx, ch in enumerate(channels):
                ch_mask = self.channel[mask] == ch
                ch_threshold = self.threshold["threshold"][ch_mask]
                ch_threshold_index = self.threshold["index"][ch_mask]
                ch_sweeps = sweep_vals[ch_mask]
                for i, i_threshold in enumerate(ch_threshold):
                    sweep_id = ch_sweeps[i]
                    sweep_color = cm.get_cmap("gist_rainbow")(
                        (sweep_id - unique_sweeps.min())
                        / (np.ptp(unique_sweeps) if np.ptp(unique_sweeps) else 1)
                    )
                    if align_threshold:
                        time_scale = 0.0
                    else:
                        time_scale = self.time.magnitude[ch_threshold_index[i]]
                    axs[idx].plot(
                        time_scale,
                        i_threshold,
                        color=sweep_color,
                        marker=".",
                        markersize=5,
                    )

        fig.tight_layout()

        if show:
            plt.show()
        if save:
            plt.savefig(save_path)
        plt.close(fig)
        return None

    def to_dict(self) -> dict | None:
        """
        Convert the action potentials to a dictionary.
        Returns:
            dict: A dictionary containing the action potentials.
        """
        if len(self.action_potentials) == 0:
            print("No action potentials to convert to dictionary")
            return None
        data = {
            "sweep_number": self.sweep_numbers,
            "channel": self.channel,
            "peak_index": self.peak_index,
            "threshold": self.threshold["threshold"],
            "threshold_index": self.threshold["index"],
            "peak_amplitude": self.peak_amplitude,
            "max_pos_slope": self.max_pos_slope,
            "max_neg_slope": self.max_neg_slope,
            "fwhm": self.fwhm,
            "ahp": self.ahp,
            "ap_number": self.ap_numbers,
            "peak_location": self.peak_location,
        }
        return data

    def to_dataframe(self) -> pd.DataFrame | None:
        """
        Convert the action potentials to a pandas DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the action potentials.
        """
        if len(self.action_potentials) == 0:
            print("No action potentials to convert to DataFrame")
            return None
        data = self.to_dict()
        df = pd.DataFrame(data)
        return df
