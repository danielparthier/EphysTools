from __future__ import annotations
from typing import TYPE_CHECKING
from scipy import signal
import numpy as np
from quantities import Quantity
from matplotlib import pyplot as plt
from matplotlib import cm
from ephys import utils
import pandas as pd

if TYPE_CHECKING:
    from ephys.classes.channels import Channel
    from ephys.classes.trace import Trace
    from ephys.classes.voltage import VoltageTrace
    from ephys.classes.events import Events


def _AP_extract(sweep: Quantity,
                window_len: float = 8,
                detection_threshold: float = 0,
                sampling_rate: float = 20000) -> tuple[Quantity, np.ndarray]:
    window_len_samples = int(window_len / 1000 * sampling_rate)
    half_window_len_samples = window_len_samples // 2
    window_len_samples = half_window_len_samples * 2
    peak_index, _ = signal.find_peaks(sweep.magnitude, height=detection_threshold)
    if len(peak_index) == 0:
        return Quantity(np.array([]), sweep.units), np.array([])
    AP_array = np.full((len(peak_index), half_window_len_samples * 2), fill_value=np.nan)
    for i, i_peak in enumerate(peak_index):
        AP_array[i] = sweep.magnitude[(i_peak - half_window_len_samples):(i_peak + half_window_len_samples)]
    AP_diff = np.diff(AP_array, axis=1, prepend=np.nan)
    max_threshold = np.nanmax(AP_diff, axis=1)*0.05
    AP_arr_onset = AP_diff > max_threshold[:, np.newaxis]
    AP_arr_onset_neg = np.flip(AP_diff < -max_threshold[:, np.newaxis], axis=1)
    def find_onset(x, half_window_len_samples=half_window_len_samples):
        index = np.argwhere(x).reshape(-1)
        diff_index = np.diff(index, prepend=index[0])
        index = index[np.where((diff_index > 1) & (index > half_window_len_samples))]
        if index.size == 0:
            return len(x)
        elif index.size > 1:
            index = index[0]
        return index
    onset_backwards_indices = -(np.apply_along_axis(find_onset, 1,
                                                    AP_arr_onset_neg,
                                                    half_window_len_samples).flatten()-window_len_samples)
    onset_indices = np.apply_along_axis(find_onset, 1,
                                        AP_arr_onset,
                                        half_window_len_samples).flatten()
    APs = np.full_like(AP_array, fill_value=np.nan)
    for i, AP_ind in enumerate(peak_index):
        APs[i, onset_backwards_indices[i]:onset_indices[i]] = sweep.magnitude[
            (AP_ind - half_window_len_samples+onset_backwards_indices[i]):
            (AP_ind + onset_indices[i]- half_window_len_samples)]
    return Quantity(APs, sweep.units), peak_index

def _find_threshold(AP_array: np.ndarray) -> tuple[np.int64, np.float64]:
    AP_diff = np.diff(AP_array, prepend=np.nan)
    max_index = np.nanargmax(AP_diff)
    threshold_05 = AP_diff[max_index] * 0.05
    indices = np.argwhere(AP_diff < threshold_05)
    if indices.min() > max_index:
        print("No threshold found")
        print("Minimum diff.: ", np.nanmin(AP_diff[:max_index]))
        print("Diff.-threshold: ", threshold_05)
        return (np.int64(0), np.float64(0))
    index = np.flip(indices[indices < max_index])[0]
    threshold_value = np.float64(AP_array[index])
    return (index, threshold_value)

def _FWHM(APs: Quantity, peak_amplitude: Quantity, thresholds: np.ndarray, time: Quantity) -> Quantity:
    FWHM = np.full_like(peak_amplitude, np.nan)
    half_height = (peak_amplitude + thresholds) / 2
    for i, AP in enumerate(APs):
        higher_val = np.where(AP > half_height[i])
        first = higher_val[0][0]
        last = higher_val[0][-1]
        FWHM[i] = (
            last
            + np.abs(AP[last].magnitude - half_height[i])
            / (
            np.abs(AP[last].magnitude - half_height[i])
            + np.abs(AP[last + 1].magnitude - half_height[i])
            )
            - first
            - 1
            + np.abs(AP[first - 1].magnitude - half_height[i])
            / (
            np.abs(AP[first].magnitude - half_height[i])
            + np.abs(AP[first - 1].magnitude - half_height[i])
            )
        )
    return Quantity(FWHM * np.diff(time[:2].magnitude), time.units)


class ActionPotentials:
    """
    A class to handle action potentials in electrophysiological data.
    Attributes:
        trace (Trace): The trace object associated with the action potentials.
        events (Events): An Events object to store event data.
        action_potentials (list): A list to store action potential data.
        sampling_rate (Quantity): The sampling rate of the trace.
    """
    def __init__(self, voltage_channel: VoltageTrace, detection_threshold: float, window_len: float = 8) -> None:
        self.sampling_rate = voltage_channel.sampling_rate
        self.detection_threshold = detection_threshold
        self.window_len = window_len
        self.APs = Quantity(np.array([]), voltage_channel.unit)
        self.APs_diff = Quantity(np.array([]), voltage_channel.unit)
        self.peak_index = np.array([])
        self.threshold = {"threshold": np.array([]),
                          "index": np.array([], dtype=np.int64)}
        self.peak_amplitude = Quantity(np.array([]), voltage_channel.unit)
        self.sweep_numbers = np.array([])
        self.AP_numbers = np.array([])
        self.channel = np.array([])
        self.peak_location = Quantity(np.array([]), voltage_channel.starting_time.units)
        self.max_pos_slope = np.array([])
        self.max_neg_slope = np.array([])
        #self.AHP = np.array([])
        window_len_half = int(window_len / 1000 * self.sampling_rate.magnitude) // 2
        self.time = Quantity((np.arange(start=0, stop=window_len_half * 2)
                             - window_len_half) / self.sampling_rate.magnitude, "s").rescale("ms")

        self.FWHM = Quantity(np.array([]), "ms")
        self.add_action_potentials(voltage_channel, detection_threshold, window_len)

    def add_action_potentials(self, voltage_channel: VoltageTrace,
                              detection_threshold: float|None = None,
                              window_len: float = 8) -> None:
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
            APs, peak_index = _AP_extract(sweep,
                                       window_len=window_len,
                                       detection_threshold=detection_threshold,
                                       sampling_rate=float(voltage_channel.sampling_rate.magnitude))
            window_len_half = self.time.size // 2
            if len(APs) == 0:
                continue
            time = Quantity(np.linspace(voltage_channel.starting_time.magnitude[sweep_index], len(sweep), len(sweep)),
                            voltage_channel.starting_time[sweep_index].units)
            if len(self.APs) == 0:
                self.APs = APs
                print(APs.units)
                self.APs_diff = Quantity(np.diff(APs.magnitude, axis=1, prepend=np.nan), APs.units)
                self.peak_index = peak_index
                self.sweep_numbers = np.full_like(peak_index, sweep_index+1)
                self.AP_numbers = np.arange(len(peak_index))
                AP_threshold_index = np.apply_along_axis(_find_threshold, 1, APs.magnitude)
                self.threshold["threshold"] = AP_threshold_index[:, 1]
                self.threshold["index"] = AP_threshold_index[:, 0].astype(np.int64)
                self.peak_location = Quantity(time[peak_index].magnitude, time.units)
                tmp_peak_amplitude = np.nanmax(APs.magnitude, axis=1)
                self.peak_amplitude = Quantity(tmp_peak_amplitude - AP_threshold_index[:, 1], APs.units)
                FWHM = _FWHM(APs, tmp_peak_amplitude, AP_threshold_index[:, 1], self.time)
                self.FWHM = FWHM
                AHP = np.nanmin(APs.magnitude[:, window_len_half:], axis=1) - AP_threshold_index[:, 1]
                self.AHP = Quantity(AHP, APs.units)

            else:
                self.APs = Quantity(np.concatenate((self.APs.magnitude, APs.magnitude), axis=0), self.APs.units)
                self.APs_diff = Quantity(np.concatenate((self.APs_diff.magnitude, np.diff(APs.magnitude, axis=1, prepend=np.nan)), axis=0), self.APs_diff.units)
                self.peak_index = np.concatenate((self.peak_index, peak_index), axis=0).astype(np.int64)
                self.sweep_numbers = np.concatenate((self.sweep_numbers,
                                                    np.full_like(peak_index, sweep_index+1)), axis=0)
                self.AP_numbers = np.concatenate((self.AP_numbers, np.arange(len(peak_index))), axis=0)
                AP_threshold_index = np.apply_along_axis(_find_threshold, 1, APs.magnitude)
                self.threshold["threshold"] = np.concatenate((self.threshold["threshold"],
                                                            AP_threshold_index[:, 1]), axis=0)
                self.threshold["index"] = np.concatenate((self.threshold["index"],
                                                        AP_threshold_index[:, 0]), axis=0).astype(np.int64)
                self.peak_location = Quantity(np.concatenate((self.peak_location.magnitude,
                                                    time[peak_index].magnitude), axis=0), time.units)
                self.peak_amplitude = Quantity(np.concatenate((self.peak_amplitude.magnitude,
                                                    np.nanmax(APs.magnitude, axis=1)-AP_threshold_index[:, 1]), axis=0), APs.units)
                FWHM = _FWHM(APs, np.nanmax(APs.magnitude, axis=1), AP_threshold_index[:, 1], self.time)
                self.FWHM = Quantity(np.concatenate((self.FWHM.magnitude, FWHM.magnitude), axis=0), FWHM.units)
                AHP = np.nanmin(APs.magnitude[:, window_len_half:], axis=1) - AP_threshold_index[:, 1]
                self.AHP = Quantity(np.concatenate((self.AHP.magnitude, AHP), axis=0), APs.units)
        self.peak_amplitude = np.nanmax(self.APs.magnitude, axis=1)
        self.max_pos_slope = np.nanmax(self.APs_diff.magnitude, axis=1)
        self.max_neg_slope = np.nanmin(self.APs_diff.magnitude, axis=1)
        if len(self.channel) == 0:
            self.channel = np.full_like(self.peak_index, voltage_channel.channel_number, dtype=np.int64)
        else:
            self.channel = np.concatenate((self.channel,
                                           np.full_like(self.peak_index, voltage_channel.channel_number, dtype=np.int64)), axis=0)
        self.remove_duplicates()

    def remove_duplicates(self):
        """
        Remove duplicate action potentials from the ActionPotentials object.
        """
        if len(self.APs) == 0:
            return None
        _, unique_APs = np.unique(np.vstack((self.peak_index,
                                             self.sweep_numbers,
                                             self.channel)), return_index=True,
                                             axis=1)
        self.APs = Quantity(self.APs[unique_APs], self.APs.units)
        self.APs_diff = Quantity(self.APs_diff[unique_APs], self.APs_diff.units)
        self.peak_index = self.peak_index[unique_APs]
        self.sweep_numbers = self.sweep_numbers[unique_APs]
        self.AP_numbers = self.AP_numbers[unique_APs]
        self.channel = self.channel[unique_APs]
        self.threshold["threshold"] = self.threshold["threshold"][unique_APs]
        self.threshold["index"] = self.threshold["index"][unique_APs]
        self.peak_location = self.peak_location[unique_APs]
        self.peak_amplitude = self.peak_amplitude[unique_APs]
        self.max_pos_slope = self.max_pos_slope[unique_APs]
        self.max_neg_slope = self.max_neg_slope[unique_APs]
        self.peak_amplitude = self.peak_amplitude[unique_APs]
        self.FWHM = self.FWHM[unique_APs]
        self.AHP = self.AHP[unique_APs]
        return None
    
    def plot(self, show: bool = True, save: bool = False, sweep_numbers: np.ndarray|list|int|None = None,
             align_threshold: bool = True, threshold: bool = False,
             save_path: str = "action_potentials.png") -> None:
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
        filtered_APs = self.APs[mask]
        sweep_vals = self.sweep_numbers[mask]
        #AP_numbers = self.AP_numbers[mask]
        # aps per sweep
        unique_sweeps = np.unique(sweep_vals)

        # make a plot for each channel
        channels = np.unique(self.channel[mask])
        fig, axs = plt.subplots(len(channels), 1, figsize=(8, 4 * len(channels)), squeeze=False)
        axs = axs.flatten()
        for idx, ch in enumerate(channels):
            ch_mask = self.channel[mask] == ch
            ch_APs = filtered_APs[ch_mask]
            ch_sweeps = sweep_vals[ch_mask]
            ch_threshold_index = self.threshold["index"][ch_mask]
            for i in range(len(ch_APs)):
                sweep_id = ch_sweeps[i]
                sweep_color = cm.get_cmap('gist_rainbow')((sweep_id - unique_sweeps.min()) /
                                                          (np.ptp(unique_sweeps) if np.ptp(unique_sweeps) else 1))
                if align_threshold:
                    time_scale = self.time.magnitude - self.time.magnitude[ch_threshold_index[i]]
                else:
                    time_scale = self.time.magnitude
                axs[idx].plot(time_scale, ch_APs[i], color=sweep_color, alpha=0.2)
            axs[idx].set_title(f"Action Potentials - Channel {ch}")
            axs[idx].set_xlabel(f"Time ({self.time.dimensionality.string})")
            axs[idx].set_ylabel(f"Amplitude ({self.APs.dimensionality.string})")
        if threshold:
            for idx, ch in enumerate(channels):
                ch_mask = self.channel[mask] == ch        
                ch_threshold = self.threshold["threshold"][ch_mask]
                ch_threshold_index = self.threshold["index"][ch_mask]
                ch_sweeps = sweep_vals[ch_mask]
                for i in range(len(ch_threshold)):
                    sweep_id = ch_sweeps[i]
                    sweep_color = cm.get_cmap('gist_rainbow')((sweep_id - unique_sweeps.min()) /
                                                          (np.ptp(unique_sweeps) if np.ptp(unique_sweeps) else 1))
                    if align_threshold:
                        time_scale = 0.0
                    else:
                        time_scale = self.time.magnitude[ch_threshold_index[i]]
                    axs[idx].plot(time_scale, ch_threshold[i], color=sweep_color, marker='.', markersize=2)

        fig.tight_layout()

        # fig, ax = plt.subplots()
        # for i in range(len(filtered_APs)):
        #     # use color map to differentiate APs from different sweeps
        #     sweep_id = sweep_vals[i]
        #     sweep_color = cm.get_cmap('gist_rainbow')((sweep_id - unique_sweeps.min()) /
        #                                               (np.ptp(unique_sweeps) if np.ptp(unique_sweeps) else 1))
        #     ax.plot(self.time.magnitude, filtered_APs[i], color=sweep_color, alpha=0.2)
        # #TODO: add labels for each AP with right units and colour
        # ax.set_xlabel(f"Time ("+self.time.dimensionality.string+")")
        # ax.set_ylabel(f"Amplitude ("+self.APs.dimensionality.string+")")
        # ax.set_title("Action Potentials")
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
        if len(self.APs) == 0:
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
            #"half_width": self.half_width,
            #"AHP": self.AHP,
        }
        return data

    def to_dataframe(self) -> pd.DataFrame | None:
        """
        Convert the action potentials to a pandas DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the action potentials.
        """
        if len(self.APs) == 0:
            print("No action potentials to convert to DataFrame")
            return None
        data = self.to_dict()
        df = pd.DataFrame(data)
        return df
    
