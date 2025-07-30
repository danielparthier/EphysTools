"""
This module defines classes for handling electrophysiological channel data.

Classes:
--------
- Channel: Represents a channel in electrophysiology experiments.
- ChannelInformation: Extracts and stores channel information from electrophysiological data.
- ChannelAverage: Calculates the average trace from a set of voltage or current traces.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from re import findall
import datetime
import numpy as np
import pandas as pd
from quantities import Quantity
from neo import WinWcpIO, AxonIO, IgorIO
from ephys.classes.class_functions import _get_sweep_subset, _is_clamp, check_clamp


if TYPE_CHECKING:
    from ephys.classes.voltage import VoltageTrace, VoltageClamp
    from ephys.classes.current import CurrentTrace, CurrentClamp


class Channel:
    """
    A class to represent a channel in electrophysiology experiments.
    Attributes:
    ----------
    sweep_count : int
        The number of sweeps in the channel.
    sweep_length : int
        The length of each sweep.
    unit : str
        The unit of the channel data.
    data : Quantity
        The data of the channel, stored as a Quantity object.
    channel_number : int | None
        The number of the channel.
    average : ChannelAverage | None
        The average of the channel data.
    clamped : bool | None
        Indicates if the channel is clamped.
    Methods:
    -------
    __init__(self, sweep_count: int, sweep_length: int, unit: str):
        Initializes the Channel with the given sweep count, sweep length, and unit.
    insert_data(self, data: np.ndarray, sweep_count: int) -> None:
        Inserts sweep data into the trace at the specified sweep count.
    append_data(self, data) -> None:
        Appends new data to the existing trace.
    load_data_block(self, data_block, channel_index) -> None:
        Loads a data block into the channel.
    check_clamp(self, quick_check: bool = False, warnings: bool = True) -> None:
        Checks the clamp status of the experiment.
    channel_average(self, sweep_subset: Any = None) -> None:
        Calculate the channel average for a given subset of sweeps.
    """

    def __init__(self, sweep_count: int, sweep_length: int, unit: str):
        self.sweep_count = sweep_count
        self.sweep_length = sweep_length
        self.unit = unit
        self.data = Quantity(np.zeros((self.sweep_count, self.sweep_length)), unit)
        self.channel_number: int | None = None
        self.average: ChannelAverage | None = None
        self.clamped: bool | None = None
        self.sampling_rate: Quantity = Quantity(0, "Hz")
        self.starting_time: Quantity = Quantity(np.zeros(self.sweep_count), "s")
        self.rec_datetime: datetime.datetime | None = None
        self.check_clamp()

    def insert_data(self, data: np.ndarray, sweep_count: int) -> None:
        """
        Inserts sweep data into the trace at the specified sweep count.

        Parameters:
        data (numpy.ndarray): The data to be inserted.
        sweep_count (int): The index at which the data should be inserted.

        Returns:
        None
        """
        self.data.magnitude[sweep_count] = data.flatten()

    def append_data(self, data) -> None:
        """
        Appends new data to the existing trace.

        Parameters:
        data (numpy.ndarray): The data to be appended. It should be a numpy array that can be
        flattened and stacked with the existing trace.

        Returns:
        None
        """

        self.data = Quantity(np.vstack((self.data, data.flatten())), self.unit)

    def load_data_block(self, data_block, channel_index) -> None:
        # TODO: Implement this method --> not sure if it is necessary
        pass

    def check_clamp(self, quick_check: bool = False, warnings: bool = True) -> None:
        """
        Checks the clamp status of the experiment.

        This method uses the `check_clamp` function from the `ephys.classes.class_functions` module
        to verify the clamp status of the experiment object.

        Args:
            quick_check (bool, optional): If True, performs a quick check. Defaults to False.
            warnings (bool, optional): If True, displays warnings. Defaults to True.

        Returns:
            None
        """

        check_clamp(self, quick_check, warnings)

    def channel_average(self, sweep_subset: Any = None) -> None:
        """
        Calculate the channel average for a given subset of sweeps.

        Parameters:
            sweep_subset (Any, optional): A subset of sweeps to be averaged. If None,
            the entire set of sweeps will be used. Defaults to None.

        Returns:
            None: The result is stored in the `self.average` attribute as a ChannelAverage object.
        """

        sweep_subset = _get_sweep_subset(self.data, sweep_subset)
        self.average = ChannelAverage(self, sweep_subset)


class ChannelInformation:
    """
    A class to extract and store channel information from electrophysiological data.

    Attributes:
    -----------
    channel_number : np.ndarray or None
        Array of channel numbers.
    recording_type : np.ndarray or None
        Array of recording types (e.g., 'field', 'cell').
    signal_type : np.ndarray or None
        Array of signal types (e.g., 'voltage', 'current').
    clamped : np.ndarray or None
        Array indicating the clamp type.
    channel_grouping : np.ndarray or None
        Array of channel groupings.
    unit : np.ndarray or None
        Array of units for each channel.

    Methods:
    --------
    __init__(data: Any) -> dict:
        Initializes the ChannelInformation object with data from a neo.io object.

    to_dict() -> dict:
        Converts the channel information to a dictionary format.

    count() -> dict:
        Counts the occurrences of unique values in the channel information attributes.
    """

    def __init__(self, data: Any = None) -> None:
        type_out = []
        channel_list = []
        signal_type = []
        clamp_type = []
        channel_groups = []
        channel_unit = []
        channel_index = 0

        self.channel_number = np.array([])
        self.recording_type = np.array([])
        self.signal_type = np.array([])
        self.clamped = np.array([])
        self.channel_grouping = np.array([])
        self.unit = np.array([])

        if data is not None:
            if isinstance(data, WinWcpIO):
                analogsignals = data.read_block().segments[0].analogsignals
                for i in data.header["signal_channels"]:
                    if len(findall(r"Vm\(AC\)", i[0])) == 1:
                        type_out.append("field")
                        signal_type.append("voltage")
                    elif len(findall(r"Vm", i[0])) == 1:
                        type_out.append("cell")
                        signal_type.append("voltage")
                    elif len(findall(r"Im|Icom", i[0])) == 1:
                        type_out.append("cell")
                        signal_type.append("current")
                    channel_groups.append(i["stream_id"].astype(int).tolist())
                    if (
                        len(analogsignals) < channel_index
                        or analogsignals[0].shape[1] > 1
                    ):
                        # check if channels are merged in signal
                        if analogsignals[0].shape[1] > 1:
                            clamp_type.append(
                                _is_clamp(
                                    analogsignals[0][
                                        :, channel_index
                                    ].magnitude.squeeze()
                                )
                            )
                        else:
                            # if so, we cannot find the channel in the data
                            print(f"Channel {channel_index} not found in data.")
                            continue
                    else:
                        clamp_type.append(
                            _is_clamp(analogsignals[channel_index].magnitude.squeeze())
                        )
                    channel_index += 1
                    channel_list.append(channel_index)
                    channel_unit.append(str(i["units"]))
            elif isinstance(data, AxonIO):
                analogsignals = data.read_block().segments[0].analogsignals
                for i in data.header["signal_channels"]:
                    if len(findall(r"V", i[4])) == 1:
                        type_out.append("cell")
                        signal_type.append("voltage")
                    elif len(findall(r"A", i[4])) == 1:
                        type_out.append("cell")
                        signal_type.append("current")
                    channel_groups.append(i["stream_id"].astype(int).tolist())
                    clamp_type.append(
                        _is_clamp(analogsignals[channel_index].magnitude.squeeze())
                    )
                    channel_index += 1
                    channel_list.append(channel_index)
                    channel_unit.append(str(i["units"]))
            elif isinstance(data, IgorIO):
                # TODO: Implement this for IgorIO
                pass
            if len(channel_list) > 0:
                self.channel_number = np.array(channel_list)
                self.recording_type = np.array(type_out)
                self.signal_type = np.array(signal_type)
                self.clamped = np.array(clamp_type)
                self.channel_grouping = np.array(channel_groups)
                self.unit = np.array(channel_unit)
            else:
                print("No channel information found.")

    def to_dict(self) -> dict:
        """
        Return the channel information as dictionary.

        Returns:
            dict: A dictionary containing the following key-value pairs:
                - 'channel_number' (int): The channel number.
                - 'recording_type' (str): The type of recording.
                - 'signal_type' (str): The type of signal.
                - 'clamped' (bool): Indicates if the signal is clamped.
                - 'channel_grouping' (str): The grouping of the channel.
                - 'unit' (str): The unit of measurement.
        """

        return {
            "channel_number": self.channel_number,
            "recording_type": self.recording_type,
            "signal_type": self.signal_type,
            "clamped": self.clamped,
            "channel_grouping": self.channel_grouping,
            "unit": self.unit,
        }

    def count(self) -> dict:
        """
        Return the count of unique values in the channel information attributes.

        Returns:
            dict: A dictionary containing the attributes and their counts.
                - 'signal_type' (dict): The count of unique signal types.
                - 'recording_type' (dict): The count of unique recording types.
                - 'clamped' (dict): The count of unique clamp types.
                - 'channel_grouping' (dict): The count of unique channel groupings.
                - 'unit' (dict): The count of unique units.
        """

        signal_type, signal_type_count = np.unique(self.signal_type, return_counts=True)
        recording_type, recording_type_count = np.unique(
            self.recording_type, return_counts=True
        )
        clamped, clamped_count = np.unique(self.clamped, return_counts=True)
        channel_grouping, channel_grouping_count = np.unique(
            self.channel_grouping, return_counts=True
        )
        unit, unit_idx = np.unique(self.unit, return_counts=True)
        return {
            "signal_type": dict(zip(signal_type, signal_type_count)),
            "recording_type": dict(zip(recording_type, recording_type_count)),
            "clamped": dict(zip(clamped, clamped_count)),
            "channel_grouping": dict(zip(channel_grouping, channel_grouping_count)),
            "unit": dict(zip(unit, unit_idx)),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the channel information to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the channel information.
        """
        return pd.DataFrame(self.to_dict(), index=self.channel_number).T


class ChannelAverage:
    """
    A class to calculate the average trace from a set of voltage or current traces.

    Attributes:
    -----------
    trace : np.ndarray
        The average trace calculated from the provided data.
    sweeps_used : np.ndarray
        The indices of the sweeps used to calculate the average trace.

    Methods:
    --------
    __init__(trace: VoltageTrace | CurrentTrace, sweep_subset: Any = None) -> None
        Initializes the ChannelAverage object and calculates the average trace.

    Parameters:
    -----------
    trace : VoltageTrace | CurrentTrace
        An object containing the voltage or current trace data.
    sweep_subset : Any, optional
        A subset of sweeps to use for calculating the average trace. If None, all sweeps are used.

    Raises:
    -------
    ValueError
        If the provided trace data is empty.
    """

    def __init__(
        self,
        trace: Channel | VoltageTrace | CurrentTrace | VoltageClamp,
        sweep_subset: Any = None,
    ) -> None:
        self.trace = None
        self.sweeps_used = None
        if len(trace.data) == 0:
            raise ValueError("No data available to calculate the average trace.")
        sweep_subset = _get_sweep_subset(trace.data, sweep_subset)
        self.trace = np.mean(trace.data[sweep_subset, :], axis=0)
        self.sweeps_used = sweep_subset
