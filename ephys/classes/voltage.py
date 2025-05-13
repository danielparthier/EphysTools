"""
This module defines classes for handling voltage traces and voltage clamping in electrophysiology
experiments.

Classes:
---------
VoltageTrace:
    Represents a voltage trace in electrophysiology experiments. It provides methods to manipulate
    and analyze the trace data, including changing units, inserting data, and checking clamping
    status.

VoltageClamp:
    Handles voltage clamping operations using a VoltageTrace instance. It provides methods to
    manipulate the voltage data and change its unit.

Dependencies:
-------------
- numpy: For numerical operations.
- quantities: For handling physical quantities with units.
- ephys.classes.class_functions: Contains utility functions for electrophysiology data.
- ephys.classes.channels: Provides base classes for channel data handling.
"""

#from typing import Any

#import numpy as np
#from quantities import Quantity
#from ephys.classes.class_functions import _get_sweep_subset, check_clamp
from ephys.classes.channels import Channel


class VoltageTrace(Channel):
    """
    A class to represent a voltage trace in electrophysiology experiments.

    Attributes:
    -----------
    sweep_count : int
        The number of sweeps in the voltage trace.
    sweep_length : int
        The length of each sweep.
    unit : str
        The unit of the voltage trace.
    data : Quantity
        The voltage trace data.
    clamped : None or bool
        The clamped state of the voltage trace.

    Methods:
    --------
    __init__(self, sweep_count: int, sweep_length: int, unit: str):
        Initializes the VoltageTrace with the given sweep count, sweep length, and unit.

    insert_data(self, data: Any, sweep_count: int) -> None:
        Inserts data into the trace at the specified sweep count.

    append_data(self, data: Any) -> None:
        Appends data to the trace.

    load_data_block(self, data_block: Any, channel_index: int) -> None:
        Loads a block of data into the trace (currently a placeholder).

    change_unit(self, unit: str) -> None:
        Changes the unit of the voltage trace to the specified unit if it is a voltage unit.

    check_clamp(self, quick_check: bool = False, warnings: bool = True) -> None:
        Checks if the voltage trace is clamped using the check_clamp function.
    """

    def change_unit(self, unit: str) -> None:
        """
        Change the unit of the trace to the specified voltage unit.

        Parameters:
        unit (str): The new unit to rescale the trace to. Must be a voltage unit.

        Returns:
        None

        Raises:
        ValueError: If the unit does not end with 'V'.
        """

        if unit.endswith("V"):
            if unit.find("µ") == 0:
                unit = unit.replace("µ", "u")
            self.data = self.data.rescale(unit)
            self.unit = unit
        else:
            raise ValueError("Unit must be voltage.")

class VoltageClamp:
    """
    VoltageClamp handles voltage clamping operations using a VoltageTrace instance.

    Attributes:
        data (Quantity): Voltage data from the VoltageTrace instance.
        unit (str): Unit of the voltage data.
        sweep_count (int): Number of sweeps in the voltage trace.
        sweep_length (float): Length of each sweep in the voltage trace.
        clamped (bool): Indicates if the voltage is clamped.

    Methods:
        __init__(channel: VoltageTrace): Initializes the VoltageClamp object.

        change_unit(unit: str) -> None: Changes the unit of the voltage trace.
    """

    def __init__(self, channel: VoltageTrace):
        """
        Initializes the VoltageClamp object with a VoltageTrace instance.

        Parameters:
        channel (VoltageTrace): The voltage trace object to be used for clamping.

        Returns:
        None
        """
        self.data = channel.data
        self.unit = channel.unit
        self.sweep_count = channel.sweep_count
        self.sweep_length = channel.sweep_length
        self.clamped = True
    def change_unit(self, unit: str) -> None:
        """
        Change the unit of the trace to the specified voltage unit.

        Parameters:
        unit (str): The new unit to rescale the trace to. Must be a voltage unit.

        Returns:
        None

        Raises:
        ValueError: If the unit does not end with 'V'.
        """

        if unit.endswith("V"):
            if unit.find("µ") == 0:
                unit = unit.replace("µ", "u")
            self.data = self.data.rescale(unit)
            self.unit = unit
        else:
            raise ValueError("Unit must be voltage.")
