"""
This module defines classes for handling current traces and current clamping in electrophysiology
experiments.

Classes:
---------
CurrentTrace:
    Represents a current trace in electrophysiology experiments. It provides methods to manipulate
    and analyze the trace data, including changing units and checking clamping status.

CurrentClamp:
    Handles current clamping operations using a CurrentTrace instance. It provides methods to
    manipulate the current data and change its unit.

Dependencies:
-------------
- ephys.classes.class_functions: Contains utility functions for electrophysiology data.
- ephys.classes.channels: Provides base classes for channel data handling.
"""

from ephys.classes.channels import Channel


class CurrentTrace(Channel):
    """
    A class to represent a current trace in electrophysiology experiments.

    Attributes:
    -----------
    sweep_count : int
        The number of sweeps in the current trace.
    sweep_length : int
        The length of each sweep.
    unit : str
        The unit of the current trace.
    data : Quantity
        The current trace data.

    Methods:
    --------
    __init__(self, sweep_count: int, sweep_length: int, unit: str):
        Initializes the CurrentTrace with the given sweep count, sweep length, and unit.

    change_unit(self, unit: str) -> None:
        Changes the unit of the current trace to the specified unit if it is a current unit.
    """
    def __init__(self, sweep_count: int, sweep_length: int, unit: str):
        """
        Initializes the CurrentTrace with the given sweep count, sweep length, and unit.

        Parameters:
        sweep_count (int): The number of sweeps in the current trace.
        sweep_length (int): The length of each sweep.
        unit (str): The unit of the current trace.

        Returns:
        None
        """
        super().__init__(sweep_count=sweep_count, sweep_length=sweep_length, unit=unit)
        self.sweep_count = sweep_count
        self.sweep_length = sweep_length
        self.unit = unit
        self.clamped = False

    def change_unit(self, unit: str) -> None:
        """
        Change the unit of the trace to the specified voltage unit.

        Parameters:
        unit (str): The new unit to rescale the trace to. Must be a current unit.

        Returns:
        None

        Raises:
        ValueError: If the unit does not end with 'A'.
        """

        if unit.endswith("A"):
            if unit.find("µ") == 0:
                unit = unit.replace("µ", "u")
            self.data = self.data.rescale(unit)
            self.unit = unit
        else:
            raise ValueError("Unit must be current.")


class CurrentClamp(Channel):
    """
    CurrentClamp handles current clamping operations using a CurrentTrace instance.

    Attributes:
        data (Quantity): Current data from the CurrentTrace instance.
        unit (str): Unit of the current data.
        sweep_count (int): Number of sweeps in the current trace.
        sweep_length (int): Length of each sweep in the current trace.
        clamped (bool): Indicates if the current is clamped.

    Methods:
        __init__(channel: CurrentTrace): Initializes the CurrentClamp object.

        change_unit(unit: str) -> None: Changes the unit of the current trace.
    """

    def __init__(self, channel: CurrentTrace):
        """
        Initializes the CurrentClamp object with a CurrentTrace instance.

        Parameters:
        channel (CurrentTrace): The current trace object to be used for clamping.

        Returns:
        None
        """
        super().__init__(sweep_count=channel.sweep_count,
                         sweep_length=channel.sweep_length,
                         unit=channel.unit)
        self.data = channel.data
        self.unit = channel.unit
        self.sweep_count = channel.sweep_count
        self.sweep_length = channel.sweep_length
        self.clamped = True

    def change_unit(self, unit: str) -> None:
        """
        Change the unit of the trace to the specified current unit.

        Parameters:
        unit (str): The new unit to rescale the trace to. Must be a current unit.

        Returns:
        None

        Raises:
        ValueError: If the unit does not end with 'A'.
        """

        if unit.endswith("A"):
            if unit.find("µ") == 0:
                unit = unit.replace("µ", "u")
            self.data = self.data.rescale(unit)
            self.unit = unit
        else:
            raise ValueError("Unit must be current.")
