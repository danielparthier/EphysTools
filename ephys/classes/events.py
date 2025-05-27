from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ephys.classes.trace import Trace


class Events:
    """
    A class to handle events in electrophysiological data.
    Attributes:
        trace (Trace): The trace object associated with the events.
        events (list): A list to store event data.
        event_times (list): A list to store event times.
    """

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.events = []
        self.event_times = []
        self.sweep_numbers = []
        self.event_names = []
    
    def add_event(self, event_name: str) -> None:
        """
        Add an event to the events list.
        
        Parameters:
            event_name (str): The name of the event to be added.
        """
        self.events.append(event_name)
        self.event_times.append(self.trace.time)
        self.sweep_numbers.append(self.trace.sweep_number)
        self.event_names.append(event_name)
        self.trace.meta_data.event_info.append(
            {
                "event_name": event_name,
                "event_time": self.trace.time,
                "sweep_number": self.trace.sweep_number,
            }
        )