"""
This module provides classes for representing experimental data and metadata.
"""

import os
import time

import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
from ephys import utils
from ephys.classes.trace import Trace

mplstyle.use("fast")


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


class MetaData:
    """
    A class representing metadata for experiment files.

    Args:
        file_path (str | list): The path(s) of the file(s) to be added.
        experimenter (str | list, optional): The name(s) of the experimenter(s).
                                             Defaults to 'unknown'.

    Attributes:
        file_info (numpy.ndarray): An array containing information about the file(s).
        experiment_info (numpy.ndarray): An array containing information about the experiment(s).

    Methods:
        __init__(self, file_path: str | list, experimenter: str | list = 'unknown') -> None:
            Initializes the MetaData object.

        add_file_info(self, file_path: str | list, experimenter: str | list = 'unknown',
                      add: bool = True) -> None:
            Adds file information to the MetaData object.

        remove_file_info(self, file_path: str | list) -> None:
            Removes file information from the MetaData object.
    """

    def __init__(
        self, file_path: str | list, experimenter: str | list = "unknown"
    ) -> None:
        self.file_info = np.array([])
        self.experiment_info = np.array([])
        self.add_file_info(file_path, experimenter, add=False)


    def add_file_info(
        self,
        file_path: str | list,
        experimenter: str | list = "unknown",
        add: bool = True,
    ) -> None:
        """
        Adds file information to the MetaData object.

        Args:
            file_path (str | list): The path(s) of the file(s) to be added.
            experimenter (str | list, optional): The name(s) of the experimenter(s).
                                                 Defaults to 'unknown'.
            add (bool, optional): Whether to append the information to existing data.
                                  Defaults to True.
        """

        if isinstance(file_path, str):
            file_path = [file_path]
        file_list = []
        experiment_list = []
        for file in file_path:
            time_created = time.ctime(os.path.getctime(file))
            time_modified = time.ctime(os.path.getmtime(file))
            estimated_exp_date = (
                time_created if time_created < time_modified else time_modified
            )
            # NOTE: abf files have date of experiment in the header
            file_list.append(
                {
                    "data_of_creation": time_created,
                    "last_modified": time_modified,
                    "file_name": os.path.basename(file),
                    "file_path": file,
                }
            )
            experiment_list.append(
                {"date_of_experiment": estimated_exp_date, "experimenter": experimenter}
            )
            print("Date of Experiment estimated. Please check for correct date.")
        if add:
            self.file_info = np.append(self.file_info, np.array(file_list))
            self.experiment_info = np.append(
                self.experiment_info, np.array(experiment_list)
            )
        else:
            self.file_info = np.array(file_list)
            self.experiment_info = np.array(experiment_list)

    def remove_file_info(self, file_path: str | list) -> None:
        """
        Removes file information from the MetaData object.

        Args:
            file_path (str | list): The path(s) of the file(s) to be removed.
        """
        if isinstance(file_path, str):
            file_path = [file_path]
        files_to_remove = [os.path.basename(file) for file in file_path]
        files_exist = [
            entry["file_name"] for entry in self.file_info if isinstance(entry, dict)
        ]
        self.file_info = self.file_info[
            utils.string_match(files_to_remove, files_exist)
        ]


class ExpData:
    """
    A class representing experimental data.

    Args:
        file_path (str | list): The path(s) to the file(s) containing the data.
        experimenter (str, optional): The name of the experimenter. Defaults to 'unknown'.

    Attributes:
        protocols (list): A list of Trace objects representing the protocols.
        meta_data: An instance of the MetaData class.

    """

    def __init__(
        self, file_path: str | list, experimenter: str = "unknown", sort: bool = True
    ) -> None:
        self.protocols = []
        if isinstance(file_path, str):
            self.protocols.append(Trace(file_path))
            self.meta_data = MetaData(file_path, experimenter)

        elif isinstance(file_path, list):
            loaded_files = []
            for file_index, file in enumerate(file_path):
                if not (file.endswith(".abf") or file.endswith(".wcp")):
                    continue
                try:
                    self.protocols.append(Trace(file))
                    loaded_files.append(file)
                    print(f"Loaded file {file_index + 1}/{len(file_path)}: {file}")
                except (FileNotFoundError, ValueError) as err:
                    print(f"Error loading file {file}: {err}")
                    print("Skipping file.")
                    continue
            self.meta_data = MetaData(loaded_files, experimenter)
        if sort:
            self.sort_by_date()

    def sort_by_date(self):
        """
        Sorts the protocols by the date of the experiment.
        """

        dates = np.array(
            [
                experiment_info_i["date_of_experiment"]
                for experiment_info_i in self.meta_data.experiment_info
            ]
        )
        sorted_indices = np.argsort(dates)
        self.protocols = [self.protocols[i] for i in sorted_indices]
        self.meta_data.file_info = self.meta_data.file_info[sorted_indices]
        self.meta_data.experiment_info = self.meta_data.experiment_info[sorted_indices]

    def meta_data_summary(self, to_dataframe: bool = True) -> dict | pd.DataFrame:
        """
        Returns a summary of the metadata information.
        """

        summary_dict = {
            "file_name": [
                file_info_i["file_name"] for file_info_i in self.meta_data.file_info
            ],
            "file_path": [
                file_info_i["file_path"] for file_info_i in self.meta_data.file_info
            ],
            "date_of_experiment": [
                experiment_info_i["date_of_experiment"]
                for experiment_info_i in self.meta_data.experiment_info
            ],
            "experimenter": [
                experiment_info_i["experimenter"]
                for experiment_info_i in self.meta_data.experiment_info
            ],
        }
        if to_dataframe:
            return pd.DataFrame(summary_dict)
        return summary_dict

    # TODO: function to add different protocols to the same experiment
    # TODO: get summary outputs for the experiment
    # TODO: get summary plots for the experiment
