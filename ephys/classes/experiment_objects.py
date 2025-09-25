"""
This module provides classes for representing experimental data and metadata.
"""

import os
import struct
from typing import Any

from datetime import datetime, timedelta
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
from ephys import utils
from ephys.classes.trace import Trace

mplstyle.use("fast")


# Function to get the experiment date from ABF and WCP files
# Can be removed once implemented in neo package
def get_abf_exp_date(file_name):
    """
    Extracts the recording date and time from an Axon Binary File (ABF or ABF2).

    This function reads the specified ABF file and attempts to extract the recording
    date and time. For ABF version 1.x files, the recording date is not available and
    the function returns None. For ABF2 files, it reads the date and time information
    from the appropriate file offsets.

    Args:
        file_name (str): Path to the ABF or ABF2 file.

    Returns:
        datetime.datetime or None: The recording date and time as a datetime object if available,
        otherwise None for ABF version 1.x files.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        struct.error: If the file does not contain the expected binary structure.
        ValueError: If the date information cannot be parsed.
    """
    with open(file_name, "rb") as f:
        byte_size = 4
        f.seek(0)
        version_flag = f.read(4).decode("ascii")
        if version_flag == "ABF":
            print("ABF version 1.x detected")
            print("rec_date is not available in this version")
            rec_date = None
        elif version_flag == "ABF2":
            f.seek(16)
            date_integer = struct.unpack("I", f.read(byte_size))
            f.seek(20)
            ms = struct.unpack("I", f.read(byte_size))[0]
            start_date_str = str(date_integer[0])
            rec_date = datetime(
                int(start_date_str[:4]),
                int(start_date_str[4:6]),
                int(start_date_str[6:8]),
            )
            rec_date += timedelta(milliseconds=ms)
        else:
            raise ValueError("Unsupported ABF file version: " + version_flag)
        return rec_date


def get_wcp_exp_date(file_name):
    """
    Extracts the recording date from the header of a WCP (WinWCP) file.

    The function reads the first 1024 bytes of the specified file, decodes it as UTF-8,
    and searches for the 'RTIME' key in the header to extract the recording date.
    If the file version ('VER') is less than 9, it prints a message and returns None,
    as the recording date is not available in earlier versions.

    Args:
        file_name (str): Path to the WCP file.

    Returns:
        datetime.datetime or None: The recording date and time if available, otherwise None.
    """
    with open(file_name, "rb") as f:
        headertext = f.read(1024).decode("utf-8")
        rec_date = None
        for line in headertext.split("\r\n"):
            if "=" not in line:
                continue
            key, val = line.split("=")
            if key == "VER":
                if int(val) < 9:
                    print("WCP file version", val, "detected.")
                    print("rec_date is not available in this version")
            if key == "RTIME":
                rec_date = datetime.strptime(val, "%d/%m/%Y %H:%M:%S")
        return rec_date


def get_exp_date(file_name):
    """
    Returns the date of the experiment from the file header.

    Args:
        file_name (str): The path to the file.

    Returns:
        datetime: The date of the experiment.
    """
    if file_name.endswith(".abf"):
        rec_date = get_abf_exp_date(file_name)
    elif file_name.endswith(".wcp"):
        rec_date = get_wcp_exp_date(file_name)
    else:
        raise ValueError(
            "Unsupported file format. Only .abf and .wcp files are supported."
        )
    return rec_date


class MetaData:
    """
    A class representing metadata for experiment files.

    Args:
        file_path (str | list, optional): The path(s) to the file(s) for which
            metadata is to be created. Defaults to an empty string.
        experimenter (str | list, optional): The name(s) of the experimenter(s).
            Defaults to 'unknown'.
        license_number (str, optional): The license number associated with the
            experiment. Defaults to 'unknown'.
        subject_id (str, optional): The ID of the subject involved in the
            experiment. Defaults to 'unknown'.
        date_of_birth (str, optional): The date of birth of the subject in
            'YYYY-MM-DD' format. Defaults to 'YYYY-MM-DD'.
        sex (str, optional): The sex of the subject. Defaults to 'unknown'.

    Attributes:
        file_info (numpy.ndarray): An array containing information about the
            file(s).
        experiment_info (numpy.ndarray): An array containing information about
            the experiment(s).

    Methods:
        __init__(self, file_path: str | list, experimenter: str | list = 'unknown')
            -> None: Initializes the MetaData object.

        add_file_info(self, file_path: str | list, experimenter: str | list =
            'unknown', add: bool = True) -> None: Adds file information to the
            MetaData object.

        remove_file_info(self, file_path: str | list) -> None: Removes file
            information from the MetaData object.
    """

    def __init__(
        self, file_path: str | list = "", experimenter: str | list = "unknown"
    ) -> None:
        self.file_info = np.array([])
        self.experiment_info = np.array([])
        self.subject_info = np.array([])
        if file_path != "":
            self.add_file_info(file_path, experimenter, add=False)

    def add_file_info(
        self,
        file_path: str | list,
        experimenter: str | list = "unknown",
        license_number: str = "unknown",
        subject_id: str = "unknown",
        species: str | list = ["mouse", "rat", "human"],
        strain: str | list = [
            "C57BL/6J",
            "C57BL/6JEi",
            "C57BL/6N",
            "129S1/SvImJ",
            "BALB/c",
        ],
        genotype: str = "WT",
        date_of_birth: str = "YYYY-MM-DD",
        sex: str = "unknown",
        add: bool = True,
    ) -> None:
        """
        Adds file information to the MetaData object.
        Args:
            file_path (str | list): The path(s) of the file(s) to be added.
            experimenter (str | list, optional): The name(s) of the experimenter(s).
                Defaults to 'unknown'.
            license_number (str, optional): The license number associated with the experiment.
                Defaults to 'unknown'.
            subject_id (str, optional): The ID of the subject involved in the experiment.
                Defaults to 'unknown'.
            species (str | list, optional): The species of the subject.
                Defaults to ['mouse', 'rat', 'human'].
            strain (str | list, optional): The strain of the subject.
                Defaults to ['C57BL/6J'].
            genotype (str, optional): The genotype of the subject.
                Defaults to 'WT'.
            date_of_birth (str, optional): The date of birth of the subject in 'YYYY-MM-DD' format.
                Defaults to 'YYYY-MM-DD'.
            sex (str, optional): The sex of the subject.
                Defaults to 'unknown'.
            add (bool, optional): If True, appends the new file information to existing data.
                Defaults to True.
        """

        if isinstance(file_path, str):
            file_path = [file_path]
        file_list = []
        experiment_list = []
        subject_list = []
        if isinstance(species, list) and len(species) > 1:
            species = "unknown"
        elif isinstance(species, str) and species in ["mouse", "rat", "human"]:
            pass
        else:
            print("Species not in default list ('mouse', 'rat', 'human').")

        for file in file_path:
            time_created = datetime.fromtimestamp(os.path.getctime(file))
            time_modified = datetime.fromtimestamp(os.path.getmtime(file))
            time_rec = get_exp_date(file)
            time_list = [time_created, time_modified]
            if time_rec is not None:
                time_list.append(time_rec)
            time_list.sort()
            estimated_exp_date = time_list[0]
            file_list.append(
                {
                    "data_of_creation": time_created,
                    "last_modified": time_modified,
                    "file_name": os.path.basename(file),
                    "file_path": file,
                }
            )
            experiment_list.append(
                {
                    "date_of_experiment": estimated_exp_date,
                    "experimenter": experimenter,
                    "license": license_number,
                }
            )
            subject_list.append(
                {
                    "species": species,
                    "strain": strain,
                    "genotype": genotype,
                    "subject_id": subject_id,
                    "date_of_birth": date_of_birth,
                    "sex": sex,
                }
            )

            print(
                f"Date of Experiment estimated. Please check for correct date: {estimated_exp_date}"
            )
        if add:
            self.file_info = np.append(self.file_info, np.array(file_list))
            self.experiment_info = np.append(
                self.experiment_info, np.array(experiment_list)
            )
            self.subject_info = np.append(self.subject_info, np.array(subject_list))
        else:
            self.file_info = np.array(file_list)
            self.experiment_info = np.array(experiment_list)
            self.subject_info = np.array(subject_list)

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
        keep_indices = np.invert(np.isin(files_exist, files_to_remove))

        self.file_info = self.file_info[keep_indices]
        self.experiment_info = self.experiment_info[keep_indices]
        self.subject_info = self.subject_info[keep_indices]

    def to_dict(self) -> dict:
        """
        Converts the MetaData object to a dictionary.

        Returns:
            dict: A dictionary representation of the MetaData object.
        """
        return {
            "file_info": self.file_info,
            "experiment_info": self.experiment_info,
            "subject_info": self.subject_info,
        }

    def get_file_path(self) -> list[str]:
        """
        Retrieves the file path(s) from the MetaData object.

        Returns:
            list[str]: The file path(s) stored in the MetaData object.
        """
        return [file["file_name"] for file in self.file_info]

    def get_file_name(self) -> list[str]:
        """
        Retrieves the file name(s) from the MetaData object.

        Returns:
            list[str]: The file name(s) stored in the MetaData object.
        """
        return [file["file_name"] for file in self.file_info]


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
        self.meta_data = MetaData()
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

    def add_file(
        self, file_path: str | list, experimenter: str = "unknown", sort: bool = True
    ) -> None:
        """
        Adds a file or files to the ExpData object.

        Args:
            file_path (str | list): The path(s) to the file(s) to be added.
            experimenter (str, optional): The name of the experimenter. Defaults to 'unknown'.
        """
        if isinstance(file_path, str):
            file_path = [file_path]

        # Check for duplicates
        existing_file_path = self.meta_data.get_file_path()
        new_files = np.array(file_path)[
            np.invert(np.isin(file_path, existing_file_path))
        ].tolist()

        for file in new_files:
            self.protocols.append(Trace(file))
        self.meta_data.add_file_info(new_files, experimenter)
        if sort:
            self.sort_by_date()

    def remove_file(self, index: int) -> None:
        """
        Removes a file from the ExpData object.

        Args:
            index (int): The index of the file to be removed.
        """
        if 0 <= index < len(self.protocols):
            removed_file = self.protocols.pop(index)
            self.meta_data.remove_file_info(removed_file.file_path)
            print(f"Removed file at index {index}: {removed_file.file_path}")

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

    def get_window_summary(self, remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Returns a summary of the window information for each protocol.
        """
        pd_collection = []
        meta_df = self.meta_data_summary()
        exp_start = np.array(
            [
                protocol.rec_datetime.timestamp()
                for protocol in self.protocols
                if protocol.rec_datetime is not None
            ]
        ).min()
        for i, protocol in enumerate(self.protocols):
            protocol: Trace
            df = protocol.window_summary.to_dataframe()
            df["file_name"] = meta_df["file_name"][i]
            if protocol.rec_datetime is not None:
                df["exp_time"] = (
                    df["time"] + protocol.rec_datetime.timestamp() - exp_start
                )
            df["protocol"] = protocol
            pd_collection.append(df)
        df_out = pd.concat(pd_collection, ignore_index=True)
        if remove_duplicates:
            df_out.drop_duplicates(inplace=True)
        return df_out

    def window_function(
        self,
        window: list | tuple | None = None,
        channels: Any = None,
        signal_type: Any = None,
        rec_type: str = "",
        function: str = "mean",
        label: str = "",
        sweep_subset: Any = None,
        return_output: bool = False,
        plot_individual: bool = False,
    ) -> None:
        for protocol in self.protocols:
            protocol: Trace
            protocol.window_function(
                window=window,
                channels=channels,
                signal_type=signal_type,
                rec_type=rec_type,
                function=function,
                label=label,
                sweep_subset=sweep_subset,
                return_output=return_output,
                plot=plot_individual,
            )

    def label_diff(
        self, labels: list | None = None, new_name: str = "", time_label: str = ""
    ) -> None:
        for protocol in self.protocols:
            protocol: Trace
            protocol.window_summary.label_diff(
                labels=labels, new_name=new_name, time_label=time_label
            )

    def label_ratio(
        self, labels: list | None = None, new_name: str = "", time_label: str = ""
    ) -> None:
        for protocol in self.protocols:
            protocol: Trace
            protocol.window_summary.label_ratio(
                labels=labels, new_name=new_name, time_label=time_label
            )

    def plot_summary(
        self,
        align_onset: bool = False,
        **kwargs,
    ) -> None:
        from ephys.classes.window_functions import FunctionOutput

        df_complete: pd.DataFrame = self.get_window_summary()
        summary_output = FunctionOutput()
        df_complete = df_complete.drop(columns=["time"], errors="ignore")
        df_complete = df_complete.rename(columns={"exp_time": "time"})
        summary_output.from_dataframe(df=df_complete)
        summary_output.plot(plot_trace=False, align_onset=align_onset, **kwargs)

    # TODO: get summary outputs for the experiment
    # TODO: get summary plots for the experiment
