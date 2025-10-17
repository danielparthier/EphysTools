"""
This module provides classes for representing experimental data and metadata.
"""

import os
import struct
import json
from typing import Any

from datetime import datetime, timedelta
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
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


class DeviceInfo:
    def __init__(
        self, device_name: str = "unknown", device_description: str = ""
    ) -> None:
        self.device_name = device_name
        self.device_description = device_description


class SubjectInfo:
    def __init__(
        self,
        subject_id: str = "unknown",
        species: str = "mouse",
        strain: str = "C57BL/6J",
        genotype: str = "WT",
        sex: str = "unknown",
        date_of_birth: str = "YYYY-MM-DD",
        date_of_experiment: str = "YYYY-MM-DD",
        post_natal_days: int | None = None,
        expression_construct: str = "unknown",
    ) -> None:
        self.subject_id = subject_id
        self.species = species
        self.strain = strain
        self.genotype = genotype
        self.sex = sex
        self.date_of_birth = date_of_birth
        self.age = post_natal_days
        self.expression_construct = expression_construct
        if date_of_birth != "YYYY-MM-DD":
            self.date_of_birth = datetime.strptime(date_of_birth, "%Y-%m-%d")
        self.date_of_experiment = date_of_experiment
        if date_of_experiment != "YYYY-MM-DD":
            self.date_of_experiment = datetime.strptime(date_of_experiment, "%Y-%m-%d")
        self.calculate_age(post_natal_days)

    def calculate_age(self, post_natal_days: int | None) -> None:
        if (
            isinstance(self.date_of_birth, datetime)
            and isinstance(self.date_of_experiment, datetime)
            and post_natal_days is None
        ):
            self.age = (self.date_of_experiment - self.date_of_birth).days
        else:
            self.age = post_natal_days

    def validate(self) -> None:
        if self.species not in ["mouse", "rat", "human"]:
            print("Species not in default list ('mouse', 'rat', 'human').")
            self.species = "unknown"
            if self.species == "human":
                self.strain = "NA"
            if self.species == "rat":
                if self.strain not in [
                    "Sprague Dawley",
                    "Wistar",
                    "Long Evans",
                ]:  # TODO: connect to owl database for rat strains
                    print(
                        f"Strain '{self.strain}' not in default list. Setting to 'unknown'."
                    )
                    self.strain = "unknown"
            if self.species == "mouse":
                if self.strain not in [
                    "C57BL/6J",
                    "129S1/Sv",
                    "BALB/c",
                ]:  # TODO: connect to owl database for mouse strains
                    print(
                        f"Strain '{self.strain}' not in default list. Setting to 'unknown'."
                    )
                    self.strain = "unknown"
        if self.sex not in ["unknown", "male", "female", "other"]:
            print("Sex not in default list ('unknown', 'male', 'female', 'other').")
            self.sex = "unknown"

        if self.genotype not in ["WT", "KO", "HET"]:
            print(
                f"Genotype '{self.genotype}' not in default list. Setting to 'unknown'."
            )
            self.genotype = "unknown"

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "species": self.species,
            "strain": self.strain,
            "genotype": self.genotype,
            "sex": self.sex,
            "date_of_birth": self.date_of_birth,
            "date_of_experiment": self.date_of_experiment,
            "age": self.age,
            "expression_construct": self.expression_construct,
        }


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
        age (str, optional): The age of the subject in PxxD format (post-natal days).
            Defaults to 'P00D'.
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
        self,
        file_path: str | list = "",
        experimenter: str | list = "unknown",
        license_number: str = "unknown",
        subject_id: str = "unknown",
        age: str = "P00D",
        species: str | list = "mouse",
        strain: str = "C57BL/6J",
        genotype: str = "WT",
        date_of_birth: str = "YYYY-MM-DD",
        sex: str = "unknown",
    ) -> None:
        self.file_info = np.array([])
        self.experiment_info = np.array([])
        self.subject_info = np.array([])
        if file_path != "":
            # Extract subject-related info from kwargs if provided
            self.add_file_info(
                file_path=file_path,
                experimenter=experimenter,
                license_number=license_number,
                subject_id=subject_id,
                age=age,
                species=species,
                strain=strain,
                genotype=genotype,
                date_of_birth=date_of_birth,
                sex=sex,
                add=False,
            )

    def add_file_info(
        self,
        file_path: str | list,
        experimenter: str | list = "unknown",
        license_number: str = "unknown",
        subject_id: str = "unknown",
        age: str | int = "unknown",
        species: str | list = "mouse",
        strain: str = "C57BL/6J",
        genotype: str = "WT",
        date_of_birth: str = "YYYY-MM-DD",
        sex: str = "unknown",
        description: str = "",
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
            age (str, optional): The age of the subject in PxxD format (post-natal days).
                Defaults to 'P00D'.
            species (str | list, optional): The species of the subject.
                Defaults to 'mouse', other options are 'rat', 'human'.
            strain (str, optional): The strain of the subject.
                Defaults to 'C57BL/6J'.
            genotype (str, optional): The genotype of the subject.
                Defaults to 'WT'.
            date_of_birth (str, optional): The date of birth of the subject in 'YYYY-MM-DD' format.
                Defaults to 'YYYY-MM-DD'.
            sex (str, optional): The sex of the subject.
                Defaults to 'unknown', other options are 'male', 'female'.
            description (str, optional): A description of the experiment.
                Defaults to an empty string.
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
            species = "unknown"

        if sex not in ["unknown", "male", "female"]:
            print("Sex not in default list ('unknown', 'male', 'female').")
            sex = "unknown"

        base_dir = os.path.dirname(__file__)
        json_path = os.path.join(base_dir, "..", "database", "mouse_strains.json")

        with open(file=json_path, mode="r", encoding="utf-8") as f:
            strain_list = json.load(f)

        if strain not in strain_list["strain"]:
            print(f"Strain '{strain}' not in default list. Setting to 'unknown'.")
            strain = "unknown"

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
                    "description": description,
                }
            )

            post_natal_days = None  # TODO: handle age in years for humans
            if isinstance(age, str) and age.startswith("P") and "D" in age:
                try:
                    days = int(age[1 : age.index("D")])
                    post_natal_days = days
                except ValueError:
                    print(
                        f"Could not parse age '{age}'. Setting post_natal_days to None."
                    )
                    post_natal_days = None
            elif isinstance(age, int):
                post_natal_days = age
            exp_date_str = estimated_exp_date.strftime("%Y-%m-%d")
            subject_info = SubjectInfo(
                subject_id=subject_id,
                species=species,
                strain=strain,
                genotype=genotype,
                date_of_birth=date_of_birth,
                date_of_experiment=exp_date_str,
                post_natal_days=post_natal_days,
                sex=sex,
            )
            subject_list.append(subject_info)

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

    def to_dict(self, unpack=False) -> dict:
        """
        Converts the MetaData object to a dictionary.
        Args:
            unpack (bool): If True, unpacks the subject information to dictionary.

        Returns:
            dict: A dictionary representation of the MetaData object.
        """
        if unpack:
            subject_info = [subject.to_dict() for subject in self.subject_info]
        else:
            subject_info = [subject for subject in self.subject_info]

        return {
            "file_info": self.file_info,
            "experiment_info": self.experiment_info,
            "subject_info": subject_info,
        }

    def get_file_path(self) -> list[str]:
        """
        Retrieves the file path(s) from the MetaData object.

        Returns:
            list[str]: The file path(s) stored in the MetaData object.
        """
        return self.get_file_attribute("file_path")

    def get_file_name(self) -> list[str]:
        """
        Retrieves the file name(s) from the MetaData object.

        Returns:
            list[str]: The file name(s) stored in the MetaData object.
        """
        return self.get_file_attribute("file_name")

    def get_experimenter(self) -> list[str]:
        """
        Retrieves the experimenter name(s) from the MetaData object.

        Returns:
            list[str]: The experimenter name(s) stored in the MetaData object.
        """
        return self.get_experiment_attribute("experimenter")

    def get_license_number(self) -> list[str]:
        """
        Retrieves the license number(s) from the MetaData object.

        Returns:
            list[str]: The license number(s) stored in the MetaData object.
        """
        return self.get_experiment_attribute("license_number")

    def get_subject_id(self) -> list[str]:
        """
        Retrieves the subject ID(s) from the MetaData object.

        Returns:
            list[str]: The subject ID(s) stored in the MetaData object.
        """
        return self.get_subject_attribute("subject_id")

    def get_strain(self) -> list[str]:
        """
        Retrieves the strain name(s) from the MetaData object.

        Returns:
            list[str]: The strain name(s) stored in the MetaData object.
        """
        return self.get_subject_attribute("strain")

    def get_species(self) -> list[str]:
        """
        Retrieves the species name(s) from the MetaData object.

        Returns:
            list[str]: The species name(s) stored in the MetaData object.
        """
        return self.get_subject_attribute("species")

    def get_date_of_birth(self) -> list[str]:
        """
        Retrieves the date(s) of birth from the MetaData object.

        Returns:
            list[str]: The date(s) of birth stored in the MetaData object.
        """
        return self.get_subject_attribute("date_of_birth")

    def get_date_of_experiment(self) -> list[Any]:
        """
        Retrieves the date(s) of the experiment from the MetaData object.

        Returns:
            list[Any]: The date(s) of the experiment stored in the MetaData object.
        """
        return self.get_experiment_attribute("date_of_experiment")

    def get_experiment_attribute(self, attribute: str) -> list[str]:
        """
        Retrieves the specified attribute(s) from the ExperimentInfo object.

        Args:
            attribute (str): The name of the attribute to retrieve.
        Returns:
            list[str]: The values of the specified attribute(s) from the ExperimentInfo object.
        """
        # check that attribute exists
        if self.experiment_info.size == 0:
            return []
        elif attribute not in self.experiment_info[0]:
            raise ValueError(f"Attribute '{attribute}' not found in experiment_info.")
        else:
            return [
                experiment[attribute]
                for experiment in self.experiment_info
                if attribute in experiment
            ]

    def get_file_attribute(self, attribute: str) -> list[str]:
        """
        Retrieves the specified attribute(s) from the MetaData object.

        Args:
            attribute (str): The name of the attribute to retrieve.
        Returns:
            list[str]: The values of the specified attribute(s) from the MetaData object.
        """
        # check that attribute exists
        if self.file_info.size == 0:
            return []
        elif attribute not in self.file_info[0]:
            raise ValueError(f"Attribute '{attribute}' not found in file_info.")
        else:
            return [file[attribute] for file in self.file_info if attribute in file]

    def get_subject_attribute(self, attribute: str) -> list[str]:
        """
        Retrieves the specified attribute(s) from the SubjectInfo object.

        Args:
            attribute (str): The name of the attribute to retrieve.

        Returns:
            list[str]: The values of the specified attribute(s) from the SubjectInfo object.
        """
        # check that attribute exists
        if self.subject_info.size == 0:
            return []
        if hasattr(self.subject_info[0], attribute):
            return [getattr(subject, attribute) for subject in self.subject_info]
        else:
            raise ValueError(f"Attribute '{attribute}' not found in subject_info.")


class ExpData:
    """
    A class representing experimental data.

    Args:
        file_path (str | list): The path(s) to the file(s) containing the data.
        experimenter (str, optional): The name of the experimenter. Defaults to 'unknown'.
        sort (bool, optional): If True, sorts the protocols by date after loading.
            Defaults to True.
        **kwargs: Additional keyword arguments for the MetaData class.

    Attributes:
        protocols (list): A list of Trace objects representing the protocols.
        meta_data: An instance of the MetaData class.

    """

    def __init__(
        self,
        file_path: str | list,
        experimenter: str = "unknown",
        sort: bool = True,
        **kwargs,
    ) -> None:
        # self.object_id =
        self.protocols = []
        self.meta_data = MetaData()
        if isinstance(file_path, str):
            self.protocols.append(Trace(file_path))
            self.meta_data = MetaData(file_path, experimenter, **kwargs)

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
            self.meta_data = MetaData(
                file_path=loaded_files, experimenter=experimenter, **kwargs
            )
        if sort:
            self.sort_by_date()

    def add_file(
        self,
        file_path: str | list,
        experimenter: str = "unknown",
        sort: bool = True,
    ) -> None:
        """
        Adds a file or files to the ExpData object.

        Args:
            file_path (str | list): The path(s) to the file(s) to be added.
            experimenter (str, optional): The name of the experimenter. Defaults to 'unknown'.
            sort (bool, optional): If True, sorts the protocols by date after adding.
                Defaults to True.
        """

        # Check for duplicates
        new_files = self.new_file_paths(file_path)

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

        Args:
            to_dataframe (bool, optional): If True, returns the summary as a pandas DataFrame
                Defaults to True.
        Returns:
            dict | pd.DataFrame: A dictionary or DataFrame containing the metadata summary.
        """

        summary_dict = {
            "file_name": self.meta_data.get_file_name(),
            "file_path": self.meta_data.get_file_path(),
            "date_of_experiment": self.meta_data.get_date_of_experiment(),
            "experimenter": self.meta_data.get_experiment_attribute("experimenter"),
        }
        if to_dataframe:
            return pd.DataFrame(summary_dict)
        return summary_dict

    def new_file_paths(self, file_paths: list | str) -> list[str]:
        """
        Returns a list of file paths that are not already in the ExpData object.

        Args:
            file_paths (list | str): The path(s) to the file(s) to be checked.

        Returns:
            list[str]: A list of file paths that are not already in the ExpData object.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        new_file_paths = [
            file_path
            for file_path in file_paths
            if file_path not in self.meta_data.get_file_path()
        ]
        return new_file_paths

    def get_window_summary(self, remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Compiles window summary data from all protocols into a single DataFrame.

        Args:
            remove_duplicates (bool, optional): If True, removes duplicate entries
                from the final DataFrame. Defaults to True.
        Returns:
            pd.DataFrame: A DataFrame containing the compiled window summary data.
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
        """
        Applies the window function to each protocol's window_summary.

        Args:
            window (list | tuple | None, optional): The window to apply. Defaults to None.
            channels (Any, optional): The channels to include. Defaults to None.
            signal_type (Any, optional): The type of signal to process. Defaults to None.
            rec_type (str, optional): The type of recording. Defaults to "".
            function (str, optional): The function to apply. Defaults to "mean".
            label (str, optional): The label for the window function. Defaults to "".
            sweep_subset (Any, optional): A subset of sweeps to include. Defaults to None.
            return_output (bool, optional): Whether to return the output. Defaults to False.
            plot_individual (bool, optional): Whether to plot individual traces. Defaults to False.

        Returns:
            None
        """
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
        """
        Applies the `label_diff` method to each protocol's window_summary.

        Args:
            labels (list, optional): Labels to use for differentiation.
            Defaults to None.
            new_name (str, optional): Name for the differentiated label.
            Defaults to "".
            time_label (str, optional): Time label for differentiation.
            Defaults to "".

        Returns:
            None
        """
        for protocol in self.protocols:
            protocol: Trace
            protocol.window_summary.label_diff(
                labels=labels, new_name=new_name, time_label=time_label
            )

    def label_ratio(
        self, labels: list | None = None, new_name: str = "", time_label: str = ""
    ) -> None:
        """
        Calculates and assigns the ratio of specified labels within each protocol's
        window summary.

        Args:
            labels (list, optional): Labels to compute the ratio for. If None,
            uses default labels.
            new_name (str, optional): Name for the computed ratio. Defaults to "".
            time_label (str, optional): Time label for the ratio calculation.
            Defaults to "".

        Returns:
            None
        """
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
        """
        Plots a summary of the experiment data using windowed function outputs.
        This method retrieves a summary DataFrame of windowed experiment data,
        processes it, and visualizes the results using the FunctionOutput class.
        The plot can optionally be aligned to the onset of events.

        Args:
            align_onset (bool, optional): If True, aligns the plot to the onset
            of events. Defaults to False.
            **kwargs: Additional keyword arguments passed to the
            FunctionOutput.plot() method.

        Returns:
            None
        """
        from ephys.classes.window_functions import (
            FunctionOutput,
        )  # pylint: disable=import-outside-toplevel

        df_complete: pd.DataFrame = self.get_window_summary()
        summary_output = FunctionOutput()
        df_complete = df_complete.drop(columns=["time"], errors="ignore")
        df_complete = df_complete.rename(columns={"exp_time": "time"})
        summary_output.from_dataframe(df=df_complete)
        summary_output.plot(plot_trace=False, align_onset=align_onset, **kwargs)

    # TODO: get summary outputs for the experiment
    # TODO: get summary plots for the experiment
