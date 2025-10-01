import uuid
import numpy as np
from ephys import MetaData


class SessionInfo:
    def __init__(self):
        self.experimenter_name_val = ""
        self.exp_date = ""
        self.file_path = ""
        self.file_list = []
        self.current_user = ""
        self.theme = ""
        self.session_id = str(uuid.uuid4())

    def set_attribute(
        self, attr_name: str, value: str | list[str] | np.ndarray
    ) -> None:
        if hasattr(self, attr_name):
            if isinstance(value, str):
                value_list = [value]
            elif isinstance(value, np.ndarray):
                value_list = value.tolist()
            elif isinstance(value, list):
                value_list = value
            else:
                raise ValueError(
                    "Value must be a string, list of strings, or numpy array."
                )
            setattr(self, attr_name, value_list)
        else:
            raise AttributeError(
                f"Attribute '{attr_name}' does not exist in SessionInfo."
            )

    def set_file_path(self, file_path: list[str] | str | np.ndarray) -> None:
        self.set_attribute("file_path", file_path)

    def set_file_list(self, file_list: list[str] | str | np.ndarray) -> None:
        self.set_attribute("file_list", file_list)

    def set_experimenter_name(
        self, experimenter_name: str | list[str] | np.ndarray
    ) -> None:
        self.set_attribute("experimenter_name_val", experimenter_name)

    def set_exp_date(self, exp_date: str | list[str] | np.ndarray) -> None:
        self.set_attribute("exp_date", exp_date)

    def set_theme(self, theme: str) -> None:
        self.theme: str = theme

    def set_current_user(self, current_user: str) -> None:
        self.current_user: str = current_user

    def sync_metadata(self, meta_data: MetaData) -> None:
        if hasattr(meta_data, "experimenter"):
            self.set_experimenter_name(meta_data.get_experimenter())
        if hasattr(meta_data, "exp_date"):
            self.set_exp_date(meta_data.get_date_of_experiment())
        if hasattr(meta_data, "file_name"):
            self.set_file_list(meta_data.get_file_name())
        if hasattr(meta_data, "file_path"):
            self.set_file_path(meta_data.get_file_path())
