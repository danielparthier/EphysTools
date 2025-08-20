import uuid
import numpy as np


class SessionInfo:
    def __init__(self):
        self.experimenter_name_val = ""
        self.exp_date = ""
        self.file_path = ""
        self.file_list = []
        self.current_user = ""
        self.theme = ""
        self.session_id = str(uuid.uuid4())

    def set_file_list(self, file_list: list[str] | str | np.ndarray) -> None:
        if isinstance(file_list, str):
            self.file_list = [file_list]
        elif isinstance(file_list, np.ndarray):
            self.file_list = file_list.tolist()
        else:
            self.file_list = file_list

    def set_file_path(self, file_path: list[str] | str | np.ndarray) -> None:
        if isinstance(file_path, str):
            self.file_path = file_path
        elif isinstance(file_path, np.ndarray):
            self.file_path = file_path.tolist()
        else:
            self.file_path = file_path
