import numpy as np
from ephys.classes.experiment_objects import ExpData


class DataContainer:
    def __init__(self, data: ExpData):
        self.data = data

    def get_data(self) -> ExpData:
        return self.data

    def set_data(self, new_data: ExpData) -> None:
        self.data = new_data

    def new_files_check(self, file_path: str | list[str]) -> list[str]:
        # Implement duplicate checking logic here
        existing_file_path = self.data.meta_data.get_file_path()
        new_files = np.array(file_path)[
            np.invert(np.isin(file_path, existing_file_path))
        ]
        if new_files.size == 0:
            print("No new files to add.")
        else:
            print(f"New files detected: {new_files}")
        return new_files.tolist()

    def add_file(self, file_path: str | list[str]) -> None:
        new_files = self.new_files_check(file_path)
        self.data.add_file(new_files)

    def __repr__(self) -> str:
        return f"DataContainer(data={self.data})"
