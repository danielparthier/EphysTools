import uuid


class SessionInfo:
    def __init__(self):
        self.experimenter_name_val = ""
        self.exp_date = ""
        self.file_path = ""
        self.file_list = []
        self.current_user = ""
        self.theme = ""
        self.session_id = str(uuid.uuid4())
