import requests
from ephys.labfolder.classes.labfolder_access import LabFolderUserInfo
from ephys.labfolder.classes.data_elements import parse_data_element

class Entry:
     def __init__(self, entry_id='', raw=False, user_info: LabFolderUserInfo = None):
        self.entry_id = entry_id
        self.title = ""
        self.author_id = ""
        self.project_id = ""
        self.tags = []
        self.creation_date = ""
        self.elements = []
        if entry_id!='':
            self.get_entry(raw=raw, user_info=user_info)
     def get_entry(self, user_info: LabFolderUserInfo, raw=False):
         self.elements = _get_entry(self, self.entry_id, user_info)
         if not raw:
              self.elements = [parse_data_element(element, user_info) for element in self.elements]
     def add_title(self, title):
         self.title = title
     def add_author_id(self, author_id):
          self.author_id = author_id
     def add_project_id(self, project_id):
          self.project_id = project_id
     def add_tags(self, tags):
          if isinstance(tags, list):
                  self.tags.extend(tags)
          if isinstance(tags, str):
               self.tags.append(tags)
          self.tags = list(set(self.tags))
     def add_creation_date(self, creation_date):
          self.creation_date = creation_date
     def add_elements(self, elements):
          self.elements.append(elements)
     def make_entry(self, user_info: LabFolderUserInfo):
          endpoint = f"{user_info.API_address}entries"
          entry = {
               "title": self.title,
               "author_id": user_info.id,
               "project_id": self.project_id,
               "tags": self.tags,
               "elements": self.elements,
          #     "locked": False
          }
          response = requests.post(endpoint, headers=user_info.auth_token, json=entry)
          if response.status_code == 201:
               response_json = response.json()
               self.entry_id = response_json['id']
          else:
               print(f"Error creating entry: {response.status_code}")
     def update_entry(self, user_info: LabFolderUserInfo):
          if self.entry_id == '':
               print("Entry ID not set.")
               return None
          endpoint = f"{user_info.API_address}entries/{self.entry_id}"
          entry = {
               "title": self.title,
               "author_id": self.author_id,
               "project_id": self.project_id,
               "tags": self.tags,
               "elements": self.elements,
               "locked": False

          }
          response = requests.put(endpoint, headers=user_info.auth_token, json=entry)
          if response.status_code != 200:
               print(f"Error updating entry: {response.status_code}")
          else:
               print(f"Entry updated successfully.")

def _get_entry(entry: Entry, entry_id: str, user_info: LabFolderUserInfo) -> list:
    endpoint = f"{user_info.API_address}entries/{entry_id}"
    entry_dict = requests.get(endpoint, headers=user_info.auth_token).json()
    entry.author_id = entry_dict['author_id']
    entry.creation_date = entry_dict['creation_date']
    entry.tags = entry_dict['tags']
    entry.project_id = entry_dict['project_id']
    entry.title = entry_dict['title']
    elements = []
    for element in entry_dict['elements']:
        if element['type'] == 'FILE':
             elements.append(requests.get(user_info.API_address + f"elements/file/{element['id']}", headers=user_info.auth_token).json())
        elif element['type'] == 'IMAGE':
             elements.append(requests.get(user_info.API_address + f"elements/image/{element['id']}", headers=user_info.auth_token).json())
        elif element['type'] == 'TEXT':
             elements.append(requests.get(user_info.API_address + f"elements/text/{element['id']}", headers=user_info.auth_token).json())
        elif element['type'] == 'TABLE':
             elements.append(requests.get(user_info.API_address + f"elements/table/{element['id']}", headers=user_info.auth_token).json())
        elif element['type'] == 'DATA_ELEMENT_GROUP':
             elements.append(requests.get(user_info.API_address + f"elements/data/{element['id']}", headers=user_info.auth_token).json())
        elif element['type'] == 'DATA':
             elements.append(requests.get(user_info.API_address + f"elements/data/{element['id']}", headers=user_info.auth_token).json())
        elif element['type'] == 'WELL_PLATE':
             elements.append(requests.get(user_info.API_address + f"elements/well-plate/{element['id']}", headers=user_info.auth_token).json())
        else:
             print(f"Element type {element['type']} not supported.")
    return elements
