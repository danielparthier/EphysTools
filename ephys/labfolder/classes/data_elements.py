import pandas as pd
import numpy as np
import requests
from ephys.labfolder.classes.labfolder_access import LabFolderUserInfo
from copy import deepcopy


class DataElement:
    def __init__(self, id='', description=''):
        self.type = 'DATA'
        self.id = id
        self.description = description
    def load_data(self, user_info: LabFolderUserInfo, id=''):
        if id == '':
            id = self.id
        if id == '':
            print("No data ID provided.")
            return None
        data = requests.get(f"{user_info.API_address}elements/data/{id}", headers=user_info.auth_token).json()
        self.description = data['description']
        self.id = data['id']
    def write_to_labfolder(self, user_info: LabFolderUserInfo, entry_id = ''):
        if entry_id == '':
            print("No entry_id provided.")
            return None
        response = requests.post(f"{user_info.API_address}elements/data",
                                 headers=user_info.auth_token,
                                 json={'entry_id': entry_id, 'description': self.description})
        if response.status_code == 201:
            print(f"Data written to Labfolder.")
            self.id = response.json()['id']
        else:
            print(f"Data could not be written to Labfolder. Status code: {response.status_code}")
            print(response.text)
    def update_on_labfolder(self, user_info: LabFolderUserInfo):
        if self.description == '':
            print("No data to write.")
            return None
        response = requests.put(f"{user_info.API_address}elements/data/{self.id}",
                                 headers=user_info.auth_token,
                                 json={'id': self.id, 'description': self.description})
        if response.status_code == 200:
            print(f"Data updated on Labfolder.")
        else:
            print(f"Data could not be updated on Labfolder. Status code: {response.status_code}")
            print(response.text)
    def __repr__(self):
        return f"DataElement(type={self.type}, description={self.description})"

class DescriptiveDataElement:
    def __init__(self, title='', id='', description=''):
        self.type = 'DESCRIPTIVE_DATA_ELEMENT'
        self.title = title
        self.id = id
        self.description = description
    def to_dict(self):
        return {'type': self.type,'title': self.title, 'description': self.description}
    def __repr__(self):
        return f"DataElement(type={self.type}, title={self.title}, description={self.description})"

class FileElement:
    def __init__(self, id):
        self.type = 'FILE'
        self.id = id

    def __repr__(self):
        return f"DataElement(type={self.type}, title={self.id})"

class ImageElement:
    def __init__(self, title='', id='', original_file_content_type='image/png'):
        self.type = 'IMAGE'
        self.title = title
        self.id = id
        self.original_file_content_type = original_file_content_type
        self.creation_date = ''
        self.image = None

    def load_image(self, user_info: LabFolderUserInfo, id=''):
        import io
        from PIL import Image
        if id == '':
            id = self.id
        if id == '':
            print("No image ID provided.")
            return None
        image_info = requests.get(user_info.API_address + f"elements/image/{id}", headers=user_info.auth_token).json()
        image = requests.get(user_info.API_address + f"elements/image/{id}/original-data", headers=user_info.auth_token)
        self.title = image_info['title']
        self.owner_id = image_info['owner_id']
        self.creation_date = image_info['creation_date']
        self.image = Image.open(io.BytesIO(image.content))
    
    def show_image(self):
        import matplotlib.pyplot as plt
        if self.image is None:
            print("No image to show.")
            return None
        plt.imshow(self.image)
        plt.axis('off')
        plt.show()

    #TODO: Implement write_to_labfolder - it probably has to be written as file (?)
    # def write_to_labfolder(self, user_info: LabFolderUserInfo, entry_id = ''):
    #     if entry_id == '':
    #         print("No entry_id provided.")
    #         return None
    #     response = requests.post(f"{user_info.API_address}elements/file?entry_id={entry_id}&file_name={self.title}&locked=false",
    #                              headers=user_info.auth_token,
    #                              json={'entry_id': entry_id, 'title': self.title, 'original_file_content_type': self.original_file_content_type})
    #     if response.status_code == 201:
    #         print(f"Image written to Labfolder.")
    #     else:
    #         print(f"Image could not be written to Labfolder. Status code: {response.status_code}")
    #         print(response.text)


    def __repr__(self):
        return f"DataElement(type={self.type}, id={self.id}, title={self.title}, original_file_content_type={self.original_file_content_type})"

class TextElement:
    def __init__(self, content='', id=''):
        self.type = 'TEXT'
        self.id = id
        self.content = content
    def load_text(self, user_info: LabFolderUserInfo, id=''):
        if id == '':
            id = self.id
        if id == '':
            print("No text ID provided.")
            return None
        text = requests.get(f"{user_info.API_address}elements/text/{id}", headers=user_info.auth_token).json()
        self.content = text['content']
        self.id = text['id']
    def write_to_labfolder(self, user_info: LabFolderUserInfo, entry_id = ''):
        if entry_id == '':
            print("No entry_id provided.")
            return None
        response = requests.post(f"{user_info.API_address}elements/text",
                                 headers=user_info.auth_token,
                                 json={'entry_id': entry_id, 'content': self.content})
        if response.status_code == 201:
            print(f"Text written to Labfolder.")
            self.id = response.json()['id']
        else:
            print(f"Text could not be written to Labfolder. Status code: {response.status_code}")
            print(response.text)
    def update_on_labfolder(self, user_info: LabFolderUserInfo):
        if self.content == '':
            print("No text to write.")
            return None
        response = requests.put(f"{user_info.API_address}elements/text/{self.id}",
                                 headers=user_info.auth_token,
                                 json={'id': self.id, 'content': self.content})
        if response.status_code == 200:
            print(f"Text updated on Labfolder.")
        else:
            print(f"Text could not be updated on Labfolder. Status code: {response.status_code}")
            print(response.text)

    def __repr__(self):
        return f"DataElement(type={self.type}, id={self.id}, content={self.content})"
    
class DataElementGroup:
    def __init__(self, title='', children=None):
        self.type = 'DATA_ELEMENT_GROUP'
        self.title = title
        self.id = ''
        if isinstance(children, list):
            self.children = children
        else:
            self.children = []

    def add_child(self, child):
        self.children.append(child)
    def to_dict(self):
        return {'type': self.type, 'title': self.title, 'children': [child.to_dict() for child in self.children]}
    def write_to_labfolder(self, user_info: LabFolderUserInfo, entry_id = ''):
        if entry_id == '':
            print("No entry_id provided.")
            return None
        content = {'entry_id': entry_id,
                   'data_elements': [self.to_dict()],
                   'locked': False}
        response = requests.post(f"{user_info.API_address}elements/data",
                                 headers=user_info.auth_token,
                                 json=content)
        if response.status_code == 201:
            print(f"Data element group written to Labfolder.")
            self.id = response.json()['id']
        else:
            print(f"Data element group could not be written to Labfolder. Status code: {response.status_code}")
            print(response.text)
    def update_on_labfolder(self, user_info: LabFolderUserInfo):
        if self.id == '':
            print("No data element group ID provided.")
            return None
        content = {'id': self.id,
                   'title': self.title,
                   'data_elements': [self.to_dict()],
                   'locked': False}
        response = requests.put(f"{user_info.API_address}elements/data/{self.id}",
                                 headers=user_info.auth_token,
                                 json=content)
        if response.status_code == 200:
            print(f"Data element group updated on Labfolder.")
        else:
            print(f"Data element group could not be updated on Labfolder. Status code: {response.status_code}")
    def __labfolder_dict__(self):
        return {'type': self.type, 'title': self.title, 'children': [child.to_dict() for child in self.children]}
            
    def __repr__(self):
        return f"DataElementGroup(title={self.title}, children={self.children})"

class TableElement:
    def __init__(self, id='', entry_id='', table=None, user_info: LabFolderUserInfo = None, import_as_pd=True, header=True):
        self.type = 'TABLE'
        self.entry_id = entry_id
        self.id = id
        self.table = table
        self.creation_date = ''
        self.owner_id = user_info.id if user_info is not None else ''
        self.title = ''
        if isinstance(user_info, LabFolderUserInfo) and id != '':
            self.load_table(user_info, to_pd=import_as_pd, header=header)
            if import_as_pd:
                self.table_to_pd(header=header)
        # TODO: Check if owner_id can be taken from somewhere else. Case: user_info is not from table owner.

    def load_table(self, user_info: LabFolderUserInfo, to_pd=True, header=True):
        if self.id == '':
            print("No table ID provided.")
            return None
        elif not isinstance(user_info, LabFolderUserInfo):
            print("Not logged into Labfolder. User information required.")
            return None
        else:
            table = requests.get(f"{user_info.API_address}elements/table/{self.id}", headers=user_info.auth_token).json()
            self.entry_id = table['entry_id']
            self.creation_date = table['creation_date']
            self.owner_id = table['owner_id']
            self.title = table['title']
            self.table = table['content']['sheets']
            if to_pd:
                self.table_to_pd(header=header)

    def write_to_labfolder(self, user_info: LabFolderUserInfo, entry_id = '', header=True):
        if entry_id == '' and self.entry_id != '':
            entry_id = self.entry_id           
        elif entry_id == '' and self.entry_id == '':
            print("No entry_id provided.")
            return None
        if self.table is None:
            print("No table to write.")
            return None
        table_content = self.convert_pd_to_export(header=header)
        if table_content is None:
            print("Could not convert table to export format.")
            return None
        response = requests.post(f"{user_info.API_address}elements/table",
                                 headers=user_info.auth_token,
                                 json={'entry_id': entry_id, 'title': self.title,'content': table_content, 'locked': False})
        if response.status_code == 201:
            print(f"Table {self.title} written to Labfolder.")
            self.id = response.json()['id']
        else:
            print(f"Table {self.title} could not be written to Labfolder. Status code: {response.status_code}")
            print(response.text)
    def update_on_labfolder(self, user_info: LabFolderUserInfo, header=True):
        if self.table is None:
            print("No table to write.")
            return None
        table_content = self.convert_pd_to_export(header=header)
        if table_content is None:
            print("Could not convert table to export format.")
            return None
        response = requests.put(f"{user_info.API_address}elements/table/{self.id}",
                                 headers=user_info.auth_token,
                                 json={'entry_id': self.entry_id, 'id': self.id, 'content': table_content, 'locked': False})
        if response.status_code == 200:
            print(f"Table {self.title} updated on Labfolder.")
        else:
            print(f"Table {self.title} could not be updated on Labfolder. Status code: {response.status_code}")
            print(response.text)
    def table_to_pd(self, header=True, in_place=True):
        if any([isinstance(element, pd.DataFrame) for element in self.table.values()]):
            print("Table already converted to pandas DataFrame.")
            print(self.table)
            return None
        def table_to_pandas(table, header: bool):
            data = []
            for row in table.keys():
                try:
                    data.append({col: table[row][col].get('value', None) for col in table[row].keys()})
                except KeyError:
                    data.append({col: None for col in table[row].keys()})
            df = pd.DataFrame(data)
            df.fillna(np.nan, inplace=True)
            df.dropna(axis=0, how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)
            if header:
                df.columns = df.iloc[0].tolist()
                df.drop(0, inplace=True)
            df.sort_index(inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        
        if in_place:
            table = self.table
        else:
            table = deepcopy(self.table)
        
        table = {sheet: table_to_pandas(self.table[sheet]['data']['dataTable'],
                                            header=header)
                                            for sheet in self.table.keys()}
        if not in_place:
            return(table)
    def add_sheet(self, sheet_name, table: pd.DataFrame):
        if not isinstance(table, pd.DataFrame):
            print("Table is not a pandas DataFrame.")
            return None
        if self.table is None:
            self.table = {}
        self.table.update({sheet_name: table})
    def table_to_dict(self):
        for sheet in self.table.keys():
            if isinstance(self.table[sheet], pd.DataFrame):
                self.table[sheet] = self.table[sheet].to_dict()
            else:
                print(f"Sheet {sheet} is not a pandas DataFrame.")

    def convert_pd_to_export(self, header = True):
        if not all([isinstance(element, pd.DataFrame) for element in self.table.values()]):
            try:
                self.table_to_pd()
            except:
                print("Table could not be converted to pandas DataFrame.")
                return None    
        tmp_table = deepcopy(self.table)
        table_content = {'sheets': {}}
        for sheet in tmp_table.keys():
            if header:
                tmp_table[sheet].index = tmp_table[sheet].index + 1
                tmp_table[sheet].loc[0] = tmp_table[sheet].columns
            tmp_table[sheet].sort_index(inplace=True)
            tmp_table[sheet].reset_index(drop=True, inplace=True)
            rowCount = tmp_table[sheet].shape[0]
            columnCount = tmp_table[sheet].shape[1]
            tmp_table[sheet].columns = range(tmp_table[sheet].shape[1])
            tmp_table[sheet].replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
            tmp_table[sheet] = tmp_table[sheet].to_dict(orient='index')
            for row in tmp_table[sheet].keys():
                for col in tmp_table[sheet][row].keys():
                    tmp_table[sheet][row][col] = {'value': tmp_table[sheet][row][col]}
            table_content['sheets'].update({sheet: {'name': sheet,
                                                    'rowCount': rowCount,
                                                    'columnCount': columnCount,
                                                    'data': {'dataTable': tmp_table[sheet]}}})
        return table_content 
    
    def dict_to_table(self, in_place=True):
        if in_place:
            table = self.table
        else:
            table = deepcopy(self.table)

        for sheet in table.keys():
            if isinstance(table[sheet], dict):
                table[sheet] = pd.DataFrame(table[sheet])
            else:
                print(f"Sheet {sheet} is not a dictionary.")
        if not in_place:
            return(table)
        
    def to_dict(self, in_place=True):
        if in_place:
            table = self.table
        else:
            table = deepcopy(self.table)
        for sheet in table.keys():
            if isinstance(table[sheet], pd.DataFrame):
                table[sheet] = table[sheet].to_dict()
            elif isinstance(table[sheet], dict):
                pass
            else:
                print(f"Sheet {sheet} is not a pandas DataFrame or dictionary.")
        if not in_place:
            return(table)

    def __repr__(self):
        return f"TableElement(type={self.type}, id={self.id}, table={self.table})"    


def parse_data_element(element, user_info):
    print(element)
    if element.get('element_type', '') == 'DATA_ELEMENT_GROUP':
        group = DataElementGroup(title=element['title'])
        for child in element.get('children', []):
            group.add_child(parse_data_element(child))
        return group
    elif element.get('element_type', '') == 'FILE':
        return FileElement(id=element['id'])
    elif element.get('element_type', '') == 'IMAGE':
        return ImageElement(id=element['id'], title=element['title'], original_file_content_type=element.get('original_file_content_type'))
    elif element.get('element_type', '') == 'TEXT':
        return TextElement(id=element['id'], content=element['content'])
    elif element.get('element_type', '') == 'DESCRIPTIVE_DATA':
        return DescriptiveDataElement(id=element['id'], title=element['title'], description=element.get('description'))
    elif element.get('element_type', '') == 'DATA':
        return DataElement(id=element['id'], description=element.get('description'))
    elif element.get('element_type', '') == 'TABLE':
        return TableElement(id=element['id'], entry_id=element['entry_id'], user_info=user_info)
    else:
        print(f"Unknown element type: {element}")