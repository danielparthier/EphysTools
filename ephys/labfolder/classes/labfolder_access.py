"""
This module provides functionality for interacting with the Labfolder API.
It includes methods for user authentication, data retrieval, and other
operations related to Labfolder.
"""
import getpass
import re
import json
import requests

class LabFolderUserInfo:
    """
    A class to represent user information retrieved from Labfolder.

    Attributes:
    -----------
    auth_token : dict
        A dictionary containing the authentication token for accessing Labfolder API.
    labfolder_url : str
        The base URL for the Labfolder instance.
    first_name : str
        The first name of the user.
    last_name : str
        The last name of the user.
    initials : str
        The initials of the user, derived from the first and last name.
    email : str
        The email address of the user.
    id : int
        The unique identifier of the user.

    Methods:
    --------
    get_user_info():
        Retrieves and sets the user information from Labfolder API.
    """

    def __init__(self, auth_token: dict, labfolder_url: str) -> None:
        self.auth_token = auth_token
        self.labfolder_url = labfolder_url
        self.API_address = labfolder_url + '/api/v2/'
        self._get_user_info()

    def _get_user_info(self):
        user_data = requests.get(
            self.API_address + 'me',
            params={'expand': 'user'},
            headers=self.auth_token,
            timeout=5,
        ).json()
        self.first_name = user_data['user']['first_name']
        self.last_name = user_data['user']['last_name']
        self.initials = re.sub(r'[^A-Z]', '', self.first_name) + re.sub(
            r'[^A-Z]', '', self.last_name
        )
        self.email = user_data['user']['email']
        self.id = user_data['user']['id']
        self.location = user_data['user_settings']['zone_id']


def labfolder_login(
    labfolder_url: str, user: str = '', password: str = '', allow_input: bool = True
) -> LabFolderUserInfo:
    """
    Logs into the Labfolder API and returns user information.

    Args:
        labfolder_url (str): The base URL of the Labfolder instance.
        user (str, optional): The user's email address. If not provided, the
        function will prompt for it. Defaults to ''.
        password (str, optional): The user's password. If not provided, the
        function will prompt for it. Defaults to ''.

    Returns:
        LabFolderUserInfo: An instance of LabFolderUserInfo containing the authentication token
        and user details if login is successful.
        None: If login fails due to incorrect credentials, incorrect input, or
        blocked login.

    Raises:
        requests.exceptions.RequestException: If there is an issue with the
        network request.

    Notes:
        - The function will prompt the user for their email and password if they
          are not provided as arguments.
        - The function handles HTTP status codes 200, 400, 401, and 403
          specifically.
    """
    if allow_input:
        if user == '':
            user = input('user email: ')
        if password == '':
            password = getpass.getpass('Password for ' + user + ': ')

    api_url = labfolder_url + '/api/v2/auth/login'

    payload = json.dumps({'user': user, 'password': password})
    headers = {'Content-Type': 'application/json'}
    response = requests.request(
        'POST', api_url, headers=headers, data=payload, timeout=5
    )
    if response.status_code == 401:
        print(
            'Username or password incorrect.\nLogin failed. Status code: '
            + str(response.status_code)
        )
        user_auth = None
    elif response.status_code == 400:
        print(
            'Incorrect input.\nLogin failed. Status code: ' + str(response.status_code)
        )
        user_auth = None
    elif response.status_code == 403:
        print('Blocked login.\nLogin failed. Status code: ' + str(response.status_code))
        user_auth = None
    elif response.status_code == 200:
        auth_token = {'Authorization': 'Bearer ' + response.json()['token']}
        user_auth = LabFolderUserInfo(auth_token, labfolder_url)
        print('Hello ' + user_auth.first_name + '!')
    else:
        print('Login failed. Status code: ' + str(response.status_code))
        user_auth = None
    return user_auth

def labfolder_logout(user: LabFolderUserInfo) -> any:
        status = requests.request('POST', user.labfolder_url + '/api/v2/auth/logout',
                         headers=user.auth_token,
                         timeout=5)
        if status.status_code == 204:
            print('Logout successful')
            return None
        else:
            print('Logout failed. Status code: ' + str(status.status_code))
