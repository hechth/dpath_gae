import glob, os
import stat
import re
from sklearn.model_selection import train_test_split
import importlib, inspect

numpy_formats = ['npy', 'npz']

def collect_files(path, pattern)->list:
    """
    Function which collects everything matching to pattern in a directory.

    Returns
    -------
    return: list of elements matching pattern in directory
    """
    return glob.glob(os.path.join(path, pattern))

def split_shuffle_list(data, first_list_part, random_state=42)->tuple:
    """
    Function that splits a list into two lists with proportions first_list_part, 1 - first_list_part and random elements from the original list.

    Returns
    -------
    result: tuple(list, list)
    """
    first, second = train_test_split(data, test_size=1-first_list_part,random_state=random_state)
    return (first,second)

def make_directory(path, make_public=True):
    """
    Function to create directory at path if it doesn't exist.
    If make_public is True, gives rwx to all users.
    """
    if os.path.exists(path) == False:
        os.mkdir(path)
    if make_public == True:
        publish(path)

def publish(path):
    """
    Function which gives every user rwx access to target at path.
    """
    os.chmod(path, stat.S_IROTH |stat.S_IWOTH | stat.S_IXOTH)

def get_extension(filename)->str:
    """
    Function to get type extension from filename.

    Returns
    -------
    extension: string
    """
    return filename.split('.')[-1]

def is_numpy_format(filename):
    """
    Function to check if file is a numpy IO format.

    Returns
    -------
    True if file is .npy or .npz file, False otherwise.
    """
    return get_extension(filename) in numpy_formats

def match_regex(pattern, string):
    """
    Function which searches for pattern in string and returns the matching strings
    
    Returns
    -------
    string or list of strings matching regex pattern.
    """
    return re.search(pattern, string).group()

def call_function(filename, function):
    module = importlib.import_module(filename, package=None)
    members = inspect.getmembers(module)
    members


    with open(filename, "r") as f:
        text = f.readlines()
        for line in text:
            if line.find("def " + function) > -1:



