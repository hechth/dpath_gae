import glob, os
import stat
import re
from sklearn.model_selection import train_test_split
import importlib, inspect
from functools import reduce
from typing import TypeVar, Callable, Sequence

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

def safe_get(key:str, dictionary:dict):
    """
    Function to get value for key which returns None if key doesn't exist
    """
    if key is not None:
        if key in dictionary:
            return dictionary[key]
        else:
            return None
    else:
        return None


def match_regex(pattern, string):
    """
    Function which searches for pattern in string and returns the matching strings
    
    Returns
    -------
    string or list of strings matching regex pattern.
    """
    match = re.search(pattern, string)
    if match is not None:
        return match.group()
    else:
        return None

def get_function(filename: str, funcname: str):
    """
    Function which imports the specified file and returns the function with given name using importlib.
    Parameters
    ----------
    filename: string pointing to file to get the function from
    funcname: string with name of the function

    Returns
    -------
    function: function object with given name
    """
    package = os.path.dirname(filename).replace('/','.')
    module = os.path.basename(filename).strip('.py')

    imported = importlib.import_module(module, package=package)
    members = inspect.getmembers(imported)
    function = [t[1] for t in members if t[0] == funcname][0]
    return function


T = TypeVar('T')

def pipeline(
        value: T,
        function_pipeline: Sequence[Callable[[T], T]],
) -> T:
    '''A generic Unix-like pipeline

    :param value: the value you want to pass through a pipeline
    :param function_pipeline: an ordered list of functions that
        comprise your pipeline
    '''
    return reduce(lambda v, f: f(v), function_pipeline, value)

def concatenate_functions(funcs):
    """
    Function which takes a list of functions and returns a lambda taking a value and passing it through the specified pipeline.

    Returns
    -------
    lambda x: pipeline(x, funcs)
    """
    return lambda x: pipeline(x, funcs)

