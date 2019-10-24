import glob, os
import stat
from sklearn.model_selection import train_test_split

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