import glob, os
from sklearn.model_selection import train_test_split

def collect_files(path, pattern)->list:
    """
    Function which collects everything matching to pattern in a directory.

    Returns
    -------
    return: list of elements matching pattern in directory
    """
    return glob.glob(os.path.join(path, pattern))

def split_shuffle_list(data, first_list_part, random_state=42)->tuple(list, list):
    """
    Function that splits a list into two lists with proportions first_list_part, 1 - first_list_part and random elements from the original list.

    Returns
    -------
    result: tuple(list, list)
    """
    first, second = train_test_split(data, test_size=1-first_list_part,random_state=random_state)
    return (first,second)