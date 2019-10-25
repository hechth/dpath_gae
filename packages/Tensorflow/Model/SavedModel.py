import sys
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

from packages.Utility import match_regex

def determine_patch_size(model_dir)->int:
    """
    Function which extracts information about the patch size from the model directory string.
    Returns
    -------
    patch_size: int
    """
    return int(match_regex('(p{1}[0-9]+)(?=-)',model_dir)[1:])

def determine_batch_size(model_dir)->int:
    """
    Function which extracts information about the batch size from the model directory string.
    Returns
    -------
    batch_size: int
    """
    return int(match_regex('(b{1}[0-9]+)(?=-)',model_dir)[1:])

def determine_latent_space_size(model_dir)->int:
    """
    Function which extracts information about the latent_space_size from the model directory string.
    Returns
    -------
    latent_space_size: int
    """
    return int(match_regex('(l{1}[0-9]+)(?=-)',model_dir)[1:])