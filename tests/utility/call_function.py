import sys, os
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import importlib, inspect

import packages.Utility as cutil

def main(argv):
    filename = 'dummy_file.py'
    path = os.path.join(git_root,'tests','utility', filename)
    function_name = 'dummy_function'
    args = 'blub'
    print(path)


    function = cutil.get_function(path,function_name)
    function(args)


if __name__ == "__main__":
    main(sys.argv[1:])