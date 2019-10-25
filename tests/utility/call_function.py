import sys, os
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import importlib, inspect

import tests.utility


def main(argv):
    filename = 'dummy_file.py'
    path = os.path.join(git_root,'tests','utility', filename)
    full_module = '.'.join([git_root.replace(os.path.sep,'.'),'tests','utility',filename.split('.')[0]])
    function_name = 'dummy_function'
    args = 'blub'
    print(path)


    # Create package and module name from path
    package = os.path.dirname(path).replace(os.path.sep,'.')
    module_name = os.path.basename(path).split('.')[0]


    # Import module and get members
    module = importlib.import_module(module_name, package)
    members = inspect.getmembers(module)

    # Find matching function
    function = [t[1] for t in members if t[0] == function_name][0]
    function(args)


if __name__ == "__main__":
    main(sys.argv[1:])