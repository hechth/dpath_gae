import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)


import packages.Utility as cutil

def increment(x):
    return x + 1

def double(x):
    return 2*x

def main(argv):
    y = cutil.pipeline(1, [increment, double])
    print(y)

if __name__ == "__main__":
    main(sys.argv[1:])