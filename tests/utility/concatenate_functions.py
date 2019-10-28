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
    incr_and_double = cutil.concatenate_functions([increment, double])
    print(incr_and_double(1))

if __name__ == "__main__":
    main(sys.argv[1:])