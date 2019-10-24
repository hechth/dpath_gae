# Packages

This folder contains all functionality which is meant to be used like a python package.

To import any of the packages, add the root folder of the repository using sys.path.append(<GIT_ROOT_FOLDER>).

```python
import sys
import git

def get_git_root()->str:
        git_repo = git.Repo(sys.path[0], search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        return git_root

sys.path.append(get_git_root())
```
