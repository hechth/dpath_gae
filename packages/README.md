# Packages

This folder contains all functionality which is meant to be used like a python package.

To import any of the packages, add the root folder of the repository using sys.path.append(<GIT_ROOT_FOLDER>).

```python
import sys
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)
```
