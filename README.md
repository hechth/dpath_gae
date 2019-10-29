# dpath_gae

![Latent Space Traversals](data/images/latent_traversals_-5_to_5.png)

## Disclaimer
Development of this project started as Master's Thesis at Technical University of Munich and is now continued as part of my work at Masaryk University in Brno, Czech Republic.
Main contributors are also [Vlad Popovici](https://www.muni.cz/en/people/118944-vlad-popovici) and [Hasan Sarhan](http://campar.in.tum.de/Main/HasanSarhan).
It is targeted towards disentangled representation learning for digital pathology data as a custom similarity metric for deformable image registration.

Please note that this project is under development and doesn't contain the main content as of today.
Collaborations and contributions are welcome!
## Setup
### Python

Packages to install via pip3:
*  [tensorflow](https://www.tensorflow.org/install/pip) v1.12, preferably with [GPU suppurt enabled](https://www.tensorflow.org/install/gpu). GPU support is not necessary, but makes stuff faster.
*   numpy
*   scipy
*   matplotlib
*   [gitpython](https://gitpython.readthedocs.io/en/stable/intro.html#installing-gitpython) & [gitdb](https://pypi.org/project/gitdb/)
*   re

### C++

None of the C++ functionality is implemented as of now, so these are future dependencies.

*  Install [CUDA](https://developer.nvidia.com/cuda-zone) if you have a NVIDIA GPU.
*  Install [ITK](https://github.com/InsightSoftwareConsortium/ITK).
    *  Clone from github and check out v4.13.1
    *  Configure and build it using CMake
    *  Install it using [checkinstall](https://debian-administration.org/article/147/Installing_packages_from_source_code_with_checkinstall), you might want to remove it later.
*  Install [tensorflowCC](https://github.com/FloopCZ/tensorflow_cc).
*  Install Qt5 v5.9.5

## Usage



## Examples

The [examples](examples) folder contains functionality which illustrates how to use this project.