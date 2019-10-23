# dpath_gae
Disentangled representation learning model for digital pathology data as a custom similarity metric for deformable image registration.

## Setup
1.  Install [tensorflow](https://www.tensorflow.org/install/pip)v1.12 preferably with [GPU suppurt enabled](https://www.tensorflow.org/install/gpu). GPU supprt is not necessary.
2. Clone [ITK](https://github.com/InsightSoftwareConsortium/ITK) and build it.
    1. Install it using CMake and [checkinstall](https://debian-administration.org/article/147/Installing_packages_from_source_code_with_checkinstall), you might want to remove it later.
