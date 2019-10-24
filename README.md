# dpath_gae
Disentangled representation learning model for digital pathology data as a custom similarity metric for deformable image registration.

[Latent Space Traversals]: data/images/latent_traversals_-5_to_5.png

## Setup
0.  Install [CUDA](https://developer.nvidia.com/cuda-zone) if you have a NVIDIA GPU.
1.  Install [tensorflow](https://www.tensorflow.org/install/pip) v1.12, preferably with [GPU suppurt enabled](https://www.tensorflow.org/install/gpu). GPU support is not necessary, but makes stuff faster.
2.  Install [ITK](https://github.com/InsightSoftwareConsortium/ITK).
    1.  Clone from github and check out v4.13.1
    1.  Configure and build it using CMake
    2.  Install it using [checkinstall](https://debian-administration.org/article/147/Installing_packages_from_source_code_with_checkinstall), you might want to remove it later.
3.  Install [tensorflowCC](https://github.com/FloopCZ/tensorflow_cc).
4.  Install Qt5 v5.9.5
