import sys, os
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
from matplotlib import pyplot as plt

import packages.Tensorflow.Image as ctfi

import tensorflow as tf
tf.enable_eager_execution()

def main(argv):
    filename = os.path.join(git_root,'data','images','tile_8_14.jpeg')
    image = ctfi.load(filename, width=1024, height=1024, channels=3)

    patches = ctfi.extract_patches(image, 64)
    image_patch = patches[0,:,:,:]
    
    fig, ax = plt.subplots()
    plt.imshow(image_patch.numpy())
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])