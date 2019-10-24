import sys, getopt
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)

import numpy as np
from matplotlib import pyplot as plt

import packages.Tensorflow.Image as ctfi

import tensorflow as tf
tf.enable_eager_execution()

def main(argv):
    filename = ''

    try:
        opts, args = getopt.getopt(argv,"hi:",["image_path="])
    except getopt.GetoptError:
        print('PlotImage.py --image <image_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h","-help","help","--help"):
            print('PlotImage.py --image <image_path>')
            sys.exit()
        elif opt in ("--image", "--image_path", "--i", "-i"):
            filename = arg

    image = ctfi.load(filename, channels=3)   
    
    fig, ax = plt.subplots()
    plt.imshow(image.numpy())
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])