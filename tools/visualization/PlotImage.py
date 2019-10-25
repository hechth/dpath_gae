import sys, argparse
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)

import numpy as np
from matplotlib import pyplot as plt

import packages.Tensorflow.Image as ctfi

import tensorflow as tf
tf.enable_eager_execution()

def main(argv):    
    parser = argparse.ArgumentParser(description='Display image readable with tensorflow.')
    parser.add_argument('filename',type=str,help='Image file to display.')
    args = parser.parse_args()

    if ctfi.is_image(args.filename) == False:
        sys.exit(-1)

    image = ctfi.load(args.filename, channels=3)   
    
    fig, ax = plt.subplots()
    plt.imshow(image.numpy())
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])