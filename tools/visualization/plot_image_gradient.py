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
    image = tf.expand_dims(image,0)
    dx, dy = tf.image.image_gradients(image)

    dxr,dxg,dxb = tf.split(dx, 3, 3)
    dyr,dyg,dyb = tf.split(dy, 3, 3)

    strides = [1, 1, 1, 1]
    padding = "SAME"

    #reconstructed = tf.nn.conv2d_transpose(dxr + dyr, tf.ones([3,3,1,1], dtype=tf.float32),[1,32,32,1],strides,padding)# + tf.nn.conv2d(dy, tf.ones([3,3,1,3], dtype=tf.float32),strides,padding)
    
    reconstructed = tf.concat([tf.nn.conv2d_transpose(c, tf.ones([3,3,1,1], dtype=tf.float32),[1,32,32,1],strides,padding) for c in tf.split(dx+dy,3,3)],3)
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(image[0].numpy())
    ax[0,1].imshow(reconstructed[0].numpy())
    ax[1,0].imshow(dx[0].numpy())
    ax[1,1].imshow(dy[0].numpy())
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])