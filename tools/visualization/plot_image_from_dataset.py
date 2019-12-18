import sys, argparse
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)

import numpy as np
from matplotlib import pyplot as plt

import packages.Tensorflow.Image as ctfi
import packages.Tensorflow.Dataset as ctfd

import tensorflow as tf
tf.enable_eager_execution()

def main(argv):    
    parser = argparse.ArgumentParser(description='Display image from dataset')
    parser.add_argument('dataset',type=str,help='Image file to display.')
    parser.add_argument('key',type=str,help='Key of feature that contains image to be displayed.')
    parser.add_argument('size',type=int,help='Size of samples in dataset.')
    parser.add_argument('position',type=int,help='Position of sample to plot in dataset.')
    args = parser.parse_args()

    features = [{'shape': [args.size, args.size, 3], 'key': args.key, 'dtype':tf.float32}]
    decode_op = ctfd.construct_decode_op(features)

    dataset = tf.data.TFRecordDataset(args.dataset).map(decode_op, num_parallel_calls=8)
    image = tf.data.experimental.get_single_element(dataset.skip(args.position).take(1))[args.key]

    plt.imshow(ctfi.rescale(image.numpy(),0.0, 1.0))
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])