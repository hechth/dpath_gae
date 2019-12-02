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
    parser = argparse.ArgumentParser(description='Create balanced dataset via rejection resampling.')
    parser.add_argument('dataset',type=str,help='Dataset to balance.')
    parser.add_argument('num_classes', type=int, help='Number of classes in dataset.')

    args = parser.parse_args()

    distribution = tf.fill([args.num_classes],(1.0/args.num_classes))

    features = [{'shape': [1], 'key': 'label', 'dtype':tf.int64}]
    decode_op = ctfd.construct_decode_op(features)

    dataset = tf.data.TFRecordDataset(args.dataset).map(decode_op, num_parallel_calls=8)

    # This causes error with add op datatypes, must be a bug inside tensorflow.
    resample_op = tf.data.experimental.rejection_resample(lambda x: x['label'], distribution)

    balanced_dataset = dataset.apply(resample_op)
    
    print(tf.data.experimental.get_single_element(balanced_dataset.take(1)))


if __name__ == "__main__":
    main(sys.argv[1:])