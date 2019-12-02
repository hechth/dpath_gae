import sys, os, argparse
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)

import packages.Utility as cutil
import packages.Tensorflow as ctf
import packages.Tensorflow.Dataset as ctfd
import packages.Tensorflow.Model as ctfm

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import numpy as np

def main(argv):
    parser = argparse.ArgumentParser(description='Estimate moments for one feature of the dataset.')

    parser.add_argument('dataset', type=str, help='Input dataset filename.')
    parser.add_argument('num_classes', type=int, help='Number of classes in dataset.')
    #parser.add_argument('num_samples', type=int, help='Number of samples to use for estimation.')

    args = parser.parse_args()

    features = [{'shape': [1], 'key': 'label', 'dtype':tf.int64}]
    decode_op = ctfd.construct_decode_op(features)

    dataset = tf.data.TFRecordDataset(args.dataset).map(decode_op, num_parallel_calls=8)
    samples_per_class = np.zeros(args.num_classes, dtype=np.int64)
    for sample in tfe.Iterator(dataset):
        label = sample['label'].numpy()
        samples_per_class[label] += 1
    
    print(samples_per_class)


if __name__ == "__main__":
    main(sys.argv[1:])