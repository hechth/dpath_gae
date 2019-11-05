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

    parser.add_argument('filename', type=str, help='Input dataset filename.')
    parser.add_argument('config',type=str, help='Input config file holding features description.')
    parser.add_argument('num_samples', type=int, help='Number of samples to use for estimation.')    
    parser.add_argument('feature', type=str, help='Key of feature for which to estimate the moments.')
    parser.add_argument('axes', type=lambda s: [int(item) for item in s.split(',')], help="Comma separated list of axis to use.")
    parser.add_argument('output', type=str, nargs=2, help='Path where to store the estimated parameters, mean and variance.')
    # parser.add_argument("--shuffle", help="If to shuffle the dataset.", action="store_true")

    args = parser.parse_args()
    
    dataset = tf.data.TFRecordDataset(args.filename, num_parallel_reads=8)

    decode_op = ctfd.construct_decode_op(ctfm.parse_json(args.config))
    dataset = dataset.map(decode_op)
    mean, variance = ctfd.estimate_mean_and_variance(dataset, args.num_samples, args.axes, args.feature)

    np.save(args.output[0], mean)
    np.save(args.output[1], variance)


if __name__ == "__main__":
    main(sys.argv[1:])