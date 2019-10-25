import sys, os, argparse
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)

import packages.Utility as cutil
import packages.Tensorflow as ctf
import packages.Tensorflow.Dataset as ctfd

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()



def main(argv):
    parser = argparse.ArgumentParser(description='Split and shuf')

    parser.add_argument('input', type=str, help='Input tfrecords file.')
    parser.add_argument('factor', type=float, help='Fraction of samples to keep in larger file.')
    parser.add_argument('output', type=str, nargs=2, help='Path where to store the dataset')
    # parser.add_argument("--shuffle", help="If to shuffle the dataset.", action="store_true")

    args = parser.parse_args()
    
    dataset = tf.data.TFRecordDataset(args.input, num_parallel_reads=8)
    
    samples=list()

    for elem in tfe.Iterator(dataset):
        samples.append(elem)
    
    first, second = cutil.split_shuffle_list(samples, args.factor)

    ctfd.write([x.numpy() for x in first], None, args.output[0])
    cutil.publish(args.output[0])
    ctfd.write([x.numpy() for x in second], None, args.output[1])
    cutil.publish(args.output[1])    


if __name__ == "__main__":
    main(sys.argv[1:])