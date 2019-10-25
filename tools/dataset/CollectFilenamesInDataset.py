import sys, os, argparse
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)

import packages.Utility as cutil
import packages.Tensorflow as ctf
import packages.Tensorflow.Dataset as ctfd

import tensorflow as tf

def main(argv):
    parser = argparse.ArgumentParser(description='Create tfrecords dataset holding filenames matching a pattern')

    parser.add_argument('input_directory', type=str, help='Path where pattern is evaluated')
    parser.add_argument('pattern', type=str, help='Pattern to be evaluated')
    parser.add_argument('output_filename', type=str, help='Path where to store the dataset')

    args = parser.parse_args()
    
    # Collect all files matching the specified pattern
    filenames = cutil.collect_files(args.input_directory,args.pattern)

    # Encoding function
    def func_encode(sample):
        feature = { 'filename': ctf.string_feature(sample) }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    ctfd.write(filenames, func_encode, args.output_filename)
    cutil.publish(args.output_filename)
    


if __name__ == "__main__":
    main(sys.argv[1:])