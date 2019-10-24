import sys, getopt, os
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)

import packages.Utility as cutil
import packages.Tensorflow.Dataset as ctfd

import tensorflow as tf


def main(argv):

    # Get folderpath from arg
    input_directory = ''
    output_directory = ''
    pattern = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:p:")
    except getopt.GetoptError:
        print('Usage -i <input_directory> -o <output_directory> -p <pattern>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h","-help","help","--help"):
            print('Usage -i <input_directory> -o <output_directory> -p <pattern>')
            sys.exit()
        elif opt in ("-o", "--out","--output",):
            output_directory = arg
        elif opt in ("-i",):
            input_directory = arg
        elif opt in ("-p", "--pattern"):
            pattern = arg

    # Collect all files matching the specified pattern
    filenames = cutil.collect_files(input_directory,pattern)

    def func_encode(sample):
        feature = { 'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample.encode()])) }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    ctfd.write(filenames, func_encode, os.path.join(output_directory,'filenames.tfrecords'))
    


if __name__ == "__main__":
  main(sys.argv[1:])