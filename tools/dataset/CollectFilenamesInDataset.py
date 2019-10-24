import sys, getopt, os
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)

import packages.Utility as cutil
import packages.Tensorflow as ctf
import packages.Tensorflow.Dataset as ctfd

import tensorflow as tf


def main(argv):

    # Get folderpath from arg
    input_directory = ''
    output_filename = ''
    pattern = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:p:")
    except getopt.GetoptError:
        print('Usage -i <input_directory> -o <outfile> -p <pattern>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h","-help","help","--help"):
            print('Usage -i <input_directory> -o <outfile> -p <pattern>')
            sys.exit()
        elif opt in ("-o", "--outfile"):
            output_filename = arg
        elif opt in ("-i","--input_directory"):
            input_directory = arg
        elif opt in ("-p", "--pattern"):
            pattern = arg

    # Collect all files matching the specified pattern
    filenames = cutil.collect_files(input_directory,pattern)

    # Encoding function
    def func_encode(sample):
        feature = { 'filename': ctf.string_feature(sample) }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    ctfd.write(filenames, func_encode, output_filename)
    cutil.publish(output_filename)
    


if __name__ == "__main__":
  main(sys.argv[1:])