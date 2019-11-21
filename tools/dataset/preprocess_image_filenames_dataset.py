import sys, os, argparse, re, itertools
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)

import packages.Utility as cutil

import packages.Tensorflow as ctf
import packages.Tensorflow.Dataset as ctfd
import packages.Tensorflow.Image as ctfi


import tensorflow as tf
tf.enable_eager_execution()

def _decode_example_filename(example_proto):
    image_feature_description = {     
        'filename': tf.FixedLenFeature([], tf.string),
    }
    return tf.parse_single_example(example_proto, image_feature_description)

   

def main(argv):
    parser = argparse.ArgumentParser(description='Create tfrecords dataset holding patches of images specified by filename in input dataset.')

    parser.add_argument('input_dataset', type=str, help='Path to dataset holding image filenames')
    parser.add_argument('output_dataset', type=str, help='Path where to store the output dataset')
    parser.add_argument('patch_size', type=int, help='Patch size which to use in the preprocessed dataset')
    parser.add_argument('labels', type=lambda s: [item for item in s.split(',')], help="Comma separated list of labels to find in filenames.")

    args = parser.parse_args()

    labels_table = tf.contrib.lookup.index_table_from_tensor(
        mapping=args.labels,
        num_oov_buckets=1,
        default_value=-1
    )
    
    filename_dataset = tf.data.TFRecordDataset(args.input_dataset, num_parallel_reads=8).map(_decode_example_filename)
    
    def _extract_label(filename):
       label = filter(lambda x : x is not None, [cutil.match_regex('/' + opt + '/', filename) for opt in args.labels]).next()
       return label

    print(_extract_label('/media/hecht/DPath/SB01/01/CD45RO_CD68/CD68/level_1'))

    images_dataset = filename_dataset.map(lambda feature: {'image': ctfi.load(feature['filename']), 'filename': feature['filename']})



        


if __name__ == "__main__":
    main(sys.argv[1:])