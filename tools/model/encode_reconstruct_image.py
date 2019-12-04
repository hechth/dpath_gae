import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import tensorflow as tf

from tensorflow.contrib import predictor

import packages.Tensorflow as ctf
import packages.Tensorflow.Model as ctfm
import packages.Tensorflow.Image as ctfi
import packages.Utility as cutil


def main(argv):
    parser = argparse.ArgumentParser(description='Plot latent space traversals for model.')
    parser.add_argument('export_dir',type=str,help='Path to saved model.')
    parser.add_argument('filename', type=str,help='Image file or numpy array to run inference on.')
    parser.add_argument('mean', type=str, help='Path to npy file holding mean for normalization.')
    parser.add_argument('variance', type=str, help='Path to npy file holding variance for normalization.')
    parser.add_argument('patch_size', type=int, help='Size of image patches.')
    parser.add_argument('stride', type=int, help='Size of stride.')
    args = parser.parse_args()

    mean = np.load(args.mean)
    variance = np.load(args.variance)
    stddev = [np.math.sqrt(x) for x in variance]

    def denormalize(image):
        channels = [np.expand_dims(image[:,:,channel] * stddev[channel] + mean[channel],-1) for channel in range(3)]
        denormalized_image = ctfi.rescale(np.concatenate(channels, 2), 0.0, 1.0)
        return denormalized_image

    def normalize(image, name=None):
        channels = [tf.expand_dims((image[:,:,:,channel] - mean[channel]) / stddev[channel],-1) for channel in range(3)]
        return tf.concat(channels, 3, name=name)

    latest_checkpoint = tf.train.latest_checkpoint(args.export_dir)   
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta', import_scope='imported')

    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:
        image = ctfi.load(args.filename)
        patches = normalize(ctfi.extract_patches(image, args.patch_size, strides=[1,args.stride,args.stride,1]))

        codes = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/code:0') ,{ sess.graph.get_tensor_by_name('imported/patch:0'): patches })
        reconstructions = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/logits:0') ,{ sess.graph.get_tensor_by_name('imported/code:0'): codes })
        
        saver.restore(sess, latest_checkpoint)

        codes_npy = sess.run(codes)
        reconstructions_npy = np.vectorize(denormalize)(sess.run(reconstructions))
        print("Done!")
                


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])