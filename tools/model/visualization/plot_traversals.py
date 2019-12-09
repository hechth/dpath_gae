import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import tensorflow as tf

from tensorflow.contrib import predictor

import packages.Tensorflow as ctf
import packages.Tensorflow.Image as ctfi
import packages.Utility as cutil

import matplotlib.pyplot as plt

mean = np.load("mean.npy")
variance = np.load("variance.npy")
stddev = [np.math.sqrt(x) for x in variance]

def denormalize(image):
    channels = [np.expand_dims(image[:,:,channel] * stddev[channel] + mean[channel],-1) for channel in range(3)]
    denormalized_image = ctfi.rescale(np.concatenate(channels, 2), 0.0, 1.0)
    return denormalized_image

def normalize(image, name=None):
    channels = [tf.expand_dims((image[:,:,:,channel] - mean[channel]) / stddev[channel],-1) for channel in range(3)]
    return tf.concat(channels, 3, name=name)

def main(argv):
    parser = argparse.ArgumentParser(description='Plot latent space traversals for model.')
    parser.add_argument('export_dir',type=str,help='Path to saved model.')
    parser.add_argument('filename', type=str,help='Image file or numpy array to run inference on.')
    args = parser.parse_args()

    latest_checkpoint = tf.train.latest_checkpoint(args.export_dir)   
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta', import_scope='imported')
    image = normalize(tf.expand_dims(ctfi.load(args.filename, width=32, height=32),0))

    plots = 21

    fig_traversal, ax_traversal = plt.subplots(18,plots)

    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:

        embedding = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/code:0') ,{ sess.graph.get_tensor_by_name('imported/patch:0'): image })
        
        offsets = tf.expand_dims(tf.lin_space(-11.0, 11.0, plots),-1)


        shifts = tf.concat([tf.pad(offsets, [[0,0],[i, 17 - i]]) for i in range(0,18)], 0)
        codes = tf.tile(embedding,[plots*18,1]) + shifts

        shift_vals = sess.run(shifts)


        reconstructions = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/logits:0') ,{ sess.graph.get_tensor_by_name('imported/code:0'): codes })
        
        saver.restore(sess, latest_checkpoint)

        images = list(map(denormalize, sess.run(reconstructions)))

        for i in range(18*plots):
            ax_traversal[int(i/plots), int(i%plots)].imshow(images[i])

        plt.show()



        



        


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])