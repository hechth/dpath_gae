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


def main(argv):
    parser = argparse.ArgumentParser(description='Plot latent space traversals for model.')
    parser.add_argument('export_dir',type=str,help='Path to saved model.')
    parser.add_argument('mean', type=str, help='Path to npy file holding mean for normalization.')
    parser.add_argument('variance', type=str, help='Path to npy file holding variance for normalization.')
    parser.add_argument('source_image', type=str,help='Source image file or numpy array to run inference on.')
    parser.add_argument('target_image', type=str,help='Target image file or numpy array to run inference on.')
    parser.add_argument('image_size', type=int,help='Size of the images, has to be expected input size of model.')
    parser.add_argument('stain_code_size', type=int, help='Size of the stain code.')
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

    source_image = normalize(tf.expand_dims(ctfi.load(args.source_image, width=args.image_size, height=args.image_size),0))
    target_image = normalize(tf.expand_dims(ctfi.load(args.target_image, width=args.image_size, height=args.image_size),0))

    num_plots = 9
    fig, ax = plt.subplots(4,num_plots)

    weights = np.linspace(0.0, 1.0, num=num_plots)

    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:

        embedding_source = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/code:0') ,{ sess.graph.get_tensor_by_name('imported/patch:0'): source_image })
        embedding_target = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/code:0') ,{ sess.graph.get_tensor_by_name('imported/patch:0'): target_image })
        
        embedding_source_stain = embedding_source[:,:args.stain_code_size]
        embedding_source_structure = embedding_source[:,args.stain_code_size:]

        embedding_target_stain = embedding_target[:,:args.stain_code_size]
        embedding_target_structure = embedding_target[:,args.stain_code_size:]

        codes_stain = tf.concat([tf.concat([(1.0 - factor) * embedding_source_stain + factor * embedding_target_stain, embedding_source_structure],-1) for factor in weights],0)
        codes_structure = tf.concat([tf.concat([embedding_target_stain, (1.0 - factor) * embedding_source_structure + factor * embedding_target_structure],-1) for factor in weights],0)
        codes_full = tf.concat([(1.0 - factor) * embedding_source + factor * embedding_target for factor in weights],0)

        reconstructions_stain = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/logits:0') ,{ sess.graph.get_tensor_by_name('imported/code:0'): codes_stain })
        reconstructions_structure = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/logits:0') ,{ sess.graph.get_tensor_by_name('imported/code:0'): codes_structure })
        reconstructions_full = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/logits:0') ,{ sess.graph.get_tensor_by_name('imported/code:0'): codes_full })
        
        saver.restore(sess, latest_checkpoint)

        reconstruction_images_full = list(map(denormalize, sess.run(reconstructions_full)))
        reconstruction_images_stain = list(map(denormalize, sess.run(reconstructions_stain)))
        reconstruction_images_structure = list(map(denormalize, sess.run(reconstructions_structure)))
        interpolations = sess.run(tf.concat([(1.0 - factor) * source_image + factor * target_image for factor in weights],0))
        interpolated_images = list(map(denormalize, interpolations))

        for i in range(num_plots):
            ax[0,i].imshow(interpolated_images[i])
            ax[1,i].imshow(reconstruction_images_stain[i])
            ax[2,i].imshow(reconstruction_images_structure[i])
            ax[3,i].imshow(reconstruction_images_full[i])

        plt.show()

       


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])