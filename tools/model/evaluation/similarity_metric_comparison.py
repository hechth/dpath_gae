import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib import predictor

import packages.Tensorflow as ctf
import packages.Tensorflow.Model as ctfm
import packages.Tensorflow.Image as ctfi
import packages.Utility as cutil

def main(argv):
    parser = argparse.ArgumentParser(description='Compute codes and reconstructions for image.')
    parser.add_argument('export_dir',type=str,help='Path to saved model.')
    parser.add_argument('mean', type=str, help='Path to npy file holding mean for normalization.')
    parser.add_argument('variance', type=str, help='Path to npy file holding variance for normalization.')
    parser.add_argument('source_filename', type=str,help='Image file from which to extract patch.')
    parser.add_argument('source_image_size', type=int, nargs=2, help='Size of the input image, HW.')
    parser.add_argument('patch_size', type=int, help='Size of image patch.')
    parser.add_argument('target_filename', type=str,help='Image file for which to create the heatmap.')
    parser.add_argument('target_image_size', type=int, nargs=2, help='Size of the input image for which to create heatmap, HW.')
    parser.add_argument('--method', dest='method', type=str, help='Method to use to measure similarity, one of KLD, SKLD, BD, HD, SQHD.')
    parser.add_argument('--stain_code_size', type=int, dest='stain_code_size', default=0,
        help='Optional: Size of the stain code to use, which is skipped for similarity estimation')
    parser.add_argument('--rotate', type=float, dest='angle', default=0,
        help='Optional: rotation angle to rotate target image')
    parser.add_argument('--subsampling_factor', type=int, dest='subsampling_factor', default=1, help='Factor to subsample source and target image.')
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

    # Load image and extract patch from it and create distribution.
    source_image = ctfi.subsample(ctfi.load(args.source_filename,height=args.source_image_size[0], width=args.source_image_size[1]),args.subsampling_factor)
    args.source_image_size = list(map(lambda x: int(x / args.subsampling_factor), args.source_image_size))

    #Load image for which to create the heatmap
    target_image = ctfi.subsample(ctfi.load(args.target_filename,height=args.target_image_size[0], width=args.target_image_size[1]),args.subsampling_factor)
    args.target_image_size = list(map(lambda x: int(x / args.subsampling_factor), args.target_image_size))

    all_source_patches = normalize(ctfi.extract_patches(source_image, args.patch_size, strides=[1,1,1,1], padding='SAME'))
    all_target_patches = normalize(ctfi.extract_patches(target_image, args.patch_size, strides=[1,1,1,1], padding='SAME'))

    num_patches = all_source_patches.get_shape().as_list()[0]

    possible_splits = cutil.get_divisors(num_patches)
    num_splits = possible_splits.pop(0)

    while num_patches / num_splits > 1000 and len(possible_splits) > 0:
        num_splits = possible_splits.pop(0)


    source_patches = tf.split(all_source_patches, num_splits)
    target_patches = tf.split(all_target_patches, num_splits)

    patches = zip(source_patches, target_patches)

    source_patches_placeholder = tf.placeholder(tf.float32, shape=source_patches[0].get_shape())
    target_patches_placeholder = tf.placeholder(tf.float32, shape=target_patches[0].get_shape())

    heatmap = []
    
    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:
        source_patches_cov, source_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_log_sigma_sq/BiasAdd:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): source_patches_placeholder })
        source_patches_distribution = tf.contrib.distributions.MultivariateNormalDiag(source_patches_mean[:,args.stain_code_size:], tf.exp(source_patches_cov[:,args.stain_code_size:]))
        
        target_patches_cov, target_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_log_sigma_sq/BiasAdd:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): target_patches_placeholder })
        target_patches_distribution = tf.contrib.distributions.MultivariateNormalDiag(target_patches_mean[:,args.stain_code_size:], tf.exp(target_patches_cov[:,args.stain_code_size:]))

        similarity = source_patches_distribution.kl_divergence(target_patches_distribution) + target_patches_distribution.kl_divergence(source_patches_distribution)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, latest_checkpoint)

        for source, target in patches:
            heatmap.extend(sess.run(similarity, feed_dict={source_patches_placeholder : sess.run(source), target_patches_placeholder: sess.run(target)}))

        sim_heatmap = np.reshape(heatmap, args.source_image_size)
        plt.imshow(sim_heatmap, cmap='plasma')
        plt.show()

        sess.close()
    return 0

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])