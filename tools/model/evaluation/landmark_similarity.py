import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

import packages.Tensorflow as ctf
import packages.Tensorflow.Model as ctfm
import packages.Tensorflow.Image as ctfi
import packages.Utility as cutil

max_buffer_size_in_byte = 64*64*4*3*1000
max_patch_buffer_size = 2477273088

def get_patch_at(keypoint, images, patch_size):
    return tf.image.extract_glimpse(images, [patch_size, patch_size], [keypoint], normalized=False, centered=False)

def get_landmarks(filename, subsampling_factor=1):
    landmarks = []
    with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                landmarks.append([float(row[2]) / subsampling_factor, float(row[1]) / subsampling_factor])
            landmarks = np.array(landmarks,dtype=np.float32)
    return landmarks

def main(argv):
    parser = argparse.ArgumentParser(description='Compute codes and reconstructions for image.')
    parser.add_argument('export_dir',type=str,help='Path to saved model.')
    parser.add_argument('mean', type=str, help='Path to npy file holding mean for normalization.')
    parser.add_argument('variance', type=str, help='Path to npy file holding variance for normalization.')
    parser.add_argument('source_filename', type=str,help='Image file from which to extract patch.')
    parser.add_argument('source_image_size', type=int, nargs=2, help='Size of the input image, HW.')
    parser.add_argument('source_landmarks', type=str,help='CSV file from which to extract the landmarks for source image.')
    parser.add_argument('target_filename', type=str,help='Image file for which to create the heatmap.')
    parser.add_argument('target_image_size', type=int, nargs=2, help='Size of the input image for which to create heatmap, HW.')
    parser.add_argument('target_landmarks', type=str,help='CSV file from which to extract the landmarks for target image.')
    parser.add_argument('patch_size', type=int, help='Size of image patch.')
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

    def normalize(image, name=None, num_channels=3):
        channels = [tf.expand_dims((image[:,:,:,channel] - mean[channel]) / stddev[channel],-1) for channel in range(num_channels)]
        return tf.concat(channels, num_channels)

    latest_checkpoint = tf.train.latest_checkpoint(args.export_dir)   
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta', import_scope='imported')

    config = tf.ConfigProto()
    config.allow_soft_placement=True
    #config.log_device_placement=True

    # Load image and extract patch from it and create distribution.
    source_image = tf.expand_dims(ctfi.subsample(ctfi.load(args.source_filename,height=args.source_image_size[0], width=args.source_image_size[1]),args.subsampling_factor),0)
    args.source_image_size = list(map(lambda x: int(x / args.subsampling_factor), args.source_image_size))

    #Load image for which to create the heatmap
    target_image = tf.expand_dims(ctfi.subsample(ctfi.load(args.target_filename,height=args.target_image_size[0], width=args.target_image_size[1]),args.subsampling_factor),0)
    args.target_image_size = list(map(lambda x: int(x / args.subsampling_factor), args.target_image_size))

    source_landmarks = get_landmarks(args.source_landmarks, args.subsampling_factor)
    source_patches = tf.squeeze(tf.map_fn(lambda x: get_patch_at(x, source_image, args.patch_size), source_landmarks))

    target_landmarks = get_landmarks(args.target_landmarks, args.subsampling_factor)
    target_patches = tf.squeeze(tf.map_fn(lambda x: get_patch_at(x, target_image, args.patch_size), target_landmarks))


    with tf.Session(config=config).as_default() as sess:
        saver.restore(sess, latest_checkpoint)

        source_patches_cov, source_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_log_sigma_sq/BiasAdd:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): normalize(source_patches) })
        source_patches_distribution = tf.contrib.distributions.MultivariateNormalDiag(source_patches_mean[:,args.stain_code_size:], tf.exp(source_patches_cov[:,args.stain_code_size:]))
        
        target_patches_cov, target_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_log_sigma_sq/BiasAdd:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): normalize(target_patches) })
        target_patches_distribution = tf.contrib.distributions.MultivariateNormalDiag(target_patches_mean[:,args.stain_code_size:], tf.exp(target_patches_cov[:,args.stain_code_size:]))

        #similarities = source_patches_distribution.kl_divergence(target_patches_distribution) + target_patches_distribution.kl_divergence(source_patches_distribution)
        #similarities = ctf.multivariate_squared_hellinger_distance(source_patches_distribution, target_patches_distribution)
        similarities = ctf.bhattacharyya_distance(source_patches_distribution, target_patches_distribution)
        sim_vals = sess.run(similarities)
        min_idx = np.argmin(sim_vals)
        max_idx = np.argmax(sim_vals)
        print(sim_vals)
        print(min_idx, sim_vals[min_idx])
        print(max_idx, sim_vals[max_idx])

        fig, ax = plt.subplots(2,3)
        ax[0,0].imshow(sess.run(source_image[0]))
        ax[0,1].imshow(sess.run(source_patches)[min_idx])
        ax[0,2].imshow(sess.run(source_patches)[max_idx])
        ax[1,0].imshow(sess.run(target_image[0]))
        ax[1,1].imshow(sess.run(target_patches)[min_idx])
        ax[1,2].imshow(sess.run(target_patches)[max_idx])
        plt.show()

        sess.close()
        
    return 0

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])