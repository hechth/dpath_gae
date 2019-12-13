import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)
import cv2
import numpy as np
import tensorflow as tf

import packages.Tensorflow as ctf
import packages.Tensorflow.Image as ctfi
import packages.Tensorflow.Model as ctfm
import packages.Utility as cutil

import matplotlib.pyplot as plt

from skimage.feature import (match_descriptors, corner_harris, corner_peaks, ORB, plot_matches)
import sklearn.neighbors

def get_mean_and_cov(X, latent_code_size):
    X_mean = X[:latent_code_size]
    X_cov = np.reshape(X[latent_code_size:],(latent_code_size, latent_code_size))
    X_cov = np.matmul(X_cov, np.transpose(X_cov))
    return X_mean, X_cov

def get_mean_and_cov_tf(X, latent_code_size):
    X_mean = X[:latent_code_size]
    X_cov = tf.reshape(X[latent_code_size:],(latent_code_size, latent_code_size))
    X_cov = tf.matmul(X_cov, X_cov, transpose_b=True)
    return X_mean, X_cov

def multivariate_squared_hellinger_distance(X, Y, latent_code_size):
    """
    See https://en.wikipedia.org/wiki/Hellinger_distance for information on hellinger distance.
    X and Y are assumend to be mean vector of length K and covariance matrix with size KxK of multivariate normal distributions.
    
    Simplified formulas:        \n
    (1):    H2(P,Q)=1-A*exp(B)  \n
    (2):    A = A1/A2           \n
    (2.1):  A1 = det(sigma_x)^(1/4) * det(sigma_y)^(1/4) \n
    (2.2):  A2 = det((sigma_x + sigma_y)/2)^(1/2) \n
    (3):    B = -(1/8) * B1^T * B2 * B1 \n
    (3.1):  B1 = (mean_x - mean_y) \n
    (3.2):  B2 = ((sigma_x + sigma_y) / 2) ^ (-1)

    Parameters
    ----------
    X: tf.contrib.distributions.MultiVariateNormal* distribution. \n
    Y: tf.contrib.distributions.MultiVariateNormal* distribution.
    Returns
    -------
    h_squared: [batch_size,] vector filled with float
    """
    X_mean, X_cov = get_mean_and_cov(X, latent_code_size)
    Y_mean, Y_cov = get_mean_and_cov(Y, latent_code_size)
    Y_cov_inv = np.linalg.inv(Y_cov)

    A1 = np.math.pow(np.linalg.det(X_cov),0.25) * np.math.pow(np.linalg.det(Y_cov),0.25)
    A2 = np.sqrt(np.linalg.det((X_cov + Y_cov)* 0.5))
    A = A1/A2
    B1 = np.expand_dims(X_mean - Y_mean, axis=-1)
    B2 = np.linalg.inv((X_cov + Y_cov)* 0.5)
    B = -0.125 * np.squeeze(np.matmul(np.transpose(B1), np.matmul(B2,B1)))
    h_squared = 1.0 - A * np.math.exp(B)
    return h_squared

def main(argv):
    parser = argparse.ArgumentParser(description='Register images using keypoints.')
    parser.add_argument('export_dir',type=str,help='Path to saved model.')
    parser.add_argument('mean', type=str, help='Path to npy file holding mean for normalization.')
    parser.add_argument('variance', type=str, help='Path to npy file holding variance for normalization.')
    parser.add_argument('patch_size', type=int, help='Size of image patch.')
    parser.add_argument('source_filename', type=str,help='Image file from which to extract patch.')
    parser.add_argument('source_image_size', type=int, nargs=2, help='Size of the input image, HW.')
    parser.add_argument('target_filename', type=str,help='Image file for which to create the heatmap.')
    parser.add_argument('target_image_size', type=int, nargs=2, help='Size of the input image for which to create heatmap, HW.')
    parser.add_argument('num_keypoints', type=int, help='Number of keypoints to detect.')
    parser.add_argument('num_matches', type=int, help='Number of matches to keep.')    
    parser.add_argument('--stain_code_size', type=int, dest='stain_code_size', default=0,
        help='Optional: Size of the stain code to use, which is skipped for similarity estimation')
    parser.add_argument('--leaf_size', type=int, dest='leaf_size', default=30,
        help='Number of elements to keep in leaf nodes of search tree.')
    parser.add_argument('--method', type=str, dest='method', default='SKLD', help='Method to use to measure similarity, one of KLD, SKLD, BD, HD, SQHD.')


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

        # Load image and extract patch from it and create distribution.
        source_image = tf.expand_dims(ctfi.load(args.source_filename,height=args.source_image_size[0], width=args.source_image_size[1]),0)
        im_source = (sess.run(source_image[0]) * 255).astype(np.uint8)
        target_image =  tf.expand_dims(ctfi.load(args.target_filename,height=args.target_image_size[0], width=args.target_image_size[1]),0)
        im_target = (sess.run(target_image[0]) * 255).astype(np.uint8)

        orb = cv2.ORB_create(args.num_keypoints)        
        source_keypoints = orb.detect(im_source, None)
        target_keypoints = orb.detect(im_target, None)

        def get_patch_at(keypoint, image):
            return tf.image.extract_glimpse(image,[args.patch_size, args.patch_size], [keypoint.pt], normalized=False, centered=False)

        source_patches = normalize(tf.concat(list(map(lambda x: get_patch_at(x, source_image), source_keypoints)),0))
        target_patches = normalize(tf.concat(list(map(lambda x: get_patch_at(x, target_image), target_keypoints)),0))

        source_cov, source_mean  = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): source_patches })
        target_cov, target_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): target_patches })
        
        batch, latent_code_size = target_mean.get_shape().as_list()
        structure_code_size = latent_code_size - args.stain_code_size

        source_descriptors = tf.concat([source_mean[:,args.stain_code_size:], tf.layers.flatten(source_cov[:,args.stain_code_size:,args.stain_code_size:])], -1)
        target_descriptors = tf.concat([target_mean[:,args.stain_code_size:], tf.layers.flatten(target_cov[:,args.stain_code_size:,args.stain_code_size:])], -1)


        def multi_kl_div(X,Y):
            X_mean, X_cov = get_mean_and_cov(X, structure_code_size)
            Y_mean, Y_cov = get_mean_and_cov(Y, structure_code_size)
            Y_cov_inv = np.linalg.inv(Y_cov)

            trace_term = np.matrix.trace(np.matmul(Y_cov_inv, X_cov))
            diff_mean = np.expand_dims(Y_mean - X_mean, axis=-1)
            middle_term = np.matmul(np.transpose(diff_mean), np.matmul(Y_cov_inv, diff_mean))
            determinant_term = np.log(np.linalg.det(Y_cov) / np.linalg.det(X_cov))

            value = 0.5 * (trace_term + middle_term - structure_code_size + determinant_term)
            return np.squeeze(value)

        def multi_kl_div_tf(X,Y):
            X_mean, X_cov = get_mean_and_cov_tf(X, structure_code_size)
            Y_mean, Y_cov = get_mean_and_cov_tf(Y, structure_code_size)
            Y_cov_inv = tf.linalg.inv(Y_cov)

            trace_term = tf.linalg.trace(tf.matmul(Y_cov_inv, X_cov))
            diff_mean = tf.expand_dims(Y_mean - X_mean, axis=-1)
            middle_term = tf.matmul(tf.transpose(diff_mean), tf.matmul(Y_cov_inv, diff_mean))
            determinant_term = tf.log(tf.linalg.det(Y_cov) / tf.linalg.det(X_cov))

            value = 0.5 * (trace_term + middle_term - structure_code_size + determinant_term)
            return np.squeeze(value)

        def sym_kl_div(X,Y):
            return multi_kl_div(X,Y) + multi_kl_div(Y,X)

        def sym_kl_div_tf(X,Y):
            return multi_kl_div_tf(X,Y) + multi_kl_div_tf(Y,X)

        def sqhd(X,Y):
            return multivariate_squared_hellinger_distance(X,Y,structure_code_size)

        saver.restore(sess,latest_checkpoint)

        source_descriptors_eval = sess.run(source_descriptors)
        target_descriptors_eval = sess.run(target_descriptors)

        #matches = match_descriptors(source_descriptors, target_descriptors, metric=lambda x,y: sym_kl_div(x,y), cross_check=True)
        if args.method == 'SKLD':
            metric = sym_kl_div
        elif args.method == 'SQHD':
            metric = sqhd
        else:
            metric = sym_kl_div

        knn_source = sklearn.neighbors.NearestNeighbors(n_neighbors=5, radius=1.0, algorithm='ball_tree', leaf_size=args.leaf_size, metric=metric)
        knn_source.fit(target_descriptors_eval)

        distances, indices = knn_source.kneighbors(source_descriptors_eval, n_neighbors=1)
        matches = list(zip(range(len(indices)), np.squeeze(indices), np.squeeze(distances)))
        # Sort matches by score
        
        matches.sort(key=lambda x: x[2], reverse=False)
        matches = matches[:args.num_matches]

        def create_dmatch(queryIdx, trainIdx, distance):
            dmatch = cv2.DMatch(queryIdx, trainIdx, 0, distance)
            return dmatch

        cv_matches = list(map(lambda x: create_dmatch(x[0], x[1], x[2]),matches))  
        # Draw top matches
        imMatches = cv2.drawMatches(im_source, source_keypoints, im_target, target_keypoints, cv_matches, None)

        fix, ax = plt.subplots(1)
        ax.imshow(imMatches)
        plt.show()

        print("Detected keypoints!")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])