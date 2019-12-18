import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf
import numpy as np
import skimage

import cv2

def get_distribution_for_patch(export_dir, patch):
    latest_checkpoint = tf.train.latest_checkpoint(export_dir)   
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta', import_scope='imported')
    graph = tf.get_default_graph()

    with tf.Session(graph=graph).as_default() as sess:
        # Load image and extract patch from it and create distribution.
        # Lower triangular cov
        patch_tensor = tf.convert_to_tensor(patch)

        if len(patch_tensor.get_shape().as_list()) == 3:
            patch_tensor = tf.expand_dims(patch_tensor, 0)

        patch_cov, patch_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): patch_tensor })
        saver.restore(sess,latest_checkpoint)

        patch_mean_np = sess.run(patch_mean)
        # Return real covariance matrix
        patch_cov_np = sess.run(tf.matmul(patch_cov, patch_cov, transpose_b=True))
        tf.reset_default_graph()
        sess.close()
    return patch_mean_np, patch_cov_np

def multivariate_squared_hellinger_distance(X_mean, X_cov, Y_mean, Y_cov):
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
    X_mean/Y_mean: mean vector of shape [n]. \n
    X_cov/Y_cov: Covariance matrix of shape [n,n]. \n

    Returns
    -------
    h_squared: scalar filled with float
    """
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
    filename_patch = os.path.join(git_root,'data','images','CD3_level_1_cropped_patch_32x32.png')
    export_dir = os.path.join(git_root,'data','models','gae')

    patch = cv2.imread(filename_patch).astype(np.float32)    
    mean, cov = get_distribution_for_patch(export_dir, patch)
    print(mean, cov)
    print(multivariate_squared_hellinger_distance(mean[0,:],cov[0,:,:],np.zeros_like(mean)[0,:], np.identity(mean.size)))

    # Test if second evaluation works as well
    mean, cov = get_distribution_for_patch(export_dir, patch)
    print(mean, cov)
    print(multivariate_squared_hellinger_distance(mean[0,:],cov[0,:,:],np.zeros_like(mean)[0,:], np.identity(mean.size)))

if __name__ == "__main__":
    main(sys.argv[1:])