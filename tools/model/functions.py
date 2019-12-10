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

    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:
        # Load image and extract patch from it and create distribution.
        patch_cov, patch_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): tf.expand_dims(tf.convert_to_tensor(patch),0) })
        saver.restore(sess,latest_checkpoint)

        patch_mean_np = sess.run(patch_mean)
        patch_cov_np = sess.run(patch_cov)
        sess.close()
    return patch_mean_np, patch_cov_np


def main(argv):
    filename_patch = os.path.join(git_root,'data','images','CD3_level_1_cropped_patch_32x32.png')
    export_dir = os.path.join(git_root,'data','models','gae')

    patch = cv2.imread(filename_patch).astype(np.float32)    
    mean, cov = get_distribution_for_patch(export_dir, patch)
    print(mean, cov)

if __name__ == "__main__":
    main(sys.argv[1:])