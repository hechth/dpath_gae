import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import tensorflow as tf

from tensorflow.contrib import predictor

def main(argv):
    latest_checkpoint = tf.train.latest_checkpoint('/sdb1/logs/examples/models/gae_sampler_v2_0')
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta', import_scope='imported')
    with tf.Session().as_default() as sess:
        saver.restore(sess, latest_checkpoint)
        kernel = sess.graph.get_tensor_by_name('imported/z_covariance_root/kernel:0')
        print(sess.run(kernel))

        cov = sess.graph.get_tensor_by_name('imported/z_covariance/MatrixBandPart:0')
        mean = sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')
        cov_val, mean_val = sess.run([cov,mean],feed_dict={'imported/patch:0' : np.zeros([1,32,32,3])})
        cov_tensor = tf.convert_to_tensor(cov_val)
        mean_tensor = tf.convert_to_tensor(mean_val)
        cov_eval = sess.run(cov_tensor)
        print(cov_eval)
        print('Done!')


if __name__ == "__main__":
    main(sys.argv[1:])