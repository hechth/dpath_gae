import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf
import tensorflow.contrib.distributions as tfcd

import packages.Utility as cutil
import packages.Tensorflow as ctf
tf.enable_eager_execution()

import numpy as np

def main(argv):
    x = tf.convert_to_tensor(np.random.rand(2,1,1,48),dtype=tf.float32)
    input_shape = x.get_shape()

    mean = tf.layers.Dense(
        3,
        name='mean',
        activation=None,
        kernel_initializer=tf.initializers.lecun_uniform(),
        bias_initializer=tf.ones_initializer())
    mean.build(input_shape)
    
    cov = tf.layers.Dense(
        3*3,
        name='cov',
        activation=None,
        kernel_initializer=tf.initializers.lecun_uniform(),
        bias_initializer=tf.ones_initializer())
    cov.build(input_shape)

    cov_shape = cov.compute_output_shape(input_shape)
    new_shape = cov_shape.as_list()[:-1]
    new_shape.extend([3,3])
    print(new_shape)
    cov_activation = tf.reshape(cov(x),new_shape)
    cov_psd = tf.linalg.matmul(cov_activation, cov_activation, transpose_a=True)

    x_mean = mean(x)
    x_cov = cov_psd
    print(x_mean, x_cov)

    N_x = tfcd.MultivariateNormalFullCovariance(x_mean, x_cov)
    print(N_x.covariance())
    print(N_x.sample())



if __name__ == "__main__":
    main(sys.argv[1:])