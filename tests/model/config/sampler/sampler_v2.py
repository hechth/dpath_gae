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
    x = tf.convert_to_tensor(np.random.rand(2,48) - 0.5,dtype=tf.float32)
    input_shape = x.get_shape()

    dims = 3

    mean = tf.layers.Dense(
        dims,
        name='mean',
        activation=None,
        kernel_initializer=tf.initializers.lecun_uniform(),
        bias_initializer=tf.zeros_initializer())
    mean.build(input_shape)
    
    cov = tf.layers.Dense(
        (dims * (dims + 1)) / 2,
        name='cov',
        activation=tf.exp,
        kernel_initializer=tf.initializers.lecun_uniform(),
        bias_initializer=tf.ones_initializer())
    cov.build(input_shape)

    cov_shape = cov.compute_output_shape(input_shape)


    act = cov(x)
    cov_activation = tf.contrib.distributions.fill_triangular(act)
    cov_psd = tf.linalg.matmul(cov_activation, cov_activation, transpose_a=True)


    x_mean = mean(x)
    x_cov = cov_psd
    print(x_mean, x_cov)

    N_x = tfcd.MultivariateNormalFullCovariance(x_mean, x_cov)

    #cov_activation = tf.linalg.set_diag(cov_activation,[[0,1,1],[0,1,1]])
    print(cov_activation)
    N_x_tril = tfcd.MultivariateNormalTriL(loc=x_mean, scale_tril=cov_activation)

    print('Full covariance:')
    print(N_x.covariance())

    print('Tril covariance:')
    print(N_x_tril.covariance())

    print(N_x_tril.kl_divergence(N_x))

    print(N_x.sample())



if __name__ == "__main__":
    main(sys.argv[1:])