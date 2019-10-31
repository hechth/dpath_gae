import sys, os, json, argparse
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm
import packages.Tensorflow.Dataset as ctfd

def my_model(features, labels, mode, params, config):
    cfg = params['config']

    tensors, labels = ctfm.parse_inputs(features, labels, cfg['inputs'])
    components = {}
   
    for comp in cfg['components']:
        components[comp['name']] = ctfm.parse_component(tensors, comp, tensors)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    loss = tf.losses.absolute_difference(tensors['val'], tensors['logits'])

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'loss' : loss})

    assert mode == tf.estimator.ModeKeys.TRAIN   
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(argv):


if __name__ == "__main__":
    main(sys.argv[1:])