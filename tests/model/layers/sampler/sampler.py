import sys, os, json, argparse
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm

def my_model(features, labels, mode, params, config):
    cfg = params['config']

    inputs, labels = ctfm.parse_inputs(features, labels, cfg['inputs'])
    outputs = {}

    layers, variables, forward_pass = ctfm.parse_component(inputs, cfg['components'][0], outputs)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    loss = tf.losses.absolute_difference(labels, outputs['logits'])

    train_op = optimizer.minimize(loss,var_list=variables, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'loss' : loss})

    assert mode == tf.estimator.ModeKeys.TRAIN   
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train_func():
    x_train = np.linspace(-100,100,num=10000000)
    y_train = [np.math.sin(x) for x in x_train]

    x_train = tf.data.Dataset.from_tensor_slices(x_train)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)

    dataset = tf.data.Dataset.zip((x_train, y_train)).map(lambda x,y: ({'val': x },y)).take(10000000).shuffle(10000000).batch(100)
    return dataset


def main(argv):
    parser = argparse.ArgumentParser(description='Run training for specified model with fixed dataset creation.')
    parser.add_argument('export_dir',type=str,help='Path to store the model.')

    args = parser.parse_args()

    config_filename = os.path.join(git_root, 'tests','model','layers','sampler','sampler.json')
    
    cfg_model = ctfm.parse_json(config_filename)['model']

    model_dir = args.export_dir

    params_dict = {
        'config': cfg_model,
        'model_dir': model_dir,
    }          

    classifier = tf.estimator.Estimator(
      model_fn=my_model,
      model_dir=model_dir,
      params=params_dict,
      config=tf.estimator.RunConfig(model_dir=model_dir, save_summary_steps=100, log_step_count_steps=100)
    )

    classifier = classifier.train(input_fn=train_func, steps=10000000)



if __name__ == "__main__":
    main(sys.argv[1:])