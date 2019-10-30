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

    inputs, labels = ctfm.parse_inputs(features, labels, cfg['inputs'])
    components = {}
   
    for comp in cfg['components']:
        components[comp['name']] = ctfm.parse_component(inputs, comp, inputs)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    loss = tf.losses.absolute_difference(inputs['patch'], inputs['logits'])

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'loss' : loss})

    assert mode == tf.estimator.ModeKeys.TRAIN   
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)




def main(argv):
    parser = argparse.ArgumentParser(description='Compute latent code for image patch by model inference.')
    parser.add_argument('export_dir',type=str,help='Path to saved model to use for inference.')

    args = parser.parse_args()
    model_dir = args.export_dir

    config_filename = os.path.join(git_root, 'examples','models','autoencoder','autoencoder.json')    
    cfg = ctfm.parse_json(config_filename)


    params_dict = {
        'config': cfg,
        'model_dir': model_dir,
    }          

    classifier = tf.estimator.Estimator(
      model_fn=my_model,
      model_dir=model_dir,
      params=params_dict,
      config=tf.estimator.RunConfig(model_dir=model_dir, save_summary_steps=100, log_step_count_steps=100)
    )

    classifier = classifier.train(input_fn=ctfd.construct_train_fn(cfg['inputs']), steps=cfg['inputs']['datasets']['training']['steps'])



if __name__ == "__main__":
    main(sys.argv[1:])