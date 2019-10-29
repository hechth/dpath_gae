import sys, os, json, argparse
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)
from psutil import virtual_memory
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm
import packages.Tensorflow.Dataset as ctfd

def my_model(features, labels, mode, params, config):
    cfg = params['config']

    inputs, labels = ctfm.parse_inputs(features, labels, cfg['inputs'])
    outputs = {}

    layers, variables, forward_pass = ctfm.parse_component(inputs, cfg['model'], outputs)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    loss = tf.losses.absolute_difference(labels, outputs['logits'])

    train_op = optimizer.minimize(loss,var_list=variables, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'loss' : loss})

    assert mode == tf.estimator.ModeKeys.TRAIN   
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def construct_train_fn(config):
    cfg_dataset = config['datasets']

    cfg_train_ds = cutil.safe_get('training', cfg_dataset)

    # Create operations
    decode_op = ctfd.construct_decode_op(config['inputs'])
    unzip_op = ctfd.construct_unzip_op(config['inputs'])

    preprocess = cutil.concatenate_functions([unzip_op])
   
    def train_fn():
        #Load the dataset
        dataset = tf.data.TFRecordDataset(cfg_train_ds['filename']).map(decode_op)
        
        element_size = sys.getsizeof(dataset.output_types)
        buffer_size = int(virtual_memory().total / 2 / element_size)

        dataset = dataset.shuffle(buffer_size)


        #dataset = dataset.apply(tf.data.experimental.map_and_batch(preprocess, cfg_train_ds['batch'], num_parallel_batches=os.cpu_count()))
        
        dataset = dataset.map(preprocess, num_parallel_calls=os.cpu_count())
        dataset = dataset.batch(cfg_train_ds['batch'])
        dataset = dataset.prefetch(buffer_size=1)
        return dataset.repeat()

    return train_fn


def main(argv):
    parser = argparse.ArgumentParser(description='Compute latent code for image patch by model inference.')
    parser.add_argument('export_dir',type=str,help='Path to saved model to use for inference.')

    args = parser.parse_args()

    config_filename = os.path.join(git_root, 'examples','training','dataset','model_with_dataset.json')
    
    cfg = ctfm.parse_json(config_filename)

    cfg_dataset = cfg['datasets']
    
    cfg_train_ds = cutil.safe_get('training', cfg_dataset)


    model_dir = args.export_dir

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

    classifier = classifier.train(input_fn=construct_train_fn(cfg), steps=cfg_train_ds['steps'])



if __name__ == "__main__":
    main(sys.argv[1:])