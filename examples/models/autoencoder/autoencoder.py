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

def train_func():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255

    # Reshape input data from (28, 28) to (28, 28, 1)
    w, h = 28, 28
    x_train = x_train.reshape(x_train.shape[0], w, h, 1)

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    train_ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x_train),tf.data.Dataset.from_tensor_slices(y_train)))
    train_ds = train_ds.map(lambda image, label: ({"val": image }, label))

    train_ds = train_ds.repeat().shuffle(60000).batch(10)
    return train_ds

def main(argv):
    parser = argparse.ArgumentParser(description='Compute latent code for image patch by model inference.')
    parser.add_argument('export_dir',type=str,help='Path to saved model to use for inference.')

    args = parser.parse_args()
    model_dir = args.export_dir

    config_filename = os.path.join(git_root, 'examples','models','autoencoder','autoencoder.json')    
    cfg = ctfm.parse_json(config_filename)['model']


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

    classifier = classifier.train(input_fn=train_func, steps=10000)



if __name__ == "__main__":
    main(sys.argv[1:])