import sys, os, json, argparse, shutil
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm
import packages.Tensorflow.Dataset as ctfd
import packages.Tensorflow.Tensorboard as ctfb

def my_model(features, labels, mode, params, config):
    cfg = params['config']
    cfg_inputs = cfg.get('inputs')
    cfg_labels = cfg_inputs.get('labels')
    
    # Get adaptive learning rate
    learning_rate = tf.train.polynomial_decay(
        learning_rate=1e-3,
        end_learning_rate=1e-4,
        global_step=tf.train.get_global_step(),
        decay_steps=2e7
    )

    tensors, labels = ctfm.parse_inputs(features, labels, cfg_inputs)

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    components = {}
    for comp in cfg['components']:
        components[comp['name']] = ctfm.parse_component(tensors, comp, tensors)


    # --------------------------------------------------------
    # Losses
    # --------------------------------------------------------   
    reconstr_loss =  tf.reduce_mean(tf.reduce_sum(tf.losses.absolute_difference(tensors['patch'], tensors['logits'], reduction=tf.losses.Reduction.NONE),axis=[1,2,3]))
    loss = reconstr_loss


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------
       

    # Image summaries of patch and reconstruction
    tf.summary.image('images', tensors['patch'], 3)
    tf.summary.image('reconstructions', tensors['logits'], 3)
    deformations_x, deformations_y = tf.split(tensors['deformation'],2,3)
    tf.summary.image('deformations_x', deformations_x, 3)
    tf.summary.image('deformations_y', deformations_y, 3)
    tf.summary.image('texture', tensors['texture'], 3)
    tf.summary.image('rotated_texture', tensors['rotated_texture'], 3)
    tf.summary.image('texture_affine', tensors['texture_affine'], 3)


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    assert mode == tf.estimator.ModeKeys.TRAIN   
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('config', type=str, help='Path to configuration file to use.')
    parser.add_argument('mean', type=str, help='Path to npy file holding mean for normalization.')
    parser.add_argument('variance', type=str, help='Path to npy file holding variance for normalization.')
    parser.add_argument('model_dir',type=str,help='Path to saved model to use for inference.')
    args = parser.parse_args()

    mean = np.load(args.mean)
    variance = np.load(args.variance)
    stddev = [np.math.sqrt(x) for x in variance]

    def _normalize_op(features):
        channels = [tf.expand_dims((features['patch'][:,:,channel] - mean[channel]) / stddev[channel],-1) for channel in range(3)]
        features['patch'] = tf.concat(channels, 2)
        return features

    cutil.make_directory(args.model_dir)
    cutil.publish(args.model_dir)

    config_path = args.config
    config = ctfm.parse_json(config_path)

    config_datasets = config.get('datasets')
    config_model = config.get('model')


    train_fn = ctfd.construct_train_fn(config_datasets, operations=[_normalize_op])
          
    steps = int(config_datasets.get('training').get('size') / config_datasets.get('batch'))

    params_dict = {
        'config': config_model,
        'model_dir': args.model_dir
    }

    classifier = tf.estimator.Estimator(
      model_fn=my_model,
      model_dir=args.model_dir,
      params=params_dict,
      config=tf.estimator.RunConfig(model_dir=args.model_dir, save_summary_steps=1000, log_step_count_steps=1000)
    )

    if not os.path.exists(os.path.join(args.model_dir, os.path.basename(config_path))):
        shutil.copy2(config_path, args.model_dir)

    for epoch in range(config_datasets.get('training').get('epochs')):
        classifier = classifier.train(input_fn=train_fn, steps=steps)

    export_dir = os.path.join(args.model_dir, 'saved_model')
    cutil.make_directory(export_dir)
    cutil.publish(export_dir)

    # TODO: Write command to create serving input receiver fn from config.
    serving_input_receiver_fn = ctfd.construct_serving_fn(config_model['inputs'])

    classifier.export_saved_model(export_dir, serving_input_receiver_fn)
    cutil.publish(args.model_dir)
    cutil.publish(export_dir)



if __name__ == "__main__":
    main(sys.argv[1:])