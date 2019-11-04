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
        components[comp['name']] = comp

    encoder = ctfm.parse_component(tensors, components['encoder'], tensors)
    sampler = ctfm.parse_component(tensors, components['sampler'], tensors)
    classifier = ctfm.parse_component(tensors, components['classifier'], tensors)
    discriminator = ctfm.parse_component(tensors, components['discriminator'], tensors)
    decoder = ctfm.parse_component(tensors, components['decoder'], tensors)


    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    
     # Losses for discriminator and classifier
    discriminator_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tensors['predictions_discriminator'])
    classifier_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tensors['predictions_classifier'])

    # get bVAE losses.
    latent_loss = ctfm.latent_loss(sampler[0][0][0](tensors['encoded_patch']), tf.exp(sampler[0][0][1](tensors['encoded_patch'])))
    reconstr_loss = tf.losses.absolute_difference(tensors['patch'], tensors['logits'])

    # Combine reconstruction loss, latent loss, prediction loss and negative discriminator loss
    loss = reconstr_loss + cfg['parameters']['beta'] * latent_loss + cfg['parameters']['alpha'] * classifier_loss - cfg['parameters']['delta'] * discriminator_loss

    # Create training op
    train_op_encoder = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    train_op_discriminator = optimizer.minimize(discriminator_loss, var_list=[discriminator[1]])
    train_op_classifier = optimizer.minimize(classifier_loss, var_list=[classifier[1]])
    train_op_decoder = optimizer.minimize(reconstr_loss, var_list=[decoder[1]])

    train_op = tf.group([train_op_encoder,train_op_discriminator,train_op_classifier,train_op_decoder])


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'loss' : loss})

    assert mode == tf.estimator.ModeKeys.TRAIN   
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(argv):
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('model_dir',type=str,help='Path to saved model to use for inference.')
    args = parser.parse_args()

    config = ctfm.parse_json(os.path.join(git_root,'examples','models','gae','configuration.json'))

    train_fn = ctfd.construct_train_fn(config['datasets'])
    steps = int(config['datasets']['training']['size'] / config['datasets']['batch'])

    params_dict = {
        'config': config['model'],
        'model_dir': args.model_dir
    }

    classifier = tf.estimator.Estimator(
      model_fn=my_model,
      model_dir=args.model_dir,
      params=params_dict,
      config=tf.estimator.RunConfig(model_dir=args.model_dir, save_summary_steps=100, log_step_count_steps=100)
    )

    classifier = classifier.train(input_fn=train_fn, steps=steps)

if __name__ == "__main__":
    main(sys.argv[1:])