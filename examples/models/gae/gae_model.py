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
import packages.Tensorflow.Image as ctfi

def my_model(features, labels, mode, params, config):
    cfg = params['config']
    cfg_inputs = cfg.get('inputs')
    cfg_labels = cfg_inputs.get('labels')
    cfg_embeddings = cfg.get('embeddings')
    cfg_embeddings.update({'model_dir': params['model_dir']})

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

    #encoder = ctfm.parse_component(tensors, components['encoder'], tensors)
    #sampler = ctfm.parse_component(tensors, components['sampler'], tensors)
    #classifier = ctfm.parse_component(tensors, components['classifier'], tensors)
    #discriminator = ctfm.parse_component(tensors, components['discriminator'], tensors)
    #decoder = ctfm.parse_component(tensors, components['decoder'], tensors)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'code': tensors['code'],
            'logits': tensors['logits']
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    # --------------------------------------------------------
    # Losses
    # --------------------------------------------------------
        
    #latent_loss = ctfm.latent_loss(tensors['mean'], tensors['log_sigma_sq'])
    
    mvn_uniform = tf.contrib.distributions.MultivariateNormalDiag(
    loc=tf.zeros([tf.shape(tensors['code'])[0], cfg['parameters'].get('latent_space_size')]),
    scale_diag=tf.ones([tf.shape(tensors['code'])[0], cfg['parameters'].get('latent_space_size')]))

    latent_loss = tf.reduce_mean(tensors['distribution'].kl_divergence(mvn_uniform))
    
    #latent_loss = tf.reduce_mean(ctfm.multivariate_latent_loss(tensors['mean'], tensors['covariance']))
    #reconstr_loss = tf.losses.mean_squared_error(tensors['patch'], tensors['logits'])
    reconstr_squared_diff = tf.math.squared_difference(tensors['patch'], tensors['logits'])
    batch_reconstruction_loss = tf.reduce_sum(reconstr_squared_diff,axis=[1,2,3])
    reconstr_loss = tf.reduce_mean(batch_reconstruction_loss)

    discriminator_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tensors['predictions_discriminator'])
    classifier_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tensors['predictions_classifier'])

    # Combine reconstruction loss, latent loss, prediction loss and negative discriminator loss
    loss = reconstr_loss + cfg['parameters'].get('beta') * latent_loss + cfg['parameters'].get('alpha') * classifier_loss - cfg['parameters'].get('delta') * discriminator_loss


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    train_op_encoder = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    train_op_discriminator = optimizer.minimize(discriminator_loss, var_list=[components['discriminator'][1]])
    train_op_classifier = optimizer.minimize(classifier_loss, var_list=[components['classifier'][1]])

    #decoder_variables = components['decoder_stain'][1] + components['decoder_structure'][1] + components['merger'][1]
    decoder_variables = components['decoder'][1]
    train_op_decoder = optimizer.minimize(reconstr_loss, var_list=decoder_variables)

    train_op = tf.group([train_op_encoder,train_op_discriminator,train_op_classifier,train_op_decoder])


    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------
    
    # Predictions from classifier and discriminator
    predicted_classes_classifier = tf.argmax(tensors['predictions_classifier'], 1)
    predicted_classes_discriminator = tf.argmax(tensors['predictions_discriminator'], 1)

    classifier_accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes_classifier, name='acc_op_classifier')
    discriminator_accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes_discriminator, name='acc_op_discriminator')

    classifier_confusion = ctfb.confusion_metric(labels, predicted_classes_classifier, cfg_labels.get('num_classes'),name='classifier')
    discriminator_confusion = ctfb.confusion_metric(labels, predicted_classes_discriminator, cfg_labels.get('num_classes'),name='discriminator')

    metrics = {
        'classifier_confusion': classifier_confusion,
        'classifier_accuracy': classifier_accuracy,
        'discriminator_confusion': discriminator_confusion,
        'discriminator_accuracy': discriminator_accuracy
    }

    tf.summary.scalar('classifier_accuracy', classifier_accuracy[1])
    ctfb.plot_confusion_matrix(classifier_confusion[1],cfg_labels.get('names'),tensor_name='confusion_matrix_classifier', normalize=True)
    
    tf.summary.scalar('discriminator_accuracy', discriminator_accuracy[1])
    ctfb.plot_confusion_matrix(discriminator_confusion[1], cfg_labels.get('names'),tensor_name='confusion_matrix_discriminator', normalize=True)

    # Losses scalar summaries
    tf.summary.scalar('reconstr_loss', reconstr_loss)
    tf.summary.scalar('latent_loss', latent_loss)
    tf.summary.scalar('classifier_loss', classifier_loss)
    tf.summary.scalar('discriminator_loss', discriminator_loss)

    # Image summaries of patch and reconstruction
    tf.summary.image('images', tensors['patch'], 1)
    tf.summary.image('reconstructions', tensors['logits'], 1)

    embedding_hook = ctfb.EmbeddingSaverHook(tf.get_default_graph(), cfg_embeddings, tensors['code'].name, tensors['code'], tensors['patch'].name, labels.name, cfg_labels.get('names'))



    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN   
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[embedding_hook])






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

    def _subsampling_op(features):
        features['patch'] = ctfi.subsample(features['patch'], 2)
        return features

    cutil.make_directory(args.model_dir)
    cutil.publish(args.model_dir)

    config_path = args.config
    config = ctfm.parse_json(config_path)

    config_datasets = config.get('datasets')
    config_model = config.get('model')


    train_fn = ctfd.construct_train_fn(config_datasets, operations=[_normalize_op])
    #def train_fn():
    #    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(256,32,32,3))
    #    dataset = dataset.map(lambda x : ({"patch": x}, 0)).batch(256).repeat()
    #    return dataset
        
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