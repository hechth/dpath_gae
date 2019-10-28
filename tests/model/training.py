import sys, os, json, argparse
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm

def my_model(features, labels, mode, params, config):
    cfg = params['config']
    inputs, encoder_layers, encoder_vars, encode = ctfm.parse_component(features, cfg['encoder'])
    outputs = encode(inputs)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=outputs)

    train_op = optimizer.minimize(loss)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
def train_func():
    x_train = tf.data.experimental.RandomDataset()
    y_train = tf.data.experimental.RandomDataset()

    dataset = tf.data.Dataset.zip((x_train, y_train)).map(lambda x,y: ({'val': tf.cast(x, tf.float32)},y)).take(10000).batch(1)
    return dataset


def main(argv):
    parser = argparse.ArgumentParser(description='Compute latent code for image patch by model inference.')
    parser.add_argument('export_dir',type=str,help='Path to saved model to use for inference.')
    parser.add_argument('classes',type=int,help='Number of classes for the trained model. Must match output of last layer.')

    args = parser.parse_args()

    config_filename = os.path.join(git_root, 'tests','model','example_config.json')
    with open(config_filename,'r') as json_file:
        cfg = json.load(json_file)

    feature_columns = [ctfm.parse_feature(feature) for feature in cfg['features']]
    model_dir = args.export_dir

    params_dict = {
        'config': cfg,
        'feature_columns': feature_columns,
        'model_dir': model_dir,
        'n_classes': args.classes
    }          

    classifier = tf.estimator.Estimator(
      model_fn=my_model,
      model_dir=model_dir,
      params=params_dict
    )

    classifier.train(input_fn=train_func, steps=10000)



if __name__ == "__main__":
    main(sys.argv[1:])