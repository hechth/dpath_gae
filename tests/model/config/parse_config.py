import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm


def main(argv):
    config_filename = os.path.join(git_root, 'tests','model','example_config.json')
    cfg = ctfm.parse_json(config_filename)

    features = {'val': tf.ones([10,1])}
    labels = tf.ones([1])

    inputs, labels = ctfm.parse_inputs(features,labels, cfg['inputs'])

    outputs = {}

    encoder_layers, encoder_vars, encode = ctfm.parse_component(inputs, cfg['encoder'], outputs) 
    print(encode(features['val']))
    print(outputs['logits'])
   
    
if __name__ == "__main__":
    main(sys.argv[1:])