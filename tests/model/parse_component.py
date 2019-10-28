import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm


def main(argv):
    cfg = {
        'encoder': {
            'input': {
                'shape': [1],
                'key': 'val',
                'dtype': 'tf.float32'
            },
            'layers': [
                {
                    'type':'dense',
                    'units': 10,
                },
                {
                    'type':'activation',
                    'function': 'relu'
                }
            ]
        }
    }


    features = {'val': tf.ones([1])}
    inputs, encoder_layers, encoder_vars, encode = ctfm.parse_component(features, cfg['encoder']) 
    print(encode(features['val']))

   
    
if __name__ == "__main__":
    main(sys.argv[1:])