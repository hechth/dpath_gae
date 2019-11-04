import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm

def main(argv):
    cfg = {
        'type':'reshape',
        'shape': [10,10]
    }

    cfg_inputs = {
        "features":[
            {
                "shape": [100],
                "key": "val",
                "dtype": tf.float32
            }
        ],
        "labels": {
            "shape": [1],
            "dtype": tf.float32
        }            
    }

    features = {'val': tf.ones([10,100])}
    labels = tf.ones([1])

    inputs, labels = ctfm.parse_inputs(features, labels, cfg_inputs)

    layer, variables, function, output_shape = ctfm.parse_layer(inputs['val'].get_shape(), cfg)
    print(function(features['val']))   
   
    
if __name__ == "__main__":
    main(sys.argv[1:])