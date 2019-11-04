import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm



def main(argv):
    cfg0 = {
        'type':'slice',
        'begin': [0],
        'size': [6]
    }

    cfg1 = {
        'type':'slice',
        'begin': [5],
        'size': [12]
    }

    cfg_inputs = {
        "features":[
            {
                "shape": [18],
                "key": "val",
                "dtype": tf.float32
            }
        ],
        "labels": {
            "shape": [1],
            "dtype": tf.float32
        }            
    }

    features = {'val': tf.ones([10,18])}
    labels = tf.ones([1])

    inputs, labels = ctfm.parse_inputs(features, labels, cfg_inputs)

    slice0 = _parse_slice(inputs['val'].get_shape(), cfg0)
    slice1 = _parse_slice(inputs['val'].get_shape(), cfg1)
    
    print(slice0[2](features['val']))   
    print(slice1[2](features['val']))   
   
    
if __name__ == "__main__":
    main(sys.argv[1:])