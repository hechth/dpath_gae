import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm



def main(argv):
    cfg_inputs = {
        "features":[
            {
                "shape": [32,32,3],
                "key": "val",
                "dtype": tf.float32
            }
        ],
        "labels": {
            "shape": [1],
            "dtype": tf.float32
        }            
    }

    cfg_resnet = {
        "type":"resnet_v2_block",
        "stride":1,
        "base_depth":3,
        "num_units": 1
    }

    features = {'val': tf.ones([10,32,32,3])}
    labels = tf.ones([1])

    inputs, labels = ctfm.parse_inputs(features, labels, cfg_inputs)

    resnet = ctfm.parse_layer(inputs['val'].get_shape().as_list(),cfg_resnet)
    print(resnet[2](features['val']))
    print(resnet[1])

    
   
    
if __name__ == "__main__":
    main(sys.argv[1:])