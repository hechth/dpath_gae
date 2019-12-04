import json
import os
import sys

import git

git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import packages.Tensorflow as ctf
import packages.Tensorflow.Image as ctfi
import packages.Tensorflow.Model as ctfm
import packages.Utility as cutil


def main(argv):
    cfg = {
        'type':'spatial_integration',
    }

    cfg_inputs = {
        "features":[
            {
                "shape": [32,32,2],
                "key": "patch",
                "dtype": tf.float32
            }
        ],
        "labels": {
            "shape": [1],
            "dtype": tf.float32
        }            
    }

    features = {'patch': tf.ones([10,32,32,2])}
    labels = tf.ones([10])

    inputs, labels = ctfm.parse_inputs(features, labels, cfg_inputs)

    layer, variables, function, output_shape = ctfm.parse_layer(inputs['patch'].get_shape(), cfg)
    print(function(features['patch']))

if __name__ == "__main__":
    main(sys.argv[1:])
