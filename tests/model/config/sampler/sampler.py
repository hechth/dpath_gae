import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf

import packages.Utility as cutil
import packages.Tensorflow.Model as ctfm

def main(argv):
    cfg = {
        'type':'sampler',
        'dims': 18,
        'name': 'z'
    }

    config = {
        "model": {
            "inputs":{
                "features":[
                    {
                        "shape": [1],
                        "key": "val",
                        "dtype": tf.float32
                    }
                ],
                "labels": {
                    "shape": [1],
                    "dtype": tf.float32
                }
            },
            "components": [
                {
                    "name":"example",
                    "input":"val",
                    "layers": [
                        {
                            "type":'dense',
                            'units':10
                        },
                        {
                            'type': 'sampler',
                            'dims':10,
                            'name': 'z'
                        }
                    ],
                    "output":'sample'
                }
            ]
        }
    }

    model = config['model']

    features = {'val': tf.ones([10,1])}
    labels = tf.ones([1])

    inputs, labels = ctfm.parse_inputs(features, labels, model['inputs'])

    layer, variables, function, output_shape = ctfm.parse_layer(inputs['val'].get_shape(), cfg)
    print(function(features['val']))


    example = ctfm.parse_component(inputs,model['components'][0], inputs)

    print(example[2](features['val']))


    
   
    
if __name__ == "__main__":
    main(sys.argv[1:])