import sys, os, json, collections
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np

import tensorflow as tf

from packages.Tensorflow import tf_datatypes
from packages.Tensorflow import tf_operations
import packages.Utility as cutil

from .Layer import parse_feature
from .Layer import parse_layer

ops_keywords = ['function', 'activation', 'kernel_initializer','bias_initializer']

def parse_json(filename: str) -> dict:
    """
    Function that parses json config file and replaces all datatype and functions specified by strings with the real objects.
    """
    with open(filename,'r') as json_file:
        cfg = json.load(json_file)
    
    # Local function which browses the config
    def browse_dict(cfg):
        # Iterate over key, value pairs
        for key, value in cfg.items():

            # If the value is a dict, browse this.
            if type(value) == dict:
                cfg[key] = browse_dict(value)
            # If value is a list, we assume that its a list of dictionaries and parse each.
            elif type(value) == list and type(value[0])==dict:
                cfg[key] = [browse_dict(entry) for entry in value]
            # Else we have a plain value which might have to be replaced
            else:
                if key == 'dtype':
                    cfg[key] = tf_datatypes[value]
                elif key in ops_keywords:
                    cfg[key] = tf_operations[value]
        return cfg
    
    cfg = browse_dict(cfg)

    return cfg
       
def parse_inputs(features:dict, labels:tf.Tensor, config: dict) -> [dict, tf.Tensor]:
    """
    Parses input description from config and returns reshaped input tensors with their key as dict and the correctly shaped labels.

    Parameters
    ----------
        features: dict mapping the input features with their 'key' to the tf.Tensor values as passed to model function of estimator.
        labels: tf.Tensor holding the labels.
        config: dict holding 'features' dict and 'labels' dict, each holding the 'key', 'dtype' and 'shape' info of the feature or label.

    Returns
    -------
        inputs: dict mapping all features with their 'key' to correctly shaped and batched tf.Tensor values.
        labels: tf.Tensor holding labels with correct shape.
    """
    inputs = {}
    cfg_features = config['features']
    for cfg_input in cfg_features:

        input_shape = np.copy(cfg_input['shape']).tolist()
        key = cfg_input['key']
        dtype = cfg_input['dtype']

        feature_column = parse_feature(cfg_input)
        
        input_shape.insert(0,-1)

        input_layer = tf.feature_column.input_layer(features, feature_column)
        inputs[key] = tf.reshape(input_layer, input_shape, name=key)

    if labels is not None:
        labels_shape = np.copy(config['labels']['shape']).tolist()
        labels_shape.insert(0,-1)
        labels = tf.reshape(labels, labels_shape)
    return inputs, labels


def parse_component(inputs:dict, config:dict, outputs: dict):
    """
    Function to parse a dict holding the description for a component.
    A component is defined by an input and a number of layers.

    This function is supposed to be called in the model function of a tf.Estimator and eases model creation.

    The input description is used to build the feature_column and input layer.
    The input is then extended with batch dimension.

    Parameters
    ----------
    inputs: dict mapping from string to input tensor.

    config: dict holding keys 'input' for input speciication and 'layers', the list of layers after the input.

    outputs: dict to which to append this config output

    Returns
    -------
    layers: list(tf.layers.Layer), all layers added for this component.
            Layers not inheriting from tf.layers.Layer are passed as functions.

    variables: list(tf.Variable), list of all variables associated with the layers of this component.

    function: callable which performs a forward pass of features through the network.
    """
   
    layers = list()
    variables = list()    
    funcs = list()

    # Get input shape for following layers
    shape = None
    if type(config['input']) != list:
        shape = inputs[config['input']].get_shape()
    else:
        shape = [inputs[key].get_shape() for key in config['input']]
    
    # Parse each layer specified in layers and append them to collections.
    for desc in config['layers']:
        layer,variable,function, shape = parse_layer(shape, desc)
        if layer is not None:
            layers.append(layer)
        if variable is not None:
            variables.append(variable)
        funcs.append(function)

    function = cutil.concatenate_functions(funcs)
    output_tensors = function(inputs[config['input']])

    if isinstance(config['output'], collections.Iterable) and isinstance(output_tensors, tuple):
        for key, value in zip(config['output'], output_tensors):
            if isinstance(value, tf.Tensor):
                outputs.update({key : tf.identity(value, name=key)})
            else:
                 outputs.update({key : value})
    else:
        outputs.update({config['output'] : tf.identity(output_tensors, name=config['output'])})   
    return layers, variables, function