import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf

import packages.Utility as cutil
from packages.Tensorflow import tf_datatypes
from packages.Tensorflow import tf_operations

def avg_unpool2d(x, factor, name=None):
    '''
    Performs "average un-pooling", i.e. nearest neighbor upsampling,
    without the faulty `tf.image.resize_nearest_neighbor` op.
    ''' 
    x = tf.transpose(x, [1, 2, 3, 0])
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [factor**2, 1, 1, 1, 1])
    x = tf.batch_to_space_nd(x, [factor, factor], [[0, 0], [0, 0]])
    x = tf.transpose(x[0], [3, 0, 1, 2], name=name)
    return x


def _parse_dense_layer(config: dict)-> tf.layers.Dense:
    """
    Function to build dense layer with specific config.

    Parameters
    ----------
    config: dict holding 'units' key.
    
    Optional Keys: 'activation','kernel_initializer','name', 'trainable

    Returns
    -------
    layer: tf.layers.Dense with specified configuration.
    """
    activation = cutil.safe_get(cutil.safe_get('activation', config), tf_operations)
    kernel_initializer = cutil.safe_get(cutil.safe_get('kernel_initializer', config), tf_operations)
    name = cutil.safe_get('name',config)
    trainable = cutil.safe_get('trainable', config)

    layer = tf.layers.Dense(
        config['units'],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name,
        trainable=trainable)
    return layer

def _parse_conv_layer(config):
    """
    Function to build convolutional 2d layer with specific config.
    Pass 'transpose': True in config to create deconvolution layer.

    Parameters
    ----------
    config: dict holding 'filters', 'strides' and 'kernel_size' keys.

    Optional Keys: 'activation','kernel_initializer','name','bias_initializer', 'trainable', 'transpose'

    Returns
    -------
    layer: tf.layers.Conv2D or tf.layers.Conv2DTranspose with specified configuration.
    """
    filters = config['filters']
    strides = cutil.safe_get('strides',config)
    kernel_size = cutil.safe_get('kernel_size',config)
    name = cutil.safe_get('name',config)
    activation = cutil.safe_get(cutil.safe_get('activation', config), tf_operations)
    kernel_initializer = cutil.safe_get(cutil.safe_get('kernel_initializer', config), tf_operations)
    bias_initializer = cutil.safe_get(cutil.safe_get('bias_initializer', config), tf_operations)
    trainable = cutil.safe_get('trainable', config)
    transpose = cutil.safe_get('transpose', config)

    if transpose is not None and transpose == True:
        layer = tf.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides,
            padding='same',
            name=name,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            activation=activation,
            trainable=trainable)
    else:
        layer = tf.layers.Conv2D(
            filters,
            kernel_size,
            strides,
            padding='same',
            name=name,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            activation=activation,
            trainable=trainable)
    return layer

def _parse_maxpool_layer(config:dict)->tf.layers.MaxPooling2D:
    """
    Function to build MaxPooling2D layer with specific config.

    Parameters
    ----------
    config: dict holding 'pool_size' and 'strides' key.
    
    Optional Keys: 'name'

    Returns
    -------
    layer: tf.layers.MaxPooling2D with specified configuration.
    """
    # Retrieve attributes from config
    pool_size = cutil.safe_get('pool_size',config)
    strides = cutil.safe_get('strides',config)
    name = cutil.safe_get('name',config)

    return tf.layers.MaxPooling2D(pool_size, strides, name=name)

def _parse_avgunpool_layer(config:dict):
    """
    Function to create and avg unpooling layer with given factor.
    This is a custom implementation.

    Parameters
    ----------
    config: dict holding key 'factor'.
    
    Optional Keys: 'name'

    Returns
    -------
    lambda x: avg_unpool2d(x, factor, name=name) callable which performs the desired operation.
    """
    name=cutil.safe_get('name', config)
    factor=cutil.safe_get('factor', config)
    return lambda x: avg_unpool2d(x, factor, name=name)
    
def _parse_activation(config:dict):
    """
    Parse activation function and return callable with specified name.

    Parameters
    ----------
    config: dict with key 'function'.
    Optional Keys: 'name'

    Returns
    -------
    lambda x: function(x, name=name)
    """
    name=cutil.safe_get('name', config)
    function = cutil.safe_get(cutil.safe_get('function', config), tf_operations)

    return lambda x: function(x, name=name)


def _parse_batchnorm_layer(config:dict)->tf.layers.BatchNormalization:
    """
    Function to create batch normalization layer on specified axis.

    Parameters
    ----------
    config: dict with key 'axis'.

    Optional Keys: 'name'

    Returns
    -------
    layer: tf.layers.BatchNormalization(axis=axis,name=name)
    """
    axis=cutil.safe_get('axis', config)
    name=cutil.safe_get('name', config)
    return tf.layers.BatchNormalization(axis=axis,name=name)

_layer_map = {
    'dense': _parse_dense_layer,
    'conv': _parse_conv_layer,
    'batch_norm': _parse_batchnorm_layer,
    'max_pool': _parse_maxpool_layer
}

def _parse_layer(input_shape:list, config:dict):
    """
    Function which parses a layer or activation or avg_unpool operation specified in a layer config.
    activations and avg_unpool ops have to be treated differently since they don't create a tf.layers.Layer object.

    Parameters
    ----------
    input_shape: shape information about input tensor

    config:dict holding layer configuration for key 'type'. Other keys depend on layer type.

    Returns
    -------
    layer: tf.layers.Layer object or None
    variables: tf.Variable object or None
    function: lambda <x> callable which applies the layer or operation
    output_shape: list holding information of shape after transformation
    """
    if config['type'] == 'activation':
        layer = None
        variables = None
        function = _parse_activation(config)
        output_shape = input_shape
    elif config['type'] == 'avg_unpool':
        layer = None
        variables = None
        function = _parse_avgunpool_layer(config)
        output_shape = function(tf.ones(input_shape)).get_shape()
    else:
        layer = _layer_map[config['type']](config)
        layer.build(input_shape)
        variables = layer.variables
        output_shape = layer.compute_output_shape(input_shape)
        function = layer.apply
    return layer, variables, function, output_shape



def parse_component(features:dict, config:dict):
    """
    Function to parse a dict holding the description for a component.
    A component is defined by an input and a number of layers.

    This function is supposed to be called in the model function of a tf.Estimator and eases model creation.

    The input description is used to build the feature_column and input layer.
    The input is then extended with batch dimension.

    Parameters
    ----------
    features: dict variable which is passed to model_function at estimator creation.
    config: dict holding keys 'input' for input speciication and 'layers', the list of layers after the input.

    Returns
    -------
    inputs: tf.Tensor holding the inputs parsed from the feature column.

    layers: list(tf.layers.Layer), all layers added for this component.
            Layers not inheriting from tf.layers.Layer are passed as functions.

    variables: list(tf.Variable), list of all variables associated with the layers of this component.

    function: callable which performs a forward pass of features through the network.    
    """
    layers = list()
    funcs = list()
    variables = list()

    # Shortcut to input config.
    cfg_input = config['input']

    # Parse input config and retrieve parameters
    input_shape= cfg_input['shape']
    key = cfg_input['key']
    dtype = tf_datatypes[cfg_input['dtype']]
    inputs_name = cutil.safe_get('name', cfg_input)

    # Create input feature layer
    feature_column = tf.feature_column.numeric_column(key=key, shape=input_shape, dtype=dtype)
    inputs = tf.Variable(tf.zeros(input_shape),dtype=dtype, expected_shape=input_shape,name=inputs_name)
    input_layer = tf.feature_column.input_layer(features, feature_column, cols_to_vars={key:inputs})

    # Add batch dimension
    input_shape.insert(0,-1)
    input_reshaped = tf.reshape(input_layer, input_shape)

    # Append reshape operation to functions to reshape input with batch dimension.
    funcs.append(lambda x: tf.reshape(x, input_shape))
    layers.append(input_reshaped)

    # Get input shape for following layers
    shape = input_reshaped.get_shape()

    # Parse each layer specified in layers and append them to collections.
    for desc in config['layers']:
        layer,variable,function, shape = _parse_layer(shape, desc)
        layers.append(layer)
        funcs.append(function)
        variables.append(variable)
    
    return inputs, layers, variables, cutil.concatenate_functions(funcs)