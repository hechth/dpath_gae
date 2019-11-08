import sys, os, json
import git
import uuid
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2

import packages.Utility as cutil


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

def max_unpool2d(x, name=None):
    """
    Taken from https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/pool.py#L66 and 
    https://github.com/tensorflow/tensorflow/issues/2169.

    Parameters
    ----------
        x: input tf.Tensor

    Returns
    -------
        out: tf.Tensor with doubled spatial dimensions and filled with max entries.
    """
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]), name=name)
        return ret


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
    activation = cutil.safe_get('activation', config)
    kernel_initializer = config.get('kernel_initializer', tf.initializers.lecun_uniform())
    bias_initializer = config.get('bias_initializer', tf.ones_initializer())
    name = cutil.safe_get('name',config)
    trainable = cutil.safe_get('trainable', config)

    layer = tf.layers.Dense(
        config['units'],
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name=name,
        trainable=trainable)
    return layer

def _parse_conv_layer(config:dict):
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
    activation = cutil.safe_get('activation', config)
    kernel_initializer = config.get('kernel_initializer', tf.initializers.lecun_uniform())
    bias_initializer = config.get('bias_initializer', tf.ones_initializer())
    trainable = cutil.safe_get('trainable', config)
    transpose = cutil.safe_get('transpose', config)
    padding = config.get('padding', 'same')

    if transpose is not None and transpose == True:
        layer = tf.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides,
            padding=padding,
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
            padding=padding,
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

def _parse_flatten_layer(config):
    name = config.get('name')
    return tf.layers.Flatten(name=name)

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

def _parse_maxunpool_layer(config:dict):
    """
    Function to create max_unpool2d layer.
    This is a custom implementation.

    Parameters
    ----------    
        dict: Optional Keys: 'name'

    Returns
    -------
        lambda x: max_unpool2d(x, name=name)
    """
    name=cutil.safe_get('name', config)
    return lambda x: max_unpool2d(x, name=name)
    
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
    function = cutil.safe_get('function', config)   
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
    'max_pool': _parse_maxpool_layer,
    'flatten': _parse_flatten_layer
}

def _parse_sampler(input_shape, config):
    """
    Function to create a sampling layer of specified dimensionality.

    Parameters
    ----------
        input_shape: shape of the input data as list of tf.TensorShape
        config: dict holding configuration of the sampling layer.

    Returns
    -------
        layer: list of tf.layers.Layer holding the mean and log_sigma_sq layer.
        variables: list of variables associated with both layers.
        function: function which produces sample from input using the mean and log_sigma_sq layers
        output_shape: shape of the output tensor
    """
    name = config.get('name')
    mean = tf.layers.Dense(
        config['dims'],
        name=name + '_mean',
        activation=None,
        kernel_initializer=tf.initializers.lecun_uniform(),
        bias_initializer=tf.ones_initializer())
    mean.build(input_shape)
    
    log_sigma_sq = tf.layers.Dense(
        config['dims'],
        name=name + '_log_sigma_sq',
        activation=None,
        kernel_initializer=tf.initializers.lecun_uniform(),
        bias_initializer=tf.ones_initializer())
    log_sigma_sq.build(input_shape)

    sample_shape = mean.compute_output_shape(input_shape)

    def _sample(mean, log_sigma_sq):
        eps_shape = tf.shape(mean)
        eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
        sample = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), eps), name=name)
        return mean, log_sigma_sq, sample

    layer = [mean, log_sigma_sq]
    variables = [mean.variables, log_sigma_sq.variables]
    function = lambda x: _sample(mean(x), log_sigma_sq(x))
    output_shape = sample_shape

    return layer, variables, function, output_shape

def _parse_reshape(input_shape, config):
    """
    Parse reshape operation specification to create reshape operation in model.

    Parameters
    ----------
        input_shape: shape of the input data as list of tf.TensorShape
        config: dict holding new shape info.

    Returns
    -------
        layer: None
        variables: None
        function: function which reshapes the input
        output_shape: shape of the output tensor
    """
    new_shape = config.get('shape')
    new_shape.insert(0, -1)
    name = config.get('name')

    function = lambda x: tf.reshape(x, new_shape)
    return None, None, function, new_shape

def _parse_slice(input_shape:list, config:dict):
    """
    Function which parses slice extraction function to use inside a model.

    Parameters
    ----------
        input_shape: shape of input data
        config: dict holding 'begin':list and 'size':list entries

    Returns
    -------
        layers: None
        variables: None
        function: lambda x: tf.slice(x, begin, size, name=config.get('name'))
        output_shape: tf.TensorShape holding output tensor shape
    """
    begin = config.get('begin')
    begin.insert(0, 0)

    size = config.get('size')
    size.insert(0, -1)

    function = lambda x: tf.slice(x, begin, size, name=config.get('name'))
    output_shape = function(tf.placeholder(tf.float32, shape=input_shape)).get_shape()
    return None, None, function, output_shape

def _parse_resnet_v2_block(shape:list, config:dict):
    """
    Function to parse resnet_v2_block which creates a preactivation bottleneck unit.
    Documentation can be found here: https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py

    Parameters
    ----------
        shape: list holding the shape of expected input.
        config: dict describing the block, holding 'base_depth', 'num_units' and 'stride' keys. Optional: 'scope'

    Returns
    -------
        layers: None
        variables: All trainable variables associated with the scope.
        function: lambda x: block[1](x, args.get('depth'), args.get('depth_bottleneck'), args.get('stride'))
        output_shape: shape of the block output.
    """
    scope_name = config.get('scope', str(uuid.uuid4()))
    scope = tf.VariableScope(scope_name)
    base_depth = config.get('base_depth')
    num_units = config.get('num_units')
    stride = config.get('stride')

    block = resnet_v2.resnet_v2_block(scope, base_depth, num_units, stride)
    args = block.args[0]

    layers = None
    variables = scope.trainable_variables()
    function = lambda x: block[1](x, args.get('depth'), args.get('depth_bottleneck'), args.get('stride'))
    output_shape = function(tf.placeholder(tf.float32, shape=shape)).get_shape()

    return layers, variables, function, output_shape

def parse_layer(input_shape:list, config:dict):
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
        output_shape = function(tf.placeholder(tf.float32, shape=input_shape)).get_shape()
    elif config['type'] == 'max_unpool':
        layer = None
        variables = None
        function = _parse_maxunpool_layer(config)
        output_shape = function(tf.placeholder(tf.float32, shape=input_shape)).get_shape()
    elif config['type'] == 'sampler':
        return _parse_sampler(input_shape, config)
    elif config['type'] == 'reshape':
        return _parse_reshape(input_shape, config)
    elif config['type'] == 'slice':
        return _parse_slice(input_shape, config)
    elif config['type'] == 'resnet_v2_block':
        return _parse_resnet_v2_block(input_shape, config)   
    else:
        layer = _layer_map[config['type']](config)
        layer.build(input_shape)
        variables = layer.variables
        output_shape = layer.compute_output_shape(input_shape)
        function = layer.apply
    return layer, variables, function, output_shape


def parse_feature(config:dict)->tf.feature_column.numeric_column:
    """
    Private function to parse a single feature into a feature columns.

    Parameters
    ----------
    config: dict describing a single input feature. Entries: 'shape', 'key', 'dtype'

    Returns
    -------
    tf.feature_column.numeric_column(key=key, shape=input_shape, dtype=dtype)
    """
    input_shape= config['shape']
    key = config['key']
    dtype = config['dtype']
    return tf.feature_column.numeric_column(key=key, shape=input_shape, dtype=dtype)

def parse_feature_columns(features:list)->dict:
    """
    Function which parses the list of feature configs and returns the required input feature columns as dict mapped by their key.

    Parameters
    ----------
    features: array of dicts describing features parsed from json holding the model configuration.

    Returns
    -------
    feature_columns: dict of tf.feature_column.numeric_column objects mapepd by their key.
    """
    return [{feature['key'] : parse_feature(feature)} for feature in features]
    