import tensorflow as tf

tf_datatypes = {
    'tf.float32':tf.float32,
    'tf.float16':tf.float16,
    'tf.int64': tf.int64
}

tf_operations = {
    'tf.nn.relu': tf.nn.relu,
    'tf.nn.sigmoid':tf.nn.sigmoid,
    'tf.nn.softmax':tf.nn.softmax,
    'tf.initializers.lecun_uniform': tf.initializers.lecun_uniform(),
    'tf.initializers.ones': tf.ones_initializer(),
    'tf.initializers.zeros': tf.zeros_initializer(),
    'None': None
}

tf_optimizers = {
    
}