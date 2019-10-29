import tensorflow as tf


def string_feature(string):
    """
    Function which returns tf.train.Feature for given string.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[string.encode()]))

def float_feature(value):
    """
    Function which returns tf.train.Feature for given float.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))