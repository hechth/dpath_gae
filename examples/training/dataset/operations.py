import tensorflow as tf

def cast_to_float16(x):
    x['val'] = tf.cast(x['val'],dtype=tf.float16)
    return x