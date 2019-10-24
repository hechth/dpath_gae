import tensorflow as tf
import numpy as np

def load(filename, dtype=tf.float32, channels=3, name=''):
    """
    Function to load a tensorflow image.
    Supports BMP, GIF, JPEG, or PNG.

    Returns
    -------
    image:  Tensor with type dtype and shape [height, width, channels] for BMP, JPEG, and PNG images and
            shape [num_frames, height, width, 3] for GIF images.

    """
    # Read raw image data
    raw_image = tf.read_file(filename)

    # Decode image
    image = tf.image.decode_image(raw_image, dtype=dtype, channels=channels, name=name)   
    return image

def rescale(image, new_min, new_max):
    """
    Rescale intensity values of an image to a new range specified by new_min and new_max.
    Expects images to be in an integer format for floating point ranges.

    Returns
    -------
    image:  Image with intensity values between new_min and new_max and same shape.
    """
    return np.interp(image, (image.min(), image.max()), (new_min, new_max))