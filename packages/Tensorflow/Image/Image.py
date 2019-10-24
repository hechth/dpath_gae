import tensorflow as tf
import numpy as np
import sys
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

from packages.Utility import get_extension

supported_extensions = ['png','bmp','jpg','jpeg','gif']

def load(filename, dtype=tf.float32, width=None, height=None, channels=3, name=''):
    """
    Function to load a tensorflow image.
    Supply information about height and width if known to make this known to the tensorflow graph.
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
    if width is not None and height is not None:
        image.set_shape([width, height, channels])
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

def extract_patches(
    image,
    patch_size,
    padding="VALID",
    rates=[1,1,1,1],
    strides=None,
    ksizes=None):

    """
    Function which splits image of shape [width, height, 3] into patches with more convenient ordering than the built in tensorflow function
    tf.image.extract_image_patches (https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/image/extract_image_patches)

    Returns
    -------
    patches: image patches of shape [num_patches, patch_size, patch_size, 3]
    """

    # Set strides and ksizes if not passed as arguments
    if strides is None:
        strides=[1, patch_size, patch_size, 1]

    if ksizes is None:
        ksizes=[1, patch_size, patch_size, 1]
    
    # Split image into rgb channels
    r,g,b = tf.split(image, 3, 2)       

    # Extract patches from the image using tensorflow function
    patches = tf.image.extract_image_patches(
      [r,g,b],
      ksizes,
      strides,
      rates,
      padding
    )

    patches = tf.expand_dims(patches, 0)
    num_patches = [int(image.shape.as_list()[0] / patch_size), int(image.shape.as_list()[1] / patch_size)]

    # Change grid layout of [1, channels, x, y, pixels] to [patches, channels, 1, 1, pixels]
    patches = tf.space_to_batch_nd(
      patches,
      [1, num_patches[0], num_patches[1]],
      [[0, 0], [0, 0], [0, 0]]
    )

    # Squeeze [patches, channels, 1, 1, pixels] to [patches, channels, pixels]
    patches = tf.squeeze(patches)

    # Change shape to channels last: [patches, channels, pixels] to [patches, pixels, channels]
    patches = tf.transpose(patches, perm=[0,2,1])

    # Reshape from [patches, pixels, channels] to [patches, rows, cols, channels]
    patches = tf.reshape(
      patches,
      [patches.shape[0], patch_size, patch_size, patches.shape[2]]
    )

    return patches
    
def subsample(image, factor, method=tf.image.ResizeMethod.BILINEAR):
    """
    Function shat subsamples image or batch of images by factor given by factor.

    Returns:
    subsampled: image or image batch of shape [ (batch_size,) image.width / factor, image.height / factor, image.channels]
    """
    image_shape = image.get_shape().as_list()

    new_width = 0
    new_height = 0

    # If batch of images is passed, add factor of 1
    if len(image_shape) == 4:
        new_width = image_shape[1] / factor
        new_height = image_shape[2] / factor
    else:
        new_width = image_shape[0] / factor
        new_height = image_shape[1] / factor
    
    return tf.image.resize_images(image, [int(new_width), int(new_height)],method=method)

def is_image(filename)->bool:
    """
    Function which checks if given filename belongs to an image file readable by the load function.

    Returns
    -------
    True if the file is a readable image, otherwise False.
    """
    return get_extension(filename) in supported_extensions