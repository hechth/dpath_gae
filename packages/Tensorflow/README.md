# Tensorflow

The packages organized in this module rely primarily on tensorflow and numpy functionality.

List of packages:
*   [Image](Image)
*   [Dataset](Dataset)
*   [Model](Model)
*   [Tensorboard](Tensorboard)

## Feature

Functions to encode datatypes into *tf.train.Feature*. This is required to serialize data into *tfrecords* files.

Supported Datatypes:
* strings
* float
* int64

## Maps

File containing dictionaries which map from strings to tensorflow objects, like operations and datatypes.

