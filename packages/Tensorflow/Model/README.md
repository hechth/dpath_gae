# Model

```python
import packages.Tensorflow.Model as ctfm
```

This package comprises all functionality related to creating models as well as most operations related to them, like saving and restoring them etc.

## Configuration

List of exposed functions:
*   [parse_json](Configuration.py): Reads a json file and replaces all string datatypes and operations with the respective python objects as specified in [Maps.py](../Maps.py)
*   [parse_component](Configuration.py): Creates component passing *inputs* through layer specified in *config* and adds entry in *outputs*.
*   [parse_inputs](Configuration.py): Parses *inputs* part of model config and returns a dict mapping to the input tensors.

## Layer

List of exposed functions:
*   [avg_unpool2d](Layer.py): Custom layer performing average unpooling for 2D data.
*   [parse_layer](Layer.py): Method that parses a layer configuration and returns the *layers, variables, function and output_shape* of the specified layer.

### Supported Layer Types

This list comprises all types of layers that can be specified in a json config file in the "layers" entry of a component.

Example Component specification:
```json
{
    "name":"classifier",
    "input": "code",
    "layers":["Insert layer specifications here!"],
    "output":"predictions_classifier"
}
```

All layers are parsed using the [parse_layer](Layer.py) method by [parse_component](Component.py) and are defined by the *layers, variables, function, output_shape* quadruple and take the *input_shape* and the dict holding the *config* as arguments.

Not all layer types populate all variables, since some layers are merely functions, not associated to a tf.layers.Layer instance, but all layers populate the *function* and *output_shape*. *function* is a callable lambda which runs the defined operation or applies the layer on the input *x*, while *output_shape* holds the shape of the function's output, which is required to build the consecutive layers.

#### Dense
Dense layers are fully connected layers having *units* hidden neurons and require an input of shape \[batch, features\].

Optional Entries:
* "trainable": They can be set as non-trainable.
* "name": A name for the layer
* "kernel_initializer": A kernel initializer.
* "bias_initializer": A bias initializer.

Example:
```json
{
    "type":"dense",
    "units":9,
    "activation":"tf.nn.sigmoid",
    "trainable": false,
    "name":"dense_layer"
}
```
#### Convolutional
A convolutional layer is specified by the number of *filters* the *kernel_size* and the *strides*. For a detailed explanation of convolutional layers and the attributes see [here](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/layers/Conv2D).

One difference from native tensorflow is that Conv2DTranspose layers are also specified using the same pattern but setting the "transpose" entry to true - which defaults to None or false.

Example:
```json
{
    "type":"conv",
    "filters": 32,
    "padding": "valid",
    "kernel_size": [2,2],
    "strides":[1,1],
    "transpose": true,
    "trainable":false
}
```

#### Flatten
A flattening layer transforms input of shape \[batch, x1, x2, ..., xk\] to \[batch, x1 * x2 * x3 * ... * xk\].

Optional Entries:
* "name": A name for the layer

Example:
```json
{
    "type":"flatten"
}
```

#### Slice
A slice layer extracts a slice from a rank 2 tensor with shape \[batch, features\]. It can be used to split a tensor into 2 sub parts by having 2 components which take the original tensor as input and add the slices as output tensors.

Example:
```json
{
    "type":"slice",
    "begin":[1],
    "size":[17],
    "name:":"z_structure"
}
```

#### Reshape
A reshape layer that reshapes the input to the specified shape without touching the batch dimension. Given an input with shape \[batch, 18\], the output of the reshape layer specified in the example has shape \[batch, 1, 1, 18\].

These reshape layers are required to transition between convolutional and dense layers, as convolutional layers require input tensors of rank 4 while other layers usually work on rank 2 tensors. It's usually used as an inverse flatten layer.

Example:
```json
{
    "type":"reshape",
    "shape":[1,1,18],
    "name":"z_reshaped"
}
```

## Saved Model

Functions to extract information from a saved model directory name.

List of exposed functions:
*   [determine_patch_size](SavedModel.py): Extracts patch size from *model_dir*.
*   [determine_batch_size](SavedModel.py): Extracts batch size from *model_dir*.
*   [determine_latent_space_size](SavedModel.py): Extracts latent space size from *model_dir*.