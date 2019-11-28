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

## Saved Model

Functions to extract information from a saved model directory name.

List of exposed functions:
*   [determine_patch_size](SavedModel.py): Extracts patch size from *model_dir*.
*   [determine_batch_size](SavedModel.py): Extracts batch size from *model_dir*.
*   [determine_latent_space_size](SavedModel.py): Extracts latent space size from *model_dir*.


from .Layer import avg_unpool2d
from .Layer import parse_layer

from .Configuration import parse_component
from .Configuration import parse_json
from .Configuration import parse_inputs

from .Losses import latent_loss
from .Losses import multivariate_latent_loss