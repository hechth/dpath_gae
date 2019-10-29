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

## Saved Model

Functions to extract information from a saved model directory name.

List of exposed functions:
*   [determine_patch_size](SavedModel.py): Extracts patch size from *model_dir*.
*   [determine_batch_size](SavedModel.py): Extracts batch size from *model_dir*.
*   [determine_latent_space_size](SavedModel.py): Extracts latent space size from *model_dir*.