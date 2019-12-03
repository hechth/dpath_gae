# Examples

## Dataset

The dataset folder contains the following examples:
*   [How to create a Dataset?](dataset/create_dataset.py): Script shows how to use the functionality provided by the [Dataset](../packages/Tensorflow/Dataset) package to write a dataset from numpy data. Note that to run the [gae_model](models/gae/gae_model.py), a dataset created using the [tools](../tools/dataset) scripts is required.
*   [How to Create a Model using a Dataset?](dataset/model_with_dataset.py): Example of simple model loading a dataset from memory using the configuration system and using a custom operation in the dataset preprocessing, specified in [operations.py](dataset/operations.py).

## Models

### GAE
The main model system which is described [here](gae). An overview is also given in the main [ReadMe](..).

### Autoencoder

### Simple Model

The simple model example illustrates how to create a model with a single *component* and how to write the code for the model.

