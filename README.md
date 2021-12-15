# dpath_gae

![Latent Space Traversals](data/images/latent_traversals_-5_to_5.png)

## Disclaimer
Development of this project started as Master's Thesis at Technical University of Munich and is now continued as part of my work at Masaryk University in Brno, Czech Republic.
Main contributors are also [Vlad Popovici](https://www.muni.cz/en/people/118944-vlad-popovici) and [Hasan Sarhan](http://campar.in.tum.de/Main/HasanSarhan).
It is targeted towards disentangled representation learning for digital pathology data as a custom similarity metric for deformable image registration.

Please note that this project is under development and doesn't contain the main content as of today.
Collaborations and contributions are welcome!

If using this code in your work please cite the following publication

[1] Hecht, H.; Sarhan, M.H.; Popovici, V. Disentangled Autoencoder for Cross-Stain Feature Extraction in Pathology Image Analysis. Appl. Sci. 2020, 10, 6427. https://doi.org/10.3390/app10186427


## Setup
### Python

Packages to install via pip3:
*  [tensorflow](https://www.tensorflow.org/install/pip) v1.12, preferably with [GPU suppurt enabled](https://www.tensorflow.org/install/gpu). GPU support is not necessary, but makes stuff faster.
*   numpy
*   scipy
*   matplotlib
*   [gitpython](https://gitpython.readthedocs.io/en/stable/intro.html#installing-gitpython) & [gitdb](https://pypi.org/project/gitdb/)
*   re

### C++

None of the C++ functionality is implemented as of now, so these are future dependencies.

*  Install [CUDA](https://developer.nvidia.com/cuda-zone) if you have a NVIDIA GPU.
*  Install [ITK](https://github.com/InsightSoftwareConsortium/ITK).
    *  Clone from github and check out v4.13.1
    *  Configure and build it using CMake
    *  Install it using [checkinstall](https://debian-administration.org/article/147/Installing_packages_from_source_code_with_checkinstall), you might want to remove it later.
*  Install [tensorflowCC](https://github.com/FloopCZ/tensorflow_cc).
*  Install Qt5 v5.9.5

## How To?

The main part of the project is the [gae_model.py](examples/models/gae/gae_model.py) script for training a custom model. The model architecture, as well as the input dataset is described in a JSON configuration file which is parsed to build the model. The model is implemented using the `tf.Estimator` API and expects a `tf.data.TFRecordsDataset`.

So in order to use the script, you have to follow these steps:
1.  Create your own dataset as described below.
2.  Adapt the dataset part of the configuration script for your dataset as well as the input part of the model configuration.
3.  Adapt the model architecture (encoder/decoder layers) to your data.
4.  Adapt the hyperparameters and embedding configuration.
5.  Train your model.

### Create a Custom Dataset?

#### Custom Dataset from Image Files
The examples contained in this repository don't come with a publicly available dataset, but with multiple tools to create a dataset in the required layout. All tools required to create a matching dataset are collected [here](tools/dataset).

The dataset required to train the models are expected to consist of fixed size image patches with 3 channel float data and a class label per image patch.

Steps to create a custom dataset:
1.  Collect the filenames of the image files in a tfrecords dataset using [this](tools/dataset/CollectFilenamesInDataset.py) script.
    1.  Make sure that all images are stored in a similar pattern where all filenames can be collected using a python glob expression.
    2.  Make sure all image files are in one of the supported formats. For a list of supported image formats see [here](packages/Tensorflow/Image/Image.py).
    3.  Make sure all images have moderate size (width < 10k, height < 10k)
2.  Run the [preprocessing script](tools/dataset/preprocess_image_filenames_dataset.py) to load the collected image files, assign a label to them and store tiles as a binary tfrecords file.
    1.  The arguments required for the preprocessing script are (1) name of the input dataset holding image filenames, (2) name of the output dataset, (3) the patch size, (4) the number of samples for the target dataset, (5) a comma separated list of labels, where each filename has to contain 1 or 0 of these labels, (optional)(6) a threshold for the filter function, between 0 and 1, usually 0.04 < t < 0.1, (optional)(7) a size for the images.
    2.  Make sure to adapt the shuffle buffer sizes for your data and patch sizes.
3.  Verify the integrity of the dataset using the [plot image from dataset](tools/visualization/plot_image_from_dataset.py) script and [estimate distribution](tools/dataset/estimate_distribution.py).
4.  Create normalization files using the [estimate moments](tools/dataset/estimate_moments.py) tool.
    1.  Specify the number of samples over which to estimate the moments and the axis which to use for estimation.
    2.  In case of using 3 color RGB images, specify [0,1,2] as axis to normalize the 3 color channels, so mean and variance are length 3 vectors.

#### MNIST Dataset

The model can also be trained and run on the MNIST dataset with minor changes.
Since the whole configuration file specification only influences the training function required for the tf.Estimator to run training, providing your own training function can override the whole dataset loading and preprocessing pipeline.

```python
def train_func():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255

    # Reshape input data from (28, 28) to (28, 28, 1)
    w, h = 28, 28
    x_train = x_train.reshape(x_train.shape[0], w, h, 1)

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    train_ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x_train),tf.data.Dataset.from_tensor_slices(y_train)))
    train_ds = train_ds.map(lambda image, label: ({"val": image }, label))

    train_ds = train_ds.repeat().shuffle(60000).batch(10)
    return train_ds

def main(argv):
    # Do stuff, initialize model, etc.

    classifier = classifier.train(input_fn=train_func, steps=10000)
```

For an example how this is used in a model see [here](examples/models/autoencoder/autoencoder.py).

### Create A Model with Custom Configuration?
Creating a custom model architecture which follows pattern described in the [publication](http://home.in.tum.de/~hechth/hechth_dpath.pdf) requires costimizing a configuration file. The basic [GAE model](examples/models/gae/gae_model.py) example comes with an example configuration for patch size 32 and 64, using a resnet_v2 block based encoder, a sampler and some preconfigured summaries for tensorboard. The following explanations will be based on the 32 size configuration.

#### Adapt the Dataset
If you're using a "in script" dataset - see the MNIST example - you can skip this step.

Dataset preparation can be implemented oneself by implementing the function passed to tf.Estimator.train(...).

```python
def train_fn(args):
    # Load and preprocess dataset.
    # dataset = tf.data.TFRecordDataset(...)
    # dataset = dataset.map(...)
    return dataset

def main(argv):
    # ...
    classifier.train(input_fn=train_fn, steps=steps)
    # ...
```

Another option is to specify the dataset in the JSON configuration file, this is illustrated in an [example](examples/dataset).

```json
{ 
    "datasets": {
        "training": {
            "filename": "examples/dataset/training_ds.tfrecords",
            "size": 100000,
            "operations": [ "..." ]
        },
        "validation": { "..." },
        "test": { "..." },
        "batch":200,
        "features":[ "..." ]
    }
}
```

Fields describing individual datasets which can be populated are *training*, *validation* and *test*.
Currently, only *training* is supported.

Each dataset contains the *filename*, the number of samples to extract from this dataset as *size*, the number of *steps* to run on this dataset and an optional array of *operations* to run in the preprocessing.

Entries valid for all datasets are the *batch* size to use and the *features* of the dataset on disk, a list of triples containing of *key*, *shape* and *dtype*, similar to the features defined as model inputs.
Note that the label has to be defined as a feature with the *key* "label".

After creating a custom dataset, adapt the dataset part of the configuration file.
Example config for dataset at "/home/xyz/ds_32_1000.tfrec" with patch_size 32, 1000 samples for training for 10 epochs with a batch size of 10 and a shuffle buffer of size 1000.

The [preprocess image filenames dataset script](tools/dataset/preprocess_image_filenames_dataset.py) stores the image patch feature under the key 'patch' and the label under the key 'label'.

```json
{
    "datasets": {
        "training": {
            "filename": "/home/xyz/ds_32_1000.tfrec",
            "size":1000 ,
            "epochs": 10
        },
        "batch":10,
        "shuffle_size":1000,
        "features":[
            {
                "shape": [32,32,3],
                "key": "patch",
                "dtype": "tf.float32"
            },
            {
                "shape": [1],
                "key": "label",
                "dtype": "tf.int64"
            }
        ]
    },
}
```

#### Adapt the Hyperparameters
The hyperparameters for the model are the *beta* factor for the KL-divergence based latent loss, *alpha* and *delta*, which control the supervised classification and adversarial loss terms. Values for *beta* should be chosen 5 < *beta* < 20. For *alpha* and *delta*, 1 < *alpha (delta)* < 10.

The *latent_space_size* determines the size of the latent code, BUT it has to be manually also set as number of dims in the sampling layer. The entry in the config file is required for the correct dimensionsons for the latent loss.

Example:
```json
{
    "parameters": {
        "beta": 16,
        "alpha": 12,
        "delta": 12,
        "latent_space_size":18
    }
}
```

#### Adapt the Architecture

The model is specified by its *inputs* and *components*.

```json
{
    "model": {
        "inputs": { "..." },           
        "components":[ "..." ]
    }
}
```
The *inputs* are supposed to have the following structure:

```json
{
    "inputs": {
        "features":[
            {
                "shape": [1],
                "key": "val",
                "dtype": "tf.float32"
            }
        ],
        "labels": {
            "shape": [1],
            "dtype": "tf.float32"
        }
    }
}
```

The input *features* to the model are an array, each element being composed of *shape*, *key* and *dtype* information. The second field is the labels, which only support a single entry now defined by *shape* and *dtype*.
The second part of the model are its *components*.

```json
{
    "components":[
        {
            "name":"network",
            "input": "val",
            "layers": [ "..." ],
            "output":"logits"
        }
    ]
}
```

A *component* is defined by its *name*, *input*, the *layers* and its *output*.

The *input* of a component is the *key* of a feature describes in the model *inputs* or the name defined by the *output* field of a preceding component. The *output* field therefore defines the key under which the output values of this component can be accessed.
*Layers* is an array, each entry defining a layer in the model. The layers are connected in the ordering in which they are defined in the array.

Which fields are used to define properties regarding layers is described in [Layer.py](packages/Tensorflow/Model/Layer.py).
A detailed list of which layers can be specified and how is available [here](packages/Tensorflow/Model)

For examples of how to specify possible configurations and the corresponding python code to create the models are given in [Examples/Models](examples/models).


The model architecture is mainly determined by the components. The architecture described in the publication consists of the following components:
1.  Encoder
2.  Sampler
3.  Classifier
4.  Discriminator
5.  Decoder

The model architecture as described in the config file also contains these components:

```json
{
    "components": [      
        {
            "name":"encoder",
            "input": "patch",
            "layers": [],
            "output":"encoded_patch"
        },
        {
            "name":"sampler",
            "input":"encoded_patch",
            "layers": [],
            "output":["distribution", "code"]
        },
        {
            "name":"classifier",
            "input": "code",
            "layers":[],
            "output":"predictions_classifier"
        },
        {
            "name":"discriminator",
            "input": "code",
            "layers":[],
            "output":"predictions_discriminator"
        },
        {
            "name": "decoder",
            "input":"code",
            "layers": [],
            "output":"logits"
        }
    ]
}
```

If the names of inputs or outputs of the different components are changed, the [script](examples/models/gae/gae_model.py) has to be adapted as well. Usually, adapting the layers in a component is enough. The default architecture uses resnet_v2 blocks for the encoder, followed by batch normalization, activation and max pooling.

```json
{
    "name":"encoder",
    "input": "patch",
    "layers": [
        {
            "type":"resnet_v2_block",
            "stride":1,
            "base_depth":16,
            "num_units": 2
        },
        {
            "type":"batch_norm",
            "axis": [1,2,3]
        },
        {
            "type":"activation",
            "function":"tf.nn.relu"
        },                    
        {
            "type":"max_pool",
            "pool_size":[2,2],
            "strides":[2,2]
        }
    ],
    "output":"encoded_patch"
}
```

The number of these blocks is usually equal to the log2 of the patch size to reduce the spatial dimensions to 1. Model complexity can be increased by changing the *base_depth* of the resnet_v2_block while the depth can be increased by increasing *num_units*.

The sampler contains a single sampling layer, it can potentially also be integrated into the encoder, while *dims* controls the latent space size. The difference between "type":"sampler" and "type":"sampler_v2" is that the base sampler produces a diagonal covariance matrix while sampler_v2 uses a full lower triangular covariance matrix.

```json
{
    "name":"sampler",
    "input":"encoded_patch",
    "layers": [
        {
            "type":"sampler_v2",
            "dims":18,
            "name":"z"
        }
    ],
    "output":["distribution", "code"]
}
```

The sampler outputs are the tf.contrib.distributions.MultivariateNormalTriL distribution and the sample drawn from besaid distribution.

The classifier and discrimator both extract only their respective parts of the latent code and have a single dense layer with *num_classes* units.

```json
{
    "name":"classifier",
    "input": "code",
    "layers":[
        {
            "type":"flatten"
        },
        {
            "type":"slice",
            "begin":[0],
            "size":[4],
            "name:":"z_stain"
        },
        {
            "type":"dense",
            "units":9,
            "activation":"tf.nn.sigmoid",
            "trainable": false
        }
    ],
    "output":"predictions_classifier"
}
```

The decoder uses plain deconvolution and average upsampling layers instead of the resnet_v2 blocks architecture since it doesn't provide a transposed implementation out of the box.

The decoder, classifier and discriminator layers are set to be non-trainable, which means they are not included in the default training step. All variables associated to these layers are collected and updated in additional training ops.

```python
# Rest of my_model_fn
train_op_encoder = optimizer.minimize(loss, global_step=tf.train.get_global_step())
train_op_discriminator = optimizer.minimize(discriminator_loss, var_list=[components['discriminator'][1]])
train_op_classifier = optimizer.minimize(classifier_loss, var_list=[components['classifier'][1]])

decoder_variables = components['decoder'][1]
train_op_decoder = optimizer.minimize(reconstr_loss, var_list=decoder_variables)

# Create combined training op
train_op = tf.group([train_op_encoder,train_op_discriminator,train_op_classifier,train_op_decoder])
```

This allows explicit updating of weights regarding certain losses.


### How to train the Model?

After having adapted the configuration, you can start training the [model](examples/models/gae/gae_model.py). The command line parameters for the script are as follows: (1) path to config file, (2) path to npy file holding estimated mean, (3) path to npy file holding estimated variance, (4) path where to store the model log files.

The number of runs over the full dataset for training is specified in the config file as epochs in the dataset branch. The number of steps for which training is run is determined as *size* / *batch*. The *batch* options controls the batch size used for training.

Example:
```json
{
    "datasets": {
        "training": {
            "filename": "/sdc1/datasets/level1/train_ds_20000000_64_v3.tfrec",
            "size": 20000000,
            "epochs": 10
        },
        "batch":256,
        "shuffle_size":250000,
        "features":[
            {
                "shape": [64,64,3],
                "key": "patch",
                "dtype": "tf.float32"
            },
            {
                "shape": [1],
                "key": "label",
                "dtype": "tf.int64"
            }
        ]
    }
}
```

## Examples

The [examples](examples) folder contains functionality which illustrates how to use this project as well as the main models.
For a detailed description about the individual examples see the [ReadMe](examples).

## Packages

The packages contain custom functionality used in this project to serve as an API providing mostly convenience wrappers. Functions to manipulate images are collected in the [Image package](packages/Tensorflow/Image), dataset related operations are part of the [Dataset package](packages/Tensorflow/Dataset) and everything related to layers and model configuration is contained in the [Model package](packages/Tensorflow/Model).

The model package documentation also contains a detailed list of possible specifications for layers which can be used in components.
