# Examples

## Dataset

## Models
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

### Autoencoder

### Simple Model

The simple model example illustrates how to create a model with a single *component* and how to write the code for the model.

