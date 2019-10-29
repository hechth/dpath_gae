# Tools

Small tools which can be used from the commandline to provide some functionality to create datasets or for visualization purposes.
All scripts can be run with *--help* to display a help message about usage.

## Dataset

*   [Collect filenames in Dataset](dataset/CollectFilenamesInDataset.py): Collects files specified by pattern and stores them as a tfrecords file.
*   [Split Dataset](dataset/SplitDataset.py): Splits and shuffles a .tfrecords file with second part containing 1/*factor* samples.

## Model

*   [Run Inference](model/RunInference.py): Use a saved model stored under *export_dir* to run inference on a file.

## Visualization

*   [Plot Image](visualization/PlotImage.py): Plot given image using pyplot.




