# Image

```python
import packages.Tensorflow.Image as ctfi
```

This package provides functionality for image processing used in the [Dataset](../Dataset/README.md) package or other parts of the developed tools.

Functionality is tested with both eager execution enabled and without.

## Features

*  Load images.
*  Extract patches with given size from an image.
*  Rescale intensity values of image to a given range.
*  Subsample an image by a given factor to lower resolution.

To be implemented:
*   Encode image as example proto.
