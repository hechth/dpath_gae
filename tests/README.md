# Tests

Test files make sure that functionality provided by [packages](../packages) is correct.
They're organized into the different modules, same as the packages.

## Image

List of tested functions:
*   [Extract Patches](image/extract_patches.py)
*   [Subsample Image](image/subsample.py)
*   [Valid image file type](image/is_image.py)

## Utility

List of tested functions:
*   [get_function](utility/call_function.py): Imports *module* and retrieves a *function* defined in this module.
*   [pipeline](utility/pipeline.py): Chains function execution with single input value.
*   [concatenate_functions](util/concatenate_functions.py): Create new *function* which performs concatenated operation of specified functions.

## Model

### Config

Tests to check correct reading and parsing of configuration files and dictionaries.

