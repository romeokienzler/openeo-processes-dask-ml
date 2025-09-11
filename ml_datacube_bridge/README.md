## Purpose

Sometimes, input and output of ML models is not directly ordered in a datacube shape.

For example, Vit-Encoders output a list of tensors. Each tensor holds embeddings for
each image patch. They must first be reordered to regain their spatial shape, and to
be reordered in a datacube.

This module serves two purposes:

1to provide basic function to reshape datacube chunks into the input which the ML
model accepts as input
2to provides basic functions to reshape raw model output into a shape that can be
reordered into a datacube


## Input method specification
- Input: n-dimensional array as np.array
- Output: Anything, must be suitable as inptu for the specific ML model


## Output method specification

- Input: the raw output, as produced by the ML model.
- Return: n-dimensional array (e.g. pyTorch- or TF-Tensor, np.ndarray etc.), that
matches the specified model output shape.
- Return datatype: must match the specific framework's implementation of n-dimensional
arrays. For example: A method called after predicting with a torch model must return a
torch.Tensor.
- Exceptions: Functions raise the following exceptions:
  - `TypeError`: if the model output that is passed to the function is different from what the function expects.
  - `ValueError`: if data passed to the function has an invalid value
