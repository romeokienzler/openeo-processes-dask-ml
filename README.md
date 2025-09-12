# OpenEO Processes Dask: Machine Learning

`openeo-processes-dask-ml` is a Python package that implements generic machine learning
(ML) processes for openEO. It is built to work alongside and integrate with
[openeo-processes-dask](https://github.com/Open-EO/openeo-processes-dask), extending it
by machine learning capabilities.

## Installation

This package is not published on PyPI yet. It can only be used from source

## Development environment

1. Clone the repository
2. Install it using [poetry](https://python-poetry.org/docs/):
   `poetry install --all-extras`
3. Run the test suite: `poetry run pytest`

### Pre-commit hooks

This repo makes use of [pre-commit](https://pre-commit.com/) hooks to enforce linting &
a few sanity checks. In a fresh development setup, install the hooks using
`poetry run pre-commit install`. These will then automatically be checked against your
changes before making the commit.

## Structure

- `minibackend` has a minimal backend implementation for executing process graphs
- `opd-ml-dev-utils` has some scripts that are helpful during development
- `openeo-processes-dask-ml` the actuall ML process specs and implementations
- `tests` for pytest

## Acknowledgement

Development within this repository are carried out as part of the
[Embed2Scale](https://embed2scale.eu/) project and is cofunded by the EU Horizon Europe
program under Grant Agreement 101131841. Additional funding for this project has been
provided by the Swiss State Secretariat for Education, Research and Innovation and UK
Research and Innovation.
