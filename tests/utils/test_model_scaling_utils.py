import numpy as np
import pytest
import xarray as xr
from pystac.extensions.mlm import ValueScaling, ValueScalingType

from openeo_processes_dask_ml.process_implementations.utils.scaling_utils import (
    _raise_value_error,
    _validate_scaling_obj,
    scale_datacube,
)


@pytest.fixture
def dc() -> xr.DataArray:
    return xr.DataArray(
        [[2, 4], [6, 10]],
        dims=["t", "bands"],
        coords={"t": ["t1", "t2"], "bands": ["red", "green"]},
    )


def test_raise_value_error():
    with pytest.raises(ValueError):
        _raise_value_error("asdf", ["foo", "bar"])


@pytest.mark.parametrize(
    "value_dict, valid",
    (
        ({"type": "min-max", "minimum": 0, "maximum": 1}, True),
        ({"type": "min-max", "minimum": 0}, False),
        ({"type": "min-max", "maximum": 1}, False),
        ({"type": "min-max"}, False),
        ({"type": "clip", "minimum": 0, "maximum": 1}, True),
        ({"type": "clip", "minimum": 0}, False),
        ({"type": "clip", "maximum": 1}, False),
        ({"type": "clip"}, False),
        ({"type": "z-score", "mean": 0, "stddev": 1}, True),
        ({"type": "z-score", "mean": 0}, False),
        ({"type": "z-score", "stddev": 1}, False),
        ({"type": "z-score"}, False),
        ({"type": "clip-min", "minimum": 1}, True),
        ({"type": "clip-min"}, False),
        ({"type": "clip-max", "maximum": 2}, True),
        ({"type": "clip-max"}, False),
        ({"type": "offset", "value": 2}, True),
        ({"type": "offset"}, False),
        ({"type": "scale", "value": 2}, True),
        ({"type": "scale"}, False),
        ({"type": "processing", "format": "a", "expression": "a"}, True),
        ({"type": "processing", "format": "a"}, False),
        ({"type": "processing", "expression": "a"}, False),
    ),
)
def test_test_req_props(value_dict: dict, valid: bool):
    scale_obj = ValueScaling(value_dict)
    if valid:
        _validate_scaling_obj(scale_obj)
    else:
        with pytest.raises(ValueError):
            _validate_scaling_obj(scale_obj)


@pytest.mark.parametrize(
    "scale, truth",
    (
        (
            ValueScaling.create(ValueScalingType.MIN_MAX, minimum=2, maximum=10),
            [[0, 0.25], [0.5, 1]],
        ),
        (
            ValueScaling.create(ValueScalingType.Z_SCORE, mean=2, stddev=2),
            [[0, 1], [2, 4]],
        ),
        (
            ValueScaling.create(ValueScalingType.CLIP, minimum=3, maximum=8),
            [[3, 4], [6, 8]],
        ),
        (ValueScaling.create(ValueScalingType.CLIP_MIN, minimum=4), [[4, 4], [6, 10]]),
        (ValueScaling.create(ValueScalingType.CLIP_MAX, maximum=5), [[2, 4], [5, 5]]),
        (ValueScaling.create(ValueScalingType.OFFSET, value=2), [[0, 2], [4, 8]]),
        (ValueScaling.create(ValueScalingType.SCALE, value=2), [[1, 2], [3, 5]]),
    ),
)
def test_scale(dc, scale: ValueScaling, truth: list[list[int]]):
    new_dc = scale_datacube(dc, scale)
    truth_arr = np.array(truth)
    assert (new_dc.data == truth_arr).all()
