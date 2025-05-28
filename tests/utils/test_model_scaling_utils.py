import pytest
from openeo_processes_dask_ml.process_implementations.utils.scaling_utils import (
    scale_datacube
)
from pystac.extensions.mlm import ValueScaling, ValueScalingType
import xarray as xr
import numpy as np


@pytest.fixture
def dc() -> xr.DataArray:
    return xr.DataArray(
        [
            [2, 4],
            [6, 10]
        ],
        dims=["t", "bands"],
        coords={"t": ["t1", "t2"], "bands": ["red", "green"]}
    )


@pytest.mark.parametrize(
    "scale, truth",
    (
        (
            ValueScaling.create(ValueScalingType.MIN_MAX, minimum=2, maximum=10),
            [[0, 0.25], [0.5, 1]]
        ),
        (
            ValueScaling.create(ValueScalingType.Z_SCORE, mean=2, stddev=2),
            [[0, 1], [2, 4]]
        ),
        (
            ValueScaling.create(ValueScalingType.CLIP, minimum=3, maximum=8),
            [[3, 4], [6, 8]]
        ),
        (
            ValueScaling.create(ValueScalingType.CLIP_MIN, minimum=4),
            [[4, 4], [6, 10]]
        ),
        (
            ValueScaling.create(ValueScalingType.CLIP_MAX, maximum=5),
            [[2, 4], [5, 5]]
        ),
        (
            ValueScaling.create(ValueScalingType.OFFSET, value=2),
            [[0, 2], [4, 8]]
        ),
        (
            ValueScaling.create(ValueScalingType.SCALE, value=2),
            [[1, 2], [3, 5]]
        )
    )
)
def test_scale(dc, scale: ValueScaling, truth: list[list[int]]):
    new_dc = scale_datacube(dc, scale)
    truth_arr = np.array(truth)
    assert (new_dc.data == truth_arr).all()




