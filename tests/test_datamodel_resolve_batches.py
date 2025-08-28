"""
This model tests the resolve_batches() function from the datamodel.
"""

import numpy as np
import pytest
import xarray as xr
from dask import array as da

import pystac
from tests.dummy.dummy_ml_model import DummyMLModel


def test_different_in_out_dims(mlm_item: pystac.Item):
    """
    Different dimensions in input and output
    """
    in_dc = xr.DataArray(
        da.random.random((1, 4, 20, 20)),
        dims=["time", "band", "width", "height"],
        coords={"time": ["a"], "width": range(20), "height": range(20)},
    )
    out_dc = xr.DataArray(
        da.random.random((4, 20, 1, 1, 1)),
        dims=["batch", "embedding", "time", "width", "height"],
    )

    out_shape = [-1, 20]
    out_dims = ["batch", "embedding"]

    mlm_item.ext.mlm.output[0].result.shape = out_shape
    mlm_item.ext.mlm.output[0].result.dim_order = out_dims
    d = DummyMLModel(mlm_item)

    idx_dict = ((0, 0, 0), (0, 0, 10), (0, 10, 0), (0, 10, 10))

    dim_mapping = d.get_datacube_dimension_mapping(in_dc)

    unbatched = d.resolve_batch(
        out_dc, idx_dict, dim_mapping, out_dims, ["time"], in_dc.coords
    )

    assert isinstance(unbatched, xr.DataArray)

    assert "embedding" in unbatched.dims

    assert "width" in unbatched.dims
    assert "width" in unbatched.coords
    assert len(unbatched.coords["width"]) == 2
    assert np.all(unbatched.coords["width"].data == np.array([0, 10]))

    assert "height" in unbatched.dims
    assert "height" in unbatched.coords
    assert len(unbatched.coords["height"]) == 2
    assert np.all(unbatched.coords["height"].data == np.array([0, 10]))

    assert "time" in unbatched.dims
    assert "time" in unbatched.coords
    assert len(unbatched.coords["time"]) == 1
    assert np.all(unbatched.coords["time"].data == np.array(["a"]))

    assert "batch" not in unbatched.dims
    assert "band" not in unbatched.dims


def test_different_in_out_dims_spatial(mlm_item: pystac.Item):
    """
    Different dimensions in input and output, but spatial dimensions exist
    """
    in_dc = xr.DataArray(
        da.random.random((1, 4, 20, 20)),
        dims=["time", "band", "x", "y"],
        coords={
            "x": np.linspace(100, 120, 20, endpoint=False),
            "y": np.linspace(200, 220, 20, endpoint=False),
            "time": ["a"],
        },
    )
    out_dc = xr.DataArray(
        da.random.random((4, 20, 1, 1, 1)),
        dims=["batch", "embedding", "x", "y", "time"],
    )

    mlm_item.ext.mlm.input[0].input.shape = [-1, 4, 10, 10]
    mlm_item.ext.mlm.input[0].input.dim_order = ["batch", "channel", "x", "y"]

    mlm_item.ext.mlm.output[0].result.shape = [-1, 20]
    mlm_item.ext.mlm.output[0].result.dim_order = ["batch", "embedding"]
    d = DummyMLModel(mlm_item)

    idx_dict = ((0, 0, 0), (0, 0, 10), (0, 10, 0), (0, 10, 10))

    dim_mapping = d.get_datacube_dimension_mapping(in_dc)

    unbatched = d.resolve_batch(
        out_dc, idx_dict, dim_mapping, ["batch", "embedding"], ["time"], in_dc.coords
    )

    assert isinstance(unbatched, xr.DataArray)

    assert "embedding" in unbatched.dims

    assert "x" in unbatched.dims
    assert "x" in unbatched.coords
    assert len(unbatched.coords["x"]) == 2
    assert np.all(unbatched.coords["x"].data == np.array([104.5, 114.5]))

    assert "y" in unbatched.dims
    assert "y" in unbatched.coords
    assert len(unbatched.coords["y"]) == 2
    assert np.all(unbatched.coords["y"].data == np.array([204.5, 214.5]))

    assert "batch" not in unbatched.dims
    assert "band" not in unbatched.dims


def test_same_in_out_dims_numeric_same_len(mlm_item: pystac.Item):
    """
    Same dimensoins in input and output, with extra time dimension not used in model
    """
    in_dc = xr.DataArray(
        da.random.random((1, 4, 448, 448)),
        dims=["time", "band", "width", "height"],
        coords={
            "band": ["B1", "B2", "B3", "B4"],
            "width": range(100, 100 + 448),
            "height": range(100, 100 + 448),
            "time": ["a"],
        },
    )
    out_dc = xr.DataArray(
        da.random.random((4, 4, 4, 4, 1)),
        dims=["batch", "band", "width", "height", "time"],
    )

    out_shape = [-1, 4, 4, 4]
    out_dims = ["batch", "band", "width", "height"]

    mlm_item.ext.mlm.output[0].result.shape = out_shape
    mlm_item.ext.mlm.output[0].result.dim_order = out_dims
    d = DummyMLModel(mlm_item)

    idx_dict = ((0, 0, 0), (0, 0, 224), (0, 224, 0), (0, 224, 224))

    dim_mapping = d.get_datacube_dimension_mapping(in_dc)

    unbatched = d.resolve_batch(
        out_dc,
        idx_dict,
        dim_mapping,
        ["batch", "band", "width", "height"],
        ["time"],
        in_dc.coords,
    )

    assert isinstance(unbatched, xr.DataArray)

    assert "time" in unbatched.dims
    assert "time" in unbatched.coords
    assert len(unbatched.coords["time"]) == 1
    assert np.all(unbatched.coords["time"] == np.array(["a"]))

    assert "width" in unbatched.dims
    assert "width" in unbatched.coords
    assert len(unbatched.coords["width"]) == 8
    coords_ref = np.round(np.linspace(128, 520, 8))
    coords_given = np.round(unbatched.coords["width"].data)
    assert np.all(coords_ref == coords_given)

    assert "height" in unbatched.dims
    assert "height" in unbatched.coords
    assert len(unbatched.coords["height"]) == 8

    assert "band" in unbatched.dims
    assert "band" in unbatched.coords
    assert len(unbatched.coords["band"]) == 4
    assert np.all(unbatched.coords["band"].data == np.array(["B1", "B2", "B3", "B4"]))

    assert "batch" not in unbatched.dims


def test_same_in_out_dims_numeric_len_1(mlm_item: pystac.Item):
    """
    Same dimension in input and output, but length of dimensions is 1
    """
    in_dc = xr.DataArray(
        da.random.random((1, 4, 448, 448)),
        dims=["time", "band", "width", "height"],
        coords={
            "band": ["B1", "B2", "B3", "B4"],
            "width": range(100, 100 + 448),
            "height": range(100, 100 + 448),
            "time": ["a"],
        },
    )
    out_dc = xr.DataArray(
        da.random.random((4, 4, 1, 1, 1)),
        dims=["batch", "band", "width", "height", "time"],
    )

    out_shape = [-1, 4, 1, 1]
    out_dims = ["batch", "band", "width", "height"]

    mlm_item.ext.mlm.output[0].result.shape = out_shape
    mlm_item.ext.mlm.output[0].result.dim_order = out_dims
    d = DummyMLModel(mlm_item)

    idx_dict = ((0, 0, 0), (0, 0, 224), (0, 224, 0), (0, 224, 224))

    dim_mapping = d.get_datacube_dimension_mapping(in_dc)

    unbatched = d.resolve_batch(
        out_dc,
        idx_dict,
        dim_mapping,
        ["batch", "band", "width", "height"],
        ["time"],
        in_dc.coords,
    )

    assert "width" in unbatched.dims
    assert "width" in unbatched.coords
    assert len(unbatched.coords["width"]) == 2
    coords_ref = np.round(np.linspace(212, 436, 2))
    coords_given = np.round(unbatched.coords["width"].data)
    assert np.all(coords_ref == coords_given)


def test_same_in_out_dims_numeric_len_higher(mlm_item: pystac.Item):
    """
    Same dimensions in input and output, and longer length in output
    """
    in_dc = xr.DataArray(
        da.random.random((1, 4, 448, 448)),
        dims=["time", "band", "width", "height"],
        coords={
            "band": ["B1", "B2", "B3", "B4"],
            "width": range(100, 100 + 448),
            "height": range(100, 100 + 448),
            "time": ["a"],
        },
    )
    out_dc = xr.DataArray(
        da.random.random((4, 4, 448, 448, 1)),
        dims=["batch", "band", "width", "height", "time"],
    )

    out_shape = [-1, 4, 448, 448]
    out_dims = ["batch", "band", "width", "height"]

    mlm_item.ext.mlm.output[0].result.shape = out_shape
    mlm_item.ext.mlm.output[0].result.dim_order = out_dims
    d = DummyMLModel(mlm_item)

    idx_dict = ((0, 0, 0), (0, 0, 224), (0, 224, 0), (0, 224, 224))

    dim_mapping = d.get_datacube_dimension_mapping(in_dc)

    unbatched = d.resolve_batch(
        out_dc,
        idx_dict,
        dim_mapping,
        ["batch", "band", "width", "height"],
        ["time"],
        in_dc.coords,
    )

    assert "width" in unbatched.dims
    assert "width" in unbatched.coords
    assert len(unbatched.coords["width"]) == 896
    coords_ref = np.round(np.linspace(99.75, 547.25, 896), 2)
    coords_given = np.round(unbatched.coords["width"].data, 2)
    assert np.all(coords_ref == coords_given)


def test_same_in_out_datetime(mlm_item: pystac.Item):
    """
    Same dimension in input and output. Time dim length is smaller in output than input
    """
    in_dc = xr.DataArray(
        da.random.random((5, 4, 2, 2)),
        dims=["time", "band", "width", "height"],
        coords={
            "time": np.array(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                dtype="datetime64[ns]",
            ),
            "band": ["B1", "B2", "B3", "B4"],
            "width": [100, 110],
            "height": [200, 210],
        },
    )
    out_dc = xr.DataArray(
        da.random.random((1, 10, 4, 2, 2)),
        dims=["batch", "time", "band", "width", "height"],
    )

    mlm_item.ext.mlm.input[0].input.shape = [-1, 5, 4]
    mlm_item.ext.mlm.input[0].input.dim_order = ["batch", "time", "band"]

    mlm_item.ext.mlm.output[0].result.shape = [-1, 10, 4]
    mlm_item.ext.mlm.output[0].result.dim_order = ["batch", "time"]

    d = DummyMLModel(mlm_item)

    idx_dict = ((0, 0),)
    # sub_slice = {"width": 100, "height": 200}

    dim_mapping = d.get_datacube_dimension_mapping(in_dc)
    unbatched = d.resolve_batch(
        out_dc,
        idx_dict,
        dim_mapping,
        ["batch", "time"],
        ["width", "height"],
        in_dc.coords,
    )

    assert "time" in unbatched.dims
    assert "time" in unbatched.coords
    assert len(unbatched.coords["time"]) == 10
    time_coords_ref = np.linspace(
        np.datetime64("2024-01-01").astype("datetime64[s]").astype(int),
        np.datetime64("2024-01-06").astype("datetime64[s]").astype(int),
        10,
        endpoint=False,
        dtype=int,
    ).astype("datetime64[s]")
    assert np.all(time_coords_ref == unbatched.coords["time"].data)


def test_same_in_out_nocoords(mlm_item: pystac.Item):
    """
    Same dims in input and output, but one in-dimension has no coords
    """
    in_dc = xr.DataArray(da.random.random(3), dims=["time"])
    out_dc = xr.DataArray(da.random.random((1, 2)), dims=["batch", "time"])

    mlm_item.ext.mlm.input[0].input.shape = [-1, 3]
    mlm_item.ext.mlm.input[0].input.dim_order = ["batch", "time"]

    mlm_item.ext.mlm.output[0].result.shape = [-1, 2]
    mlm_item.ext.mlm.output[0].result.dim_order = ["batch", "time"]

    d = DummyMLModel(mlm_item)

    idx_dict = ((0,),)

    dim_mapping = d.get_datacube_dimension_mapping(in_dc)
    unbatched = d.resolve_batch(
        out_dc, idx_dict, dim_mapping, ["batch", "time"], [], in_dc.coords
    )

    assert "time" in unbatched.dims
    assert "time" in unbatched.coords
    assert len(unbatched.coords["time"]) == 2
    coord_ref = np.array([0, 1])
    assert np.all(unbatched.coords["time"].data == coord_ref)


def test_same_in_out_other(mlm_item: pystac.Item):
    """
    Same dimensoin in input and output, but coords are not numeric or datetime
    """
    in_dc = xr.DataArray(
        da.random.random(3), dims=["time"], coords={"time": ["t1", "t2", "t3"]}
    )
    out_dc = xr.DataArray(da.random.random((1, 2)), dims=["batch", "time"])

    mlm_item.ext.mlm.input[0].input.shape = [-1, 3]
    mlm_item.ext.mlm.input[0].input.dim_order = ["batch", "time"]

    mlm_item.ext.mlm.output[0].result.shape = [-1, 2]
    mlm_item.ext.mlm.output[0].result.dim_order = ["batch", "time"]

    d = DummyMLModel(mlm_item)

    idx_dict = ((0,),)

    dim_mapping = d.get_datacube_dimension_mapping(in_dc)
    unbatched = d.resolve_batch(
        out_dc, idx_dict, dim_mapping, ["batch", "time"], [], in_dc.coords
    )

    assert "time" in unbatched.dims
    assert "time" in unbatched.coords
    assert len(unbatched.coords["time"]) == 2
    coord_ref = np.array(["t1.t2.t3-0", "t1.t2.t3-1"])
    assert np.all(unbatched.coords["time"].data == coord_ref)
