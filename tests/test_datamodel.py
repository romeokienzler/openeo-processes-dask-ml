import os
import unittest.mock
from datetime import datetime
from typing import Type
import pytest
import pystac
from pystac.extensions import mlm

from vcr.cassette import Cassette
from openeo_processes_dask_ml.process_implementations.constants import MODEL_CACHE_DIR
import xarray as xr
import dask.array as da

from tests.dummy.dummy_ml_model import DummyMLModel

from openeo_processes_dask.process_implementations.exceptions import (
    DimensionMissing, DimensionMismatch
)
from openeo_processes_dask_ml.process_implementations.exceptions import (
    LabelDoesNotExist
)

def prepare_tmp_folder(dir_path: str = "./tmp", file_name: str = "file.bin") -> tuple[str, str]:
    file_path = dir_path + "/" + file_name
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(file_path):
        os.remove(file_path)
    return dir_path, file_path


def clear_tmp_folder(dir_path: str = "./tmp", file_name: str = "file.bin"):
    file_path = dir_path + "/" + file_name
    os.remove(file_path)
    os.rmdir(dir_path)


@pytest.fixture
def blank_stac_item() -> pystac.Item:
    geom = {
        "type": "Polygon",
        "coordinates": [
          [
            [5.8663153, 47.2701114],
            [5.8663153, 55.099161],
            [15.0419319, 55.099161],
            [15.0419319, 47.2701114],
            [5.8663153, 47.2701114]
          ]
        ]
      }
    return pystac.Item("asdf", geom, None, datetime.now(), {})


@pytest.fixture
def random_asset() -> pystac.Asset:
    return pystac.Asset(
        "https://example.com",
        "asdf",
        "asdf",
        "asdf",
        []
    )

@pytest.fixture
def mlm_model_asset(random_asset: pystac.Asset) -> pystac.Asset:
    return pystac.Asset(
        "https://example.com",
        "model",
        "foo",
        "asdf",
        ["mlm:model"]
    )

@pytest.fixture
def mlm_item(
        blank_stac_item: pystac.Item, mlm_model_asset: pystac.Asset
) -> pystac.Item:
    blank_stac_item.stac_extensions.append(
        "https://stac-extensions.github.io/mlm/v1.4.0/schema.json"
    )
    blank_stac_item.properties["mlm:name"] = "Test"
    blank_stac_item.properties["mlm:architecture"] = "CNN"
    blank_stac_item.properties["mlm:tasks"] = "classification"

    inp = {
        "name": "test",
        "bands": [],
        "input": {
            "shape": [-1, 4, 224, 224],
            "dim_order": ["batch", "channel", "width", "height"],
            "data_type": "float64"
        }
    }
    outp = {
        "name": "classification",
        "tasks": ["classification"],
        "result": {
            "shape": [-1, 1, 1, 1],
            "dim_order": ["batch", "channel", "width", "height"],
            "data_type": "uint8"
        }
    }

    blank_stac_item.properties["mlm:input"] = [inp]
    blank_stac_item.properties["mlm:output"] = [outp]

    mlm_model_asset.href = "https://filesamples.com/samples/font/bin/slick.bin"
    blank_stac_item.add_asset("weights", mlm_model_asset)

    return blank_stac_item


def test_correct_asset_selection(blank_stac_item, random_asset, mlm_model_asset) -> None:
    d = DummyMLModel(blank_stac_item)
    with pytest.raises(Exception):
        d._get_model_asset()

    blank_stac_item.add_asset("asset1", random_asset)
    d = DummyMLModel(blank_stac_item)
    print(random_asset.title)
    with pytest.raises(Exception):
        d._get_model_asset()
    with pytest.raises(Exception):
        d._get_model_asset("asset1")

    blank_stac_item.add_asset("asset2", mlm_model_asset)
    d = DummyMLModel(blank_stac_item)
    assert d._get_model_asset().title == "model"
    assert d._get_model_asset("asset2").title == "model"

    blank_stac_item.add_asset("asset3", mlm_model_asset)
    d = DummyMLModel(blank_stac_item)
    with pytest.raises(Exception):
        d._get_model_asset()
    assert d._get_model_asset("asset3").title == "model"


@pytest.mark.vcr()
def test_download_model_http(
        mocker: unittest.mock.Mock,
        blank_stac_item: pystac.Item,
        vcr: Cassette
):
    # random binary data
    model_url = "https://filesamples.com/samples/font/bin/slick.bin"
    mock_file_path = "/fake/file/path/slick.bin"

    mock_file = unittest.mock.mock_open()
    mocker.patch("builtins.open", mock_file)

    d = DummyMLModel(blank_stac_item)
    d._download_model_http(model_url, mock_file_path)

    mock_file.assert_called_once_with(mock_file_path, 'wb')

    mock_file_handle = mock_file()

    written_parts = [call_args[0][0] for call_args in mock_file_handle.write.call_args_list]
    actual_content = b"".join(written_parts)
    expected_content = vcr.responses[0]["body"]["string"]

    assert actual_content == expected_content


@pytest.mark.vcr()
def test_download_model_http_fail(
        mocker: unittest.mock.Mock,
        blank_stac_item: pystac.Item
):
    invalid_model_url = "https://filesamples.com/fake/url/slick.bin"
    mock_file_path = "/fake/file/path/slick.bin"
    mock_file = unittest.mock.mock_open()
    mocker.patch("builtins.open", mock_file)

    d = DummyMLModel(blank_stac_item)

    with pytest.raises(Exception):
        d._download_model_http(invalid_model_url, mock_file_path)


@pytest.mark.vcr()
def test_download_model_s3(blank_stac_item: pystac.Item, mocker: unittest.mock.Mock):
    url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/35/X/LE/2025/5/S2C_35XLE_20250513_0_L2A/S2C_35XLE_20250513_0_L2A.json"

    dir_path, file_path = prepare_tmp_folder()

    d = DummyMLModel(blank_stac_item)
    d._download_model_s3(url, file_path)

    clear_tmp_folder()


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "url",
    (
        "https://filesamples.com/samples/font/bin/slick.bin",
        "s3://sentinel-cogs/sentinel-s2-l2a-cogs/35/X/LE/2025/5/S2C_35XLE_20250513_0_L2A/S2C_35XLE_20250513_0_L2A.json"
    )
)
def test_download_model(
        blank_stac_item: pystac.Item,url: str,
        vcr: Cassette
):

    dir_path, file_path = prepare_tmp_folder()

    d = DummyMLModel(blank_stac_item)
    d._download_model(url, file_path)

    with open(file_path, "rb") as file:
        actual_content = file.read()

    # filter out the GET request
    for request in vcr.requests:
        if request.method != "GET":
            continue
        expected_content = vcr.responses_of(request)[0]["body"]["string"]
        break
    # expected_content = vcr.responses[0]["body"]["string"]
    assert actual_content == expected_content

    clear_tmp_folder()


@pytest.mark.vcr()
def test_get_model(mlm_item: pystac.Item, monkeypatch):
    mock_opener: unittest.mock.MagicMock = unittest.mock.mock_open()

    monkeypatch.setattr("builtins.open", mock_opener)
    monkeypatch.setattr("os.makedirs", lambda x: None)

    d = DummyMLModel(mlm_item)
    model_file_path = d._get_model()

    # assert that the method was called once
    mock_opener.assert_called_once()

    # mock path exists to use
    monkeypatch.setattr("os.path.exists", lambda x: True)

    # should not download the mdoel again as it is cached
    model_file_path = d._get_model()

    # assert that the method was STILL called only once (cached file exists)
    mock_opener.assert_called_once()


@pytest.mark.parametrize(
    "model_dim_names, dc_dim_names, idx",
    (
        (("bands", "x", "y", "time"), ("band", "x", "y", "time"), (0, 1, 2, 3)),
        (("band", "x", "y", "time"), ("band", "lon", "lat", "t"), (0, 1, 2, 3)),
        (("t", "x", "y", "channel"), ("band", "x", "y", "time"), (3, 1, 2, 0)),
        (("x", "y", "asdf"), ("x", "y", "bands", "t"), (0, 1, None)),
    )
)
def test_get_datacube_dimension_mapping(
        mlm_item: pystac.Item,
        model_dim_names: tuple[str],
        dc_dim_names: tuple[str],
        idx: list[int|None]
):
    d = DummyMLModel(mlm_item)
    mlm_item.ext.mlm.input[0].input.dim_order = model_dim_names

    cube = xr.DataArray(
        da.random.random((1,1,1,1)),
        dims=dc_dim_names
    )

    mapping = d.get_datacube_dimension_mapping(cube)
    assert len(idx) == len(mapping)
    assert len(model_dim_names) == len(mapping)

    for i, model_dim_name in enumerate(model_dim_names):

        if mapping[i] is not None:
            mapped_dim_name = mapping[i][0]
            map_idx = mapping[i][1]
            assert mapped_dim_name == dc_dim_names[map_idx]


@pytest.mark.parametrize(
    "dc_dims, ignore_batch, valid",
    (
        (["batch", "channel", "width", "height"], True, True),
        (["asdf", "channel", "width", "height"], False, False),
        (["batch", "channel", "width", "asdf"], True, False),
        (["batch", "channel", "width", "asdf"], False, False),
    )
)
def test_check_dimensions_present_in_datacube(
    mlm_item: pystac.Item,
    dc_dims: list[str],
    ignore_batch: bool,
    valid: bool
):
    d = DummyMLModel(mlm_item)
    dc = xr.DataArray(
        da.random.random((1,1,1,1)),
        dims=dc_dims
    )

    if valid:
        d._check_dimensions_present_in_datacube(dc, ignore_batch)
    else:
        with pytest.raises(DimensionMissing):
            d._check_dimensions_present_in_datacube(dc, ignore_batch)


@pytest.mark.parametrize(
    "dc_shape, ignore_batch, valid",
    (
        ([10, 4, 224, 224], False, True),
        ([10, 4, 224, 224], True, True),
        ([10, 10, 230, 230], True, True),
        ([10, 10, 230, 230], False, True),
        ([10, 10, 230, 230], True, True),
        ([10, 2, 230, 230], True, False),
        ([10, 10, 15, 230], False, False),
    )
)
def test_check_datacube_dimension_size(
    mlm_item: pystac.Item,
    dc_shape: list[int],
    ignore_batch: bool,
    valid: bool
):
    d = DummyMLModel(mlm_item)
    dc_dims = ["batch", "channel", "width", "height"]

    dc = xr.DataArray(
        da.random.random(dc_shape),
        dims=dc_dims
    )
    if valid:
        d._check_datacube_dimension_size(dc, ignore_batch)
    else:
        with pytest.raises(DimensionMismatch):
            d._check_datacube_dimension_size(dc, ignore_batch)


@pytest.mark.parametrize(
    "m_bands, dc_bands, dc_band_dim_name, exception",
    (
        ([], ["B02", "B03"], "band", None),
        (["B02", "B03"], ["B02", "B03"], "band", None),
        (["B02", "B03"], ["B02", "B03", "B04"], "band", None),
        (["B02", "B03"], ["B02", "B03"], "asdf", DimensionMissing),
        (["B02", "B03"], ["B02", "B04"], "band", LabelDoesNotExist),
        (
            [mlm.ModelBand({"name": "B02"}), mlm.ModelBand({"name": "B03"})],
            ["B02", "B03"],
            "band",
            None
        ),
        (
            [mlm.ModelBand({"name": "B02"}), mlm.ModelBand({"name": "B03"})],
            ["B02", "B03", "B04"],
            "band",
            None
        ),
        (
            [mlm.ModelBand({"name": "B02"}), mlm.ModelBand({"name": "B03"})],
            ["B02", "B04"],
            "band",
            LabelDoesNotExist
        ),
        (
            [
                mlm.ModelBand({"name": "NDVI", "format": "asdf"}),
                mlm.ModelBand({"name": "B02"})
            ],
            ["B02", "B04"],
            "band",
            ValueError
        ),
        (
            [
                mlm.ModelBand({"name": "NDVI", "expression": "asdf"}),
                mlm.ModelBand({"name": "B02"})
            ],
            ["B02", "B04"],
            "band",
            ValueError
        ),
        (
            [
                mlm.ModelBand({"name": "B04"}),
                mlm.ModelBand({"name": "B08"}),
                mlm.ModelBand({
                    "name": "NDVI",
                    "format": "python",
                    "expression": "(B08-B04)/(B08+B04)"}
                )
            ],
            ["B04", "B08"],
            "band",
            None
        ),
    )
)
def test_check_datacube_bands(
        mlm_item: pystac.Item, m_bands: list[str|mlm.ModelBand], dc_bands: list[str],
        dc_band_dim_name: str, exception: Type[Exception]|None
):
    mlm_item.ext.mlm.input[0].bands = m_bands
    d = DummyMLModel(mlm_item)

    dc = xr.DataArray(
        da.random.random((1,1,len(dc_bands))),
        dims=["x", "y", dc_band_dim_name],
        coords={"x": [1], "y": [1], dc_band_dim_name: dc_bands}
    )

    if exception is None:
        d._check_datacube_bands(dc)
    else:
        with pytest.raises(exception):
            d._check_datacube_bands(dc)


@pytest.mark.parametrize(
    "dc_dims, dc_dim_shp, exception_raised",
    (
        (("time", "x", "y", "bands"), (4, 1000, 1000, 8), None),
        (("x", "y", "t", "bands"), (1000, 1000, 4, 8), None),
        (("times", "x", "y", "channel"), (4, 1000, 1000, 8), None),
        (("time", "x", "y"), (4, 1000, 1000), DimensionMissing),
        (("time", "x", "y", "bands"), (4, 100, 100, 8), DimensionMismatch)
    )
)
def test_check_datacube_dimensions(
        mlm_item: pystac.Item,
        dc_dims: list[str],
        dc_dim_shp: list[int],
        exception_raised: Type[Exception]
):
    dc = xr.DataArray(
        da.random.random(dc_dim_shp),
        dims=dc_dims
    )

    assert len(dc_dim_shp) == len(dc_dim_shp)

    mlm_item.ext.mlm.input[0].input.shape = (-1, 1, 128, 128, 4)
    mlm_item.ext.mlm.input[0].input.dim_order = ("batch", "time", "x", "y", "bands")

    d = DummyMLModel(mlm_item)

    if exception_raised is None:
        # positive tests: should work flawlessly
        d.check_datacube_dimensions(dc, True)
    else:
        # negative test: when an exception is raised
        with pytest.raises(exception_raised):
            d.check_datacube_dimensions(dc, True)
