import os
import unittest.mock
from datetime import datetime
import pytest
import pystac

from vcr.cassette import Cassette
from openeo_processes_dask_ml.process_implementations.constants import MODEL_CACHE_DIR
import xarray as xr
import dask.array as da

from tests.dummy.dummy_ml_model import DummyMLModel


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
