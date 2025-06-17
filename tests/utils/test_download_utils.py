import os

import pytest
import unittest.mock
from vcr.cassette import Cassette

from openeo_processes_dask_ml.process_implementations.utils import download_utils


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


@pytest.mark.vcr()
def test_download_http(
        mocker: unittest.mock.Mock,
        vcr: Cassette
):
    # random binary data
    model_url = "https://filesamples.com/samples/font/bin/slick.bin"
    mock_file_path = "/fake/file/path/slick.bin"

    mock_file = unittest.mock.mock_open()
    mocker.patch("builtins.open", mock_file)

    download_utils.download_http(model_url, mock_file_path)

    mock_file.assert_called_once_with(mock_file_path, 'wb')

    mock_file_handle = mock_file()

    written_parts = [call_args[0][0] for call_args in
                     mock_file_handle.write.call_args_list]
    actual_content = b"".join(written_parts)
    expected_content = vcr.responses[0]["body"]["string"]

    assert actual_content == expected_content


@pytest.mark.vcr()
def test_download_model_http_fail(
        mocker: unittest.mock.Mock
):
    invalid_model_url = "https://filesamples.com/fake/url/slick.bin"
    mock_file_path = "/fake/file/path/slick.bin"
    mock_file = unittest.mock.mock_open()
    mocker.patch("builtins.open", mock_file)

    with pytest.raises(Exception):
        download_utils.download_http(invalid_model_url, mock_file_path)

    mock_file.assert_not_called()


@pytest.mark.vcr()
def test_download_model_s3():
    url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/35/X/LE/2025/5/S2C_35XLE_20250513_0_L2A/S2C_35XLE_20250513_0_L2A.json"

    dir_path, file_path = prepare_tmp_folder()
    download_utils.download_s3(url, file_path)
    clear_tmp_folder()


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "url",
    (
        "https://filesamples.com/samples/font/bin/slick.bin",
        "s3://sentinel-cogs/sentinel-s2-l2a-cogs/35/X/LE/2025/5/S2C_35XLE_20250513_0_L2A/S2C_35XLE_20250513_0_L2A.json"
    )
)
def test_download(url: str, vcr: Cassette):

    dir_path, file_path = prepare_tmp_folder()
    download_utils.download(url, file_path)

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
