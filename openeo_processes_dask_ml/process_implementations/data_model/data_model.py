import os.path

import botocore.exceptions

import pystac
from pystac.extensions.mlm import MLMExtension
from typing import Any
from abc import ABC, abstractmethod
import requests
from io import BytesIO
import sys
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from openeo_processes_dask_ml.process_implementations.constants import (
    MODEL_CACHE_DIR, S3_MODEL_REPO_ENDPOINT, S3_MODEL_REPO_ACCESS_KEY_ID,
    S3_MODEL_REPO_SECRET_ACCESS_KEY
)
from openeo_processes_dask_ml.process_implementations.utils import model_cache_utils

# todo: replace sys.out.write() and print() with logger actions


class MLModel(ABC):
    stac_item: pystac.Item

    def __init__(self, stac_item: pystac.Item):
        self.stac_item = stac_item
        self._model_object = None

    @property
    def model_metadata(self) -> MLMExtension:
        # todo: account for if metadata is stored with the asset
        return MLMExtension.ext(self.stac_item)

    def _get_model_asset(self, asset_name: str = None) -> pystac.Asset:
        """
        Determine which asset holds the model (has mlm:model in roles)
        Determines whcih asset to use if multiple assets with role mlm:models are found
        :param asset_name:
        :return:
        """
        assets = self.stac_item.assets
        model_assets = {
            key: assets[key] for key in assets if "mlm:model" in assets[key].roles
        }

        # case 1: no assets with mlm:model role
        if not model_assets:
            raise Exception(
                "The given STAC Item does not have an asset with role mlm:model"
            )

        # case 2: asset_name is given
        if asset_name:
            if asset_name in model_assets:
                return model_assets[asset_name]
            else:
                raise Exception(
                    f"Provided STAC Item does not have an asset named {asset_name} which "
                    f"also lists mlm:model as its asset role"
                )

        # case 3: asset name is not given and there is only one mlm:model asset
        if len(model_assets) == 1:
            return next(iter(model_assets.values()))

        # case 4: multiple mlm:model exist but asset_name is not specified
        raise Exception(
            "Multiple assets with role=mlm:model are found in the provided STAC-Item. "
            "Please sepcify which one to use."
        )

    @staticmethod
    def _download_model_http(url: str, target_path: str):
        chunk_size = 8192

        total_size = 0

        # todo: download to disk instead of RAM

        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()

                # Check for Content-Length header to estimate size (optional)
                # content_length = response.headers.get('content-length')
                # if content_length:
                #     total_expected_size = int(content_length)
                #     print(f"Downloading {url}")
                #     print(f"Total expected size: {total_expected_size / (1024 * 1024):.2f} MB")
                # else:
                #     total_expected_size = None
                #     print(f"Downloading {url} (size unknown)")

                with open(target_path, "wb") as f:
                    # Iterate over the response data in chunks
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            # total_size += len(chunk)
                            # # Optional: Print progress
                            # if total_expected_size:
                            #     done = int(50 * total_size / total_expected_size)
                            #     sys.stdout.write(
                            #         f"\r[{'=' * done}{' ' * (50 - done)}] {total_size / (1024 * 1024):.2f} MB / {total_expected_size / (1024 * 1024):.2f} MB")
                            #     sys.stdout.flush()
                            #
                            # else:
                            #     sys.stdout.write(
                            #         f"\rDownloaded: {total_size / (1024 * 1024):.2f} MB")
                            #     sys.stdout.flush()

            print("\nDownload complete.")  # Newline after progress bar

            # IMPORTANT: Reset the stream position to the beginning
            # so it can be read from the start.

        except requests.exceptions.RequestException as e:
            raise Exception(f"\nError downloading {url}: {e}")

        except Exception as e:
            raise Exception(f"\nAn unexpected error occurred: {e}")

    def _download_model_s3(self, url: str, target_path: str):
        #, aws_access_key_id = ..., aws_secret_access_key = ...
        object_path = url[5:].split("/")  # remove s3://, then split by /
        bucket_name = object_path.pop(0)
        object_key = "/".join(object_path)

        try:
            if S3_MODEL_REPO_ACCESS_KEY_ID and S3_MODEL_REPO_SECRET_ACCESS_KEY:
                s3 = boto3.client(
                    's3',
                    aws_access_key_id=S3_MODEL_REPO_ACCESS_KEY_ID,
                    aws_secret_access_key=S3_MODEL_REPO_SECRET_ACCESS_KEY,
                    endpoint_url=S3_MODEL_REPO_ENDPOINT
                )
            else:
                s3 = boto3.client(
                    's3',
                    endpoint_url=S3_MODEL_REPO_ENDPOINT,
                    config=Config(signature_version=UNSIGNED)
                )
            s3.download_file(bucket_name, object_key, target_path)

        except FileNotFoundError:
            raise Exception(
                f"Could not locate file with {object_key=} in {bucket_name=}"
            )

        except botocore.exceptions.ClientError:
            raise Exception(
                f"Error connecting to s3 storage to download model"
            )

    def _download_model(self, url: str, target_path: str):
        protocol = url.split("://")[0]

        # download the model
        if protocol == "http" or protocol == "https":
            self._download_model_http(url, target_path)
        elif protocol == "s3":
            self._download_model_s3(url, target_path)

    def _get_model(self, asset_name=None) -> str:
        model_asset = self._get_model_asset(asset_name)
        url = model_asset.href

        # encode URL to directory name and file name
        model_dir_name = model_cache_utils.url_to_dir_string(url)
        model_file_name = model_cache_utils.url_to_dir_string(url.split("/")[-1])

        model_cache_dir = os.path.join(MODEL_CACHE_DIR, model_dir_name)
        model_cache_file = os.path.join(model_cache_dir, model_file_name)

        # check if model file has been downloaded to cache already
        if os.path.exists(model_cache_file):
            return model_cache_file

        # check if directory exists already in cache and create if not
        if not os.path.exists(model_cache_dir):
            os.mkdir(model_cache_dir)

        self._download_model(url, model_cache_file)
        return model_cache_file

    @abstractmethod
    def create_object(self):
        # convert to object (in derived chilc classes
        pass

    @abstractmethod
    def run_model(self):
        pass
