import pystac
from pystac.extensions.mlm import MLMExtension

from typing import Any
from abc import ABC, abstractmethod
import requests
from io import BytesIO
import sys

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
    def _download_model_http(url) -> BytesIO:
        chunk_size = 8192

        file_in_memory = BytesIO()
        total_size = 0

        try:
            # Use stream=True to avoid loading the whole content into memory at once
            with requests.get(url, stream=True, timeout=30) as response:
                # Raise an exception for bad status codes (4xx or 5xx)
                response.raise_for_status()

                # Check for Content-Length header to estimate size (optional)
                content_length = response.headers.get('content-length')
                if content_length:
                    total_expected_size = int(content_length)
                    print(f"Downloading {url}")
                    print(
                        f"Total expected size: {total_expected_size / (1024 * 1024):.2f} MB")
                else:
                    total_expected_size = None
                    print(f"Downloading {url} (size unknown)")

                # Iterate over the response data in chunks
                for chunk in response.iter_content(chunk_size=chunk_size):

                    if chunk:  # filter out keep-alive new chunks
                        file_in_memory.write(chunk)
                        total_size += len(chunk)
                        # Optional: Print progress
                        if total_expected_size:
                            done = int(50 * total_size / total_expected_size)
                            sys.stdout.write(
                                f"\r[{'=' * done}{' ' * (50 - done)}] {total_size / (1024 * 1024):.2f} MB / {total_expected_size / (1024 * 1024):.2f} MB")
                            sys.stdout.flush()

                        else:
                            sys.stdout.write(
                                f"\rDownloaded: {total_size / (1024 * 1024):.2f} MB")
                            sys.stdout.flush()

            print("\nDownload complete.")  # Newline after progress bar

            # IMPORTANT: Reset the stream position to the beginning
            # so it can be read from the start.
            file_in_memory.seek(0)

        except requests.exceptions.RequestException as e:
            raise Exception(f"\nError downloading {url}: {e}")

        except Exception as e:
            raise Exception(f"\nAn unexpected error occurred: {e}")

        return file_in_memory

    def _download_model_s3(self, url) -> BytesIO:
        pass

    def _download_model(self, asset_name=None):

        model_asset = self._get_model_asset(asset_name)
        url = model_asset.href
        protocol = url.split("://")[0]

        # download the model
        if protocol == "http" or protocol == "https":
            self._download_model_http(url)
        elif protocol == "s3":
            self._download_model_s3(url)

        # save it to disk? Store it in RAM?
        # Both so we can use it immediately and we are able to re-usee the model?
        # todo: use caching!
        # First look if the downloaded asset is already on disk disk and only DL if if is not there?

        # Return the File object (or path)? Or save it in a variable?

    @abstractmethod
    def create_object(self):
        # convert to object (in derived chilc classes
        pass

    @abstractmethod
    def run_model(self):
        pass
