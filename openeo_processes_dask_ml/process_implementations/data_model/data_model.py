import os.path

import botocore.exceptions

import pystac
from pystac.extensions.mlm import MLMExtension
from abc import ABC, abstractmethod
import requests
import boto3
from botocore import UNSIGNED
from botocore.config import Config

import xarray as xr

from openeo_processes_dask_ml.process_implementations.constants import (
    MODEL_CACHE_DIR, S3_MODEL_REPO_ENDPOINT, S3_MODEL_REPO_ACCESS_KEY_ID,
    S3_MODEL_REPO_SECRET_ACCESS_KEY
)
from openeo_processes_dask_ml.process_implementations.utils import model_cache_utils

from openeo_processes_dask.process_implementations.exceptions import (
    DimensionMissing, DimensionMismatch
)
from openeo_processes_dask_ml.process_implementations.exceptions import (
    LabelDoesNotExist
)

# todo: replace sys.out.write() and print() with logger actions


class MLModel(ABC):
    stac_item: pystac.Item

    def __init__(self, stac_item: pystac.Item, model_asset_name: str = None):
        self.stac_item = stac_item
        self._model_asset_name = model_asset_name
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

            # print("\nDownload complete.")  # Newline after progress bar

        except requests.exceptions.RequestException as e:
            raise Exception(f"\nError downloading {url}: {e}")

        except Exception as e:
            raise Exception(f"\nAn unexpected error occurred: {e}")

    @staticmethod
    def _download_model_s3(url: str, target_path: str):
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
        model_file_name = model_cache_utils.url_to_dir_string(url.split("/")[-1], True)

        model_cache_dir = os.path.join(MODEL_CACHE_DIR, model_dir_name)
        model_cache_file = os.path.join(model_cache_dir, model_file_name)

        # check if model file has been downloaded to cache already
        if os.path.exists(model_cache_file):
            return model_cache_file

        # check if directory exists already in cache and create if not
        if not os.path.exists(model_cache_dir):
            os.makedirs(model_cache_dir)

        self._download_model(url, model_cache_file)
        return model_cache_file

    def get_datacube_dimension_mapping(self, datacube: xr.DataArray) -> list[None|tuple[str,int]]:
        """
        Maps the model input dimension names to datacube dimension names, as dimension
        names can sometimes differ, e.g. t -> time
        :param datacube: The datacube to map the dimeions agains
        :return: Dict containing the model dimsnion names as keys and datacube
        dimension names as values. Value is None if a match could not be found.
        """
        model_dims = self.model_metadata.input[0].input.dim_order
        dc_dims = datacube.dims

        def get_dc_dim_name(model_dim_name: str) -> str|None:
            if model_dim_name in dc_dims:
                return model_dim_name

            t_dim_names = ["time", "times", "t", "date", "dates", "DATE"]
            if model_dim_name in t_dim_names:
                return next((t_dim for t_dim in t_dim_names if t_dim in dc_dims), None)

            b_dim_names = ["band", "bands", "b", "channel", "channels"]
            if model_dim_name in b_dim_names:
                return next((b_dim for b_dim in b_dim_names if b_dim in dc_dims), None)

            x_dim_names = ["x", "lon", "lng", "longitude"]
            if model_dim_name in x_dim_names:
                return next((x_dim for x_dim in x_dim_names if x_dim in dc_dims), None)

            y_dim_names = ["y", "lat", "latitude"]
            if model_dim_name in y_dim_names:
                return next((y_dim for y_dim in y_dim_names if y_dim in dc_dims), None)

            return None

        dim_mapping = []
        for m_dim_name in model_dims:

            dc_dim_name = get_dc_dim_name(m_dim_name)
            if dc_dim_name is None:
                dim_mapping.append(None)
            else:
                dim_mapping.append((dc_dim_name, dc_dims.index(dc_dim_name)))

        return dim_mapping

    def _check_dimensions_present_in_datacube(
            self, datacube: xr.DataArray, ignore_batch_dim: bool = False
    ) -> None:
        """
        Checkl whether the datacube contains all dimensions required by the model input
        :param datacube: The datacube to be checked
        :param ignore_batch_dim: Ignore a missing "batch" dimension in the datacube
        :raise DimensionMissing: Raised when a dimension requqired by the model input
        is missing
        :return: None
        """

        input_dims = self.model_metadata.input[0].input.dim_order
        dim_mapping = self.get_datacube_dimension_mapping(datacube)

        unmatched_dims = [input_dims[i] for i, d in enumerate(dim_mapping) if d is None]

        # check if all model input dimensions could be matched to dc dimensions
        # ignore batch dimension, we will take care of this later
        if ignore_batch_dim and "batch" in unmatched_dims:
            unmatched_dims.remove("batch")
        if any(unmatched_dims):
            raise DimensionMissing(
                f"Datacube is missing the following dimensions required by the "
                f"STAC-MLM Input: {', '.join(unmatched_dims)}"
            )

    def _check_datacube_dimension_size(
            self, datacube: xr.DataArray, ignore_batch_dim: bool = False
    ) -> None:
        """
        Check whether each datacube dimension is long enough to satisfy the model
        input requriements
        :param datacube: The datacube to be checked
        :param ignore_batch_dim: Ignore a missing "batch" dimension in the datacube
        :raise DimensionMismatch: raised when a datacube dimension has fewer
        coordinates than requried by the model input
        :return: None
        """

        input_dims = self.model_metadata.input[0].input.dim_order
        dim_mapping = self.get_datacube_dimension_mapping(datacube)

        input_shape = self.model_metadata.input[0].input.shape  # e.g. [-1, 3, 128, 128]
        dc_shape = datacube.shape  # e.g. (12, 1000, 1000, 5)

        # reorder dc_shape to match input_shape
        # xor: (a and not b) or (not a and b)
        dc_shape_reorder = [
            dc_shape[dim_mapping[i][1]] for i, inp_dim in enumerate(input_dims)
            if inp_dim != "batch" or (inp_dim == "batch" and not ignore_batch_dim)
        ]

        # ignore "batch" dimension for now, we will take care of that later
        input_shape_reorder = [
            d for i, d in zip(input_dims, input_shape)
            if i != "batch" or (i == "batch" and not ignore_batch_dim)
        ]

        # check whether dc shape is big enough to suffice input:
        # input size must be smaller than dc size in every input dimension
        for dc_dim_size, inp_dim_size in zip(dc_shape_reorder, input_shape_reorder):
            if inp_dim_size == -1:
                # -1 as input shape size means all values are allowed
                # e.g. batch=-1 means the models allows for arbitrary batch size
                continue
            if dc_dim_size >= inp_dim_size:
                continue
            raise DimensionMismatch(
                "The model input requires dimension DIM_NAME to have X values. "
                "The datacube only has Y values for dimension DIM_NAME."
            )

    def _check_datacube_bands(self, datacube: xr.DataArray):
        """
        Checks if the required input bands for the model are present in the provided
        `xarray.DataArray` datacube.
        This function verifies that all bands specified in the `model_metadata.input`
        are available within the `datacube`. It handles cases where bands might be
        directly present or need to be computed from other bands.
        :param datacube: The input data as an `xarray.DataArray`, expected to contain
        geospatial and spectral data.
        :raises DimensionMissing: If a 'bands' dimension is required by the model but
        not found in the datacube, or if the dimension is named unconventionally.
        :raises ValueError: If a band definition in `model_metadata` has either
        `format` or `expression` but not both, when a computation is expected.
        :raises LabelDoesNotExist: If any required band (that cannot be computed) is
        missing from the datacube.
        """
        input_bands = self.model_metadata.input[0].bands

        # bands prorety not utilized, list is empty
        if not input_bands:
            return

        # possibilities how the "bands" dimension could be called
        band_dim_options = ["band", "bands", "b", "channel", "channels"]
        band_dim_name = None
        for b in band_dim_options:
            if b not in datacube.dims:
                continue
            band_dim_name = b

        if not band_dim_name:
            raise DimensionMissing(
                f"The loaded model requires a bands dimension in its input, but none "
                f"was found. If this is a mistake, please rename the bands dimension "
                f"to one of the following: {', '.join(band_dim_options)}"
            )

        dc_bands = datacube.coords[band_dim_name]
        band_available_in_datacube: list[bool] = []
        bands_unavailable: list[str] = []
        for band in input_bands:
            if isinstance(band, str):
                if band in dc_bands:
                    band_available_in_datacube.append(True)
                else:
                    band_available_in_datacube.append(False)
                    bands_unavailable.append(band)

            else:
                # this means type(band) must be ModelInput
                band_name = band.name

                if band_name in dc_bands:
                    band_available_in_datacube.append(True)
                    continue

                # two possibilities here:
                # 1) band not in datacube 2) band must be computed via expression
                if band.format is None and band.expression is None:
                    band_available_in_datacube.append(False)
                    bands_unavailable.append(band_name)
                    continue

                if (
                        (band.format is None and band.expression is not None) or
                        (band.format is not None and band.expression is None)
                ):
                    raise ValueError(
                        f"Properties \"format\" and \"expression\" are both required,"
                        f"but only one was given for band with name {band_name}."
                    )

                # if execution gets up to here, it means that bands either available,
                # or may be computed.
                # todo: Check if bands involved in computation are available
                # todo: check if computation is viable

                # if execution of code gets all the way here, this means that the band
                # is unavailable in the datacube, but can computed from other bands
                band_available_in_datacube.append(True)

        if not all(band_available_in_datacube):
            raise LabelDoesNotExist(
                f"The following bands are unavailable in the datacube, but are "
                f"required in the model input: {', '.join(bands_unavailable)}"
            )

    def check_datacube_dimensions(
            self, datacube: xr.DataArray, ignore_batch_dim: bool = False
    ) -> None:
        """
        Check whether the datacube has all dimensions which the model requires.
        :param datacube: The datacube to check
        :param ignore_batch_dim: Ignore a missing "batch" dimension in the datacube
        :raise DimensionMissing: When the stac:mlm item requires an input dimension
        that is not present in the datacube
        :raise DimensionMismatch: When a dimension is smaller in the datacube than
        required by the stac:mlm input shape
        :return:
        """

        self._check_dimensions_present_in_datacube(datacube, ignore_batch_dim)
        self._check_datacube_dimension_size(datacube, ignore_batch_dim)
        self._check_datacube_bands(datacube)

    def run_model(self, datacube: xr.DataArray) -> xr.DataArray:
        self.check_datacube_dimensions(datacube)

        # todo: was tun wenn DC extra dimensionen hat? Öfters anwenden entlang der dimension?
        # todo: z.b. model hat x,y; cube hat x,y,t: Anwenden für jeden Zeitschritt

        self.check_datacube_dimensions(datacube)

        if self._model_object is None:
            self.create_object()

        pre_datacube = self.preprocess_datacube(datacube)

        # todo: datacube rechunk?
        # todo: datacube in Einzelteile hacken, zu batches zusammenfassen

        b = None  # dummy variable for batches, will be properly filled later
        result = self.execute_model(b)

        post_cube = self.postprocess_datacube(result)

        return post_cube

    def preprocess_datacube(self, datacube: xr.DataArray) -> xr.DataArray:
        # todo: datacube compute new bands?
        # todo: datacube value scaling
        # todo: datacube value preprocessing
        # todo: datacube padding?
        pass

    def postprocess_datacube(self, result_cube) -> xr.DataArray:
        # todo: output gemäß mlm-spec post-processing transformieren
        # todo: output zu neuem datacube zusammenführen
        pass

    def create_object(self):
        if self._model_object is not None:
            # model object has already been created
            return

        model_filepath = self._get_model(self._model_asset_name)
        self.create_model_object(model_filepath)

    @abstractmethod
    def create_model_object(self, filepath: str):
        pass

    @abstractmethod
    def execute_model(self, batch):
        pass
