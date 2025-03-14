import re
import requests
import requests.exceptions
from stac_validator.validate import StacValidate


def load_ml_model(uri: str):
    if type(uri) is not str:
        raise ValueError("Type of URI parameter must be a string.")

    # fetch STAC Item
    r = requests.get(uri)
    if r.status_code != 200:
        raise requests.exceptions.HTTPError("Error while fetching STAC Item from URI: "
                                            "Server did not respond with status code 200")

    try:
        stac = r.json()
    except requests.exceptions.JSONDecodeError:
        raise Exception("The provided URI does not point to a valid JSON file")

    # check if downloaded JSON is valid STAC
    stac_validator = StacValidate()
    stac_valid = stac_validator.validate_dict(stac)
    if not stac_valid:
        raise Exception("The provided URI does not point to a valid STAC-Item")

    # check if downloaded JSON is valid STAC Item
    stac_type = stac["type"]
    if stac_type != "Feature":
        raise Exception("The provided URI does not point to a STAC-Item.")

    # Check if downloaded STAC Item implements the STAC:MLM extension
    extensions = stac["stac_extensions"]
    regex = r'^https:\/\/stac-extensions\.github\.io\/mlm\/v(\d+\.){0,2}\d*\/schema\.json$'
    pattern = re.compile(regex)
    follows_mlm = any(pattern.match(s) for s in extensions)
    if not follows_mlm:
        raise Exception("The provided STAC Item does not implement the STAC:MLM standard")
    # todo: validate json against extension schema

    # parse STAC:MLM item

    # Check if model runtime is supported (ONNX!, torch? tf?)

    # download model
    # question: Download here, or when mdoel is actually executed?
    # Answer: Here, so we dont do image preprocessing steps unnecessarily in case the download fails
    # question: where to download the model to?

    # construct model from object

    return uri
