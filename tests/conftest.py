from datetime import datetime

import pystac
import pytest


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
                [5.8663153, 47.2701114],
            ]
        ],
    }
    return pystac.Item("asdf", geom, None, datetime.now(), {})


@pytest.fixture
def random_asset() -> pystac.Asset:
    return pystac.Asset("https://example.com", "asdf", "asdf", "asdf", [])


@pytest.fixture
def mlm_model_asset(random_asset: pystac.Asset) -> pystac.Asset:
    return pystac.Asset("https://example.com", "model", "foo", "asdf", ["mlm:model"])


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
            "data_type": "float64",
        },
    }
    outp = {
        "name": "classification",
        "tasks": ["classification"],
        "result": {
            "shape": [-1, 1, 1, 1],
            "dim_order": ["batch", "channel", "width", "height"],
            "data_type": "uint8",
        },
    }

    blank_stac_item.properties["mlm:input"] = [inp]
    blank_stac_item.properties["mlm:output"] = [outp]

    mlm_model_asset.href = "https://filesamples.com/samples/font/bin/slick.bin"
    blank_stac_item.add_asset("weights", mlm_model_asset)

    return blank_stac_item
