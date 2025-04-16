
from datetime import datetime
import pytest
import pystac
from openeo_processes_dask_ml.process_implementations.data_model import MLModel


class DummyMLModel(MLModel):
    # Only for testing purposes

    def create_object(self):
        pass

    def run_model(self):
        pass

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
