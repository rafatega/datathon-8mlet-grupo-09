import pytest
from fastapi.testclient import TestClient

from datathon import api
from datathon.golden_set import GOLDEN_SET


@pytest.fixture(scope="module")
def client():
    api.get_model.cache_clear()
    return TestClient(api.app)


@pytest.mark.parametrize("case", GOLDEN_SET, ids=lambda c: c["segment"])
def test_golden_set(client, case):
    body = client.post("/recommend", json={"age": case["age"], "poutcome": case["poutcome"]}).json()
    assert body["segment"] == case["segment"]
    if case["expected_channel"] is None:
        assert 0.05 < body["prob_cellular_better"] < 0.95
    else:
        assert body["recommended_channel"] == case["expected_channel"]
