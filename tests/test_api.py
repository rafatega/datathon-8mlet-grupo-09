import json

import pytest
from fastapi.testclient import TestClient

from datathon import api
from datathon.features import ARMS, SEGMENTS, segment_of


def _write_policy(path):
    segments = {seg: {"cellular": {"alpha": 30.0, "beta": 70.0},
                      "telephone": {"alpha": 10.0, "beta": 90.0}} for seg in SEGMENTS}
    segments["senior_com_sucesso"] = {"cellular": {"alpha": 2.0, "beta": 2.0},
                                      "telephone": {"alpha": 2.0, "beta": 2.0}}
    path.write_text(json.dumps({"algorithm": "discounted_thompson_sampling", "gamma": 0.995,
                                "prior": {"alpha": 1.0, "beta": 1.0}, "arms": list(ARMS),
                                "segments": segments, "dataset_sha256": "x" * 64,
                                "trained_at": "2026-09-24T00:00:00+00:00"}), encoding="utf-8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    _write_policy(path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    api.get_model.cache_clear()
    yield TestClient(api.app)
    api.get_model.cache_clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_recommend_valid(client):
    body = client.post("/recommend", json={"age": 25, "poutcome": "nonexistent"}).json()
    assert body["segment"] == "jovem_sem_sucesso"
    assert body["recommended_channel"] == "cellular"
    assert body["estimated_conversion_rate"] == pytest.approx({"cellular": 0.3, "telephone": 0.1})
    assert body["prob_cellular_better"] > 0.99
    assert body["mode"] == "explotacao"


def test_api_uses_shared_segment_function(client):
    for age, pout in [(30, "success"), (31, "failure"), (60, "nonexistent")]:
        body = client.post("/recommend", json={"age": age, "poutcome": pout}).json()
        assert body["segment"] == segment_of(age, pout)


@pytest.mark.parametrize("payload", [
    {"age": 17, "poutcome": "nonexistent"},
    {"age": 101, "poutcome": "nonexistent"},
    {"age": 40, "poutcome": "Success"},
    {"age": 40},
    {"age": "quarenta", "poutcome": "success"},
    {"age": 40, "poutcome": "success", "mode": "aleatorio"},
])
def test_invalid_input_returns_422(client, payload):
    assert client.post("/recommend", json=payload).status_code == 422


def test_extra_fields_are_ignored(client):
    base = client.post("/recommend", json={"age": 45, "poutcome": "success"}).json()
    extra = client.post("/recommend", json={"age": 45, "poutcome": "success",
                                            "duration": 999, "job": "admin."}).json()
    assert base == extra


def test_thompson_mode_reproducible_with_seed(client):
    payload = {"age": 65, "poutcome": "success", "mode": "thompson", "seed": 11}
    first = client.post("/recommend", json=payload).json()
    assert first == client.post("/recommend", json=payload).json()
    assert first["mode"] == "thompson"


def test_missing_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nao_existe.json"))
    api.get_model.cache_clear()
    c = TestClient(api.app)
    assert c.get("/health").status_code == 503
    assert c.post("/recommend", json={"age": 30, "poutcome": "success"}).status_code == 503
    api.get_model.cache_clear()
