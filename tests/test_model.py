import pytest

from datathon.features import ARMS, SEGMENTS, segment_of
from datathon.golden_set import GOLDEN_SET
from datathon.model import PosteriorModel


def policy_dict(overrides=None):
    segments = {seg: {"cellular": {"alpha": 30.0, "beta": 70.0},
                      "telephone": {"alpha": 10.0, "beta": 90.0}} for seg in SEGMENTS}
    segments.update(overrides or {})
    return {"algorithm": "discounted_thompson_sampling", "gamma": 0.995,
            "prior": {"alpha": 1.0, "beta": 1.0}, "arms": list(ARMS),
            "segments": segments, "dataset_sha256": "x" * 64, "trained_at": "2026-09-24T00:00:00+00:00"}


def test_estimated_rates_and_exploit_recommendation():
    model = PosteriorModel.from_dict(policy_dict())
    rates = model.estimated_rates("jovem_sem_sucesso")
    assert rates == pytest.approx({"cellular": 0.30, "telephone": 0.10})
    assert model.recommend("jovem_sem_sucesso") == "cellular"


def test_prob_cellular_better_is_deterministic():
    model = PosteriorModel.from_dict(policy_dict())
    p1 = model.prob_cellular_better("adulto_sem_sucesso")
    assert p1 == model.prob_cellular_better("adulto_sem_sucesso")
    assert p1 > 0.99


def test_thompson_mode_reproducible_with_seed():
    uncertain = {"senior_com_sucesso": {"cellular": {"alpha": 2.0, "beta": 2.0},
                                        "telephone": {"alpha": 2.0, "beta": 2.0}}}
    model = PosteriorModel.from_dict(policy_dict(uncertain))
    picks = [model.recommend("senior_com_sucesso", "thompson", seed=s) for s in range(40)]
    assert set(picks) == {"cellular", "telephone"}
    assert model.recommend("senior_com_sucesso", "thompson", seed=3) == picks[3]


def test_from_dict_rejects_missing_segment_or_wrong_arms():
    bad = policy_dict()
    del bad["segments"]["jovem_sem_sucesso"]
    with pytest.raises(ValueError):
        PosteriorModel.from_dict(bad)
    bad = policy_dict()
    bad["arms"] = ["telephone", "cellular"]
    with pytest.raises(ValueError):
        PosteriorModel.from_dict(bad)


def test_invalid_mode():
    with pytest.raises(ValueError):
        PosteriorModel.from_dict(policy_dict()).recommend("jovem_sem_sucesso", "outro")


def test_golden_set_is_consistent():
    assert len(GOLDEN_SET) == 5
    assert len({c["segment"] for c in GOLDEN_SET}) == 5
    for case in GOLDEN_SET:
        assert segment_of(case["age"], case["poutcome"]) == case["segment"]
        assert case["expected_channel"] in (None, *ARMS)
