import json

import numpy as np
import pandas as pd
import pytest

from datathon.evaluation import ReplayResult
from datathon.features import ARMS, SEGMENTS
from datathon.model import PosteriorModel
from datathon.train import EvalData, build_policy_artifact, summarize


@pytest.fixture
def small_df():
    return pd.DataFrame({"segment_idx": [0, 0, 5, 5], "arm": [0, 1, 0, 1],
                         "reward": [1, 0, 1, 1]})


def test_policy_artifact_schema_and_loadable(small_df):
    art = build_policy_artifact(small_df, gamma=1.0, sha="a" * 64)
    assert art["algorithm"] == "discounted_thompson_sampling"
    assert art["arms"] == list(ARMS)
    assert set(art["segments"]) == set(SEGMENTS)
    assert art["segments"]["jovem_sem_sucesso"]["cellular"] == {"alpha": 2.0, "beta": 1.0}
    assert art["segments"]["senior_com_sucesso"]["telephone"] == {"alpha": 2.0, "beta": 1.0}
    assert art["dataset_sha256"] == "a" * 64
    PosteriorModel.from_dict(json.loads(json.dumps(art)))


def test_policy_artifact_is_deterministic(small_df):
    a = build_policy_artifact(small_df, 0.995, "a" * 64)
    b = build_policy_artifact(small_df, 0.995, "a" * 64)
    a.pop("trained_at"), b.pop("trained_at")
    assert a == b


def test_summarize_uses_only_test_events():
    data = EvalData(segments=np.zeros(4, int), arms=np.array([0, 0, 1, 1]),
                    rewards=np.array([1, 1, 0, 1]), propensities=np.full(4, 0.5),
                    blocks=np.array([1, 1, 2, 2]),
                    arm_rates=np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
                    test_mask=np.array([False, False, True, True]))
    res = ReplayResult(chosen=np.array([0, 0, 1, 1]), matched=np.array([True, True, True, True]))
    out = summarize([res], data)
    assert out["snips_mean"] == pytest.approx(0.5)
    assert out["accepted_events"] == 2
    assert out["best_arm_share"] == 1.0
    assert out["pseudo_regret_final"] == 0.0
