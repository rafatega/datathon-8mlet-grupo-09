import numpy as np
import pytest

from datathon.evaluation import (
    best_arm_share,
    block_sums,
    bootstrap_diff_ci,
    effective_sample_size,
    pseudo_regret,
    replay,
    snips,
)
from datathon.policies import AlternatingAB, FixedArm, Policy


class _Recorder(Policy):
    """Registra a ordem em que recebe atualizações."""

    def __init__(self):
        self.seen = []

    def select(self, segment, rng):
        return 0

    def update(self, segment, arm, reward):
        self.seen.append(reward)


def test_replay_keeps_order_and_skips_mismatches():
    logged = np.array([0, 1, 0, 0, 1])
    rewards = np.array([1, 1, 0, 1, 0])
    rec = _Recorder()
    res = replay(rec, np.zeros(5, int), logged, rewards, np.random.default_rng(0))
    assert res.matched.tolist() == [True, False, True, True, False]
    assert rec.seen == [1, 0, 1]  # só eventos com match, na ordem original
    assert res.chosen.tolist() == [0, 0, 0, 0, 0]


def test_snips_manual_example():
    matched = np.array([True, True, False])
    rewards = np.array([1, 0, 1])
    props = np.array([0.5, 0.25, 0.5])
    # pesos 2 e 4 -> (2*1 + 4*0) / 6
    assert snips(matched, rewards, props) == pytest.approx(1 / 3)
    assert effective_sample_size(matched, props) == pytest.approx(36 / 20)


def test_snips_recovers_true_rate_under_known_propensity():
    rng = np.random.default_rng(0)
    n = 200_000
    blocks = np.repeat(np.arange(4), n // 4)
    p_arm0 = np.array([0.9, 0.2, 0.5, 0.7])[blocks]
    logged = np.where(rng.random(n) < p_arm0, 0, 1)
    true = np.array([[0.30, 0.10], [0.10, 0.05], [0.20, 0.20], [0.40, 0.10]])
    rewards = (rng.random(n) < true[blocks, logged]).astype(int)
    props = np.where(logged == 0, p_arm0, 1 - p_arm0)
    res = replay(FixedArm(0), np.zeros(n, int), logged, rewards, rng)
    assert snips(res.matched, rewards, props) == pytest.approx(0.25, abs=0.01)
    naive = rewards[res.matched].mean()
    assert abs(naive - 0.25) > 0.02  # sem correção, o estimador é enviesado


def test_ab_alternates_under_replay():
    res = replay(AlternatingAB(), np.zeros(6, int), np.zeros(6, int), np.ones(6, int),
                 np.random.default_rng(0))
    assert res.chosen.tolist() == [0, 1, 0, 1, 0, 1]


def test_pseudo_regret_and_best_arm_share():
    rates = np.array([[0.3, 0.1], [0.3, 0.1], [0.1, 0.4]])
    chosen = np.array([0, 1, 0])
    np.testing.assert_allclose(pseudo_regret(chosen, rates), [0.0, 0.2, 0.5])
    assert best_arm_share(chosen, rates) == pytest.approx(1 / 3)
    mask = np.array([False, True, True])
    np.testing.assert_allclose(pseudo_regret(chosen, rates, mask), [0.2, 0.5])


def test_block_sums():
    matched = np.array([True, True, False, True])
    rewards = np.array([1, 0, 1, 1])
    props = np.array([0.5, 0.5, 0.5, 0.25])
    blocks = np.array([7, 7, 8, 8])
    num, den = block_sums(matched, rewards, props, blocks, np.array([7, 8]))
    np.testing.assert_allclose(num, [2.0, 4.0])
    np.testing.assert_allclose(den, [4.0, 4.0])


def test_bootstrap_diff_ci_detects_clear_difference():
    rng = np.random.default_rng(1)
    den = np.full((3, 20), 100.0)
    num_a = rng.normal(30, 2, (3, 20))
    num_b = rng.normal(20, 2, (1, 20))
    out = bootstrap_diff_ci(num_a, den, num_b, den[:1], n_boot=500, seed=0)
    assert out["diff"] == pytest.approx(0.10, abs=0.01)
    assert 0 < out["ci_low"] < out["diff"] < out["ci_high"]
