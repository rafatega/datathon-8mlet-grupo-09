import numpy as np
import pytest

from datathon.policies import (
    AlternatingAB,
    DiscountedEpsilonGreedy,
    DiscountedThompsonSampling,
    FixedArm,
)


def _simulate(policy, rates_by_phase, n_per_phase, seed=0):
    """Bandit online com recompensas sintéticas; retorna os braços escolhidos."""
    rng = np.random.default_rng(seed)
    chosen = []
    for rates in rates_by_phase:
        for _ in range(n_per_phase):
            arm = policy.select(0, rng)
            reward = int(rng.random() < rates[arm])
            policy.update(0, arm, reward)
            chosen.append(arm)
    return np.array(chosen)


def test_fixed_arm():
    rng = np.random.default_rng(0)
    assert all(FixedArm(1).select(3, rng) == 1 for _ in range(10))


def test_alternating_ab_is_deterministic_50_50():
    policy, rng = AlternatingAB(), np.random.default_rng(0)
    assert [policy.select(0, rng) for _ in range(4)] == [0, 1, 0, 1]


def test_beta_update_without_discount():
    ts = DiscountedThompsonSampling(gamma=1.0)
    ts.update(2, 0, 1)
    ts.update(2, 0, 0)
    ts.update(2, 1, 1)
    alpha, beta = ts.posterior_params(2)
    np.testing.assert_allclose(alpha, [2.0, 2.0])
    np.testing.assert_allclose(beta, [2.0, 1.0])
    alpha_other, _ = ts.posterior_params(0)
    np.testing.assert_allclose(alpha_other, [1.0, 1.0])  # outro segmento intocado


def test_discount_applies_only_to_updated_segment():
    ts = DiscountedThompsonSampling(gamma=0.5)
    ts.update(0, 0, 1)   # S[0] = [1, 0]
    ts.update(1, 1, 1)   # não desconta o segmento 0
    ts.update(0, 1, 0)   # S[0] = [0.5, 0], F[0] = [0, 1]
    alpha, beta = ts.posterior_params(0)
    np.testing.assert_allclose(alpha, [1.5, 1.0])
    np.testing.assert_allclose(beta, [1.0, 2.0])


def test_non_contextual_shares_counts():
    ts = DiscountedThompsonSampling(contextual=False)
    ts.update(5, 0, 1)
    alpha, _ = ts.posterior_params(0)
    assert alpha[0] == 2.0


def test_invalid_gamma():
    with pytest.raises(ValueError):
        DiscountedThompsonSampling(gamma=0.0)


def test_epsilon_zero_always_exploits():
    eg = DiscountedEpsilonGreedy(epsilon=0.0)
    eg.update(0, 1, 1)
    rng = np.random.default_rng(0)
    assert all(eg.select(0, rng) == 1 for _ in range(50))


def test_ts_converges_on_stationary_problem():
    chosen = _simulate(DiscountedThompsonSampling(), [(0.30, 0.10)], 3000)
    assert (chosen[-1000:] == 0).mean() > 0.9


def test_discounted_ts_adapts_when_best_arm_changes():
    phases = [(0.6, 0.2), (0.1, 0.6)]
    adaptive = _simulate(DiscountedThompsonSampling(gamma=0.99), phases, 3000)
    static = _simulate(DiscountedThompsonSampling(gamma=1.0), phases, 3000)
    # Measure adaptation speed in first 1000 events of phase 2 (indices 3000-4000), not final convergence.
    # The static TS eventually switches to arm 1; what discounting buys is adaptation speed.
    assert (adaptive[3000:4000] == 1).mean() > 0.8
    assert (static[3000:4000] == 1).mean() < 0.5


def test_seed_reproducibility():
    a = _simulate(DiscountedThompsonSampling(gamma=0.99), [(0.3, 0.2)], 500, seed=7)
    b = _simulate(DiscountedThompsonSampling(gamma=0.99), [(0.3, 0.2)], 500, seed=7)
    np.testing.assert_array_equal(a, b)
