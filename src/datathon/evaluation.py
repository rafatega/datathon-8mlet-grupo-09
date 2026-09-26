"""Avaliação offline: replay cronológico com correção de propensão (SNIPS)."""
from dataclasses import dataclass

import numpy as np

from datathon.policies import Policy


@dataclass
class ReplayResult:
    chosen: np.ndarray   # braço escolhido pela política em cada evento
    matched: np.ndarray  # True quando coincide com o canal logado (recompensa observada)


def replay(policy: Policy, segments, logged_arms, rewards,
           rng: np.random.Generator) -> ReplayResult:
    """Percorre os eventos em ordem; a política só aprende quando escolhe o canal logado."""
    n = len(logged_arms)
    chosen = np.empty(n, dtype=int)
    matched = np.zeros(n, dtype=bool)
    for t in range(n):
        segment = int(segments[t])
        arm = policy.select(segment, rng)
        chosen[t] = arm
        if arm == logged_arms[t]:
            matched[t] = True
            policy.update(segment, arm, int(rewards[t]))
    return ReplayResult(chosen=chosen, matched=matched)


def _mask(matched, mask):
    return matched if mask is None else matched & mask


def snips(matched, rewards, propensities, mask=None) -> float:
    """Conversão estimada da política: média de recompensas ponderada por 1/propensão."""
    m = _mask(matched, mask)
    weights = 1.0 / propensities[m]
    if weights.sum() == 0:
        return float("nan")
    return float((weights * rewards[m]).sum() / weights.sum())


def effective_sample_size(matched, propensities, mask=None) -> float:
    weights = 1.0 / propensities[_mask(matched, mask)]
    if weights.size == 0:
        return 0.0
    return float(weights.sum() ** 2 / (weights ** 2).sum())


def pseudo_regret(chosen, arm_rates, mask=None) -> np.ndarray:
    """Regret acumulado usando a conversão logada de cada braço no bloco do evento."""
    idx = np.arange(len(chosen))
    regret = arm_rates.max(axis=1) - arm_rates[idx, chosen]
    if mask is not None:
        regret = regret[mask]
    return np.cumsum(regret)


def best_arm_share(chosen, arm_rates, mask=None) -> float:
    hits = chosen == arm_rates.argmax(axis=1)
    if mask is not None:
        hits = hits[mask]
    return float(hits.mean())


def block_sums(matched, rewards, propensities, blocks, block_ids, mask=None):
    """Numerador e denominador do SNIPS agregados por bloco."""
    m = _mask(matched, mask)
    weights = np.where(m, 1.0 / propensities, 0.0)
    num = np.array([(weights * rewards)[blocks == b].sum() for b in block_ids])
    den = np.array([weights[blocks == b].sum() for b in block_ids])
    return num, den


def bootstrap_diff_ci(num_a, den_a, num_b, den_b, n_boot=1000, seed=0, level=0.95) -> dict:
    """IC da diferença de SNIPS (A - B) reamostrando blocos; seeds agregadas."""
    rng = np.random.default_rng(seed)
    n_blocks = num_a.shape[1]

    def diff(idx):
        rate_a = num_a[:, idx].sum() / den_a[:, idx].sum()
        rate_b = num_b[:, idx].sum() / den_b[:, idx].sum()
        return rate_a - rate_b

    point = diff(np.arange(n_blocks))
    boots = np.array([diff(rng.integers(0, n_blocks, n_blocks)) for _ in range(n_boot)])
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(boots, [tail, 100 - tail])
    return {"diff": float(point), "ci_low": float(low), "ci_high": float(high)}
