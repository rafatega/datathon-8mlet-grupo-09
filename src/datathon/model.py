"""Posterior servida pela API (sem pandas/mlflow)."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datathon.features import ARMS, SEGMENTS

MODES = ("explotacao", "thompson")


@dataclass(frozen=True)
class PosteriorModel:
    alphas: dict[str, np.ndarray]
    betas: dict[str, np.ndarray]

    @classmethod
    def from_dict(cls, data: dict) -> "PosteriorModel":
        if tuple(data.get("arms", ())) != ARMS:
            raise ValueError(f"Braços do modelo devem ser {ARMS}")
        missing = set(SEGMENTS) - set(data.get("segments", {}))
        if missing:
            raise ValueError(f"Segmentos ausentes no modelo: {sorted(missing)}")
        alphas, betas = {}, {}
        for seg in SEGMENTS:
            params = data["segments"][seg]
            alphas[seg] = np.array([params[arm]["alpha"] for arm in ARMS], dtype=float)
            betas[seg] = np.array([params[arm]["beta"] for arm in ARMS], dtype=float)
        return cls(alphas=alphas, betas=betas)

    @classmethod
    def load(cls, path) -> "PosteriorModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def estimated_rates(self, segment: str) -> dict[str, float]:
        means = self.alphas[segment] / (self.alphas[segment] + self.betas[segment])
        return {arm: float(m) for arm, m in zip(ARMS, means)}

    def prob_cellular_better(self, segment: str, n_samples: int = 20000) -> float:
        rng = np.random.default_rng(0)
        draws = rng.beta(self.alphas[segment], self.betas[segment], size=(n_samples, len(ARMS)))
        return float((draws[:, 0] > draws[:, 1]).mean())

    def recommend(self, segment: str, mode: str = "explotacao", seed: int | None = None) -> str:
        if mode not in MODES:
            raise ValueError(f"mode deve ser um de {MODES}")
        if mode == "explotacao":
            scores = self.alphas[segment] / (self.alphas[segment] + self.betas[segment])
        else:
            scores = np.random.default_rng(seed).beta(self.alphas[segment], self.betas[segment])
        return ARMS[int(np.argmax(scores))]
