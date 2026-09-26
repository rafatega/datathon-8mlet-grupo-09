"""Políticas de decisão: baselines e bandits com esquecimento."""
import numpy as np

from datathon.features import ARMS, SEGMENTS

N_ARMS = len(ARMS)


class Policy:
    def select(self, segment: int, rng: np.random.Generator) -> int:
        raise NotImplementedError

    def update(self, segment: int, arm: int, reward: int) -> None:
        """Baselines não aprendem."""


class FixedArm(Policy):
    """Regra fixa: sempre o mesmo canal."""

    def __init__(self, arm: int):
        self.arm = arm

    def select(self, segment: int, rng: np.random.Generator) -> int:
        return self.arm


class AlternatingAB(Policy):
    """Teste A/B determinístico: alterna os canais evento a evento (50/50)."""

    def __init__(self):
        self._t = 0

    def select(self, segment: int, rng: np.random.Generator) -> int:
        arm = self._t % N_ARMS
        self._t += 1
        return arm


class _DiscountedBeta(Policy):
    """Contagens Beta por segmento x braço com esquecimento por segmento."""

    def __init__(self, gamma: float = 1.0, contextual: bool = True,
                 alpha0: float = 1.0, beta0: float = 1.0):
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma deve estar em (0, 1]")
        self.gamma = gamma
        self.contextual = contextual
        self.alpha0 = alpha0
        self.beta0 = beta0
        n_rows = len(SEGMENTS) if contextual else 1
        self.successes = np.zeros((n_rows, N_ARMS))
        self.failures = np.zeros((n_rows, N_ARMS))

    def _row(self, segment: int) -> int:
        return segment if self.contextual else 0

    def update(self, segment: int, arm: int, reward: int) -> None:
        row = self._row(segment)
        self.successes[row] *= self.gamma
        self.failures[row] *= self.gamma
        self.successes[row, arm] += reward
        self.failures[row, arm] += 1 - reward

    def posterior_params(self, segment: int) -> tuple[np.ndarray, np.ndarray]:
        row = self._row(segment)
        return self.alpha0 + self.successes[row], self.beta0 + self.failures[row]

    def posterior_mean(self, segment: int) -> np.ndarray:
        alpha, beta = self.posterior_params(segment)
        return alpha / (alpha + beta)


class DiscountedThompsonSampling(_DiscountedBeta):
    """Amostra uma taxa de conversão de cada Beta e escolhe o maior valor."""

    def select(self, segment: int, rng: np.random.Generator) -> int:
        alpha, beta = self.posterior_params(segment)
        return int(np.argmax(rng.beta(alpha, beta)))


class DiscountedEpsilonGreedy(_DiscountedBeta):
    """Com probabilidade epsilon explora ao acaso; senão escolhe a maior média."""

    def __init__(self, epsilon: float, gamma: float = 1.0, contextual: bool = True,
                 alpha0: float = 1.0, beta0: float = 1.0):
        super().__init__(gamma, contextual, alpha0, beta0)
        self.epsilon = epsilon

    def select(self, segment: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(N_ARMS))
        return int(np.argmax(self.posterior_mean(segment)))
