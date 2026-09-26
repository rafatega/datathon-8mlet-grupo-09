"""API de recomendação de canal (FastAPI)."""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from datathon.features import segment_of
from datathon.model import PosteriorModel

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "policy.json"


@lru_cache(maxsize=1)
def get_model() -> PosteriorModel:
    return PosteriorModel.load(Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)))


def _model_or_503() -> PosteriorModel:
    try:
        return get_model()
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=f"Modelo indisponível: {exc}") from exc


class Cliente(BaseModel):
    model_config = ConfigDict(extra="ignore")

    age: int = Field(ge=18, le=100, description="Idade do cliente")
    poutcome: Literal["success", "failure", "nonexistent"] = Field(
        description="Resultado da campanha anterior")
    mode: Literal["explotacao", "thompson"] = "explotacao"
    seed: int | None = None


class Recomendacao(BaseModel):
    segment: str
    recommended_channel: str
    estimated_conversion_rate: dict[str, float]
    prob_cellular_better: float
    mode: str


app = FastAPI(title="Datathon MLET — Recomendação de canal de oferta",
              description="Thompson Sampling contextual com esquecimento (celular × telefone).")


@app.get("/health")
def health() -> dict:
    _model_or_503()
    return {"status": "ok"}


@app.post("/recommend", response_model=Recomendacao)
def recommend(cliente: Cliente) -> Recomendacao:
    model = _model_or_503()
    segment = segment_of(cliente.age, cliente.poutcome)
    return Recomendacao(
        segment=segment,
        recommended_channel=model.recommend(segment, cliente.mode, cliente.seed),
        estimated_conversion_rate=model.estimated_rates(segment),
        prob_cellular_better=model.prob_cellular_better(segment),
        mode=cliente.mode,
    )
